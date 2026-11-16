"""Hybrid Tensor Core blocks versus native CSR/BSR/ELL and dense references."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from reference import digest
from solver_reference import read_matrix_market
from spmm_formats import prepare, reconstruct_format
from spmm_native import Spmm, library
from spmm_reference import csr
from spmm_validation import expected_buffers, hybrid_product, metrics, sparse_product, whole_product

ROOT = Path(__file__).resolve().parent
SOURCES = ("spmm_study.py", "spmm_native.py", "spmm_formats.py", "spmm_validation.py", "spmm_reference.py",
           "spmm_preflight.py", "solver_reference.py", "reference.py", "native/spmm_bridge.cu", "build_spmm.sh")
METHODS = [
    *[{"name": f"csr_{label}_a{algorithm_name}_{layout_name}", "kind": 0, "precision": precision, "algorithm": algorithm, "layout": layout}
      for precision, label in ((0, "fp32"), (1, "fp16"), (2, "bf16"))
      for algorithm_name, algorithm, layout_name, layout in ((1, 4, "col", 1), (2, 6, "row", 0), (3, 12, "row", 0))],
    *[{"name": f"{label}_b{block}_{precision_name}", "kind": kind, "block": block, "precision": precision, "algorithm": algorithm}
      for kind, label, algorithm in ((2, "bsr", 14), (3, "ell", 13)) for block in (16, 32, 64)
      for precision, precision_name in ((0, "fp32"), (1, "fp16"), (2, "bf16"))],
    *[{"name": f"dense_{name}", "kind": 4, "precision": precision} for precision, name in ((0, "fp32"), (1, "fp16"), (2, "bf16"))],
    *[{"name": f"hybrid_b{block}_f{int(fill*100)}_{name}", "kind": 1, "block": block, "fill": fill, "precision": precision, "algorithm": 6}
      for block in (16, 32, 64) for fill in (.25, .5, .75) for precision, name in ((1, "fp16"), (2, "bf16"))]
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def groups(quick=False):
    if quick:
        return [{"kind": "synthetic", "m": 128, "k": 128, "profile": "planted", "width": 64, "order": "natural"},
                {"kind": "synthetic", "m": 193, "k": 257, "profile": "mixed", "width": 64, "order": "scrambled"},
                {"kind": "real", "name": "bcsstk09", "width": 64, "order": "natural"},
                {"kind": "real", "name": "bcsstk10", "width": 64, "order": "degree"}]
    matrices = [{"kind": "synthetic", "m": 128, "k": 128, "profile": "planted"},
                {"kind": "synthetic", "m": 1024, "k": 1024, "profile": "mixed"},
                {"kind": "synthetic", "m": 4096, "k": 4096, "profile": "mixed"},
                {"kind": "synthetic", "m": 2048, "k": 2048, "profile": "unstructured"},
                {"kind": "real", "name": "bcsstk09"}, {"kind": "real", "name": "bcsstk10"}]
    result = [dict(matrix, width=width, order=order) for matrix in matrices for width in (64, 256) for order in ("natural", "degree")]
    result += [dict(matrices[1], width=64, order="scrambled"), dict(matrices[4], width=64, order="scrambled")]
    return result


def inputs(specification, seed):
    if specification["kind"] == "real":
        a = read_matrix_market(ROOT / f"fixtures/{specification['name']}.mtx.gz").astype(np.float32)
        a = np.ldexp(a, -int(np.frexp(abs(a).max())[1]))
    else:
        m, k, profile = specification["m"], specification["k"], specification["profile"]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, m, k, 4401])))
        a = np.zeros((m, k), dtype=np.float32)
        if profile == "unstructured":
            a = rng.integers(-16384, 16385, (m, k), dtype=np.int32).astype(np.float32) / 16384
            a[rng.integers(0, 100, (m, k), dtype=np.int32) != 0] = 0
        else:
            for row in range((m + 31) // 32):
                for column in rng.choice((k + 31) // 32, min(4 if profile == "mixed" else 2, (k + 31) // 32), replace=False):
                    height, width = min(32, m - row*32), min(32, k - int(column)*32)
                    tile = rng.integers(-16384, 16385, (height, width), dtype=np.int32).astype(np.float32) / 16384
                    density = 7 if profile == "planted" or row % 2 == 0 else 3
                    tile[rng.integers(0, 8, tile.shape, dtype=np.int32) >= density] = 0
                    a[row*32:row*32+height, int(column)*32:int(column)*32+width] = tile
            noise_count = max(1, int(m*k*.0005))
            locations = rng.integers(0, m*k, noise_count)
            a.flat[locations] = rng.integers(-16384, 16385, noise_count, dtype=np.int32).astype(np.float32) / 16384
        a[0] = 0
    rights = []
    for variant in range(4):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, len(a), specification["width"], variant, 4402])))
        b = rng.integers(-16384, 16385, (a.shape[1], specification["width"]), dtype=np.int32).astype(np.float32) / 16384
        b[:, 0] = 0
        rights.append(b)
    return a, rights


def coordinates(shape):
    m, n = shape
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([m, n, 4403])))
    points = [(0, 0), (m-1, n-1), (0, n-1), (m-1, 0)]
    while len(points) < 64:
        point = int(rng.integers(m)), int(rng.integers(n))
        if point not in points:
            points.append(point)
    return np.array(points, dtype=np.int64)


def keep_full(specification, method):
    if specification["width"] != 64 or specification["order"] != "natural":
        return False
    if specification["kind"] == "synthetic" and specification["m"] == 128:
        return True
    return specification["kind"] == "real" and method["name"] in {
        "csr_fp32_a2_row", "csr_fp16_a2_row", "csr_bf16_a2_row", "bsr_b32_fp32", "bsr_b32_fp16",
        "ell_b32_fp32", "ell_b32_fp16", "ell_b32_bf16", "dense_fp32", "hybrid_b32_f50_fp16", "hybrid_b32_f50_bf16"}


def environment(lib):
    native = lib.mm_environment()
    if native is None:
        raise RuntimeError(lib.mm_error().decode())
    return {"native_runtime": json.loads(native), "gpu_index": 0, "gpu_exclusive": False,
            "python": platform.python_version(), "numpy": np.__version__,
            "nvcc": subprocess.check_output(["/usr/local/cuda-13.3/bin/nvcc", "--version"], text=True),
            "nvidia_smi": subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"], text=True),
            "source_sha256": {name: sha(ROOT / name) for name in SOURCES}, "binary_sha256": sha(ROOT / "build/libspmm.so"),
            "data_sha256": {f"fixtures/{name}.mtx.gz": sha(ROOT / f"fixtures/{name}.mtx.gz") for name in ("bcsstk09", "bcsstk10")}}


def controls(lib):
    a, right = np.zeros((19, 27), dtype=np.float32), np.ones((27, 64), dtype=np.float32)
    result = []
    for kind, precision in ((0, 0), (1, 1), (1, 2)):
        method = {"kind": kind, "precision": precision, "block": 16 if kind else 1, "fill": .5}
        formatted = prepare(csr(a), a.shape, method, "degree", 7)
        plan = Spmm(lib, formatted, right, method)
        try:
            plan.run()
            np.testing.assert_array_equal(plan.result(), 0)
            try:
                plan.load_b(right[:-1].copy())
            except ValueError:
                pass
            else:
                raise AssertionError("Wrong dense input shape accepted")
            plan.call("mm_capture", 2)
            for boundary in range(3):
                plan.call("mm_time", boundary)
                np.testing.assert_array_equal(plan.result(), 0)
            result.append({"kind": kind, "precision": precision, "zero_matrix_graphs_pass": True, "wrong_b_shape_rejected": True})
        finally:
            plan.close()
    if lib.mm_run(None, 0) == 0:
        raise AssertionError("Null native plan accepted")
    result.append({"null_plan_rejected": True, "error": lib.mm_error().decode()})
    return result


def run(args):
    if Path(args.output).name != args.output or not args.output.endswith(".json"):
        raise ValueError("Output must be a JSON filename inside results/")
    lib = library()
    config = {"seed": args.seed, "quick": args.quick, "reverse": args.reverse, "observations": 3 if args.quick else 30,
              "host_observations": 4 if args.quick else 30, "warmups": 1 if args.quick else 5,
              "error_budgets": [1e-5, 1e-3, 1e-2],
              "accuracy_contract": "Full relative Frobenius AND maximum per-column relative L2 error versus original FP32-input FP64 product; exact-zero input column must remain exactly zero. Rounded-input products independently checked at relative error <=2e-5, except approximate FP32-storage ELL whose original-product quality is measured without certifying its opaque quantizer.",
              "boundaries": "Prepared: resident encodings, compute, crop/unpermute/scatter. Online B: additionally encode/gather changing FP32 B. Online both: also re-encode FP32 A format values with unchanged structure. CUDA graphs repeat one operation eight times, or once when m*k*n>=2^30. Host: pageable FP32 B H2D, uncaptured online-B operation, full FP32 C D2H; four B inputs cycle. CPU format conversion, native plan allocation/A upload/preprocessing/tuning and graph construction are separately measured and charged in reuse amortization; they are not inside device timers.",
              "memory_scope": "Explicit owned CUDA allocations, including original/encoded values, gathered B, intermediate C tiles, workspaces and final output; opaque library/graph allocations and other contexts are additional. Not whole-device peak memory.",
              "fp32_ell_scope": "FP32-storage ELL observed to be approximate; not a strict-FP32 arithmetic reference. Actual original-product errors and full/sample output witnesses retained."}
    record = {"study": 8, "complete": False, "configuration": config, **environment(lib), "controls": controls(lib), "groups": []}
    file = ROOT / "results" / args.output
    def checkpoint():
        temporary = file.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(file)
    todo = groups(args.quick)
    if args.reverse:
        todo.reverse()
    for group_index, spec in enumerate(todo):
        a, rights = inputs(spec, args.seed)
        source, shape = csr(a), a.shape
        originals = [sparse_product(source, shape, b) for b in rights]
        coords = coordinates((len(a), spec["width"]))
        group = {"specification": spec, "shape": list(shape), "a_sha256": digest(a), "b_sha256": [digest(b) for b in rights],
                 "csr_sha256": {name: digest(array) for name, array in source.items()}, "nnz": len(source["values"]),
                 "coordinates": coords.tolist(), "formats": {}, "cases": []}
        formats, whole_cache, witness_arrays = {}, {}, {}
        for case_index, method in enumerate(list(reversed(METHODS)) if args.reverse else METHODS):
            format_key = f"k{method['kind']}-b{method.get('block',1)}-f{method.get('fill',0)}"
            if format_key not in formats:
                begin = time.perf_counter_ns()
                formatted = prepare(source, shape, method, spec["order"], args.seed)
                conversion_ms = (time.perf_counter_ns() - begin) / 1e6
                np.testing.assert_array_equal(reconstruct_format(formatted), a)
                formats[format_key] = formatted
                group["formats"][format_key] = {"conversion_wall_ms": conversion_ms, "kind": method["kind"], "block": formatted["block"],
                                                 "padded_shape": list(formatted["padded_shape"]), "block_count": formatted["block_count"], "ell_columns": formatted["ell_columns"],
                                                 "array_sha256": {name: digest(value) for name, value in formatted.items() if isinstance(value, np.ndarray)},
                                                 "host_format_bytes": sum(value.nbytes for value in formatted.values() if isinstance(value, np.ndarray)),
                                                 "lossless_entries_checked": int(a.size)}
            formatted = formats[format_key]
            case = {"method": method, "format": format_key, "available": False, "checks": {}}
            begin = time.perf_counter_ns()
            try:
                plan = Spmm(lib, formatted, rights[0], method)
            except RuntimeError as error:
                case["setup_error"] = str(error)
                group["cases"].append(case)
                print(group_index+1, method["name"], "unavailable", str(error), flush=True)
                continue
            case["plan_setup_wall_ms"] = (time.perf_counter_ns() - begin) / 1e6
            models, seen = {}, {}
            def model(variant):
                if variant not in models:
                    if method["kind"] == 1:
                        models[variant] = hybrid_product(formatted, rights[variant], method["precision"])
                    else:
                        key = method["precision"], variant
                        if key not in whole_cache:
                            whole_cache[key] = whole_product(source, shape, rights[variant], method["precision"])
                        models[variant] = whole_cache[key]
                return models[variant]
            def check(output, variant):
                digest_key = variant, digest(output)
                if digest_key in seen:
                    return seen[digest_key]
                key = f"check_{len(case['checks']):03d}"
                value = {"variant": variant, "output_sha256": digest_key[1],
                         "metrics": metrics(output, originals[variant], model(variant), method["kind"] == 3 and method["precision"] == 0),
                         "samples": output[coords[:, 0], coords[:, 1]].tolist(), "full_output_entries_checked": int(output.size)}
                if keep_full(spec, method):
                    witness_key = f"case_{case_index:03d}_{key}"
                    witness_arrays[witness_key] = output.copy()
                    value["witness_key"] = witness_key
                case["checks"][key], seen[digest_key] = value, key
                return key
            def preparation_check(variant):
                expected = expected_buffers(formatted, rights[variant], method["precision"], method.get("layout", 0))
                actual = plan.snapshot()
                for got, wanted in zip(actual, expected):
                    np.testing.assert_array_equal(got, wanted)
                return {"variant": variant, "bytes_checked": sum(v.size for v in actual), "buffer_sha256": [digest(v) for v in actual]}
            try:
                case["prepared_before"] = preparation_check(0)
                plan.run(2)
                case["before"] = check(plan.result(), 0)
                repetitions = 2 if args.quick else (1 if int(shape[0])*shape[1]*spec["width"] >= 2**30 else 8)
                begin = time.perf_counter_ns()
                plan.call("mm_capture", repetitions)
                case["graph_setup_wall_ms"] = (time.perf_counter_ns() - begin) / 1e6
                case["timings_ms"], case["graph_wall_ms"], case["after_device"] = {}, {}, {}
                for boundary, name in enumerate(("prepared", "online-b", "online-both")):
                    for _ in range(config["warmups"]):
                        plan.call("mm_time", boundary)
                    cuda_values, wall_values = [], []
                    for _ in range(config["observations"]):
                        plan.call("mm_time", boundary)
                        meta = plan.metadata()
                        cuda_values.append(meta["cuda_ms"])
                        wall_values.append(meta["graph_wall_ms"])
                    case["timings_ms"][name], case["graph_wall_ms"][name] = cuda_values, wall_values
                    case["after_device"][name] = check(plan.result(), 0)
                case["prepared_after_device"] = preparation_check(0)
                host = []
                for observation in range(config["host_observations"]):
                    variant = observation % 4
                    begin = time.perf_counter_ns()
                    plan.load_b(rights[variant])
                    plan.run(1)
                    output = plan.result()
                    elapsed_ms = (time.perf_counter_ns() - begin) / 1e6
                    host.append({"variant": variant, "wall_ms": elapsed_ms, "check": check(output, variant)})
                case["host_observations"] = host
                case["prepared_after_host"] = preparation_check((config["host_observations"]-1) % 4)
                case["native"] = plan.metadata()
                case["available"] = True
                group["cases"].append(case)
                print(f"group {group_index+1}/{len(todo)} {spec['order']} {shape}x{spec['width']} {method['name']} "
                      f"error={case['checks'][case['before']]['metrics']['max_relative_column_error']:.5g} "
                      f"online_b_us={np.median(case['timings_ms']['online-b'])*1000:.4f}", flush=True)
            finally:
                plan.close()
        witness_file = f"spmm-{args.seed}-{'quick' if args.quick else 'full'}-group-{group_index:02d}.npz"
        np.savez_compressed(ROOT / "results" / witness_file, **witness_arrays)
        group["witness_file"], group["witness_sha256"] = witness_file, sha(ROOT / "results" / witness_file)
        record["groups"].append(group)
        checkpoint()
    record["complete"] = True
    checkpoint()
    print(f"Complete SpMM campaign: {file.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
