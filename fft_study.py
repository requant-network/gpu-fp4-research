"""Complete native window/FFT/filter/inverse experiment on one SM120 device."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from fft_inputs import METHODS, PROFILES, SHAPES, inputs
from fft_native import Fft, decode_spectrum, library
from fft_reference import pipeline_reference
from fft_validation import check_output, quality, tensor_stages
from reference import digest

ROOT = Path(__file__).resolve().parent
SOURCES = ("fft_study.py", "fft_native.py", "fft_inputs.py", "fft_validation.py", "fft_reference.py", "fft_coefficients.py",
           "native/fft_bridge.cu", "native/fft_dx.cuh", "build_fft.sh", "native/spmm_bridge.cu", "spmm_preflight.py",
           "spmm_reference.py", "solver_reference.py", "reference.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(a, b):
    return np.array_equal(a, b, equal_nan=True)


def environment(lib):
    lib.mm_environment.restype = ct.c_char_p
    native = lib.mm_environment()
    if native is None:
        raise RuntimeError("Native environment query failed")
    result = json.loads(native)
    result.update(numpy_version=np.__version__, gpu_index=0, gpu_exclusive=False,
                  nvcc=subprocess.check_output(["/usr/local/cuda-13.3/bin/nvcc", "--version"], text=True).strip(),
                  nvidia_smi_before=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"], text=True).strip())
    return result


def run(args):
    target = ROOT/"results"/f"{args.output}.json"
    if target.exists():
        raise FileExistsError(f"Refusing to replace an existing executed record: {target.name}")
    lib = library()
    dependency = ROOT/"results/fft-dx-dependency.json"
    if not dependency.is_file():
        raise FileNotFoundError("Run prepare_fft_dx.py before the FFT experiment")
    record = {"schema_version": 1, "study": 9, "complete": False, "seed": args.seed, "quick": args.quick,
              "reverse": args.reverse, "environment": environment(lib),
              "source_sha256": {name: sha(ROOT/name) for name in SOURCES},
              "binary_sha256": sha(ROOT/"build/libfft.so"),
              "dependency_sha256": sha(dependency), "dependency_file": dependency.name,
              "protocol": {"warmups": 5, "observations": 30, "host_variants": 4,
                           "input": "Stored FP32 complex pairs; stored FP32 windows and complex responses",
                           "device": "Complete window, forward FFT, filter, inverse FFT and normalization; input encoding/packing included",
                           "host": "Pageable FP32 H2D, complete uncaptured native pipeline, full FP32 D2H and host output allocation",
                           "setup": "Hot-process CPU coefficient loading, native planning/allocation/upload/tuning and graph construction recorded separately",
                           "diagnostics": "Separate untimed spectral pass; actual timed outputs read before diagnostics and compared exactly including nonfinite patterns",
                           "replay": "Full runtime quality on every distinct host output; retained full small/selected outputs, samples elsewhere; full runtime stage checks and selected-column witnesses for noise cases"},
              "groups": []}
    shapes = ((128, 1), (1000, 1), (16384, 16)) if args.quick else SHAPES
    profiles = ("noise", "weak120", "highrange") if args.quick else PROFILES
    grid = [(n, batch, profile) for n, batch in shapes for profile in profiles]
    if args.reverse:
        grid.reverse()
    methods = tuple(reversed(METHODS)) if args.reverse else METHODS

    def save():
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temporary.replace(target)

    save()
    for gi, (n, batch, profile) in enumerate(grid):
        fixture = ROOT/"fixtures/fft"/f"coefficients-{n}.npz"
        started = time.perf_counter_ns()
        with np.load(fixture, allow_pickle=False) as z:
            coefficients = {key: z[key] for key in z.files}
        coefficient_load_ms = (time.perf_counter_ns()-started)/1e6
        started = time.perf_counter_ns()
        generated = [inputs(n, batch, profile, args.seed, variant, coefficients) for variant in range(4)]
        input_generation_ms = (time.perf_counter_ns()-started)/1e6
        values = [item[0] for item in generated]
        window, response, parameters = generated[0][1:]
        references = [pipeline_reference(x, window, response) for x in values]
        input_norms = [float(np.linalg.norm(x.astype(np.complex128)*window)) for x in values]
        group = {"n": n, "batch": batch, "profile": profile, "parameters": parameters,
                 "coefficient_file": fixture.relative_to(ROOT).as_posix(), "coefficient_sha256": sha(fixture),
                 "coefficient_load_ms": coefficient_load_ms, "input_generation_ms_common": input_generation_ms,
                 "input_sha256": [digest(x) for x in values], "window_sha256": digest(window), "response_sha256": digest(response),
                 "cases": []}
        arrays = {}
        sample = np.unique(np.concatenate([np.linspace(0, n*batch-1, min(64, n*batch), dtype=np.int64),
                                           np.arange(batch)*n+parameters["weak_bin"] if parameters["weak_bin"] is not None else np.empty(0, dtype=np.int64)]))
        arrays["sample_indices"] = sample
        for mi, method in enumerate(methods):
            case = {"method": method, "available": False}
            group["cases"].append(case)
            plan = None
            started = time.perf_counter_ns()
            try:
                plan = Fft(lib, values[0], window, response, method, coefficients)
            except (RuntimeError, ValueError) as error:
                case["plan_setup_ms"] = (time.perf_counter_ns()-started)/1e6
                case["error"] = str(error)
                continue
            case["plan_setup_ms"] = (time.perf_counter_ns()-started)/1e6
            try:
                case["available"] = True
                meta = plan.metadata()
                case["initial_metadata"] = meta
                expected = {2: values[0], 3: window, 4: response}
                if method["backend"] == 1:
                    expected.update({5: coefficients[f"permutation{method['radix']}"], 6: coefficients[f"phases{method['radix']}"], 7: coefficients["basis"]})
                case["parameter_snapshots_sha256"] = {}
                for which, value in expected.items():
                    actual = plan.snapshot(which)
                    if actual.tobytes() != value.tobytes():
                        raise AssertionError("Uploaded input/coefficient bytes differ")
                    case["parameter_snapshots_sha256"][str(which)] = digest(actual)
                outputs = {}
                case["outputs"] = []

                def observe(output, variant):
                    check_output(output, references[variant][2], method, input_norms[variant])
                    output_sha = digest(output)
                    key = (variant, output_sha)
                    if key in outputs:
                        return outputs[key]["id"]
                    plan.run(True)
                    if not equal(output, plan.result()):
                        raise AssertionError("Diagnostic output differs from the actual observed output")
                    spectrum_precision = method["precision"] if method["backend"] == 0 else 0
                    spectra = [decode_spectrum(plan.snapshot(which), spectrum_precision, output.shape, meta["spectrum_scale"]) for which in (0, 1)]
                    metrics = quality(output, *spectra, references[variant], parameters, input_norms[variant])
                    prefix = f"m{mi}_v{variant}"
                    full = ((n <= 512 and batch == 1)
                            or (profile in ("noise", "weak120") and batch == 1 and method["name"] in ("cufft-fp64", "matrix-r16-bf16", "cufftdx-fp32-pack1"))
                            or (n == 16384 and batch == 16 and profile == "noise" and method["name"] in ("cufft-fp64", "matrix-r16-bf16", "cufftdx-fp32-pack1")))
                    for label, actual in (("output", output), ("spectrum", spectra[0]), ("filtered", spectra[1])):
                        arrays[prefix+"_"+label] = actual.copy() if full else actual.ravel()[sample].copy()
                    result = {"id": len(case["outputs"]), "variant": variant, "prefix": prefix, "full": full,
                              "output_sha256": output_sha, "spectrum_sha256": digest(spectra[0]), "filtered_sha256": digest(spectra[1]),
                              "quality": metrics, "checked_output_entries": output.size}
                    case["outputs"].append(result)
                    outputs[key] = result
                    return result["id"]

                plan.run()
                original = plan.result()
                case["initial_output"] = observe(original, 0)
                if method["backend"] == 1 and profile == "noise":
                    case["stage_checks"] = tensor_stages(plan, method, coefficients, arrays, f"m{mi}_stage")
                    plan.run()
                    if not equal(original, plan.result()):
                        raise AssertionError("Stage diagnostics changed the native complete pipeline")
                repetitions = 1 if n*batch >= 262144 else 8
                started = time.perf_counter_ns()
                plan.call("ff_capture", repetitions)
                case["graph_setup_ms"] = (time.perf_counter_ns()-started)/1e6
                case["boundaries"] = {}
                for captured, label in ((1, "captured_complete"), (0, "uncaptured_complete")):
                    for _ in range(5):
                        plan.call("ff_time", captured)
                    observations = []
                    for _ in range(30):
                        plan.call("ff_time", captured)
                        current = plan.metadata()
                        observations.append({"cuda_ms": current["cuda_ms"], "wall_ms": current["wall_ms"]})
                    actual = plan.result()
                    if not equal(actual, original):
                        raise AssertionError("Actual timed device output differs from checked input-zero output")
                    case["boundaries"][label] = {"observations": observations, "actual_output_sha256": digest(actual),
                                                 "output": case["initial_output"], "repetitions": repetitions if captured else 1}
                for i in range(5):
                    plan.load(values[i % 4])
                    plan.run()
                host = []
                for i in range(30):
                    variant = i % 4
                    started = time.perf_counter_ns()
                    plan.load(values[variant])
                    plan.call("ff_time", 0)
                    actual = plan.result()
                    elapsed = (time.perf_counter_ns()-started)/1e6
                    current = plan.metadata()
                    output_id = observe(actual, variant)
                    host.append({"variant": variant, "wall_ms": elapsed, "cuda_ms": current["cuda_ms"], "output": output_id,
                                 "actual_output_sha256": digest(actual)})
                case["host_requests"] = host
                for which, value in expected.items():
                    expected_value = values[29 % 4] if which == 2 else value
                    if plan.snapshot(which).tobytes() != expected_value.tobytes():
                        raise AssertionError("Final native input/parameter bytes changed unexpectedly")
                case["final_metadata"] = plan.metadata()
            finally:
                plan.close()
        witness = f"{args.output}-n{n}-b{batch}-{profile}.npz"
        path = ROOT/"results"/witness
        if path.exists():
            raise FileExistsError("Refusing to replace an executed FFT witness")
        np.savez_compressed(path, **arrays)
        group.update(witness_file=witness, witness_sha256=sha(path))
        record["groups"].append(group)
        save()
        print(f"{gi+1}/{len(grid)} n={n} batch={batch} {profile}: {sum(c['available'] for c in group['cases'])}/{len(methods)} available", flush=True)
    record["environment"]["nvidia_smi_after"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"], text=True).strip()
    record["complete"] = True
    save()
    print("Complete native FFT campaign finished", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if not args.output or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.output):
        parser.error("Output must be a lowercase alphanumeric basename with optional hyphens")
    run(args)


if __name__ == "__main__":
    main()
