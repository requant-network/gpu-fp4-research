"""Native cuSOLVER SPD study: fresh solves, online residual gates and factor reuse."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from reference import digest
from solver_reference import (FORWARD_TARGET, RESIDUAL_TARGET, accuracy, controlled,
                              exact_residual_rows, read_matrix_market, real_system)

ROOT = Path(__file__).resolve().parent
SOURCES = ("solver_study.py", "solver_reference.py", "reference.py", "native/solver_bridge.cu", "build_solver.sh")
METHODS = [
    {"name": "cholesky64", "mode": 1},
    {"name": "lu64", "precision": 1203},
    *[{"name": f"fixed_{name}", "precision": precision} for name, precision in
      (("fp32", 1204), ("fp16", 1205), ("bf16", 1206), ("tf32", 1207))],
    *[{"name": f"ir_{name}", "precision": precision, "refinement": 1102} for name, precision in
      (("fp32", 1204), ("fp16", 1205), ("bf16", 1206), ("tf32", 1207))],
    {"name": "gmres_fp16", "precision": 1205, "refinement": 1104},
    {"name": "library_fallback_fp16", "precision": 1205, "refinement": 1102, "fallback": 1},
    {"name": "equilibrated_library_fp16", "precision": 1205, "refinement": 1102, "fallback": 1, "equilibrate": 1},
    {"name": "adaptive", "mode": 2},
    {"name": "equilibrated_adaptive", "mode": 2, "equilibrate": 1},
    {"name": "cholesky64_reuse", "mode": 3},
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library():
    lib = ct.CDLL(str(ROOT / "build/libsolver.so"))
    lib.sp_new.argtypes = [ct.c_int] * 8
    lib.sp_new.restype = ct.c_void_p
    lib.sp_error.restype = ct.c_char_p
    lib.sp_delete.argtypes = [ct.c_void_p]
    lib.sp_load.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p]
    lib.sp_run.argtypes = lib.sp_factor.argtypes = [ct.c_void_p]
    lib.sp_output.argtypes = [ct.c_void_p, ct.c_void_p]
    lib.sp_prepared.argtypes = [ct.c_void_p] * 4
    lib.sp_metadata.argtypes = [ct.c_void_p]
    lib.sp_metadata.restype = ct.c_char_p
    lib.sp_environment.restype = ct.c_char_p
    return lib


class Solver:
    def __init__(self, lib, n, nrhs, method):
        self.lib, self.n, self.nrhs = lib, n, nrhs
        self.output = np.empty((n, nrhs), dtype=np.float64, order="F")
        self.pointer = lib.sp_new(n, nrhs, method.get("mode", 0), method.get("precision", 1203),
                                  method.get("refinement", 1101), method.get("fallback", 0),
                                  method.get("max_iterations", 30), method.get("equilibrate", 0))
        if not self.pointer:
            raise RuntimeError(lib.sp_error().decode())

    def close(self):
        if self.pointer:
            self.lib.sp_delete(self.pointer)
            self.pointer = None

    def call(self, name, *args):
        if getattr(self.lib, name)(self.pointer, *args):
            raise RuntimeError(self.lib.sp_error().decode())

    def load(self, a, b):
        for value in (a, b):
            if value is not None and (value.dtype != np.float64 or not value.flags.f_contiguous):
                raise ValueError("Fortran-contiguous FP64 arrays required")
        self.call("sp_load", None if a is None else a.ctypes.data, None if b is None else b.ctypes.data)

    def run(self):
        self.call("sp_run")

    def result(self):
        self.call("sp_output", self.output.ctypes.data)
        return self.output.copy(order="F")

    def metadata(self):
        value = self.lib.sp_metadata(self.pointer)
        if value is None:
            raise RuntimeError(self.lib.sp_error().decode())
        return json.loads(value)

    def prepared(self):
        a = np.empty((self.n, self.n), order="F")
        b = np.empty((self.n, self.nrhs), order="F")
        scale = np.empty(self.n)
        self.call("sp_prepared", a.ctypes.data, b.ctypes.data, scale.ctypes.data)
        return a, b, scale


def groups(quick=False):
    if quick:
        return [dict(kind="controlled", n=128, exponent=e, nrhs=r) for e in (0, 24, 40) for r in (1, 4)] + [
            dict(kind="real", name="bcsstk09", scale_exponent=0, nrhs=1),
            dict(kind="real", name="bcsstk10", scale_exponent=20, nrhs=4)]
    return [dict(kind="controlled", n=n, exponent=e, nrhs=r)
            for n in (128, 512, 1024, 2048) for e in (0, 12, 24, 36, 40) for r in (1, 4)] + [
        dict(kind="real", name=name, scale_exponent=e, nrhs=r)
        for name in ("bcsstk09", "bcsstk10") for e in (-20, 0, 20) for r in (1, 4)]


def system(group, seed, variant=0):
    if group["kind"] == "controlled":
        return controlled(group["n"], group["exponent"], group["nrhs"], seed, variant)
    a = read_matrix_market(ROOT / f"fixtures/{group['name']}.mtx.gz")
    return real_system(a, group["nrhs"], seed, variant, group["scale_exponent"])


def compact_trace(meta):
    return [[s["requested_precision"], s["status"], s["device_info"], s["iterations"],
             s["gate_pass"], s["max_relative_residual"]] for s in meta["stages"]]


def environment(lib):
    def capture(args):
        return subprocess.check_output(args, text=True).strip()
    native = lib.sp_environment()
    if native is None:
        raise RuntimeError(lib.sp_error().decode())
    return {"python": platform.python_version(), "numpy": np.__version__, "gpu_exclusive": False,
            "native_runtime": json.loads(native),
            "nvcc": capture(["/usr/local/cuda-13.3/bin/nvcc", "--version"]),
            "nvidia_smi": capture(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,compute_cap", "--format=csv,noheader"]),
            "gpu_index": 0, "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
            "binary_sha256": sha(ROOT / "build/libsolver.so"),
            "data_sha256": {name: sha(ROOT / name) for name in
                            ("fixtures/solver-matrices.json", "fixtures/bcsstk09.mtx.gz", "fixtures/bcsstk10.mtx.gz")},
            "blas_threads": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}}


def controls(lib):
    results = []
    for name, dimensions, method in (("zero_dimension", (0, 1), {}), ("invalid_rhs", (128, 2), {}),
                                     ("invalid_mode", (128, 1), {"mode": 8}),
                                     ("gmres_multiple_rhs", (128, 4), {"refinement": 1104})):
        try:
            solver = Solver(lib, *dimensions, method)
        except RuntimeError as error:
            results.append({"control": name, "rejected": True, "error": str(error)})
        else:
            solver.close()
            raise AssertionError(f"Invalid input unexpectedly accepted: {name}")
    if lib.sp_run(None) == 0:
        raise AssertionError("Null plan accepted")
    results.append({"control": "null_plan", "rejected": True, "error": lib.sp_error().decode()})
    solver = Solver(lib, 128, 1, {"mode": 1})
    try:
        a, b = np.eye(128, order="F"), np.ones((128, 1), order="F")
        a[0, 0] = 0
        solver.load(a, b)
        solver.run()
        meta = solver.metadata()
        assert meta["stages"][0]["device_info"] > 0 and not meta["stages"][0]["gate_pass"]
        results.append({"control": "singular_matrix", "metadata": meta, "rejected": True})
    finally:
        solver.close()
    a, b, truth = controlled(128, 12, 1, 714)
    solver = Solver(lib, 128, 1, {"precision": 1204, "refinement": 1102})
    try:
        solver.load(a, b)
        repeated = []
        for _ in range(4):
            solver.run()
            meta = solver.metadata()
            quality = accuracy(a, b, truth, solver.result())
            assert meta["stages"][0]["status"] == 0 and quality["target_pass"]
            repeated.append({"metadata": meta, "accuracy": quality})
        results.append({"control": "repeated_irs_infos_lifecycle", "solves": repeated, "passed": True})
    finally:
        solver.close()
    return results


def run(args):
    lib = library()
    configuration = {"seed": args.seed, "quick": args.quick, "reverse": args.reverse,
                     "observations": 3 if args.quick else 30, "warmups": 1 if args.quick else 5,
                     "residual_target": RESIDUAL_TARGET, "forward_target": FORWARD_TARGET,
                     "adaptive_rule": "FP16 classical IR (4 iterations), then FP32 classical IR (12), then FP64 Cholesky. Each stage solves from original A/B and is gated only by status, device info and max per-RHS original-system relative L2 residual <=1e-12. No forward-error oracle is used online.",
                     "precision_scope": "FP64 inputs/output; requested lowest precision and library iteration/fallback codes are retained. Requested flags do not prove Tensor Core execution of every internal operation.",
                     "boundaries": "Resident: fresh restore/equilibration/factorization/solve and full original-system GPU residual with host decision. CUDA event intervals include IRS host gaps; host wall recorded separately. Host request: pageable FP64 A+B H2D, complete solve/gate, full FP64 X D2H; RHS cycles through four variants. Reuse mode excludes a separately measured FP64 factor setup and uploads only changing B. Plan allocation, library initialization, CPU input generation, witnesses and history extraction are outside solve timings.",
                     "memory_scope": "Explicit owned CUDA buffers/workspaces only. cuSOLVER/cuBLAS handles and opaque internal allocations are additional; these are not whole-device peak measurements."}
    if Path(args.output).name != args.output or not args.output.endswith(".json"):
        raise ValueError("Output must be a JSON filename inside results/")
    record = {"study": 7, "complete": False, "configuration": configuration, **environment(lib),
              "controls": controls(lib), "groups": []}
    filename = ROOT / "results" / args.output
    filename.parent.mkdir(exist_ok=True)
    def checkpoint():
        temporary = filename.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(filename)
    todo = groups(args.quick)
    if args.reverse:
        todo.reverse()
    for group_index, group in enumerate(todo):
        inputs = [system(group, args.seed, variant) for variant in range(4)]
        a, b, truth = inputs[0]
        assert all(np.array_equal(a, value[0]) for value in inputs)
        n, nrhs = b.shape
        rows = np.unique(np.linspace(0, n - 1, 16, dtype=int))
        row = {"specification": group, "n": n, "nrhs": nrhs, "input_sha256": [
            {"a": digest(aa), "b": digest(bb), "truth": digest(xx)} for aa, bb, xx in inputs],
            "exact_rows": rows.tolist(), "solutions": {}, "cases": []}
        arrays, solution_lookup = {}, {}
        def witness(solution, variant):
            key = (variant, digest(solution))
            if key in solution_lookup:
                return solution_lookup[key]
            label = f"solution_{len(arrays):04d}"
            aa, bb, xx = inputs[variant]
            quality = accuracy(aa, bb, xx, solution)
            exact = exact_residual_rows(aa, bb, solution, rows) if np.isfinite(solution).all() else None
            row["solutions"][label] = {"variant": variant, "sha256": key[1], "accuracy": quality,
                                       "exact_residual_samples": exact.tolist() if exact is not None and np.isfinite(exact).all() else None}
            arrays[label] = solution
            solution_lookup[key] = label
            return label
        methods = list(reversed(METHODS)) if args.reverse else METHODS
        for method in methods:
            case = {"method": method, "available": False}
            begin = time.perf_counter_ns()
            try:
                solver = Solver(lib, n, nrhs, method)
            except RuntimeError as error:
                case["setup_error"] = str(error)
                row["cases"].append(case)
                continue
            case["plan_setup_wall_ms"] = (time.perf_counter_ns() - begin) / 1e6
            try:
                solver.load(a, b)
                prepared_a, prepared_b, scales = solver.prepared()
                expected_scale = 1 / np.sqrt(np.diag(a)) if method.get("equilibrate") else np.ones(n)
                np.testing.assert_allclose(scales, expected_scale, rtol=3e-15, atol=0)
                np.testing.assert_allclose(prepared_a, a * expected_scale[:, None] * expected_scale[None, :], rtol=5e-15, atol=0)
                np.testing.assert_allclose(prepared_b, b * expected_scale[:, None], rtol=5e-15, atol=0)
                case["preparation"] = {"a_sha256": digest(prepared_a), "b_sha256": digest(prepared_b),
                                       "scale_sha256": digest(scales), "verified_entries": n * n + n * nrhs + n}
                if method.get("mode") == 3:
                    factor_observations = []
                    for _ in range(configuration["observations"]):
                        begin = time.perf_counter_ns()
                        solver.load(a, b)
                        solver.call("sp_factor")
                        host_ms = (time.perf_counter_ns() - begin) / 1e6
                        meta = solver.metadata()
                        factor_observations.append({"cuda_ms": meta["cuda_ms"], "wall_ms": meta["wall_ms"], "host_ms": host_ms,
                                                    "trace": compact_trace(meta)})
                        assert meta["factor_ready"], "SPD native factor setup failed"
                    case["factor_observations"] = factor_observations
                solver.run()
                case["before"] = {"solution": witness(solver.result(), 0), "metadata": solver.metadata()}
                for _ in range(configuration["warmups"]):
                    solver.run()
                resident = []
                for _ in range(configuration["observations"]):
                    solver.run()
                    solution = solver.result()
                    meta = solver.metadata()
                    resident.append({"cuda_ms": meta["cuda_ms"], "wall_ms": meta["wall_ms"],
                                     "solution": witness(solution, 0), "trace": compact_trace(meta)})
                case["resident_observations"] = resident
                host = []
                for observation in range(configuration["observations"]):
                    variant = observation % 4
                    aa, bb, _ = inputs[variant]
                    begin = time.perf_counter_ns()
                    solver.load(None if method.get("mode") == 3 else aa, bb)
                    solver.run()
                    solution = solver.result()
                    host_ms = (time.perf_counter_ns() - begin) / 1e6
                    meta = solver.metadata()
                    host.append({"cuda_ms": meta["cuda_ms"], "wall_ms": meta["wall_ms"], "host_ms": host_ms,
                                 "variant": variant, "solution": witness(solution, variant), "trace": compact_trace(meta)})
                case["host_observations"] = host
                solver.load(None, b)
                solver.run()
                case["after"] = {"solution": witness(solver.result(), 0), "metadata": solver.metadata()}
                case["available"] = True
                row["cases"].append(case)
                print(f"group {group_index + 1}/{len(todo)} n={n} rhs={nrhs} {method['name']} "
                      f"pass={row['solutions'][case['after']['solution']]['accuracy']['target_pass']} "
                      f"resident_ms={np.median([v['cuda_ms'] for v in resident]):.4f}", flush=True)
            finally:
                solver.close()
        witness_name = f"solver-{args.seed}-{'quick' if args.quick else 'full'}-group-{group_index:02d}.npz"
        np.savez_compressed(ROOT / "results" / witness_name, **arrays)
        row["witness_file"], row["witness_sha256"] = witness_name, sha(ROOT / "results" / witness_name)
        record["groups"].append(row)
        checkpoint()
    record["complete"] = True
    checkpoint()
    print(f"Complete: {filename.name}, {len(record['groups'])} groups", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
