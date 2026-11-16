"""Actual cuSPARSE precision/layout/algorithm probe, not a completed SpMM study."""
import ctypes as ct
import json
from pathlib import Path
import time

import numpy as np

from reference import digest
from solver_reference import read_matrix_market
from spmm_reference import csr

ROOT = Path(__file__).resolve().parent
SOURCES = ("spmm_preflight.py", "spmm_reference.py", "solver_reference.py", "reference.py",
           "native/spmm_preflight.cu", "build_spmm_preflight.sh")


def encode(values, precision):
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Finite FP32 inputs required")
    if precision == 0:
        return values.copy(), values.copy()
    if precision == 1:
        encoded = values.astype(np.float16)
        return encoded, encoded.astype(np.float32)
    if precision != 2:
        raise ValueError("Unknown precision")
    bits = values.view(np.uint32)
    encoded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return encoded, np.left_shift(encoded.astype(np.uint32), 16).view(np.float32)


def inputs(name, width):
    rng = np.random.Generator(np.random.PCG64(20261125 + width))
    if name == "constructed":
        a = rng.integers(-16, 17, (128, 128)).astype(np.float32) / 16
        a[rng.integers(0, 16, a.shape) != 0] = 0
        a[0] = 0
    else:
        a = read_matrix_market(ROOT / f"fixtures/{name}.mtx.gz").astype(np.float32)
        a = np.ldexp(a, -int(np.frexp(abs(a).max())[1]))
    b = rng.integers(-128, 129, (a.shape[1], width)).astype(np.float32) / 128
    return a, b


def main():
    import hashlib
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    lib = ct.CDLL(str(ROOT / "build/libspmm-preflight.so"))
    lib.mmp_new.argtypes = [ct.c_int] * 8 + [ct.c_void_p] * 4
    lib.mmp_new.restype = ct.c_void_p
    lib.mmp_error.restype = ct.c_char_p
    lib.mmp_delete.argtypes = lib.mmp_run.argtypes = [ct.c_void_p]
    lib.mmp_output.argtypes = lib.mmp_update_b.argtypes = [ct.c_void_p, ct.c_void_p]
    lib.mmp_metadata.argtypes = [ct.c_void_p]
    lib.mmp_metadata.restype = ct.c_char_p
    def call(name, pointer, *args):
        if getattr(lib, name)(pointer, *args):
            raise RuntimeError(lib.mmp_error().decode())
    record = {"study": 8, "complete": False,
              "scope": "Capability/reference probe only: three FP32-input matrices, two dense widths, CSR algorithms 1/2/3, FP32/FP16/BF16 inputs with FP32 compute/output, matching row/column-major B/C, optional preprocessing. No hybrid kernels, full conversion/host costs or study completion are claimed.",
              "gpu_index": 0, "gpu_exclusive": False,
              "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
              "data_sha256": {f"fixtures/{name}.mtx.gz": sha(ROOT / f"fixtures/{name}.mtx.gz") for name in ("bcsstk09", "bcsstk10")},
              "binary_sha256": sha(ROOT / "build/libspmm-preflight.so"), "groups": []}
    for name in ("constructed", "bcsstk09", "bcsstk10"):
        for width in (16, 64):
            a, b = inputs(name, width)
            sparse = csr(a)
            original = a.astype(np.float64) @ b.astype(np.float64)
            group = {"matrix": name, "width": width, "shape": list(a.shape), "a_sha256": digest(a), "b_sha256": digest(b),
                     "nnz": len(sparse["values"]), "cases": [], "outputs": {}}
            arrays, lookup = {}, {}
            for precision in (0, 1, 2):
                av, decoded_a = encode(sparse["values"], precision)
                dense_a = np.zeros_like(a)
                for row in range(len(a)):
                    start, end = sparse["rowptr"][row:row+2]
                    dense_a[row, sparse["columns"][start:end]] = decoded_a[start:end]
                for row_major in (0, 1):
                    encoded_b, decoded_b = encode(b, precision)
                    encoded_b = np.array(encoded_b, order="C" if row_major else "F")
                    gold = dense_a.astype(np.float64) @ decoded_b.astype(np.float64)
                    magnitude = np.linalg.norm(abs(dense_a.astype(np.float64)) @ abs(decoded_b.astype(np.float64)))
                    maximum_nnz = int(np.diff(sparse["rowptr"]).max())
                    gamma = maximum_nnz * np.finfo(np.float32).eps / (1 - maximum_nnz * np.finfo(np.float32).eps)
                    bound = 4 * gamma * magnitude + np.finfo(np.float32).eps * np.linalg.norm(gold)
                    for algorithm in (4, 6, 12):
                        for preprocess in (0, 1):
                            case = {"precision": precision, "algorithm": algorithm, "row_major": bool(row_major), "preprocess": bool(preprocess), "available": False}
                            start = time.perf_counter_ns()
                            pointer = lib.mmp_new(*a.shape, width, len(av), precision, algorithm, row_major, preprocess,
                                                  sparse["rowptr"].ctypes.data, sparse["columns"].ctypes.data, av.ctypes.data, encoded_b.ctypes.data)
                            case["plan_setup_wall_ms"] = (time.perf_counter_ns() - start) / 1e6
                            if not pointer:
                                case["error"] = lib.mmp_error().decode()
                                group["cases"].append(case)
                                continue
                            try:
                                call("mmp_run", pointer)
                                observations = []
                                for observation in range(5):
                                    call("mmp_run", pointer)
                                    output = np.empty((len(a), width), dtype=np.float32, order="C" if row_major else "F")
                                    call("mmp_output", pointer, output.ctypes.data)
                                    meta = json.loads(lib.mmp_metadata(pointer))
                                    difference = float(np.linalg.norm(output.astype(np.float64) - gold))
                                    assert np.isfinite(output).all() and difference <= bound, (name, precision, algorithm, difference, bound)
                                    key = digest(output)
                                    if key not in lookup:
                                        label = f"output_{len(arrays):04d}"
                                        lookup[key] = label
                                        arrays[label] = output.copy()
                                        group["outputs"][label] = {"sha256": key, "precision": precision, "variant": 0,
                                                                  "relative_error_original": float(np.linalg.norm(output - original) / np.linalg.norm(original)),
                                                                  "error_quantized_gold": difference, "arithmetic_bound": float(bound)}
                                    observations.append({"metadata": meta, "output": lookup[key]})
                                case["observations"] = observations
                                changed, changed_decoded = encode(-b, precision)
                                changed = np.array(changed, order="C" if row_major else "F")
                                call("mmp_update_b", pointer, changed.ctypes.data)
                                call("mmp_run", pointer)
                                call("mmp_output", pointer, output.ctypes.data)
                                difference = float(np.linalg.norm(output.astype(np.float64) + gold))
                                assert np.isfinite(output).all() and difference <= bound
                                key = digest(output)
                                if key not in lookup:
                                    label = f"output_{len(arrays):04d}"
                                    lookup[key] = label
                                    arrays[label] = output.copy()
                                    group["outputs"][label] = {"sha256": key, "precision": precision, "variant": 1,
                                                              "relative_error_original": float(np.linalg.norm(output + original) / np.linalg.norm(original)),
                                                              "error_quantized_gold": difference, "arithmetic_bound": float(bound)}
                                case["changed_b_output"] = lookup[key]
                                case["available"] = True
                            finally:
                                lib.mmp_delete(pointer)
                            group["cases"].append(case)
            file = f"spmm-preflight-{name}-{width}.npz"
            np.savez_compressed(ROOT / "results" / file, **arrays)
            group["witness_file"], group["witness_sha256"] = file, sha(ROOT / "results" / file)
            record["groups"].append(group)
            print(name, width, sum(c["available"] for c in group["cases"]), "/", len(group["cases"]), flush=True)
    record["complete"] = True
    (ROOT / "results/spmm-preflight.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("SpMM capability probe complete", flush=True)


if __name__ == "__main__":
    main()
