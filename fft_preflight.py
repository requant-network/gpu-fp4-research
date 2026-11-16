"""Record native cuFFT format/shape capability and independent FFT/roundtrip checks."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from reference import digest
from spmm_preflight import encode

ROOT = Path(__file__).resolve().parent
SOURCES = ("fft_preflight.py", "native/fft_preflight.cu", "build_fft_preflight.sh", "reference.py",
           "spmm_preflight.py", "spmm_reference.py", "solver_reference.py")


def inputs(n, batch):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261126, n, batch])))
    pairs = rng.integers(-2048, 2049, (batch, n, 2), dtype=np.int32).astype(np.float32) / 2**19
    index = np.arange(n)
    pairs[:, :, 0] += .5 * ((index // 2) % 2 * 2 - 1).astype(np.float32)
    pairs[:, :, 1] += .25 * (index % 2 * 2 - 1).astype(np.float32)
    pairs[:, :, 0] += np.float32(2**-14) * ((index // 4) % 2 * 2 - 1).astype(np.float32)
    return pairs


def complex_values(pairs):
    return pairs[..., 0].astype(np.float64) + 1j * pairs[..., 1].astype(np.float64)


def prepare(pairs, precision):
    if precision == 1:
        values = pairs.astype(np.float64)
        return values, values
    return encode(pairs, {0: 0, 2: 1, 3: 2}[precision])


def decode(raw, precision):
    if precision != 3:
        return raw.astype(np.float64)
    return np.left_shift(raw.astype(np.uint32), 16).view(np.float32).astype(np.float64)


def relative(output, expected):
    return float(np.linalg.norm(output - expected) / np.linalg.norm(expected))


def main():
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    lib = ct.CDLL(str(ROOT / "build/libfft-preflight.so"))
    lib.ffp_new.argtypes = [ct.c_int] * 3 + [ct.c_void_p]
    lib.ffp_new.restype = ct.c_void_p
    lib.ffp_error.restype = ct.c_char_p
    lib.ffp_delete.argtypes = [ct.c_void_p]
    lib.ffp_run.argtypes = [ct.c_void_p, ct.c_int]
    lib.ffp_output.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]
    lib.ffp_metadata.argtypes = [ct.c_void_p]
    lib.ffp_metadata.restype = ct.c_char_p
    def call(name, pointer, *args):
        if getattr(lib, name)(pointer, *args):
            raise RuntimeError(lib.ffp_error().decode())
    record = {"study": 9, "complete": False, "gpu_index": 0, "gpu_exclusive": False,
              "scope": "cuFFT capability/numerical probe only; no complete window/filter/inverse pipeline, Tensor Core FFT implementation, full latency/energy or completed study claim.",
              "timing_scope": "Five raw cuFFT forward and inverse event intervals after one warmup each. Immutable-input D2D restores, host encoding/transfers and CPU verification normalization are excluded; these are not complete pipeline times.",
              "source_sha256": {name: sha(ROOT / name) for name in SOURCES}, "binary_sha256": sha(ROOT / "build/libfft-preflight.so"),
              "nvcc": subprocess.check_output(["/usr/local/cuda-13.3/bin/nvcc", "--version"], text=True), "groups": []}
    for n in (128, 256, 512, 1000, 1024, 4096):
        for batch in (1, 16):
            original = inputs(n, batch)
            gold = np.fft.fft(complex_values(original), axis=-1)
            group = {"n": n, "batch": batch, "original_sha256": digest(original), "cases": []}
            arrays = {"original": original}
            for precision in (0, 1, 2, 3):
                encoded, rounded = prepare(original, precision)
                case = {"precision": precision, "available": False, "encoded_sha256": digest(encoded)}
                begin = time.perf_counter_ns()
                pointer = lib.ffp_new(n, batch, precision, encoded.ctypes.data)
                case["plan_setup_wall_ms"] = (time.perf_counter_ns() - begin) / 1e6
                if not pointer:
                    case["error"] = lib.ffp_error().decode()
                    group["cases"].append(case)
                    continue
                try:
                    call("ffp_run", pointer, -1)
                    forward = []
                    for _ in range(5):
                        call("ffp_run", pointer, -1)
                        forward.append(json.loads(lib.ffp_metadata(pointer))["cuda_ms"])
                    fft_raw = np.empty_like(encoded)
                    call("ffp_output", pointer, 1, fft_raw.ctypes.data)
                    spectrum = complex_values(decode(fft_raw, precision))
                    call("ffp_run", pointer, 1)
                    backward = []
                    for _ in range(5):
                        call("ffp_run", pointer, 1)
                        backward.append(json.loads(lib.ffp_metadata(pointer))["cuda_ms"])
                    inverse_raw, source_raw = np.empty_like(encoded), np.empty_like(encoded)
                    call("ffp_output", pointer, 2, inverse_raw.ctypes.data)
                    call("ffp_output", pointer, 0, source_raw.ctypes.data)
                    np.testing.assert_array_equal(source_raw, encoded)
                    restored = complex_values(decode(inverse_raw, precision)) / n
                    errors = {"forward_vs_original": relative(spectrum, gold),
                              "forward_vs_rounded_input": relative(spectrum, np.fft.fft(complex_values(rounded), axis=-1)),
                              "inverse_vs_actual_forward_spectrum": relative(restored, np.fft.ifft(spectrum, axis=-1)),
                              "roundtrip_vs_original": relative(restored, complex_values(original))}
                    threshold = {0: 2e-5, 1: 1e-12, 2: .01, 3: .05}[precision]
                    assert np.isfinite(spectrum).all() and np.isfinite(restored).all() and max(errors.values()) < threshold, (n, batch, precision, errors)
                    arrays[f"fft_p{precision}"] = fft_raw
                    arrays[f"inverse_p{precision}"] = inverse_raw
                    case.update({"available": True, "forward_ms": forward, "inverse_ms": backward, "errors": errors,
                                 "screening_threshold": threshold, "metadata": json.loads(lib.ffp_metadata(pointer)),
                                 "fft_sha256": digest(fft_raw), "inverse_sha256": digest(inverse_raw)})
                finally:
                    lib.ffp_delete(pointer)
                group["cases"].append(case)
            file = f"fft-preflight-n{n}-b{batch}.npz"
            np.savez_compressed(ROOT / "results" / file, **arrays)
            group["witness_file"], group["witness_sha256"] = file, sha(ROOT / "results" / file)
            record["groups"].append(group)
            print(n, batch, sum(c["available"] for c in group["cases"]), "/4", flush=True)
    record["complete"] = True
    (ROOT / "results/fft-preflight.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("cuFFT capability probe complete", flush=True)


if __name__ == "__main__":
    main()
