"""Small shared helpers for additive experiments; historical pilot sources stay unchanged."""
import ctypes
import hashlib
import json
from pathlib import Path

import cupy as cp
import numpy as np

from gpu_study import Plan, load_library
from reference import digest, quantize, relative_error

ROOT = Path(__file__).resolve().parent


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_extended(tile=128):
    baseline = load_library()
    library = ctypes.CDLL(str(ROOT / f"build/libextended-{tile}.so"))
    for name in ("rq_error", "rq_create", "rq_destroy", "rq_quantize", "rq_run", "rq_candidates",
                 "rq_algorithm_id", "rq_buffer", "rq_bytes", "rq_cublas_version"):
        function, prototype = getattr(library, name), getattr(baseline, name)
        function.argtypes, function.restype = prototype.argtypes, prototype.restype
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    library.rq_quantize_variant.argtypes = [ptr, integer, integer, ptr]
    library.rq_quantize_variant.restype = integer
    library.rq_prepare_fused.argtypes = [ptr, ptr, ptr, integer, ptr]
    library.rq_prepare_fused.restype = integer
    library.rq_activate.argtypes = [ptr, ptr, ptr, integer, integer, ptr]
    library.rq_activate.restype = integer
    return library


class ExtendedPlan(Plan):
    def __init__(self, library, variant, a, b, stream, bias=None, fused=False, mode="fp4"):
        self.variant, self.bias, self.fused, self.source = variant, bias, fused, a
        actual_input = cp.empty_like(a) if bias is not None else a
        super().__init__(library, mode, actual_input, b, stream)

    def prepare(self, mask=3):
        if self.bias is not None and mask & 1:
            if self.fused:
                self.checked(self.library.rq_prepare_fused(self.pointer, self.source.data.ptr,
                             self.bias.data.ptr, self.variant, self.stream.ptr))
                mask &= ~1
            else:
                self.checked(self.library.rq_activate(self.source.data.ptr, self.bias.data.ptr,
                             self.inputs[0].data.ptr, int(self.source.size), self.source.shape[1], self.stream.ptr))
        if mask:
            self.checked(self.library.rq_quantize_variant(self.pointer, mask, self.variant, self.stream.ptr))


def oracle(a, b):
    encoded, scales, values, hashes = [], [], [], {}
    for operand, original in enumerate((a, b)):
        raw, sf, q = quantize(original, "fp4")
        encoded.append(raw)
        scales.append(sf)
        values.append(q)
        hashes[f"operand_{operand}_encoded"] = digest(raw)
        hashes[f"operand_{operand}_scales"] = digest(sf)
    x, y = (v.astype(np.float64) for v in values)
    expected = x @ y.T
    original = a.astype(np.float64) @ b.astype(np.float64).T
    gamma = a.shape[1] * 2**-24 / (1 - a.shape[1] * 2**-24)
    bound = 8 * gamma * (np.abs(x) @ np.abs(y).T) + np.finfo(np.float32).tiny
    return {"encoded": encoded, "scales": scales, "values": values, "hashes": hashes,
            "expected": expected, "original": original, "bound": bound}


def validate(plan, reference):
    plan.prepare()
    plan.run()
    plan.stream.synchronize()
    for operand in (0, 1):
        np.testing.assert_array_equal(plan.buffer(operand), reference["encoded"][operand].ravel())
        np.testing.assert_array_equal(plan.buffer(operand + 2), reference["scales"][operand].ravel())
    actual = cp.asnumpy(plan.output)
    if not np.isfinite(actual).all():
        raise ValueError("Nonfinite output")
    maximum_ratio = float(np.max(np.abs(actual.astype(np.float64) - reference["expected"]) / reference["bound"]))
    if maximum_ratio > 1:
        raise ValueError(f"Arithmetic bound exceeded: {maximum_ratio}")
    return {"encoded_sha256": reference["hashes"], "arithmetic_max_bound_ratio": maximum_ratio,
            "relative_frobenius_error": relative_error(actual, reference["original"]),
            "maximum_absolute_error": float(np.max(np.abs(actual.astype(np.float64)-reference["original"]))),
            "output_sha256": digest(actual), "checked_output_entries": int(actual.size)}, actual


def write_record(path, record):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
