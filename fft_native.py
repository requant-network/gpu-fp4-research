"""Validated NumPy/ctypes binding for the complete native FFT pipelines."""
import ctypes as ct
import json
from pathlib import Path

import numpy as np


def library():
    lib = ct.CDLL(str(Path(__file__).resolve().parent / "build/libfft.so"))
    lib.ff_new.argtypes = [ct.c_int] * 8 + [ct.c_void_p] * 6
    lib.ff_new.restype = ct.c_void_p
    lib.ff_error.restype = ct.c_char_p
    lib.ff_delete.argtypes = [ct.c_void_p]
    for name in ("ff_load", "ff_output"):
        getattr(lib, name).argtypes = [ct.c_void_p, ct.c_void_p]
    for name in ("ff_run", "ff_capture", "ff_time", "ff_tc_begin"):
        getattr(lib, name).argtypes = [ct.c_void_p, ct.c_int]
    lib.ff_tc_step.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
    lib.ff_snapshot.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]
    lib.ff_metadata.argtypes = [ct.c_void_p]
    lib.ff_metadata.restype = ct.c_char_p
    return lib


def require(array, dtype, shape):
    if array.dtype != np.dtype(dtype) or array.shape != shape or not array.flags.c_contiguous:
        raise ValueError(f"Expected a contiguous {dtype} array of shape {shape}")


class Fft:
    def __init__(self, lib, values, window, response, method, coefficients):
        if values.ndim != 2:
            raise ValueError("Expected batch-by-length inputs")
        self.shape = values.shape
        batch, n = self.shape
        require(values, np.complex64, self.shape)
        require(window, np.float32, (n,))
        require(response, np.complex64, (n,))
        radix = method.get("radix", 16)
        perm = coefficients.get(f"permutation{radix}", np.empty(0, dtype=np.int32))
        phases = coefficients.get(f"phases{radix}", np.empty(0, dtype=np.complex64))
        basis = coefficients["basis"]
        require(perm, np.int32, perm.shape)
        require(phases, np.complex64, phases.shape)
        require(basis, np.float32, (3584,))
        if method["backend"] == 1 and perm.shape != (n,):
            raise ValueError("Tensor FFT requires a complete mixed-radix permutation")
        self.lib = lib
        self.pointer = lib.ff_new(n, batch, method["backend"], method["precision"], radix,
                                  method.get("pack", 1), phases.size, basis.size,
                                  *[a.ctypes.data for a in (values, window, response, perm, phases, basis)])
        if not self.pointer:
            raise RuntimeError(lib.ff_error().decode())

    def call(self, name, *args):
        if getattr(self.lib, name)(self.pointer, *args):
            raise RuntimeError(self.lib.ff_error().decode())

    def close(self):
        if self.pointer:
            self.lib.ff_delete(self.pointer)
            self.pointer = None

    def load(self, values):
        require(values, np.complex64, self.shape)
        self.call("ff_load", values.ctypes.data)

    def run(self, diagnostic=False):
        self.call("ff_run", int(diagnostic))

    def metadata(self):
        result = self.lib.ff_metadata(self.pointer)
        if result is None:
            raise RuntimeError(self.lib.ff_error().decode())
        return json.loads(result)

    def result(self):
        result = np.empty(self.shape, dtype=np.complex64)
        self.call("ff_output", result.ctypes.data)
        return result

    def snapshot(self, which):
        if which not in range(13):
            raise ValueError("Invalid FFT snapshot selection")
        result = np.empty(self.metadata()["buffer_bytes"][which], dtype=np.uint8)
        self.call("ff_snapshot", which, result.ctypes.data)
        return result

    def spectrum(self, filtered=False):
        raw = self.snapshot(int(filtered))
        element = self.metadata()["spectrum_element_bytes"]
        if element == 16:
            result = raw.view(np.complex128)
        elif element == 8:
            result = raw.view(np.complex64).astype(np.complex128)
        else:
            # The half/BF16 cuFFT format is selected by the constructor method.
            raise ValueError("Decode low-precision spectra with decode_spectrum")
        return result.reshape(self.shape) * self.metadata()["spectrum_scale"]


def decode_spectrum(raw, precision, shape, scale=1):
    if precision in (0, 1):
        result = raw.view(np.complex64 if precision == 0 else np.complex128).reshape(shape).astype(np.complex128)
        return result if scale == 1 else result * scale
    values = raw.view(np.float16).astype(np.float32) if precision == 2 else (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    result = values.reshape(*shape, 2).astype(np.float64).view(np.complex128).reshape(shape)
    return result if scale == 1 else result * scale
