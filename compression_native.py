"""Owned CPU encoder plus stream-borrowing CUDA ndzip decoder. Use a context manager."""
import ctypes as C
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


class NdzipPlan:
    def __init__(self, shape, dtype, threads=1, stream=None):
        import cupy as cp
        self.shape = tuple(shape)
        assert 1 <= len(self.shape) <= 3
        assert all(isinstance(n, (int, np.integer)) and 0 < n < 2**32 for n in self.shape)
        assert int(np.prod(self.shape, dtype=np.uint64)) <= 2**32 - 1
        self.dtype = np.dtype(dtype)
        assert self.dtype in (np.dtype('<f4'), np.dtype('<f8'))
        assert isinstance(threads, int) and threads > 0
        self.word_dtype = np.dtype('<u4' if self.dtype.itemsize == 4 else '<u8')
        self.stream = stream if stream is not None else cp.cuda.get_current_stream()
        self.device = cp.cuda.Device().id
        self._retained = []
        lib = self.lib = C.CDLL(str(ROOT / 'build/libcompression_ndzip.so'))
        lib.ndz_error.restype = C.c_char_p
        lib.ndz_create.argtypes = [C.c_int, C.c_int, C.c_void_p, C.c_uint, C.c_void_p]
        lib.ndz_create.restype = C.c_void_p
        lib.ndz_bound.argtypes = [C.c_void_p]; lib.ndz_bound.restype = C.c_uint64
        lib.ndz_encode.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint64, C.POINTER(C.c_uint64)]
        lib.ndz_encode.restype = C.c_int
        lib.ndz_decode.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]; lib.ndz_decode.restype = C.c_int
        lib.ndz_free.argtypes = [C.c_void_p]; lib.ndz_free.restype = C.c_int
        sizes = np.asarray(self.shape, np.uint32)
        self.handle = lib.ndz_create(self.dtype.itemsize, len(self.shape), sizes.ctypes.data, threads, self.stream.ptr)
        if not self.handle:
            raise RuntimeError(lib.ndz_error().decode())
        self.bound_words = int(lib.ndz_bound(self.handle))
        if not self.bound_words:
            error = lib.ndz_error().decode(); lib.ndz_free(self.handle); self.handle = None
            raise RuntimeError(error)

    def _check(self, status):
        if status:
            raise RuntimeError(self.lib.ndz_error().decode())

    def encode(self, original):
        assert self.handle and isinstance(original, np.ndarray)
        assert original.dtype == self.dtype and original.shape == self.shape and original.flags.c_contiguous
        output = np.empty(self.bound_words, self.word_dtype)
        length = C.c_uint64()
        self._check(self.lib.ndz_encode(self.handle, original.ctypes.data, output.ctypes.data,
                                        self.bound_words, C.byref(length)))
        assert 0 < length.value <= self.bound_words
        return output[:length.value]

    def decode(self, encoded, output):
        import cupy as cp
        assert self.handle and isinstance(encoded, cp.ndarray) and isinstance(output, cp.ndarray)
        assert encoded.dtype == self.word_dtype and encoded.flags.c_contiguous
        assert 0 < encoded.size <= self.bound_words
        assert output.dtype == self.dtype and output.shape == self.shape and output.flags.c_contiguous
        assert encoded.device.id == output.device.id == self.device == cp.cuda.Device().id
        assert cp.cuda.get_current_stream().ptr == self.stream.ptr, 'Use the plan CUDA stream'
        self._retained.append((encoded, output))
        self._check(self.lib.ndz_decode(self.handle, encoded.data.ptr, output.data.ptr))

    def synchronize(self):
        self.stream.synchronize()
        self._retained.clear()

    def close(self):
        if self.handle:
            try:
                self.synchronize()
            finally:
                handle, self.handle = self.handle, None
                self._check(self.lib.ndz_free(handle))

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
