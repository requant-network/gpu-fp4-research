"""Thin native bridge binding; NumPy inputs remain FP32 until timed GPU encoding."""
import ctypes as ct
import json
from pathlib import Path

import numpy as np


def library():
    lib = ct.CDLL(str(Path(__file__).resolve().parent / "build/libspmm.so"))
    lib.mm_new.argtypes = [ct.c_int] * 14 + [ct.c_void_p] * 8
    lib.mm_new.restype = ct.c_void_p
    lib.mm_error.restype = ct.c_char_p
    lib.mm_delete.argtypes = [ct.c_void_p]
    lib.mm_load_b.argtypes = lib.mm_output.argtypes = [ct.c_void_p, ct.c_void_p]
    lib.mm_run.argtypes = lib.mm_capture.argtypes = lib.mm_time.argtypes = [ct.c_void_p, ct.c_int]
    lib.mm_snapshot.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]
    lib.mm_metadata.argtypes = [ct.c_void_p]
    lib.mm_metadata.restype = ct.c_char_p
    lib.mm_environment.restype = ct.c_char_p
    return lib


class Spmm:
    def __init__(self, lib, formatted, right, method):
        if right.dtype != np.float32 or not right.flags.c_contiguous:
            raise ValueError("A row-major FP32 dense input is required")
        self.lib = lib
        m, k = formatted["shape"]
        if right.ndim != 2 or right.shape[0] != k:
            raise ValueError("Invalid dense input shape")
        self.shape = m, right.shape[1]
        self.right_shape = right.shape
        self.pointer = lib.mm_new(m, k, right.shape[1], *formatted["padded_shape"], method["kind"], method["precision"], formatted["block"],
                                  method.get("algorithm", 6), method.get("layout", 0), method.get("preprocess", 1),
                                  len(formatted["values"]) if method["kind"] in (0, 1) else 0,
                                  formatted["block_count"], formatted["ell_columns"],
                                  *[formatted[key].ctypes.data for key in ("rowptr", "columns", "values", "block_rowptr", "block_columns", "block_values", "permutation")],
                                  right.ctypes.data)
        if not self.pointer:
            raise RuntimeError(lib.mm_error().decode())

    def call(self, name, *args):
        if getattr(self.lib, name)(self.pointer, *args):
            raise RuntimeError(self.lib.mm_error().decode())

    def close(self):
        if self.pointer:
            self.lib.mm_delete(self.pointer)
            self.pointer = None

    def load_b(self, right):
        if right.dtype != np.float32 or not right.flags.c_contiguous or right.shape != self.right_shape:
            raise ValueError("Expected a row-major FP32 dense input with the original shape")
        self.call("mm_load_b", right.ctypes.data)

    def run(self, boundary=2):
        self.call("mm_run", boundary)

    def metadata(self):
        result = self.lib.mm_metadata(self.pointer)
        if result is None:
            raise RuntimeError(self.lib.mm_error().decode())
        return json.loads(result)

    def result(self):
        output = np.empty(self.shape, dtype=np.float32)
        self.call("mm_output", output.ctypes.data)
        return output

    def snapshot(self):
        arrays = []
        for which, size in enumerate(self.metadata()["buffer_bytes"]):
            array = np.empty(size, dtype=np.uint8)
            self.call("mm_snapshot", which, array.ctypes.data)
            arrays.append(array)
        return arrays
