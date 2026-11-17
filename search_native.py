"""Validated host binding for native quantization/GEMM/select/refine search."""
import ctypes as ct
import json
from pathlib import Path

import numpy as np


def library():
    import libcuvs
    libcuvs.load_library()
    lib = ct.CDLL(str(Path(__file__).resolve().parent/"build/libsearch.so"))
    lib.sq_new.argtypes = [ct.c_int]*8+[ct.c_void_p]*3
    lib.sq_new.restype = ct.c_void_p
    lib.sq_delete.argtypes = lib.sq_run.argtypes = [ct.c_void_p]
    lib.sq_load.argtypes = [ct.c_void_p,ct.c_void_p]
    lib.sq_time.argtypes = lib.sq_capture.argtypes = [ct.c_void_p,ct.c_int]
    lib.sq_snapshot.argtypes = [ct.c_void_p,ct.c_int,ct.c_void_p]
    lib.sq_error.restype = ct.c_char_p
    lib.sq_metadata.argtypes = [ct.c_void_p]
    lib.sq_metadata.restype = ct.c_char_p
    return lib


def validate(values, shape, dtype):
    if values.shape != shape or values.dtype != np.dtype(dtype) or not values.flags.c_contiguous:
        raise ValueError(f"Expected contiguous {dtype} values with shape {shape}")


class Search:
    def __init__(self,lib,database,queries,allowed,precision,candidates=64,k=10,norm_mode=1,refine=True):
        if database.ndim != 2 or queries.ndim != 2 or queries.shape[1] != database.shape[1]:
            raise ValueError("Invalid database/query matrix dimensions")
        validate(database,database.shape,np.float32)
        validate(queries,queries.shape,np.float32)
        validate(allowed,(len(database),),np.bool_)
        if not np.isfinite(database).all() or not np.isfinite(queries).all():
            raise ValueError("Finite FP32 vectors are required")
        self.lib,self.shape,self.query_shape = lib,database.shape,queries.shape
        self.k,self.count = k,candidates
        self.pointer = lib.sq_new(*database.shape,len(queries),precision,candidates,k,norm_mode,int(refine),database.ctypes.data,queries.ctypes.data,allowed.ctypes.data)
        if not self.pointer:
            raise RuntimeError(lib.sq_error().decode())

    def call(self,name,*args):
        if getattr(self.lib,name)(self.pointer,*args):
            raise RuntimeError(self.lib.sq_error().decode())

    def close(self):
        if self.pointer:
            self.lib.sq_delete(self.pointer)
            self.pointer = None

    def load(self,queries):
        validate(queries,self.query_shape,np.float32)
        if not np.isfinite(queries).all():
            raise ValueError("Finite queries are required")
        self.call("sq_load",queries.ctypes.data)

    def run(self):
        self.call("sq_run")

    def metadata(self):
        raw = self.lib.sq_metadata(self.pointer)
        if raw is None:
            raise RuntimeError(self.lib.sq_error().decode())
        return json.loads(raw)

    def snapshot(self,which):
        if which not in range(16):
            raise ValueError("Invalid search snapshot selector")
        output = np.empty(self.metadata()["buffer_bytes"][which],dtype=np.uint8)
        self.call("sq_snapshot",which,output.ctypes.data)
        return output

    def result(self):
        return (self.snapshot(14).view(np.float32).reshape(self.query_shape[0],self.k),
                self.snapshot(13).view(np.int64).reshape(self.query_shape[0],self.k))
