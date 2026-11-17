"""Complete native search adapter including static compaction and GPU global IDs."""
import ctypes as ct
import json
from pathlib import Path
import time

import numpy as np

from search_native import Search,validate


def library():
    import libcuvs
    libcuvs.load_library()
    lib = ct.CDLL(str(Path(__file__).resolve().parent/"build/libsearch-full.so"))
    lib.sq_new.argtypes,lib.sq_new.restype = [ct.c_int]*8+[ct.c_void_p]*3,ct.c_void_p
    lib.sq_delete.argtypes = lib.sq_run.argtypes = [ct.c_void_p]
    lib.sq_delete.restype = None
    lib.sq_load.argtypes = [ct.c_void_p,ct.c_void_p]
    lib.sq_snapshot.argtypes = [ct.c_void_p,ct.c_int,ct.c_void_p]
    lib.sq_metadata.argtypes,lib.sq_metadata.restype = [ct.c_void_p],ct.c_char_p
    lib.sq_error.restype = ct.c_char_p
    lib.sf_wrap.argtypes,lib.sf_wrap.restype = [ct.c_void_p,ct.c_void_p],ct.c_void_p
    lib.sf_delete.argtypes,lib.sf_delete.restype = [ct.c_void_p],None
    lib.sf_run.argtypes = [ct.c_void_p]
    lib.sf_time.argtypes = lib.sf_capture.argtypes = [ct.c_void_p,ct.c_int]
    lib.sf_output.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_void_p]
    lib.sf_gather.argtypes = [ct.c_void_p,ct.c_int,ct.c_size_t,ct.c_size_t,ct.c_void_p,ct.c_void_p]
    lib.sf_metadata.argtypes,lib.sf_metadata.restype = [ct.c_void_p],ct.c_char_p
    return lib


class Native:
    def __init__(self,lib,database,queries,allowed,method,k=10):
        validate(database,database.shape,np.float32)
        validate(queries,queries.shape,np.float32)
        validate(allowed,(len(database),),np.bool_)
        self.base = self.wrapper = None
        self.lib,self.method,self.query_shape,self.k = lib,method,queries.shape,k
        started = time.perf_counter_ns()
        self.mapping = np.flatnonzero(allowed) if method.get("compact",False) else None
        source = np.ascontiguousarray(database[self.mapping]) if self.mapping is not None else database
        padded = max(128,(len(source)+127)//128*128)
        if padded == len(source):
            self.local_database = source
        else:
            self.local_database = np.zeros((padded,database.shape[1]),dtype=np.float32)
            self.local_database[:len(source)] = source
        self.local_allowed = np.zeros(padded,dtype=bool)
        self.local_allowed[:len(source)] = True if self.mapping is not None else allowed
        self.device_mapping = None
        if self.mapping is not None:
            self.device_mapping = np.full(padded,-1,dtype=np.int64)
            self.device_mapping[:len(source)] = self.mapping
        self.preparation_ms = (time.perf_counter_ns()-started)/1e6
        try:
            self.base = Search(lib,self.local_database,queries,self.local_allowed,method["precision"],method.get("candidates",k),k,method.get("norm_mode",1),method.get("refine",True))
            self.wrapper = lib.sf_wrap(self.base.pointer,None if self.device_mapping is None else self.device_mapping.ctypes.data)
            if not self.wrapper:
                raise RuntimeError(lib.sq_error().decode())
        except Exception:
            self.close()
            raise

    def call(self,name,*args):
        if getattr(self.lib,name)(self.wrapper,*args):
            raise RuntimeError(self.lib.sq_error().decode())

    def run(self):
        self.call("sf_run")

    def load(self,queries):
        self.base.load(queries)

    def capture(self,repetitions):
        self.call("sf_capture",repetitions)

    def time(self,captured):
        self.call("sf_time",int(captured))

    def result(self):
        distances = np.empty((self.query_shape[0],self.k),dtype=np.float32)
        ids = np.empty((self.query_shape[0],self.k),dtype=np.int64)
        self.call("sf_output",distances.ctypes.data,ids.ctypes.data)
        return distances,ids

    def gather(self,which,indices,dtype):
        indices = np.ascontiguousarray(indices,dtype=np.uint64)
        dtype = np.dtype(dtype)
        if indices.ndim != 1 or not len(indices) or dtype.itemsize not in (1,2,4,8):
            raise ValueError("Expected nonempty scalar snapshot indices")
        offsets = indices*np.uint64(dtype.itemsize)
        output = np.empty(len(indices),dtype=dtype)
        self.call("sf_gather",which,len(indices),dtype.itemsize,offsets.ctypes.data,output.ctypes.data)
        return output

    def metadata(self):
        raw = self.lib.sf_metadata(self.wrapper)
        if raw is None:
            raise RuntimeError(self.lib.sq_error().decode())
        record = json.loads(raw)
        record.update(cuda_ms=record["native"]["cuda_ms"],wall_ms=record["native"]["wall_ms"],
                      cpu_compaction_padding_ms=self.preparation_ms,padded_database_shape=list(self.local_database.shape),
                      host_prepared_database_bytes=self.local_database.nbytes)
        record["owned_device_bytes"] = record["native"]["owned_device_bytes"]+record["mapping_owned_bytes"]
        return record

    def close(self):
        if self.wrapper:
            self.lib.sf_delete(self.wrapper)
            self.wrapper = None
        if self.base is not None:
            self.base.close()
            self.base = None
