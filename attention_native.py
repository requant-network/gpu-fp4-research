"""Typed native batched CUDA attention bridge for Torch-owned device buffers."""
import ctypes
from pathlib import Path

import numpy as np
import torch

from reference import MODES
from attention_reference import MASKS

ROOT=Path(__file__).resolve().parent


def load():
    library=ctypes.CDLL(str(ROOT/"build/libattention.so"));ptr,integer=ctypes.c_void_p,ctypes.c_int
    library.rq_error.restype=ctypes.c_char_p;library.rq_cublas_version.restype=ctypes.c_size_t
    library.att_create.argtypes=[integer]*7+[ptr]*4;library.att_create.restype=ptr
    library.att_destroy.argtypes=[ptr];library.att_destroy.restype=None
    library.att_prepare.argtypes=[ptr,ptr];library.att_prepare.restype=integer
    for name in ("att_run","att_stage"):getattr(library,name).argtypes=[ptr,integer,integer,ptr];getattr(library,name).restype=integer
    for name in ("att_stage_plan","att_buffer","rq_buffer"):getattr(library,name).argtypes=[ptr,integer];getattr(library,name).restype=ptr
    library.att_encoded_bytes.argtypes=[ptr,integer,integer];library.att_encoded_bytes.restype=ctypes.c_size_t
    library.att_owned_bytes.argtypes=[ptr];library.att_owned_bytes.restype=ctypes.c_size_t
    library.rq_candidates.argtypes=[ptr];library.rq_candidates.restype=integer
    library.rq_algorithm_id.argtypes=[ptr,integer];library.rq_algorithm_id.restype=integer
    runtime=ctypes.CDLL("/usr/local/cuda-13.3/lib64/libcudart.so.13")
    runtime.cudaMemcpy.argtypes=[ptr,ptr,ctypes.c_size_t,integer];runtime.cudaMemcpy.restype=integer
    return library,runtime


class NativeAttention:
    def __init__(self,libraries,q,k,v,modes,mask):
        if q.shape!=k.shape or q.shape!=v.shape or q.ndim!=4 or any(x.dtype!=torch.float32 or not x.is_cuda or not x.is_contiguous() for x in (q,k,v)):raise ValueError("Contiguous CUDA FP32 Q/K/V required")
        self.library,self.runtime=libraries;self.inputs=q,k,v;self.shape=tuple(q.shape);self.modes=modes;self.mask=mask
        self.output=torch.empty_like(q);self.algorithms=[0,0];self.pointer=self.library.att_create(*(MODES.index(mode) for mode in modes),*q.shape,MASKS.index(mask),q.data_ptr(),k.data_ptr(),v.data_ptr(),self.output.data_ptr())
        if not self.pointer:raise RuntimeError(self.library.rq_error().decode())

    def checked(self,status):
        if status:raise RuntimeError(self.library.rq_error().decode())

    def prepare(self):self.checked(self.library.att_prepare(self.pointer,torch.cuda.current_stream().cuda_stream))

    def run(self):self.checked(self.library.att_run(self.pointer,*self.algorithms,torch.cuda.current_stream().cuda_stream));return self.output

    def __call__(self):self.prepare();return self.run()

    def read(self,index,shape):
        output=np.empty(shape,np.float32);pointer=self.library.att_buffer(self.pointer,index)
        if not pointer:raise ValueError("Invalid native attention buffer")
        if self.runtime.cudaMemcpy(output.ctypes.data,pointer,output.nbytes,2):raise RuntimeError("Attention buffer copy failed")
        return output

    def encoded(self,stage,index):
        pointer=self.library.att_stage_plan(self.pointer,stage);size=self.library.att_encoded_bytes(self.pointer,stage,index);output=np.empty(size,np.uint8)
        if size and self.runtime.cudaMemcpy(output.ctypes.data,self.library.rq_buffer(pointer,index),size,2):raise RuntimeError("Attention encoding copy failed")
        return output

    def tune(self):
        self.prepare();self.run();torch.cuda.synchronize();result=[]
        for stage in (0,1):
            pointer=self.library.att_stage_plan(self.pointer,stage);rows=[]
            for candidate in range(self.library.rq_candidates(pointer)):
                def run():self.checked(self.library.att_stage(self.pointer,stage,candidate,torch.cuda.current_stream().cuda_stream))
                for _ in range(2):run()
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(4):run()
                values=[]
                for _ in range(3):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();graph.replay();end.record();end.synchronize();values.append(start.elapsed_time(end)/4)
                rows.append({"candidate":candidate,"algorithm_id":self.library.rq_algorithm_id(pointer,candidate),"ms":values})
            self.algorithms[stage]=min(rows,key=lambda row:np.median(row["ms"]))["candidate"];result.append(rows)
        self.prepare();self.run();torch.cuda.synchronize();return result

    def close(self):
        if self.pointer:self.library.att_destroy(self.pointer);self.pointer=None
