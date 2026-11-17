"""Ungrouped short-q cuBLASLt: no artificial 128-wide contraction padding."""
import ctypes as ct
import hashlib
from pathlib import Path
import cupy as cp
import numpy as np
from pde_cuda import PdeCUDA

ROOT=Path(__file__).resolve().parent
MODES=('fp4','fp8','fp16','bf16','fp32')
_MODULE=None


def library():
    lib=ct.CDLL(str(ROOT/'build/libpde_small.so'))
    specs={'pde_small_create':([ct.c_int,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p],ct.c_void_p),
           'pde_small_destroy':([ct.c_void_p],None),'pde_small_error':([],ct.c_char_p),
           'pde_small_prepare':([ct.c_void_p,ct.c_int,ct.c_void_p],ct.c_int),
           'pde_small_run':([ct.c_void_p,ct.c_int,ct.c_void_p],ct.c_int),
           'pde_small_candidates':([ct.c_void_p],ct.c_int),
           'pde_small_flags':([ct.c_void_p,ct.c_int],ct.c_uint64),
           'pde_small_id':([ct.c_void_p,ct.c_int],ct.c_int),
           'pde_small_buffer':([ct.c_void_p,ct.c_int],ct.c_void_p),
           'pde_small_bytes':([ct.c_void_p,ct.c_int],ct.c_size_t),
           'pde_small_owned_bytes':([ct.c_void_p],ct.c_size_t),
           'pde_small_cublas_version':([],ct.c_size_t)}
    for name,(args,result) in specs.items():
        function=getattr(lib,name);function.argtypes=args;function.restype=result
    return lib


def module():
    global _MODULE
    if _MODULE is None:_MODULE=cp.RawModule(code=(ROOT/'native/pde_small_kernels.cu').read_text(),options=('--std=c++17',))
    return _MODULE


class PdeSmall(PdeCUDA):
    def __init__(self,mesh,mode,threads,reaction=0.):
        if mode not in MODES[1:]:raise ValueError('Short-q reference supports FP8/FP16/BF16/pedantic-FP32')
        self.pointer=None
        self.lib=library()
        super().__init__(mesh,'fp64',threads,reaction)
        self.mode,self.algorithm,self.tuning=mode,0,[]
        self.pad=16 if mode=='fp8' else max(8,mesh.q) if mode in ('fp16','bf16') else mesh.q
        self.columns=3*mesh.elements**3*mesh.q**2
        try:
            self.pack_kernel=module().get_function('pde_small_pack')
            self.unweight_kernel=module().get_function('pde_small_unweight')
            left=np.zeros((self.pad,self.pad),np.float32);left[:mesh.q,:mesh.q]=mesh.stiffness.astype(np.float32)
            with self.stream:
                self.left=cp.asarray(left);self.right=cp.zeros((self.columns,self.pad),cp.float32)
                self.product=cp.empty((self.pad,self.columns),cp.float32)
            self.stream.synchronize()
            self.pointer=self.lib.pde_small_create(MODES.index(mode),self.pad,self.columns,self.left.data.ptr,self.right.data.ptr,self.product.data.ptr)
            if not self.pointer:raise RuntimeError(self.lib.pde_small_error().decode())
            self.checked(self.lib.pde_small_prepare(self.pointer,1,self.stream.ptr));self.stream.synchronize()
        except Exception:self.close();raise

    def checked(self,status):
        if status:raise RuntimeError(self.lib.pde_small_error().decode())

    def pack(self):
        m=self.mesh
        with self.stream:
            self.pack_kernel(((self.right.size+self.threads-1)//self.threads,),(self.threads,),
                             (self.data,self.indices,self.boundary,self.right,np.int32(m.elements**3),np.int32(m.q),np.int32(self.pad)))

    def calibrate(self):
        self.pack();self.checked(self.lib.pde_small_prepare(self.pointer,2,self.stream.ptr))
        self.tuning=[];available=[]
        with self.stream:
            for candidate in range(self.lib.pde_small_candidates(self.pointer)):
                flags=int(self.lib.pde_small_flags(self.pointer,candidate))
                if self.mode in ('fp16','bf16') and not flags&2:
                    self.tuning.append(dict(candidate=candidate,flags=flags,available=False));continue
                times=[]
                for _ in range(3):
                    start,end=cp.cuda.Event(),cp.cuda.Event();start.record(self.stream)
                    self.checked(self.lib.pde_small_run(self.pointer,candidate,self.stream.ptr))
                    end.record(self.stream);end.synchronize();times.append(float(cp.cuda.get_elapsed_time(start,end)))
                row=dict(candidate=candidate,flags=flags,algorithm_id=int(self.lib.pde_small_id(self.pointer,candidate)),available=True,cuda_ms=times)
                self.tuning.append(row);available.append(row)
        if not available:raise RuntimeError('No measured eligible short-q cuBLAS candidate')
        self.algorithm=min(available,key=lambda row:float(np.median(row['cuda_ms'])))['candidate']

    def run(self):
        assert not self.closed and self.pointer
        m=self.mesh
        with self.stream:
            self.pack();self.checked(self.lib.pde_small_prepare(self.pointer,2,self.stream.ptr))
            self.checked(self.lib.pde_small_run(self.pointer,self.algorithm,self.stream.ptr))
            self.unweight_kernel(((self.local.size+self.threads-1)//self.threads,),(self.threads,),
                                 (self.product,self.weights,self.local,np.int32(m.elements**3),np.int32(m.q),np.float64(m.h*.5)))
            self.assembly_kernel(((m.size+self.threads-1)//self.threads,),(self.threads,),
                                 (self.local,self.data,self.mass,self.boundary,self.output,np.int32(m.size),np.int32(m.points_per_axis),
                                  np.int32(m.elements),np.int32(m.q),np.float64(self.reaction)))
        cp.cuda.runtime.deviceSynchronize()

    def raw_buffer(self,index):
        result=np.empty(self.lib.pde_small_bytes(self.pointer,index),np.uint8)
        if len(result):cp.cuda.runtime.memcpy(result.ctypes.data,self.lib.pde_small_buffer(self.pointer,index),len(result),cp.cuda.runtime.memcpyDeviceToHost)
        return result

    def metadata(self):
        result:dict=super().metadata();result.pop('local_kernel_attributes')
        extras={n:getattr(self,n).nbytes for n in ('left','right','product')}
        result['named_device_buffers'].update(extras)
        python=result['named_owned_device_bytes']+sum(extras.values());native=int(self.lib.pde_small_owned_bytes(self.pointer))
        result.update(family='short-q-cublas',component_precision=self.mode,product_storage='fp32',post_arithmetic='fp64',
                      pad=self.pad,columns=self.columns,selected_candidate=self.algorithm,selected_flags=int(self.lib.pde_small_flags(self.pointer,self.algorithm)),
                      tuning=self.tuning,python_named_owned_device_bytes=python,native_owned_device_bytes=native,named_owned_device_bytes=python+native,
                      compiled_cublas_version=int(self.lib.pde_small_cublas_version()),native_binary_sha256=hashlib.sha256((ROOT/'build/libpde_small.so').read_bytes()).hexdigest(),
                      pack_kernel_attributes=self.pack_kernel.attributes,unweight_kernel_attributes=self.unweight_kernel.attributes,
                      completion='Changing-original direct short-q packing, encoding, cuBLASLt, FP64 all-axis weighting, deterministic assembly and current-context acknowledgment; API/flags not an instruction trace')
        return result

    def close(self):
        if self.pointer:self.stream.synchronize();self.lib.pde_small_destroy(self.pointer);self.pointer=None
        for n in ('left','right','product'):setattr(self,n,None)
        super().close()
