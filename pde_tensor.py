"""Complete changing-input grouped native GEMM physical operator.

Static S encoding is setup. Every operation repacks original observations,
re-encodes B, runs the actual product, reweights all axes and assembles globally.
API precision/flags are not a new hardware instruction trace.
"""
import hashlib
from pathlib import Path

import cupy as cp
import numpy as np

from pca_native import library
from pde_cuda import PdeCUDA
from pde_packing import layout,static_operator
from reference import MODES

ROOT=Path(__file__).resolve().parent
_MODULE=None


def tensor_module():
    global _MODULE
    if _MODULE is None:_MODULE=cp.RawModule(code=(ROOT/'native/pde_tensor_kernels.cu').read_text(),options=('--std=c++17',))
    return _MODULE


class PdeTensor(PdeCUDA):
    def __init__(self,mesh,mode,threads,reaction=0.):
        if mode not in MODES:raise ValueError('Unknown bounded native GEMM precision')
        self.pointer=None
        self.lib=library()
        super().__init__(mesh,'fp64',threads,reaction)
        self.mode,self.encoder_variant,self.algorithm=mode,12,0
        self.layout=layout(mesh)
        self.tuning=[]
        try:
            self.pack_kernel=tensor_module().get_function('pde_tensor_pack')
            self.unweight_kernel=tensor_module().get_function('pde_tensor_unweight')
            with self.stream:
                self.left=cp.asarray(static_operator(mesh,np.float32))
                self.right=cp.zeros((self.layout['columns'],128),cp.float32)
                self.product=cp.empty((128,self.layout['columns']),cp.float32)
            self.stream.synchronize()
            self.pointer=self.lib.rq_create(MODES.index(mode),128,self.layout['columns'],128,
                                            self.left.data.ptr,self.right.data.ptr,self.product.data.ptr)
            if not self.pointer:raise RuntimeError(self.lib.rq_error().decode())
            self.prepare(1)
            self.stream.synchronize()
        except Exception:
            self.close()
            raise

    def checked(self,status):
        if status:raise RuntimeError(self.lib.rq_error().decode())

    def prepare(self,mask):
        if self.mode=='fp4':self.checked(self.lib.rq_quantize_native(self.pointer,mask,self.encoder_variant,self.stream.ptr))
        else:self.checked(self.lib.rq_quantize(self.pointer,mask,self.stream.ptr))

    def pack(self):
        with self.stream:
            self.pack_kernel(((self.right.size+self.threads-1)//self.threads,),(self.threads,),
                             (self.data,self.indices,self.boundary,self.right,np.int32(self.mesh.elements**3),np.int32(self.mesh.q),np.int32(self.layout['columns'])))

    def calibrate(self):
        self.pack()
        self.prepare(2)
        available=[]
        self.tuning=[]
        with self.stream:
            for candidate in range(self.lib.rq_candidates(self.pointer)):
                flags=int(self.lib.rq_pca_flags(self.pointer,candidate))
                if self.mode in ('fp16','bf16') and not flags&2:
                    self.tuning.append(dict(candidate=candidate,flags=flags,available=False))
                    continue
                times=[]
                for _ in range(3):
                    start,end=cp.cuda.Event(),cp.cuda.Event()
                    start.record(self.stream)
                    self.checked(self.lib.rq_run(self.pointer,candidate,self.stream.ptr))
                    end.record(self.stream);end.synchronize()
                    times.append(float(cp.cuda.get_elapsed_time(start,end)))
                row=dict(candidate=candidate,flags=flags,algorithm_id=int(self.lib.rq_algorithm_id(self.pointer,candidate)),available=True,cuda_ms=times)
                self.tuning.append(row);available.append(row)
        if not available:raise RuntimeError('No measured eligible native GEMM candidate')
        self.algorithm=min(available,key=lambda row:float(np.median(row['cuda_ms'])))['candidate']

    def run(self):
        assert not self.closed and self.pointer
        m=self.mesh
        with self.stream:
            self.pack()
            self.prepare(2)
            self.checked(self.lib.rq_run(self.pointer,self.algorithm,self.stream.ptr))
            self.unweight_kernel(((self.local.size+self.threads-1)//self.threads,),(self.threads,),
                                 (self.product,self.weights,self.local,np.int32(m.elements**3),np.int32(m.q),
                                  np.int32(self.layout['columns']),np.float64(m.h*.5)))
            self.assembly_kernel(((m.size+self.threads-1)//self.threads,),(self.threads,),
                                 (self.local,self.data,self.mass,self.boundary,self.output,np.int32(m.size),np.int32(m.points_per_axis),
                                  np.int32(m.elements),np.int32(m.q),np.float64(self.reaction)))
        cp.cuda.runtime.deviceSynchronize()

    def raw_buffer(self,index):
        value=np.empty(self.lib.rq_bytes(self.pointer,index),np.uint8)
        if len(value):cp.cuda.runtime.memcpy(value.ctypes.data,self.lib.rq_buffer(self.pointer,index),len(value),cp.cuda.runtime.memcpyDeviceToHost)
        return value

    def metadata(self):
        result=super().metadata()
        result.pop('local_kernel_attributes')
        extras={name:getattr(self,name).nbytes for name in ('left','right','product')}
        result['named_device_buffers'].update(extras)
        python=result['named_owned_device_bytes']+sum(extras.values())
        native=int(self.lib.rq_pca_owned_bytes(self.pointer))
        result.update(family='grouped-native-gemm',component_precision=self.mode,product_storage='fp32',post_arithmetic='fp64',
                      layout=self.layout,encoder_variant=self.encoder_variant if self.mode=='fp4' else None,
                      selected_candidate=self.algorithm,selected_flags=int(self.lib.rq_pca_flags(self.pointer,self.algorithm)),tuning=self.tuning,
                      python_named_owned_device_bytes=python,native_owned_device_bytes=native,named_owned_device_bytes=python+native,
                      pack_kernel_attributes=self.pack_kernel.attributes,unweight_kernel_attributes=self.unweight_kernel.attributes,
                      native_binary_sha256=hashlib.sha256((ROOT/'build/libpca.so').read_bytes()).hexdigest(),
                      setup='Static left source encoding once; measured eligible GEMM candidate calibration is separate and charged by caller',
                      completion='Changing-original pack, dynamic B encoding, native GEMM, all-axis FP64 weighting, deterministic assembly, context acknowledgment; no prepacked-input or dispatch-trace claim')
        return result

    def close(self):
        if self.pointer:
            self.stream.synchronize()
            self.lib.rq_destroy(self.pointer)
            self.pointer=None
        for name in ('left','right','product'):setattr(self,name,None)
        super().close()
