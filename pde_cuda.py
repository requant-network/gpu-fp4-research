"""Direct CUDA FP32/FP64 sumfactorization strong controls for the physical study.

Input/common output storage is FP64. The named precision determines all local
products, weighting and deterministic assembly, not the output container label.
No assembled global matrix is created. Legacy default stream has process lifetime;
completion acknowledgment covers the whole current CUDA context.
"""
from pathlib import Path

import cupy as cp
import numpy as np

ROOT=Path(__file__).resolve().parent
_MODULE=None


def module():
    global _MODULE
    if _MODULE is None:
        expressions=tuple(f'pde_local<{kind},{q}>' for kind in ('float','double') for q in (4,8,16))
        expressions+=tuple(f'pde_assemble<{kind}>' for kind in ('float','double'))
        _MODULE=cp.RawModule(code=(ROOT/'native/pde_kernels.cu').read_text(),options=('--std=c++17',),name_expressions=expressions)
    return _MODULE


class PdeCUDA:
    def __init__(self,mesh,precision,threads,reaction=0.):
        if mesh.q not in (4,8,16) or precision not in ('fp32','fp64') or threads not in (128,256):raise ValueError('Unknown bounded direct CUDA configuration')
        if not np.isfinite(reaction) or reaction<0:raise ValueError('Nonnegative reaction required')
        self.mesh,self.precision,self.threads,self.reaction=mesh,precision,threads,float(reaction)
        self.stream=cp.cuda.Stream.null
        self.closed=False
        dtype=cp.float32 if precision=='fp32' else cp.float64
        kind='float' if precision=='fp32' else 'double'
        self.local_kernel=module().get_function(f'pde_local<{kind},{mesh.q}>')
        self.assembly_kernel=module().get_function(f'pde_assemble<{kind}>')
        with self.stream:
            self.data=cp.empty(mesh.size,cp.float64)
            self.output=cp.empty_like(self.data)
            self.local=cp.empty((mesh.elements**3,mesh.q**3),dtype)
            self.indices=cp.asarray(mesh.indices.ravel(),dtype=cp.int32)
            self.stiffness=cp.asarray(mesh.stiffness,dtype=cp.float64)
            self.weights=cp.asarray(mesh.weights,dtype=cp.float64)
            self.mass=cp.asarray(mesh.mass,dtype=cp.float64)
            self.boundary=cp.asarray(mesh.boundary,dtype=cp.uint8)
        self.stream.synchronize()

    def load(self,values):
        values=np.asarray(values)
        if values.shape!=(self.mesh.size,) or values.dtype not in (np.float32,np.float64) or not np.isfinite(values).all():
            raise ValueError('Finite original nodal vector required')
        self.data.set(np.ascontiguousarray(values,dtype=np.float64),stream=self.stream)
        self.stream.synchronize()

    def run(self):
        assert not self.closed
        m=self.mesh
        with self.stream:
            self.local_kernel((m.elements**3,),(self.threads,),
                              (self.data,self.indices,self.boundary,self.stiffness,self.weights,self.local,np.float64(m.h*.5)),
                              shared_mem=m.q**3*self.local.dtype.itemsize)
            self.assembly_kernel(((m.size+self.threads-1)//self.threads,),(self.threads,),
                                 (self.local,self.data,self.mass,self.boundary,self.output,np.int32(m.size),np.int32(m.points_per_axis),
                                  np.int32(m.elements),np.int32(m.q),np.float64(self.reaction)))
        cp.cuda.runtime.deviceSynchronize()

    def result(self):
        return cp.asnumpy(self.output,stream=self.stream)

    def metadata(self):
        arrays=('data','output','local','indices','stiffness','weights','mass','boundary')
        return dict(family='direct-cuda-sumfactorization',component_precision=self.precision,input_output_storage='fp64',threads=self.threads,
                    local_kernel_attributes=self.local_kernel.attributes,assembly_kernel_attributes=self.assembly_kernel.attributes,
                    named_owned_device_bytes=sum(getattr(self,name).nbytes for name in arrays),
                    named_device_buffers={name:getattr(self,name).nbytes for name in arrays},
                    memory_scope='Named buffers only; excludes allocator cache, module/JIT/driver and temporary physical peaks',
                    completion='Two complete uncaptured kernels plus current-context device synchronization; default stream0, no graph/cold-system claim')

    def close(self):
        if self.closed:return
        self.stream.synchronize()
        for name in ('data','output','local','indices','stiffness','weights','mass','boundary'):setattr(self,name,None)
        self.closed=True
