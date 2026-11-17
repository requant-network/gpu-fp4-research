"""Complete libCEED CUDA physical operator, with explicit Dirichlet projection.

Requested/root resource and API FP64 are recorded, not a trace proving which
internal optimized/fallback kernels ran. Whole-operation completion is required.
"""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess

import cupy as cp
import numpy as np

ROOT=Path(__file__).resolve().parent
PIN='3e98d546622c05ba46cfce6d4ba2e7d0b9ad4b97'
_LIB=None
_VENDOR=None
_MASK=cp.ElementwiseKernel('float64 value, bool boundary','float64 masked','masked = boundary ? 0.0 : value;','pde_ceed_mask_original')
_RESTORE=cp.ElementwiseKernel('float64 value, bool boundary, float64 computed','float64 corrected','corrected = boundary ? value : computed;','pde_ceed_restore_original')


def library():
    global _LIB,_VENDOR
    if _LIB is None:
        if 'CEED_DIR' not in os.environ:raise RuntimeError('Set CEED_DIR to the pinned libCEED checkout')
        _VENDOR=Path(os.environ['CEED_DIR']).resolve()
        actual=subprocess.check_output(['git','-C',str(_VENDOR),'rev-parse','HEAD'],text=True).strip()
        assert actual==PIN,'libCEED source revision differs'
        lib=ct.CDLL(str(ROOT/'build/libpde_ceed.so'))
        pointer,integer,string=ct.c_void_p,ct.c_int,ct.c_char_p
        lib.rq_pde_ceed_error.restype=string
        lib.rq_pde_ceed_create.argtypes=[string,integer,integer,integer]+[pointer]*6+[string]
        lib.rq_pde_ceed_create.restype=pointer
        lib.rq_pde_ceed_apply.argtypes=[pointer,pointer,pointer]
        lib.rq_pde_ceed_apply.restype=integer
        lib.rq_pde_ceed_resource.argtypes=[pointer]
        lib.rq_pde_ceed_resource.restype=string
        lib.rq_pde_ceed_destroy.argtypes=[pointer]
        lib.rq_pde_ceed_destroy.restype=None
        _LIB=lib
    return _LIB


class PdeCEED:
    def __init__(self,mesh,backend,reaction=0.):
        if backend not in ('ref','shared','gen') or not np.isfinite(reaction) or reaction<0:raise ValueError('Unknown bounded libCEED operator')
        self.mesh,self.backend,self.reaction=mesh,backend,float(reaction)
        self.stream=cp.cuda.Stream.null
        self.handle=None
        self.lib=library()
        with self.stream:
            self.data=cp.empty(mesh.size,cp.float64)
            self.masked=cp.empty_like(self.data)
            self.output=cp.empty_like(self.data)
            self.boundary=cp.asarray(mesh.boundary,dtype=cp.bool_)
        weight=mesh.weights[:,None,None]*mesh.weights[None,:,None]*mesh.weights[None,None,:]
        qdata=np.empty((mesh.elements**3,4,mesh.q**3),np.float64)
        qdata[:,:3,:]=(mesh.h*.5)*weight.ravel()[None,None,:]
        qdata[:,3,:]=self.reaction*(mesh.h*.5)**3*weight.ravel()[None,:]
        interpolation=np.eye(mesh.q,dtype=np.float64)
        self.qdata_requested_bytes=qdata.nbytes
        self.indices_requested_bytes=mesh.indices.nbytes
        pointers=[value.ctypes.data for value in (mesh.indices,interpolation,mesh.derivative,mesh.nodes,mesh.weights,qdata)]
        requested='/gpu/cuda/'+backend
        source=str(ROOT/'native/pde_ceed_qfunction.h')+':PdeApply'
        self.handle=self.lib.rq_pde_ceed_create(requested.encode(),mesh.elements**3,mesh.q,mesh.size,*pointers,source.encode())
        if not self.handle:raise RuntimeError(self.lib.rq_pde_ceed_error().decode())
        self.actual_resource=self.lib.rq_pde_ceed_resource(self.handle).decode()
        cp.cuda.runtime.deviceSynchronize()

    def load(self,values):
        values=np.asarray(values)
        if values.shape!=(self.mesh.size,) or values.dtype not in (np.float32,np.float64) or not np.isfinite(values).all():raise ValueError('Finite original nodal vector required')
        self.data.set(np.ascontiguousarray(values,dtype=np.float64),stream=self.stream)
        self.stream.synchronize()

    def run(self):
        assert self.handle
        with self.stream:
            _MASK(self.data,self.boundary,self.masked)
            code=self.lib.rq_pde_ceed_apply(self.handle,self.masked.data.ptr,self.output.data.ptr)
            if code:raise RuntimeError(self.lib.rq_pde_ceed_error().decode())
            _RESTORE(self.data,self.boundary,self.output,self.output)
        cp.cuda.runtime.deviceSynchronize()

    def result(self):return cp.asnumpy(self.output,stream=self.stream)

    def metadata(self):
        buffers={name:getattr(self,name).nbytes for name in ('data','masked','output','boundary')}
        return dict(family='libceed-cuda',requested_resource='/gpu/cuda/'+self.backend,root_resource=self.actual_resource,
                    precision='FP64 CeedScalar API; not an executed instruction trace',source_commit=PIN,
                    libceed_binary_sha256=hashlib.sha256((_VENDOR/'lib/libceed.so').read_bytes()).hexdigest(),
                    bridge_binary_sha256=hashlib.sha256((ROOT/'build/libpde_ceed.so').read_bytes()).hexdigest(),
                    python_named_device_buffers=buffers,python_named_owned_device_bytes=sum(buffers.values()),
                    libceed_requested_qdata_bytes=self.qdata_requested_bytes,libceed_requested_index_bytes=self.indices_requested_bytes,
                    memory_scope='CuPy named buffers plus separate libCEED request categories; opaque library workspace/cache/temporary physical peak excluded',
                    completion='Mask, actual CEED apply, original boundary identity, current-context device sync; legacy stream0; no graph/cold-system/dispatch-trace claim')

    def close(self):
        if self.handle:
            cp.cuda.runtime.deviceSynchronize()
            self.lib.rq_pde_ceed_destroy(self.handle)
            self.handle=None
        for name in ('data','masked','output','boundary'):setattr(self,name,None)
