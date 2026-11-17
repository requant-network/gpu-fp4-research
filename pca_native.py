"""Complete GPU centering/scaling/streamed products, moment merge and PCA.

FP4 uses the already pinned native SM120 CUTLASS product/encoder, other low
precisions use cuBLASLt, and the FP64 API control uses CuPy/cuBLAS. There is no
prepacked-input shortcut: each operation redoes centering, transpose, optional
power-of-two scaling, encoding, products, accumulation and the complete eigh.
"""
import ctypes as ct
import gc
from pathlib import Path

import cupy as cp
import cupyx
import numpy as np

from reference import MODES,digest,quantize

ROOT=Path(__file__).resolve().parent


def library():
    lib=ct.CDLL(str(ROOT/'build/libpca.so'))
    pointer,integer=ct.c_void_p,ct.c_int
    lib.rq_error.restype=ct.c_char_p
    lib.rq_create.argtypes,lib.rq_create.restype=[integer]*4+[pointer]*3,pointer
    lib.rq_destroy.argtypes,lib.rq_destroy.restype=[pointer],None
    for name in ('rq_quantize','rq_run'):
        getattr(lib,name).argtypes,getattr(lib,name).restype=[pointer,integer,pointer],integer
    lib.rq_quantize_native.argtypes,lib.rq_quantize_native.restype=[pointer,integer,integer,pointer],integer
    lib.rq_candidates.argtypes,lib.rq_candidates.restype=[pointer],integer
    lib.rq_algorithm_id.argtypes,lib.rq_algorithm_id.restype=[pointer,integer],integer
    lib.rq_buffer.argtypes,lib.rq_buffer.restype=[pointer,integer],pointer
    lib.rq_bytes.argtypes,lib.rq_bytes.restype=[pointer,integer],ct.c_size_t
    lib.rq_pca_owned_bytes.argtypes,lib.rq_pca_owned_bytes.restype=[pointer],ct.c_size_t
    lib.rq_pca_flags.argtypes,lib.rq_pca_flags.restype=[pointer,integer],ct.c_uint64
    return lib


class PcaPlan:
    def __init__(self,lib,data,method):
        self.pointer=None
        self.stream=None
        self.lib,self.method=lib,dict(method)
        self.validate(data)
        self.shape=data.shape
        self.precision=method['precision']
        self.centering=method['centering']
        self.scaling=method['scaling']
        self.acc_dtype=np.dtype(method['accumulation'].replace('fp','float'))
        self.chunk=method['chunk']
        self.d=data.shape[1]
        self.algorithm=0
        self.tuning=[]
        if self.precision not in (*MODES,'fp64') or self.centering not in ('raw','global','merge') or self.scaling not in (True,False) or self.acc_dtype not in (np.dtype('float32'),np.dtype('float64')):
            raise ValueError('Unknown bounded PCA arithmetic choice')
        if self.chunk not in (256,512,1024,2048,4096,8192) or self.d not in (128,256):
            raise ValueError('Expected D128/256 and an aligned bounded chunk')
        cp.cuda.Device(0).use()
        assert cp.cuda.Device(0).compute_capability=='120'
        self.host_data=data.copy()
        self.stream=cp.cuda.Stream(non_blocking=True)
        try:
            with self.stream:
                self.data=cp.asarray(data)
                self.operand=cp.zeros((self.d,self.chunk),dtype=cp.float64 if self.precision=='fp64' else cp.float32)
                self.product=cp.empty((self.d,self.d),dtype=self.operand.dtype if self.precision=='fp64' else cp.float32)
            self.stream.synchronize()
            if self.precision!='fp64':
                self.pointer=lib.rq_create(MODES.index(self.precision),self.d,self.d,self.chunk,self.operand.data.ptr,self.operand.data.ptr,self.product.data.ptr)
                if not self.pointer:raise RuntimeError(lib.rq_error().decode())
                cp.cuda.runtime.deviceSynchronize()
                with self.stream:
                    self.prepare()
                    for candidate in range(lib.rq_candidates(self.pointer)):
                        flags=int(lib.rq_pca_flags(self.pointer,candidate))
                        if self.precision in ('fp16','bf16') and not flags&2:
                            self.tuning.append(dict(candidate=candidate,flags=flags,available=False))
                            continue
                        values=[]
                        for _ in range(3):
                            start,end=cp.cuda.Event(),cp.cuda.Event()
                            start.record(self.stream)
                            self.checked(lib.rq_run(self.pointer,candidate,self.stream.ptr))
                            end.record(self.stream)
                            end.synchronize()
                            values.append(float(cp.cuda.get_elapsed_time(start,end)))
                        self.tuning.append(dict(candidate=candidate,algorithm_id=int(lib.rq_algorithm_id(self.pointer,candidate)),flags=flags,available=True,cuda_ms=values))
                eligible=[row for row in self.tuning if row['available']]
                if not eligible:raise RuntimeError('No required native Tensor Core product candidate')
                self.algorithm=min(eligible,key=lambda row:np.median(row['cuda_ms']))['candidate']
            self.stream.synchronize()
        except Exception:
            self.close()
            raise

    @staticmethod
    def validate(data):
        if not isinstance(data,np.ndarray) or data.dtype!=np.float32 or data.ndim!=2 or not data.flags.c_contiguous or len(data)<2 or not np.isfinite(data).all():
            raise ValueError('Expected contiguous finite FP32 observations')

    def checked(self,status):
        if status:raise RuntimeError(self.lib.rq_error().decode())

    def prepare(self):
        if self.precision=='fp4':
            self.checked(self.lib.rq_quantize_native(self.pointer,3,12,self.stream.ptr))
        else:
            self.checked(self.lib.rq_quantize(self.pointer,3,self.stream.ptr))

    def load(self,data):
        self.validate(data)
        if data.shape!=self.shape:raise ValueError('Changed observation geometry')
        self.host_data=data.copy()
        self.data.set(data,stream=self.stream)
        self.stream.synchronize()

    def raw_buffer(self,index):
        output=np.empty(self.lib.rq_bytes(self.pointer,index),np.uint8)
        if len(output):
            cp.cuda.runtime.memcpy(output.ctypes.data,self.lib.rq_buffer(self.pointer,index),output.nbytes,cp.cuda.runtime.memcpyDeviceToHost)
        return output

    def witness(self,begin,count,scale,center):
        self.stream.synchronize()
        operand=cp.asnumpy(self.operand,stream=self.stream)
        actual=cp.asnumpy(self.product,stream=self.stream)
        host_scale=cp.asnumpy(scale,stream=self.stream)
        host_center=cp.asnumpy(center,stream=self.stream)
        expected_source=np.zeros_like(operand)
        block=self.host_data[begin:begin+count].astype(np.float64)
        base=block if self.centering=='raw' else block-host_center
        base=base.astype(operand.dtype)
        np.testing.assert_array_equal(operand[:,:count],(base/host_scale).T)
        np.testing.assert_array_equal(operand[:,count:],expected_source[:,count:])
        if self.precision=='fp64':
            decoded=operand
            encoded_sha={}
            epsilon=2.**-53
        else:
            raw,sf,decoded=quantize(operand,self.precision)
            encoded_sha={}
            for index in (0,1):
                actual_raw=self.raw_buffer(index)
                np.testing.assert_array_equal(actual_raw,raw.view(np.uint8).ravel())
                encoded_sha[str(index)]=digest(actual_raw)
            if self.precision=='fp4':
                for index in (2,3):
                    actual_sf=self.raw_buffer(index)
                    np.testing.assert_array_equal(actual_sf,sf.ravel())
                    encoded_sha[str(index)]=digest(actual_sf)
            epsilon=2.**-24
        typed=decoded.astype(np.float64)
        expected=typed@typed.T
        gamma=self.chunk*epsilon/(1-self.chunk*epsilon)
        bound=8*gamma*(abs(typed)@abs(typed).T)+np.finfo(np.float64 if self.precision=='fp64' else np.float32).tiny
        assert np.all(abs(actual.astype(np.float64)-expected)<=bound),'PCA Gram product outside its arithmetic envelope'
        return dict(begin=begin,count=count,operand_sha256=digest(operand),encoded_sha256=encoded_sha,product_sha256=digest(actual),
                    maximum_product_bound_ratio=float(np.max(abs(actual.astype(np.float64)-expected)/bound)),
                    source=operand,scale=host_scale,center=host_center,product=actual)

    def device_compute(self,witness=False):
        stages=[]
        with self.stream:
            n=len(self.data)
            global_mean=cp.mean(self.data,axis=0,dtype=cp.float64) if self.centering!='merge' else None
            self.mean=cp.zeros(self.d,cp.float64) if self.centering=='merge' else global_mean
            self.moment=cp.zeros((self.d,self.d),self.acc_dtype)
            count_so_far=0
            for begin in range(0,n,self.chunk):
                block=self.data[begin:begin+self.chunk]
                count=len(block)
                center=cp.mean(block,axis=0,dtype=cp.float64) if self.centering=='merge' else global_mean
                values=block if self.centering=='raw' else block.astype(cp.float64)-center
                values=values.astype(self.operand.dtype,copy=False)
                if self.scaling:
                    maximum=cp.max(cp.abs(values),axis=0)
                    _,exponent=cp.frexp(maximum)
                    scale=cp.where(maximum==0,1,cp.ldexp(cp.ones(self.d,dtype=values.dtype),exponent)).astype(values.dtype)
                else:scale=cp.ones(self.d,dtype=values.dtype)
                self.operand.fill(0)
                self.operand[:,:count]=(values/scale).T
                if self.precision=='fp64':
                    cp.matmul(self.operand,self.operand.T,out=self.product)
                else:
                    self.prepare()
                    self.checked(self.lib.rq_run(self.pointer,self.algorithm,self.stream.ptr))
                if witness:stages.append(self.witness(begin,count,scale,center))
                typed_scale=scale.astype(self.acc_dtype)
                gram=(self.product.astype(self.acc_dtype)*typed_scale[:,None])*typed_scale[None,:]
                if self.centering=='merge':
                    total=count_so_far+count
                    delta=center-self.mean
                    correction=(delta[:,None]*delta[None,:])*(count_so_far*count/total)
                    self.moment+=gram+correction.astype(self.acc_dtype)
                    self.mean+=delta*(count/total)
                    count_so_far=total
                else:self.moment+=gram
            if self.centering=='raw':
                typed_mean=self.mean.astype(self.acc_dtype)
                self.moment-=np.float64(n)*(typed_mean[:,None]*typed_mean[None,:])
            self.covariance=((self.moment+self.moment.T)*np.float64(.5/(n-1))).astype(self.acc_dtype)
            with cupyx.errstate(linalg='raise'):
                self.eigenvalues,self.eigenvectors=cp.linalg.eigh(self.covariance)
        return stages

    def result(self):
        self.stream.synchronize()
        return tuple(cp.asnumpy(array,stream=self.stream) for array in (self.mean,self.covariance,self.eigenvalues,self.eigenvectors))

    def compute(self,witness=False):
        stages=self.device_compute(witness)
        return self.result(),stages

    def metadata(self):
        arrays={}
        for name in ('data','operand','product','mean','moment','covariance','eigenvalues','eigenvectors'):
            value=getattr(self,name,None)
            if value is not None:arrays[value.data.ptr]=value.nbytes
        native=int(self.lib.rq_pca_owned_bytes(self.pointer)) if self.pointer else 0
        return dict(python_owned_device_bytes=sum(arrays.values()),native_owned_device_bytes=native,
                    owned_device_bytes=sum(arrays.values())+native,host_original_bytes=self.host_data.nbytes,
                    selected_candidate=self.algorithm,tuning=self.tuning,
                    precision=self.precision,centering=self.centering,scaling=self.scaling,accumulation=str(self.acc_dtype),chunk=self.chunk,
                    memory_scope='Live named Python buffers plus native encoded/scales/workspace allocation requests; excludes cuSOLVER/cuBLAS internal allocations, allocator cache and temporary Python expression high-watermark; not physical peak')

    def close(self):
        if self.stream is not None:self.stream.synchronize()
        if self.pointer:
            self.lib.rq_destroy(self.pointer)
            self.pointer=None
        for name in ('data','operand','product','mean','moment','covariance','eigenvalues','eigenvectors'):
            if hasattr(self,name):setattr(self,name,None)
        gc.collect()


class PcaBatch(PcaPlan):
    """Strong complete full-matrix library reference, without chunk dispatch.

    The FP32 reference is the recorded CuPy/cuBLAS FP32 API, not an unsupported
    assertion about a particular internal instruction path. The independent
    product envelope and complete task quality remain checked.
    """
    def __init__(self,data,precision):
        self.validate(data)
        if precision not in ('fp32','fp64'):raise ValueError('Expected FP32/FP64 library control')
        self.pointer=None
        self.lib=None
        self.shape=data.shape
        self.d=data.shape[1]
        self.chunk=len(data)
        self.precision=precision
        self.centering='global'
        self.scaling=False
        self.acc_dtype=np.dtype('float64')
        self.algorithm=0
        self.tuning=[]
        self.method=dict(family='batch',precision=precision,centering='global',scaling=False,accumulation='fp64',chunk=len(data))
        self.host_data=data.copy()
        cp.cuda.Device(0).use()
        assert cp.cuda.Device(0).compute_capability=='120'
        self.stream=cp.cuda.Stream(non_blocking=True)
        with self.stream:
            self.data=cp.asarray(data)
            self.product=cp.empty((self.d,self.d),cp.float64 if precision=='fp64' else cp.float32)
        self.stream.synchronize()

    def device_compute(self,witness=False):
        stages=[]
        with self.stream:
            self.mean=cp.mean(self.data,axis=0,dtype=cp.float64)
            centered=(self.data.astype(cp.float64)-self.mean).astype(cp.float64 if self.precision=='fp64' else cp.float32)
            self.operand=cp.ascontiguousarray(centered.T)
            cp.matmul(self.operand,self.operand.T,out=self.product)
            self.moment=self.product.astype(cp.float64)
            self.covariance=(self.moment+self.moment.T)*np.float64(.5/(len(self.data)-1))
            with cupyx.errstate(linalg='raise'):
                self.eigenvalues,self.eigenvectors=cp.linalg.eigh(self.covariance)
            if witness:
                self.stream.synchronize()
                source=cp.asnumpy(self.operand,stream=self.stream)
                product=cp.asnumpy(self.product,stream=self.stream)
                center=cp.asnumpy(self.mean,stream=self.stream)
                expected_source=(self.host_data.astype(np.float64)-center).astype(source.dtype).T
                np.testing.assert_array_equal(source,expected_source)
                typed=source.astype(np.float64)
                expected=typed@typed.T
                epsilon=2.**(-53 if self.precision=='fp64' else -24)
                gamma=len(self.data)*epsilon/(1-len(self.data)*epsilon)
                bound=8*gamma*(abs(typed)@abs(typed).T)+np.finfo(product.dtype).tiny
                assert np.all(abs(product.astype(np.float64)-expected)<=bound)
                stages.append(dict(begin=0,count=len(self.data),operand_sha256=digest(source),encoded_sha256={},product_sha256=digest(product),
                                   maximum_product_bound_ratio=float(np.max(abs(product.astype(np.float64)-expected)/bound)),
                                   source=source,scale=np.ones(self.d,dtype=source.dtype),center=center,product=product))
        return stages
