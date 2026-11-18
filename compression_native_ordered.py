"""Separate ordered-copy reference. Never silently reuse an unmodified same-SONAME library."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import numpy as np
from compression_native import NdzipPlan

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_loaded_reference(root,record):
    root=Path(root).resolve()
    expected={Path(name).name:(root/name).resolve() for name in record['binaries'] if name.endswith('.so')}
    observed={name:set() for name in expected}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)==6 and fields[5].startswith('/'):
            path=Path(fields[5])
            if path.name in observed:observed[path.name].add(path.resolve())
    for name,path in expected.items():
        if observed[name]!={path}:raise RuntimeError(f'Wrong/mixed ndzip reference object: {name}')
    return {'reference':record['reference'], 'build_manifest_sha256':sha(root/'manifest.json'),
            'binaries':{name:sha(path) for name,path in expected.items()}}


class OrderedNdzipPlan(NdzipPlan):
    def __init__(self,shape,dtype,threads=1,stream=None,*,build_root=None):
        import cupy as cp
        root=Path(build_root) if build_root is not None else ROOT/'build/ndzip-ordered-gpu-v1'
        root=root.resolve();record=json.loads((root/'manifest.json').read_text())
        assert record['schema']=='ndzip-ordered-gpu-build-v1' and record['complete']
        assert record['reference']=='ndzip-ff4e6702+ordered-copy'
        assert all(sha(root/name)==expected for name,expected in record['binaries'].items())
        assert sha(ROOT/'native/compression_ndzip_bridge.cpp')==record['adapter_source_sha256']
        self.shape=tuple(shape)
        assert 1<=len(self.shape)<=3 and all(isinstance(n,(int,np.integer)) and 0<n<2**32 for n in self.shape)
        assert int(np.prod(self.shape,dtype=np.uint64))<=2**32-1
        self.dtype=np.dtype(dtype);assert self.dtype in (np.dtype('<f4'),np.dtype('<f8'))
        assert isinstance(threads,int) and threads>0
        self.word_dtype=np.dtype('<u4' if self.dtype.itemsize==4 else '<u8')
        self.stream=stream if stream is not None else cp.cuda.get_current_stream()
        self.device=cp.cuda.Device().id;self._retained=[];self.handle=None
        lib=self.lib=C.CDLL(str(root/'adapter/libcompression_ndzip_ordered.so'))
        self.loaded_reference=verify_loaded_reference(root,record)
        lib.ndz_error.restype=C.c_char_p
        lib.ndz_create.argtypes=[C.c_int,C.c_int,C.c_void_p,C.c_uint,C.c_void_p];lib.ndz_create.restype=C.c_void_p
        lib.ndz_bound.argtypes=[C.c_void_p];lib.ndz_bound.restype=C.c_uint64
        lib.ndz_encode.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_uint64,C.POINTER(C.c_uint64)];lib.ndz_encode.restype=C.c_int
        lib.ndz_decode.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p];lib.ndz_decode.restype=C.c_int
        lib.ndz_free.argtypes=[C.c_void_p];lib.ndz_free.restype=C.c_int
        sizes=np.asarray(self.shape,np.uint32)
        self.handle=lib.ndz_create(self.dtype.itemsize,len(self.shape),sizes.ctypes.data,threads,self.stream.ptr)
        if not self.handle:raise RuntimeError(lib.ndz_error().decode())
        self.bound_words=int(lib.ndz_bound(self.handle))
        if not self.bound_words:
            error=lib.ndz_error().decode();lib.ndz_free(self.handle);self.handle=None;raise RuntimeError(error)
