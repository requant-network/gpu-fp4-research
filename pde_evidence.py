"""Exact executed-source/binary/environment bindings for the physical campaign."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import cupy as cp
import numpy as np

ROOT=Path(__file__).resolve().parent
PRODUCERS=('pde_study.py','pde_evidence.py','pde_cases.py','pde_protocol.py','pde_methods.py','pde_cg.py',
           'pde_reference.py','pde_coefficients.py','prepare_pde_coefficients.py','reference.py','pde_cuda.py','native/pde_kernels.cu',
           'pde_ceed.py','native/pde_ceed_bridge.cpp','native/pde_ceed_qfunction.h','build_pde_ceed.sh',
           'pde_tensor.py','native/pde_tensor_kernels.cu','pde_packing.py','pca_native.py','native/pca_bridge.cu',
           'native/hardware_bridge.cu','native/application_bridge.cu','native/optimized_bridge.cu','native/bridge.cu','build_pca.sh',
           'pde_small.py','native/pde_small_bridge.cu','native/pde_small_kernels.cu','build_pde_small.sh')


def file_hash(path):
    hasher=hashlib.sha256()
    with open(path,'rb') as source:
        while data:=source.read(1024*1024):hasher.update(data)
    return hasher.hexdigest()


def sources():return {name:file_hash(ROOT/name) for name in PRODUCERS}


def binaries():
    return {name:file_hash(path) for name,path in
            (('native_core',ROOT/'build/libpca.so'),('ceed_bridge',ROOT/'build/libpde_ceed.so'),
             ('short_q_bridge',ROOT/'build/libpde_small.so'),('libceed',Path(os.environ['CEED_DIR'])/'lib/libceed.so'))}


def environment():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Only authorized physical GPU0 may be visible'
    properties=cp.cuda.runtime.getDeviceProperties(0)
    assert properties['name'].decode()=='NVIDIA GeForce RTX 5060 Ti'
    version=subprocess.check_output(['/usr/local/cuda-13.3/bin/nvcc','--version'],text=True,timeout=20).strip()
    commit=subprocess.check_output(['git','-C',os.environ['CEED_DIR'],'rev-parse','HEAD'],text=True,timeout=20).strip()
    assert commit=='3e98d546622c05ba46cfce6d4ba2e7d0b9ad4b97'
    return dict(python=sys.version.split()[0],os=platform.system(),numpy=np.__version__,cupy=cp.__version__,
                cupy_package=importlib.metadata.version('cupy-cuda13x'),cuda_runtime_api=int(cp.cuda.runtime.runtimeGetVersion()),
                cuda_driver_api=int(cp.cuda.runtime.driverGetVersion()),compiler_banner=version,
                device_name=properties['name'].decode(),compute_capability=[properties['major'],properties['minor']],
                total_device_memory=int(properties['totalGlobalMem']),libceed_commit=commit,exclusive_device=False,
                source_sha256=sources(),binary_sha256=binaries(),
                scope='Package/compiler/API observations are separate; no loader-cause, instruction-trace or exclusive-performance assertion')


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def save_payloads(directory,prefix,arrays):
    """Bound each raw shard below48MiB, not a guessed compressed-size limit."""
    shards=[];batch={};size=0
    def flush():
        nonlocal batch,size
        if not batch:return
        path=Path(directory)/(prefix+'-part-'+str(len(shards))+'.npz')
        np.savez_compressed(path,**batch)
        shards.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=file_hash(path),keys=sorted(batch)))
        batch={};size=0
    for key,value in arrays.items():
        if value.nbytes>48*1024*1024:raise ValueError('Single array exceeds declared raw shard bound; split explicitly')
        if size+value.nbytes>48*1024*1024:flush()
        batch[key]=value;size+=value.nbytes
    flush()
    return shards
