"""Portable PCA inputs: offset, weak dependence, drift and real SIFT descriptors."""
import hashlib
import json
from pathlib import Path

import numpy as np

from reference import digest

ROOT=Path(__file__).resolve().parent
COHORTS={'a':20261117,'b':20261217}
PROFILES=('uniform','shifted','dependent','drift','sift')
DATASETS={f'{profile}-n{n}-d{d}':(n,d,profile) for profile in PROFILES[:-1] for n,d in ((8192,128),(32768,128),(32768,256))}
DATASETS.update({'sift-n8192-d128':(8192,128,'sift'),'sift-n65536-d128':(65536,128,'sift'),'tail-n8193-d128':(8193,128,'uniform')})


def sift_window(begin,count):
    directory=ROOT/'fixtures/search-sift'
    manifest=json.loads((directory/'manifest.json').read_text())
    selected=[]
    for item in manifest['files']:
        if item['kind']!='database' or item['stop']<=begin or item['start']>=begin+count:continue
        path=directory/item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        with np.load(path,allow_pickle=False) as archive:vectors=archive['vectors']
        assert digest(vectors)==item['vectors_sha256']
        lo,hi=max(begin,item['start'])-item['start'],min(begin+count,item['stop'])-item['start']
        selected.append(vectors[lo:hi])
    data=np.ascontiguousarray(np.concatenate(selected),dtype=np.float32)
    assert data.shape==(count,128)
    return data,hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()


def inputs(n,d,profile,cohort):
    if cohort not in COHORTS or profile not in PROFILES or not isinstance(n,int) or n<2 or n>500000 or d not in (128,256):
        raise ValueError('Unknown bounded PCA input')
    if profile=='sift':
        if d!=128:raise ValueError('The genuine SIFT source has 128 features')
        start=0 if cohort=='a' else 500000
        data,manifest=sift_window(start,n)
        return data,dict(profile=profile,source='Canonical SIFT1M descriptor rows',start=start,stop=start+n,fixture_manifest_sha256=manifest)
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([COHORTS[cohort],n,d,PROFILES.index(profile)])))
    integers=rng.integers(-2048,2049,(n,d),dtype=np.int32).astype(np.float32)
    if profile=='uniform':data=integers/4096
    elif profile=='shifted':data=np.float32(32)+integers/65536
    elif profile=='drift':
        sign=np.where(np.arange(d)%2,1,-1).astype(np.float32)
        drift=((np.arange(n,dtype=np.int64)*8)//n-4).astype(np.float32)/4
        data=integers/65536+drift[:,None]*sign[None,:]
    else:
        latent=rng.integers(-2048,2049,(n,16),dtype=np.int32).astype(np.float32)/4096
        weights=np.ldexp(np.ones(d,np.float32),-((np.arange(d)%16)//4))
        data=latent[:,np.arange(d)%16]*weights+integers/16777216
        exponents=(np.arange(d)//16)%5-2
        data=np.ldexp(data,exponents[None,:])
    return np.ascontiguousarray(data,dtype=np.float32),dict(profile=profile,seed=COHORTS[cohort],generator='PCG64 SeedSequence cohort/N/D/profile; integer-derived dyadic FP32 inputs')


def methods():
    result=[dict(precision=precision,centering=centering,scaling=scaling,accumulation=accumulation,chunk=2048)
            for precision in ('fp4','fp8','fp16','bf16','fp32','fp64') for centering in ('raw','global','merge')
            for scaling in (False,True) for accumulation in ('fp32','fp64')]
    result.extend(dict(precision=precision,centering='merge',scaling=True,accumulation='fp64',chunk=chunk)
                  for precision in ('fp4','fp8','fp16','bf16','fp32','fp64') for chunk in (256,8192))
    return result
