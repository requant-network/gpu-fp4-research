"""Independent base-128 components and exact finite-FP32 product witnesses."""
import math

import numpy as np

from reference import digest,relative_error

PROFILES=("normal","cancellation","spread","subnormal")


def dataset(seed,shape,profile):
    if profile not in PROFILES:raise ValueError("Unknown precision input profile")
    m,n,k=shape
    rng=np.random.default_rng(np.random.SeedSequence([seed,m,n,k,PROFILES.index(profile)]))
    def values(rows):
        raw=rng.bit_generator.random_raw((rows,k))
        return ((raw&np.uint64((1<<24)-1)).astype(np.int32)-(1<<23)).astype(np.float32)/np.float32(1<<23)
    a,b=values(m),values(n)
    if profile=="cancellation":
        a[:,1:-2:2]=a[:,0:-2:2]
        b[:,1:-2:2]=-b[:,0:-2:2]
        a[:,-2]=0;b[:,-2]=0;a[:,-1]*=np.float32(2**-30)
    elif profile=="spread":
        exponents=np.resize(np.array([-60,-30,0,30,60],np.int32),k)
        a=np.ldexp(a,exponents).astype(np.float32);b=np.ldexp(b,-exponents).astype(np.float32)
    elif profile=="subnormal":a=np.ldexp(a,-140).astype(np.float32)
    a[0]=0;b[0]=0
    return np.ascontiguousarray(a),np.ascontiguousarray(b)


def row_statistics(x):
    if x.dtype!=np.float32 or x.ndim!=2 or not np.isfinite(x).all():raise ValueError("Finite FP32 matrix required")
    raw=x.view(np.uint32)&np.uint32(0x7fffffff);field=(raw>>23).astype(np.int32)
    mantissa=(raw&np.uint32(0x7fffff))|np.where(field!=0,np.uint32(0x800000),np.uint32(0))
    lowest=mantissa&(~mantissa+np.uint32(1));nonzero=mantissa!=0
    trailing=np.log2(np.where(nonzero,lowest,np.uint32(1)).astype(np.float64)).astype(np.int32)
    lsb=np.where(nonzero,np.where(field!=0,field-150,-149)+trailing,np.iinfo(np.int32).max)
    exponent=np.frexp(np.max(np.abs(x),axis=1).astype(np.float64))[1].astype(np.int32)
    minimum=lsb.min(axis=1);required=np.where(np.any(nonzero,axis=1),(exponent-minimum+6)//7,0).astype(np.int32)
    return exponent,required


def components(x,count):
    if count<1 or count>8:raise ValueError("Component count must be1..8")
    exponent,required=row_statistics(x)
    residual=np.ldexp(x.astype(np.float64),-exponent[:,None]);parts=[];decoded=np.zeros_like(residual)
    for index in range(count):
        residual*=128
        part=np.trunc(residual).astype(np.int8);residual-=part
        parts.append(part);decoded+=np.ldexp(part.astype(np.float64),-7*(index+1))
    return np.stack(parts),exponent,required,np.ldexp(decoded,exponent[:,None])


def scaled_integers(x):
    if x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError("Exact witnesses require finite FP32")
    def convert(raw):
        raw=int(raw);field=(raw>>23)&255;mantissa=(raw&0x7fffff)|(0x800000 if field else 0)
        value=mantissa<<max(field-1,0)
        return -value if raw>>31 else value
    return np.array([convert(v) for v in x.view(np.uint32).ravel()],dtype=object).reshape(x.shape)


def exact_product(a,b):
    integers=scaled_integers(a)@scaled_integers(b).T
    return np.array([math.ldexp(float(v),-298) for v in integers.ravel()],np.float64).reshape(integers.shape)


def exact_samples(a,b,indices):
    rows_a,map_a=np.unique(indices[:,0],return_inverse=True);rows_b,map_b=np.unique(indices[:,1],return_inverse=True)
    aa,bb=scaled_integers(a[rows_a]),scaled_integers(b[rows_b])
    return np.array([math.ldexp(float(sum(int(x)*int(y) for x,y in zip(aa[i],bb[j]))),-298) for i,j in zip(map_a,map_b)],np.float64)


def gold(a,b,profile):
    if profile=="cancellation":return a[:,-1:].astype(np.float64)@b[:,-1:].astype(np.float64).T
    return a.astype(np.float64)@b.astype(np.float64).T


def arithmetic_reference(a,b,mode,selected,retained):
    hashes={};metadata={}
    if mode>=4 and selected:
        values=[]
        for operand,x in enumerate((a,b)):
            parts,exponent,required,q=components(x,selected);values.append(q)
            hashes[f"operand_{operand}_components"]=digest(parts)
            metadata[f"operand_{operand}_exponents"]=digest(exponent);metadata[f"operand_{operand}_required"]=digest(required)
        x,y=values
    else:x,y=a.astype(np.float64),b.astype(np.float64)
    expected=x@y.T
    u=2**-24 if mode in (0,3) else 2**-53
    gamma=(a.shape[1]+20)*u/(1-(a.shape[1]+20)*u)
    bound=8*gamma*(np.abs(x)@np.abs(y).T)+np.finfo(np.float64).tiny
    if mode in (0,3):bound+=np.finfo(np.float32).tiny
    if mode==2 and retained>0:
        ea,_=row_statistics(a);eb,_=row_statistics(b)
        sa=np.ldexp(np.ones(a.shape[0]),ea-retained+2);sb=np.ldexp(np.ones(b.shape[0]),eb-retained+2)
        bound+=sa[:,None]*np.sum(np.abs(b.astype(np.float64)),axis=1)[None,:]+np.sum(np.abs(a.astype(np.float64)),axis=1)[:,None]*sb[None,:]+a.shape[1]*sa[:,None]*sb[None,:]
    return expected,bound,hashes,metadata


def check_output(actual,expected,bound,original):
    if not np.isfinite(actual).all():raise ValueError("Nonfinite precision output")
    ratio=float(np.max(np.abs(actual-expected)/bound))
    if ratio>1:raise ValueError(f"Precision arithmetic bound exceeded: {ratio}")
    return {"arithmetic_max_bound_ratio":ratio,"relative_frobenius_error":relative_error(actual,original),
            "output_sha256":digest(actual),"checked_output_entries":int(actual.size)}
