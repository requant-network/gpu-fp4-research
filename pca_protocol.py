"""CPU-only lossless PCA artifact identity and retained-subspace protocol."""
import hashlib
import json

import numpy as np


def array_binding(array):
    return dict(shape=list(array.shape),dtype=array.dtype.str,sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def retain_output(outputs,rank=16):
    mean,covariance,eigenvalues,eigenvectors=outputs
    d=len(mean)
    if covariance.shape!=(d,d) or eigenvalues.shape!=(d,) or eigenvectors.shape!=(d,d) or not 1<=rank<d:
        raise ValueError('Invalid complete PCA output geometry or retained rank')
    arrays={name:np.ascontiguousarray(value).copy() for name,value in zip(('mean','covariance','eigenvalues','leading'),(mean,covariance,eigenvalues,eigenvectors[:,-rank:]))}
    binding={name:array_binding(value) for name,value in arrays.items()}
    key=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return arrays,binding,key


def validate_binding(arrays,binding):
    assert set(arrays)==set(binding)
    for name,value in arrays.items():
        assert array_binding(value)==binding[name],name


def retain_stage(stage,full=False):
    source,product=stage['source'],stage['product']
    d=len(product)
    assert product.shape==(d,d) and source.ndim==2 and len(source)==d
    rng=np.random.Generator(np.random.PCG64(1811))
    source_indices=np.unique(np.concatenate((np.array([0,source.size-1],np.int64),rng.choice(source.size,min(64,source.size),replace=False))))
    product_indices=np.unique(np.concatenate((np.arange(d,dtype=np.int64)*(d+1),rng.choice(product.size,min(64,product.size),replace=False))))
    arrays=dict(center=stage['center'].copy(),scale=stage['scale'].copy(),source_indices=source_indices,
                source_samples=source.ravel()[source_indices].copy(),product_indices=product_indices,
                product_samples=product.ravel()[product_indices].copy())
    if full:arrays['full_product']=product.copy()
    record={key:stage[key] for key in ('begin','count','operand_sha256','encoded_sha256','product_sha256','maximum_product_bound_ratio')}
    record.update(source_shape=list(source.shape),source_dtype=source.dtype.str,product_dtype=product.dtype.str,
                  full_product_retained=bool(full),array_bindings={key:array_binding(value) for key,value in arrays.items()})
    return arrays,record
