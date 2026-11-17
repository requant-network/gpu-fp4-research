"""CPU-only independent reconstruction of retained PCA stages.

No GPU producer is imported. Recreate original-data means/preparation/encoding,
and use two interleaved FP64 reductions rather than the producer's single
matrix product. Full retained matrices and samples use the declared arithmetic
envelope, not the producer's recorded maximum-error diagnostic.
"""
import numpy as np

from pca_protocol import validate_binding
from reference import MODES,digest,quantize


def replay_stage(data,method,stage,arrays):
    validate_binding(arrays,stage['array_bindings'])
    family,precision=method['family'],method['precision']
    assert family in ('stream','batch') and precision in (*MODES,'fp64')
    assert method['centering'] in ('raw','global','merge') and type(method['scaling']) is bool
    assert method['accumulation'] in ('fp32','fp64')
    n,d=data.shape
    chunk=n if family=='batch' else method['chunk']
    begin,count=stage['begin'],stage['count']
    assert type(begin) is int and type(count) is int and type(chunk) is int
    assert 0<=begin<n and begin%chunk==0 and count==min(chunk,n-begin)
    dtype=np.dtype('float64' if precision=='fp64' else 'float32')
    assert stage['source_shape']==[d,chunk] and stage['source_dtype']==dtype.str
    assert stage['product_dtype']==dtype.str and type(stage['full_product_retained']) is bool
    expected_keys={'center','scale','source_indices','source_samples','product_indices','product_samples'}
    if stage['full_product_retained']:expected_keys.add('full_product')
    assert set(arrays)==expected_keys
    center,scale=arrays['center'],arrays['scale']
    assert center.shape==(d,) and center.dtype==np.dtype('float64') and np.isfinite(center).all()
    assert scale.shape==(d,) and scale.dtype==dtype and np.isfinite(scale).all() and np.all(scale>0)
    rows=data[begin:begin+count].astype(np.float64)
    mean_rows=rows if method['centering']=='merge' else data.astype(np.float64)
    mean=mean_rows.sum(axis=0,dtype=np.float64)/len(mean_rows)
    unit=2.**-53
    gamma=len(mean_rows)*unit/(1-len(mean_rows)*unit)
    mean_bound=8*gamma*np.mean(abs(mean_rows),axis=0)+np.finfo(np.float64).tiny
    assert np.all(abs(center-mean)<=mean_bound),'Stage mean does not represent its original observations'
    base=(rows if method['centering']=='raw' else rows-center).astype(dtype)
    if method['scaling']:
        maximum=np.max(abs(base),axis=0)
        _,exponent=np.frexp(maximum)
        expected_scale=np.where(maximum==0,1,np.ldexp(np.ones(d,dtype=dtype),exponent)).astype(dtype)
    else:expected_scale=np.ones(d,dtype=dtype)
    np.testing.assert_array_equal(scale,expected_scale)
    source=np.zeros((d,chunk),dtype=dtype)
    source[:,:count]=(base/scale).T
    assert digest(source)==stage['operand_sha256'],'Prepared stage source differs from original observations'
    if precision=='fp64' or family=='batch':
        assert stage['encoded_sha256']=={}
        decoded=source
        buffers=0
    else:
        encoded,scales,decoded=quantize(source,precision)
        encoded_hash=digest(encoded.view(np.uint8).ravel())
        expected_encoding={'0':encoded_hash,'1':encoded_hash}
        if precision=='fp4':expected_encoding.update({'2':digest(scales.ravel()),'3':digest(scales.ravel())})
        assert stage['encoded_sha256']==expected_encoding,'Native encoding differs from independent format reconstruction'
        buffers=len(expected_encoding)
    rng=np.random.Generator(np.random.PCG64(1811))
    source_indices=np.unique(np.concatenate((np.array([0,source.size-1],np.int64),rng.choice(source.size,min(64,source.size),replace=False))))
    product_indices=np.unique(np.concatenate((np.arange(d,dtype=np.int64)*(d+1),rng.choice(d*d,min(64,d*d),replace=False))))
    assert arrays['source_indices'].dtype==np.int64 and arrays['product_indices'].dtype==np.int64
    np.testing.assert_array_equal(arrays['source_indices'],source_indices)
    np.testing.assert_array_equal(arrays['product_indices'],product_indices)
    np.testing.assert_array_equal(arrays['source_samples'],source.ravel()[source_indices])
    typed=decoded.astype(np.float64)
    even,odd=typed[:,::2].copy(),typed[:,1::2].copy()
    expected=even@even.T+odd@odd.T
    absolute=abs(even)@abs(even).T+abs(odd)@abs(odd).T
    epsilon=2.**(-53 if precision=='fp64' else -24)
    gamma=chunk*epsilon/(1-chunk*epsilon)
    bound=8*gamma*absolute+np.finfo(dtype).tiny
    samples=arrays['product_samples']
    assert samples.dtype==dtype and samples.shape==product_indices.shape and np.isfinite(samples).all()
    assert np.all(abs(samples.astype(np.float64)-expected.ravel()[product_indices])<=bound.ravel()[product_indices]),'Sampled Gram arithmetic is invalid'
    full_entries=0
    product=None
    if stage['full_product_retained']:
        product=arrays['full_product']
        assert product.dtype==dtype and product.shape==(d,d) and np.isfinite(product).all()
        assert digest(product)==stage['product_sha256']
        np.testing.assert_array_equal(samples,product.ravel()[product_indices])
        assert np.all(abs(product.astype(np.float64)-expected)<=bound),'Full Gram arithmetic is invalid'
        full_entries=product.size
    return dict(source_entries=source.size,encoded_buffers=buffers,full_product_entries=full_entries,
                sampled_product_entries=len(samples),expected=expected,bound=bound,actual_product=product)
