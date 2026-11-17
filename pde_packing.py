"""Explicit grouped-128 GEMM layout for the three original GLL contractions.

This is a layout/model, not a speed claim. Static block-diagonal S and changing
original-data packing, padding, post-weighting and assembly must all be charged.
"""
import numpy as np


def layout(mesh)->dict:
    assert mesh.q in (4,8,16)
    vectors=3*mesh.elements**3*mesh.q**2
    groups=128//mesh.q
    columns=((vectors+groups-1)//groups+127)//128*128
    return dict(rows=128,reduction=128,columns=columns,groups=groups,physical_vectors=vectors,
                physical_values=vectors*mesh.q,padded_values=columns*128,
                scope='Axis-concatenated physical vectors grouped into fixed128-row block-diagonal contractions; padding explicitly retained')


def static_operator(mesh,dtype=np.float32):
    dtype=np.dtype(dtype)
    if dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('FP32/FP64 preparation only')
    left=np.zeros((128,128),dtype=dtype)
    for group in range(128//mesh.q):
        begin=group*mesh.q
        left[begin:begin+mesh.q,begin:begin+mesh.q]=mesh.stiffness.astype(dtype)
    return left


def pack_operator(mesh,values,dtype=np.float32):
    values=np.asarray(values)
    if values.shape!=(mesh.size,) or values.dtype not in (np.float32,np.float64) or not np.isfinite(values).all():raise ValueError('Finite full original nodal vector required')
    left=static_operator(mesh,dtype)
    metadata=layout(mesh)
    masked=values.astype(dtype,copy=True)
    masked[mesh.boundary]=0
    local=masked[mesh.indices]
    vectors=np.concatenate([np.moveaxis(local,1+axis,-1).reshape(-1,mesh.q) for axis in range(3)],axis=0)
    columns=metadata['columns']
    right=np.zeros((columns,128),dtype=dtype)
    for group in range(metadata['groups']):
        begin=group*columns
        count=min(columns,max(0,len(vectors)-begin))
        if count:right[:count,group*mesh.q:(group+1)*mesh.q]=vectors[begin:begin+count]
    return left,right


def unpack_product(mesh,product,original,reaction=0.):
    metadata=layout(mesh)
    product=np.asarray(product)
    original=np.asarray(original)
    if product.shape!=(128,metadata['columns']) or product.dtype not in (np.float32,np.float64):raise ValueError('Full grouped GEMM output required')
    if original.shape!=(mesh.size,) or original.dtype not in (np.float32,np.float64) or not np.isfinite(original).all():raise ValueError('Finite original nodal vector required')
    if not np.isfinite(reaction) or reaction<0:raise ValueError('Nonnegative finite reaction required')
    q,columns=mesh.q,metadata['columns']
    vectors=np.empty((metadata['physical_vectors'],q),np.float64)
    for group in range(metadata['groups']):
        begin=group*columns
        count=min(columns,max(0,len(vectors)-begin))
        if count:vectors[begin:begin+count]=product[group*q:(group+1)*q,:count].T
    per_axis=mesh.elements**3*q*q
    weight=mesh.weights
    factors=(weight[None,None,:,None]*weight[None,None,None,:],
             weight[None,:,None,None]*weight[None,None,None,:],
             weight[None,:,None,None]*weight[None,None,:,None])
    local=np.zeros((mesh.elements**3,q,q,q),np.float64)
    for axis in range(3):
        moved=vectors[axis*per_axis:(axis+1)*per_axis].reshape(mesh.elements**3,q,q,q)
        local+=np.moveaxis(moved,-1,1+axis)*factors[axis]
    result=np.bincount(mesh.indices.ravel(),weights=(local*(mesh.h*.5)).ravel(),minlength=mesh.size)
    masked=original.astype(np.float64,copy=True)
    masked[mesh.boundary]=0
    result+=float(reaction)*mesh.mass*masked
    result[mesh.boundary]=original[mesh.boundary]
    return result
