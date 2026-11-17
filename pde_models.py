"""Independent NumPy physical and decoded arithmetic models, not GPU producers.

Grouping is reconstructed with Kronecker/reshape/transposes rather than the
producer's per-thread index formula. Final arithmetic validity remains distinct
from approximation error against the original high-precision physical operator.
"""
import numpy as np
from reference import quantize
from pde_protocol import array_hash


def gamma(count,unit):return count*unit/(1-count*unit)


def contract(matrix,local,axis):
    equation=('il,eljk->eijk','jl,eilk->eijk','kl,eijl->eijk')[axis]
    return np.einsum(equation,matrix,local,optimize=True)


def scatter(mesh,local,original,reaction=0.):
    value=np.bincount(mesh.indices.ravel(),weights=local.ravel(),minlength=mesh.size)
    masked=original.copy();masked[mesh.boundary]=0
    value+=reaction*mesh.mass*masked
    value[mesh.boundary]=original[mesh.boundary]
    return value


def directional_weights(weights):
    return (weights[None,None,:,None]*weights[None,None,None,:],
            weights[None,:,None,None]*weights[None,None,None,:],
            weights[None,:,None,None]*weights[None,None,:,None])


def ungroup(mesh,product,family,columns,original,reaction=0.):
    q=mesh.q;length=3*mesh.elements**3*q*q;per_axis=length//3
    if family=='tensor':
        vectors=product.reshape(128//q,q,columns).transpose(0,2,1).reshape(-1,q)[:length]
    else:vectors=product[:q,:length].T
    local=np.zeros((mesh.elements**3,q,q,q),np.float64)
    for axis,factor in enumerate(directional_weights(mesh.weights)):
        moved=vectors[axis*per_axis:(axis+1)*per_axis].reshape(mesh.elements**3,q,q,q)
        local+=np.moveaxis(moved,-1,axis+1)*factor
    return scatter(mesh,local*(mesh.h*.5),original,reaction)


def operator_model(mesh,original,configuration,reaction=0.):
    original=np.asarray(original,dtype=np.float64)
    assert original.shape==(mesh.size,) and np.isfinite(original).all()
    family,precision=configuration['family'],configuration['precision']
    masked=original.copy();masked[mesh.boundary]=0
    if family=='cuda':
        rounded=np.float32 if precision=='fp32' else np.float64
        values=masked.astype(rounded).astype(np.float64)[mesh.indices]
        stiffness=mesh.stiffness.astype(rounded).astype(np.float64)
        weights=mesh.weights.astype(rounded).astype(np.float64)
        local=np.zeros_like(values);absolute=np.zeros_like(values)
        for axis,factor in enumerate(directional_weights(weights)):
            local+=contract(stiffness,values,axis)*factor
            absolute+=contract(abs(stiffness),abs(values),axis)*factor
        expected=scatter(mesh,local*(mesh.h*.5),original,reaction)
        flow=scatter(mesh,absolute*(mesh.h*.5),abs(original),reaction)
        unit=np.finfo(rounded).eps*.5
        bound=8*gamma(mesh.q+16,unit)*flow+np.finfo(rounded).tiny
        bound[mesh.boundary]=0
        return dict(expected=expected,bound=bound)
    if family=='ceed':
        values=masked[mesh.indices];derivative=mesh.derivative
        qdata=(mesh.h*.5)*mesh.weights[:,None,None]*mesh.weights[None,:,None]*mesh.weights[None,None,:]
        local=np.zeros_like(values);absolute=np.zeros_like(values)
        for axis in range(3):
            gradient=contract(derivative,values,axis)
            local+=contract(derivative.T,gradient*qdata[None,:,:,:],axis)
            absolute+=contract(abs(derivative.T),contract(abs(derivative),abs(values),axis)*qdata[None,:,:,:],axis)
        expected=scatter(mesh,local,original,reaction)
        flow=scatter(mesh,absolute,abs(original),reaction)
        bound=8*gamma(2*mesh.q+64,np.finfo(np.float64).eps*.5)*flow+np.finfo(np.float64).tiny
        bound[mesh.boundary]=0
        return dict(expected=expected,bound=bound)
    assert family in ('tensor','small')
    q=mesh.q;gathered=masked.astype(np.float32)[mesh.indices]
    vectors=np.concatenate([np.moveaxis(gathered,axis+1,-1).reshape(-1,q) for axis in range(3)])
    if family=='tensor':
        groups=128//q;columns=((len(vectors)+groups-1)//groups+127)//128*128
        left=np.kron(np.eye(groups,dtype=np.float32),mesh.stiffness.astype(np.float32))
        left[left==0]=np.float32(0)
        arranged=np.zeros((groups*columns,q),np.float32);arranged[:len(vectors)]=vectors
        right=arranged.reshape(groups,columns,q).transpose(1,0,2).reshape(columns,128)
    else:
        pad=16 if precision=='fp8' else max(8,q) if precision in ('fp16','bf16') else q
        columns=len(vectors)
        left=np.pad(mesh.stiffness.astype(np.float32),((0,pad-q),(0,pad-q)),constant_values=0)
        right=np.pad(vectors,((0,0),(0,pad-q)),constant_values=0)
    decoded=[];encoded={}
    for index,value in enumerate((left,right)):
        raw,scale,reconstruction=quantize(value,precision)
        encoded[str(index)]=array_hash(raw.view(np.uint8).ravel())
        if precision=='fp4':encoded[str(index+2)]=array_hash(scale.ravel())
        decoded.append(reconstruction.astype(np.float64))
    product=decoded[0]@decoded[1].T
    absolute=abs(decoded[0])@abs(decoded[1]).T
    product_bound=8*gamma(left.shape[1],2.**-24)*absolute+np.finfo(np.float32).tiny
    expected=ungroup(mesh,product,family,columns,original,reaction)
    post_flow=ungroup(mesh,absolute,family,columns,abs(original),reaction)
    propagated=ungroup(mesh,product_bound,family,columns,np.zeros_like(original),0.)
    bound=propagated+256*np.finfo(np.float64).eps*(post_flow+abs(expected))+np.finfo(np.float64).tiny
    bound[mesh.boundary]=0
    return dict(expected=expected,bound=bound,left=left,right=right,encoded=encoded,product=product,
                product_bound=product_bound,family=family,columns=columns)


def check_output(model,actual,mesh,original):
    assert actual.shape==(mesh.size,) and actual.dtype==np.float64 and np.isfinite(actual).all(),'Invalid full physical output'
    assert np.array_equal(actual[mesh.boundary].view(np.uint64),original[mesh.boundary].view(np.uint64)),'Original boundary identity failed'
    assert np.all(abs(actual-model['expected'])<=model['bound']),'Final physical output fails original-data arithmetic'


def check_stage(model,witness,arrays,actual,mesh,original):
    if 'product' not in model:
        assert witness is None,'Unexpected retained GEMM stage';return
    assert witness is not None and witness['left_sha256']==array_hash(model['left']) and witness['right_sha256']==array_hash(model['right']),'Native preparation is not original-data packing'
    assert witness['encoded_sha256']==model['encoded'],'Native encoding is not independently reconstructed'
    product=np.asarray(arrays[witness['product_key']])
    assert product.dtype==np.float32 and product.shape==model['product'].shape and array_hash(product)==witness['product_sha256'],'Retained full product changed'
    assert np.all(abs(product.astype(np.float64)-model['product'])<=model['product_bound']),'Retained GEMM fails decoded arithmetic'
    expected=ungroup(mesh,product.astype(np.float64),model['family'],model['columns'],original)
    flow=ungroup(mesh,abs(product.astype(np.float64)),model['family'],model['columns'],abs(original))
    bound=256*np.finfo(np.float64).eps*(flow+abs(expected))+np.finfo(np.float64).tiny
    bound[mesh.boundary]=0
    assert np.all(abs(actual-expected)<=bound),'Retained product does not generate its full physical output'
