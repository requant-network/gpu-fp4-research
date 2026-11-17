"""Independent original-data checks for ungrouped short-q cuBLASLt controls."""
import hashlib
import json
from pathlib import Path
import cupy as cp
import numpy as np
from pde_reference import Mesh,apply_operator
from pde_small import PdeSmall
from reference import digest,quantize

ROOT=Path(__file__).resolve().parent
MODES=('fp8','fp16','bf16','fp32')


def unpack(mesh,product,original,reaction):
    q,w=mesh.q,mesh.weights
    count=mesh.elements**3*q*q
    tensors=[]
    for axis in range(3):
        moved=product[:q,axis*count:(axis+1)*count].T.reshape(mesh.elements**3,q,q,q).astype(np.float64)
        tensors.append(np.moveaxis(moved,-1,axis+1))
    local=mesh.h*.5*(tensors[0]*w[None,None,:,None]*w[None,None,None,:]+
                     tensors[1]*w[None,:,None,None]*w[None,None,None,:]+
                     tensors[2]*w[None,:,None,None]*w[None,None,:,None])
    result=np.bincount(mesh.indices.ravel(),weights=local.ravel(),minlength=mesh.size)
    masked=original.copy();masked[mesh.boundary]=0
    result+=reaction*mesh.mass*masked
    result[mesh.boundary]=original[mesh.boundary]
    return result


def main():
    cp.cuda.Device(0).use()
    assert cp.cuda.runtime.getDeviceProperties(0)['name'].decode()=='NVIDIA GeForce RTX 5060 Ti'
    rows=[]
    for elements in (1,2):
        for q in (4,8,16):
            mesh=Mesh(elements,q)
            original=np.random.default_rng(1240+elements*100+q).integers(-4096,4097,mesh.size).astype(np.float64)/8192
            for mode in MODES:
                plan=PdeSmall(mesh,mode,256,reaction=3.)
                try:
                    for variant in (0,1):
                        values=original.copy() if variant==0 else -original
                        values.setflags(write=False)
                        identity=digest(values)
                        plan.load(values)
                        if variant==0:plan.calibrate()
                        plan.run()
                        actual=plan.result()
                        masked=values.astype(np.float32);masked[mesh.boundary]=0
                        gathered=masked[mesh.indices]
                        vectors=np.concatenate([np.moveaxis(gathered,axis+1,-1).reshape(-1,q) for axis in range(3)])
                        left=np.zeros((plan.pad,plan.pad),np.float32);left[:q,:q]=mesh.stiffness.astype(np.float32)
                        right=np.zeros((len(vectors),plan.pad),np.float32);right[:,:q]=vectors
                        np.testing.assert_array_equal(cp.asnumpy(plan.left),left)
                        np.testing.assert_array_equal(cp.asnumpy(plan.right),right)
                        decoded=[];encoded={}
                        for index,value in enumerate((left,right)):
                            raw,_,reconstruction=quantize(value,mode)
                            native=plan.raw_buffer(index)
                            np.testing.assert_array_equal(native,raw.view(np.uint8).ravel())
                            encoded[str(index)]=digest(native);decoded.append(reconstruction.astype(np.float64))
                        product=cp.asnumpy(plan.product)
                        expected=decoded[0]@decoded[1].T
                        absolute=abs(decoded[0])@abs(decoded[1]).T
                        gamma=plan.pad*2.**-24/(1-plan.pad*2.**-24)
                        assert np.all(abs(product.astype(np.float64)-expected)<=8*gamma*absolute+np.finfo(np.float32).tiny),'Short-q product arithmetic failure'
                        post=unpack(mesh,product,values,3.)
                        absolute_post=unpack(mesh,abs(product),abs(values),3.)
                        assert np.all(abs(actual-post)<=256*np.finfo(np.float64).eps*(absolute_post+abs(post)+np.finfo(np.float64).tiny)),'Short-q original post-assembly failure'
                        np.testing.assert_array_equal(actual[mesh.boundary],values[mesh.boundary])
                        assert digest(values)==identity and digest(cp.asnumpy(plan.data))==identity
                        truth=apply_operator(mesh,values,reaction=3.)
                        rows.append(dict(elements=elements,q=q,mode=mode,variant=variant,input_sha256=identity,encoded_sha256=encoded,
                                         left_sha256=digest(left),right_sha256=digest(right),product_sha256=digest(product),output_sha256=digest(actual),
                                         relative_original_operator_error=float(np.linalg.norm(actual-truth)/np.linalg.norm(truth)),metadata=plan.metadata()))
                finally:plan.close()
    sources=('check_pde_small.py','pde_small.py','native/pde_small_bridge.cu','native/pde_small_kernels.cu','build_pde_small.sh',
             'pde_cuda.py','native/pde_kernels.cu','pde_reference.py','pde_coefficients.py','prepare_pde_coefficients.py','reference.py')
    result=dict(scope='Ungrouped short-q library preflight only, not completed PDE/solver/performance study',cases=rows,
                source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources},
                data_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('fixtures/pde-gll.json','fixtures/pde-gll.npz')})
    (ROOT/'results/pde-small-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PDE_SHORT_Q_CUBLAS_ARITHMETIC_PREFLIGHT_PASS',len(rows),flush=True)


if __name__=='__main__':main()
