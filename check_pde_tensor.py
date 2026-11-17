"""Original-data packing/encoding/product/post-assembly controls for native GEMM.

Arithmetic validity is separate from original PDE operator quality. Poor FP4
quality is retained, never turned into a passed physical-accuracy requirement.
"""
import hashlib
import json
from pathlib import Path

import cupy as cp
import numpy as np

from pde_packing import pack_operator,unpack_product
from pde_reference import Mesh,apply_operator
from pde_tensor import PdeTensor
from reference import MODES,digest,quantize

ROOT=Path(__file__).resolve().parent


def main():
    cp.cuda.Device(0).use()
    assert cp.cuda.runtime.getDeviceProperties(0)['name'].decode()=='NVIDIA GeForce RTX 5060 Ti'
    rows=[]
    for elements in (1,2):
        for q in (4,8,16):
            mesh=Mesh(elements,q)
            rng=np.random.Generator(np.random.PCG64(1230+elements*100+q))
            original=rng.integers(-4096,4097,mesh.size,dtype=np.int32).astype(np.float64)/8192
            for mode in MODES:
                plan=PdeTensor(mesh,mode,256,reaction=3.)
                try:
                    for variant in (0,1):
                        values=original.copy() if variant==0 else -original
                        values.setflags(write=False)
                        before=digest(values)
                        plan.load(values)
                        if variant==0:plan.calibrate()
                        plan.run()
                        actual=plan.result()
                        left,right=pack_operator(mesh,values,np.float32)
                        np.testing.assert_array_equal(cp.asnumpy(plan.left),left)
                        np.testing.assert_array_equal(cp.asnumpy(plan.right),right)
                        decoded=[]
                        encoded={}
                        for operand,value in enumerate((left,right)):
                            raw,scale,reconstructed=quantize(value,mode)
                            native=plan.raw_buffer(operand)
                            np.testing.assert_array_equal(native,raw.view(np.uint8).ravel())
                            encoded[str(operand)]=digest(native)
                            if mode=='fp4':
                                native=plan.raw_buffer(operand+2)
                                np.testing.assert_array_equal(native,scale.ravel())
                                encoded[str(operand+2)]=digest(native)
                            decoded.append(reconstructed.astype(np.float64))
                        product=cp.asnumpy(plan.product)
                        expected=decoded[0]@decoded[1].T
                        absolute=abs(decoded[0])@abs(decoded[1]).T
                        gamma=128*2.**-24/(1-128*2.**-24)
                        product_bound=8*gamma*absolute+np.finfo(np.float32).tiny
                        assert np.all(abs(product.astype(np.float64)-expected)<=product_bound),'Native physical-contraction product is outside its arithmetic envelope'
                        post=unpack_product(mesh,product,values,reaction=3.)
                        absolute_post=unpack_product(mesh,abs(product),abs(values),reaction=3.)
                        post_bound=256*np.finfo(np.float64).eps*(absolute_post+abs(post)+np.finfo(np.float64).tiny)
                        assert np.all(abs(actual-post)<=post_bound),'Post-weighting/assembly does not represent retained GEMM'
                        np.testing.assert_array_equal(actual[mesh.boundary],values[mesh.boundary])
                        assert digest(values)==before and digest(cp.asnumpy(plan.data))==before
                        physical=apply_operator(mesh,values,reaction=3.)
                        rows.append(dict(elements=elements,q=q,mode=mode,variant=variant,input_sha256=before,
                                         left_sha256=digest(left),right_sha256=digest(right),encoded_sha256=encoded,
                                         product_sha256=digest(product),output_sha256=digest(actual),
                                         relative_original_operator_error=float(np.linalg.norm(actual-physical)/np.linalg.norm(physical)),metadata=plan.metadata()))
                finally:plan.close()
    sources=('check_pde_tensor.py','pde_tensor.py','pde_packing.py','pde_cuda.py','native/pde_kernels.cu','native/pde_tensor_kernels.cu',
             'pde_reference.py','pde_coefficients.py','prepare_pde_coefficients.py','pca_native.py','reference.py',
             'native/pca_bridge.cu','native/hardware_bridge.cu','native/application_bridge.cu','native/optimized_bridge.cu','native/bridge.cu','build_pca.sh')
    result=dict(scope='Native grouped-contraction arithmetic preflight only; no full PDE/solver/performance completion or useful-accuracy claim',
                cases=rows,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
                data_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('fixtures/pde-gll.json','fixtures/pde-gll.npz')})
    (ROOT/'results/pde-tensor-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PDE_NATIVE_GROUPED_CONTRACTION_ARITHMETIC_PREFLIGHT_PASS',len(rows),flush=True)


if __name__=='__main__':main()
