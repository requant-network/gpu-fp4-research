"""GPU0 direct-CUDA physical-operator regression; preflight, not study completion."""
import copy
import hashlib
import json
from pathlib import Path

import cupy as cp
import numpy as np

from pde_cuda import PdeCUDA
from pde_reference import Mesh,apply_operator

ROOT=Path(__file__).resolve().parent


def digest(value):return hashlib.sha256(value.tobytes()).hexdigest()


def main():
    cp.cuda.Device(0).use()
    properties=cp.cuda.runtime.getDeviceProperties(0)
    assert properties['name'].decode()=='NVIDIA GeForce RTX 5060 Ti' and cp.cuda.Device(0).compute_capability=='120'
    rows=[]
    for elements in (1,2):
        for q in (4,8,16):
            mesh=Mesh(elements,q)
            rng=np.random.Generator(np.random.PCG64(1200+elements*100+q))
            original=rng.integers(-4096,4097,mesh.size,dtype=np.int32).astype(np.float64)/8192
            absolute_mesh=copy.copy(mesh)
            absolute_mesh.stiffness=abs(mesh.stiffness)
            for precision in ('fp32','fp64'):
                epsilon=np.finfo(np.float32 if precision=='fp32' else np.float64).eps
                for threads in (128,256):
                    plan=PdeCUDA(mesh,precision,threads,reaction=3.)
                    try:
                        for variant in (0,1):
                            values=original.copy() if variant==0 else -original
                            values.setflags(write=False)
                            expected_hash=digest(values)
                            expected=apply_operator(mesh,values,reaction=3.)
                            absolute=apply_operator(absolute_mesh,abs(values),reaction=3.)
                            envelope=128*(q+16)*epsilon*(absolute+abs(expected)+np.finfo(np.float64).tiny)
                            plan.load(values)
                            assert digest(cp.asnumpy(plan.data))==expected_hash
                            plan.run()
                            actual=plan.result()
                            assert np.isfinite(actual).all() and np.all(abs(actual-expected)<=envelope),'Direct CUDA arithmetic exceeds original weak-form envelope'
                            np.testing.assert_array_equal(actual[mesh.boundary],values[mesh.boundary])
                            assert digest(values)==expected_hash and digest(cp.asnumpy(plan.data))==expected_hash
                            rows.append(dict(elements=elements,q=q,precision=precision,threads=threads,variant=variant,
                                             input_sha256=expected_hash,output_sha256=digest(actual),
                                             relative_original_operator_error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),metadata=plan.metadata()))
                    finally:plan.close()
    sources=('check_pde_cuda.py','pde_cuda.py','pde_reference.py','pde_coefficients.py','prepare_pde_coefficients.py','native/pde_kernels.cu')
    result=dict(scope='Direct CUDA operator preflight only: full original weak-form output and boundary/input integrity; not full PDE/solver completion',
                cases=rows,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
                data_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('fixtures/pde-gll.json','fixtures/pde-gll.npz')})
    (ROOT/'results/pde-cuda-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PDE_DIRECT_CUDA_OPERATOR_PREFLIGHT_PASS',len(rows),flush=True)


if __name__=='__main__':main()
