"""Changed-input libCEED weak-form output/teardown controls; not a full PDE study."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import cupy as cp
import numpy as np

from pde_ceed import PdeCEED
from pde_reference import Mesh,apply_operator

ROOT=Path(__file__).resolve().parent


def digest(value):return hashlib.sha256(value.tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=('ref','shared','gen'),required=True)
    args=parser.parse_args()
    cp.cuda.Device(0).use()
    assert cp.cuda.runtime.getDeviceProperties(0)['name'].decode()=='NVIDIA GeForce RTX 5060 Ti'
    rows=[]
    for elements in (1,2):
        for q in (4,8,16):
            mesh=Mesh(elements,q)
            rng=np.random.Generator(np.random.PCG64(1210+elements*100+q))
            original=rng.integers(-4096,4097,mesh.size,dtype=np.int32).astype(np.float64)/8192
            absolute_mesh=copy.copy(mesh)
            absolute_mesh.stiffness=abs(mesh.derivative).T@(mesh.weights[:,None]*abs(mesh.derivative))
            plan=PdeCEED(mesh,args.backend,reaction=5.)
            try:
                for variant in (0,1):
                    values=original.copy() if variant==0 else -original
                    values.setflags(write=False)
                    expected_hash=digest(values)
                    expected=apply_operator(mesh,values,reaction=5.)
                    absolute=apply_operator(absolute_mesh,abs(values),reaction=5.)
                    envelope=512*(q+16)*np.finfo(np.float64).eps*(absolute+abs(expected)+np.finfo(np.float64).tiny)
                    plan.load(values)
                    assert digest(cp.asnumpy(plan.data))==expected_hash
                    plan.run()
                    actual=plan.result()
                    assert np.isfinite(actual).all() and np.all(abs(actual-expected)<=envelope),'libCEED output differs from independent original weak form'
                    np.testing.assert_array_equal(actual[mesh.boundary],values[mesh.boundary])
                    assert digest(values)==expected_hash and digest(cp.asnumpy(plan.data))==expected_hash
                    rows.append(dict(elements=elements,q=q,variant=variant,input_sha256=expected_hash,output_sha256=digest(actual),
                                     relative_original_operator_error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),metadata=plan.metadata()))
            finally:plan.close()
    sources=('check_pde_ceed.py','pde_ceed.py','pde_reference.py','pde_coefficients.py','prepare_pde_coefficients.py',
             'native/pde_ceed_bridge.cpp','native/pde_ceed_qfunction.h','build_pde_ceed.sh')
    result=dict(scope='libCEED original weak-form/input/boundary controls only; not full PDE/solver/performance completion',
                cases=rows,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
                data_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('fixtures/pde-gll.json','fixtures/pde-gll.npz')})
    (ROOT/'results'/f'pde-ceed-{args.backend}-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PDE_CEED_ORIGINAL_OPERATOR_PREFLIGHT_PASS',args.backend,len(rows),flush=True)


if __name__=='__main__':main()
