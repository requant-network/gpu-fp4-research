"""Retain the actually attempted libCEED cuda/ref q16 static-shared limit."""
import hashlib
import json
from pathlib import Path
import re

import cupy as cp

from pde_ceed import PdeCEED
from pde_reference import Mesh

ROOT=Path(__file__).resolve().parent


def main():
    cp.cuda.Device(0).use()
    assert cp.cuda.runtime.getDeviceProperties(0)['name'].decode()=='NVIDIA GeForce RTX 5060 Ti'
    mesh=Mesh(1,16)
    try:
        plan=PdeCEED(mesh,'ref',reaction=5.)
    except RuntimeError as error:
        original=str(error)
        assert 'uses too much shared data' in original and '0xc000 max' in original,'Unexpected failure, not the established static-shared limit'
        message=re.sub(r'/(?:home|pyenv|__w)/[^\s:]+','<vendor-path>',original)
        result=dict(scope='Observed unavailable libCEED reference-basis construction only; not a successful operator or a completed PDE study',
                    available=False,backend='/gpu/cuda/ref',elements=1,q=16,precision='FP64 API',stage='Canonical basis construction',
                    error=message,raw_error_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                                   ('check_pde_ceed_ref_limit.py','pde_ceed.py','pde_reference.py','pde_coefficients.py','prepare_pde_coefficients.py',
                                    'native/pde_ceed_bridge.cpp','native/pde_ceed_qfunction.h','build_pde_ceed.sh')},
                    data_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('fixtures/pde-gll.json','fixtures/pde-gll.npz')})
        (ROOT/'results/pde-ceed-ref-limit.json').write_text(json.dumps(result,indent=2)+'\n')
        print('PDE_CEED_REF_UNAVAILABLE_LIMIT_RETAINED',flush=True)
    else:
        plan.close()
        raise AssertionError('Established limit did not reproduce; investigate before replacing old evidence')


if __name__=='__main__':main()
