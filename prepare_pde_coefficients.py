"""Prepare canonical high-precision GLL coefficients once; not a GPU study.

80-decimal-digit preparation is rounded once to retained IEEE FP64. CI validates
polynomials, quadrature and weak-form identities instead of requiring bitwise
regeneration from a different platform's polynomial eigensolver.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform

import mpmath as mp
import numpy as np

from pde_coefficients import CHOICES,ROOT,array_binding,validate_coefficients


def generate(q):
    degree=q-1
    with mp.workdps(80):
        polynomial=lambda x:mp.legendre(degree,x)
        prime=lambda x:mp.diff(polynomial,x)
        roots=[mp.findroot(prime,-mp.cos(mp.pi*i/degree),solver='newton',tol=mp.mpf('1e-70')) for i in range(1,degree)]
        nodes=[mp.mpf(-1),*sorted(roots),mp.mpf(1)]
        assert all(nodes[i]<nodes[i+1] for i in range(q-1))
        # Exact antisymmetry avoids retaining the arbitrary final root iteration.
        for i in range(q//2):
            left=(nodes[i]-nodes[q-1-i])/2
            nodes[i],nodes[q-1-i]=left,-left
        values=[polynomial(x) for x in nodes]
        weights=[2/(degree*(degree+1)*value**2) for value in values]
        derivative=mp.matrix(q,q)
        for i in range(q):
            for j in range(q):
                derivative[i,j]=values[i]/(values[j]*(nodes[i]-nodes[j])) if i!=j else (mp.mpf(-degree*(degree+1))/4 if i==0 else mp.mpf(degree*(degree+1))/4 if i==degree else mp.mpf(0))
        stiffness=mp.matrix(q,q)
        for i in range(q):
            for j in range(i,q):
                value=mp.fsum(derivative[k,i]*weights[k]*derivative[k,j] for k in range(q))
                stiffness[i,j]=stiffness[j,i]=value
        arrays=[np.asarray([float(x) for x in nodes],dtype='<f8'),np.asarray([float(x) for x in weights],dtype='<f8'),
                np.asarray([[float(derivative[i,j]) for j in range(q)] for i in range(q)],dtype='<f8'),
                np.asarray([[float(stiffness[i,j]) for j in range(q)] for i in range(q)],dtype='<f8')]
    validate_coefficients(*arrays)
    return arrays


def prepare(directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    assert not (directory/'pde-gll.npz').exists() and not (directory/'pde-gll.json').exists(),'Archive existing canonical coefficients before replacement'
    arrays={}
    for q in CHOICES:
        for field,value in zip(('nodes','weights','derivative','stiffness'),generate(q)):arrays[f'q{q}_{field}']=value
    witness=directory/'pde-gll.npz'
    np.savez_compressed(witness,**arrays)
    manifest=dict(complete=True,study=12,scope='Canonical80-decimal-digit GLL preparation rounded once to retained FP64; not an executed PDE experiment',
                  node_counts=list(CHOICES),decimal_digits=80,numpy_version=np.__version__,mpmath_version=mp.__version__,python=platform.python_version(),
                  prepared_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('prepare_pde_coefficients.py','pde_coefficients.py')},
                  witness_file=witness.name,witness_sha256=hashlib.sha256(witness.read_bytes()).hexdigest(),
                  array_bindings={name:array_binding(value) for name,value in arrays.items()})
    (directory/'pde-gll.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=ROOT/'fixtures')
    args=parser.parse_args()
    result=prepare(args.directory)
    print('PDE_CANONICAL_GLL_PREPARATION_PASS',json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':main()
