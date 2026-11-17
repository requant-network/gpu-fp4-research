"""Byte-frozen FP64 GLL data and independent polynomial/weak-form checks.

The same NumPy version produced different q8/q16 root/derivative bytes on the two
hosts. Production must load retained coefficients, not silently regenerate them
with a platform-dependent eigensolver. This is distinct from an input bit fault.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import Legendre

ROOT=Path(__file__).resolve().parent
CHOICES=(4,8,16)


def array_binding(value):
    return dict(shape=list(value.shape),dtype=value.dtype.str,sha256=hashlib.sha256(value.tobytes()).hexdigest())


def validate_coefficients(nodes,weights,derivative,stiffness):
    q=len(nodes)
    assert q in CHOICES
    for value,shape in ((nodes,(q,)),(weights,(q,)),(derivative,(q,q)),(stiffness,(q,q))):
        assert value.dtype==np.dtype('float64') and value.shape==shape and np.isfinite(value).all()
    assert nodes[0]==-1 and nodes[-1]==1 and np.all(np.diff(nodes)>0) and np.all(weights>0)
    epsilon=np.finfo(np.float64).eps
    vandermonde=np.vander(nodes,N=q,increasing=True)
    expected=np.zeros_like(vandermonde)
    for degree in range(1,q):expected[:,degree]=degree*nodes**(degree-1)
    envelope=128*q*epsilon*(abs(derivative)@abs(vandermonde)+abs(expected)+1)
    assert np.all(abs(derivative@vandermonde-expected)<=envelope),'Canonical derivative fails polynomial differentiation'
    polynomial=Legendre.basis(q-1)
    assert np.max(abs(polynomial.deriv()(nodes[1:-1])))<=128*q*epsilon*max(1.,(q-1)**2)
    for degree in range(2*q-2):
        exact=0. if degree%2 else 2./(degree+1)
        values=weights*nodes**degree
        assert abs(float(values.sum())-exact)<=128*q*epsilon*(float(abs(values).sum())+abs(exact)+1),'Canonical quadrature moment is invalid'
    product=derivative.T@(weights[:,None]*derivative)
    bound=128*q*epsilon*(abs(derivative).T@(weights[:,None]*abs(derivative))+abs(stiffness)+1)
    assert np.all(abs(stiffness-product)<=bound),'Canonical weak stiffness is not D.T W D'
    np.testing.assert_array_equal(stiffness,stiffness.T)
    assert float(np.linalg.eigvalsh(stiffness).min())>=-float(np.linalg.norm(bound,ord=2))
    return True


def canonical_gll(q,directory=None):
    if q not in CHOICES:raise ValueError('Canonical GLL node count must be4,8,16')
    directory=Path(directory) if directory is not None else ROOT/'fixtures'
    manifest=json.loads((directory/'pde-gll.json').read_text())
    assert manifest['complete'] and manifest['node_counts']==list(CHOICES)
    for name,expected in manifest['source_sha256'].items():
        assert Path(name).name==name and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
    assert manifest['witness_file']=='pde-gll.npz'
    witness=directory/manifest['witness_file']
    assert hashlib.sha256(witness.read_bytes()).hexdigest()==manifest['witness_sha256'],'Canonical coefficient bytes changed'
    values=[]
    with np.load(witness,allow_pickle=False) as archive:
        assert set(archive.files)=={f'q{n}_{field}' for n in CHOICES for field in ('nodes','weights','derivative','stiffness')}
        for field in ('nodes','weights','derivative','stiffness'):
            key=f'q{q}_{field}'
            value=archive[key]
            assert array_binding(value)==manifest['array_bindings'][key]
            value.setflags(write=False)
            values.append(value)
    validate_coefficients(*values)
    return tuple(values)
