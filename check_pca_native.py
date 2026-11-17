"""GPU arithmetic/lifecycle preflight for streaming covariance and PCA.

This is bounded correctness evidence, not the complete study-11 campaign.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np


def main():
    assert importlib.util.find_spec('pca_native') is not None,'Native streaming PCA implementation is missing'
    from pca_native import PcaPlan,library
    from pca_reference import quality,stable_reference
    rng=np.random.Generator(np.random.PCG64(20261117))
    data=rng.integers(-2048,2049,(2048,128),dtype=np.int32).astype(np.float32)/4096
    lib=library()
    count=0
    for shift in (0.,32.):
        source=np.ascontiguousarray(data+np.float32(shift))
        reference=stable_reference(source)
        for precision in ('fp4','fp8','fp16','bf16','fp32','fp64'):
            for centering in ('raw','global','merge'):
                for scaling in (False,True):
                    method=dict(precision=precision,centering=centering,scaling=scaling,accumulation='fp64',chunk=1024)
                    p=PcaPlan(lib,source,method)
                    try:
                        outputs,stages=p.compute(witness=True)
                        metrics=quality(*outputs,reference,rank=16)
                        assert metrics['finite'] and stages
                        assert metrics['eigensolver_relative_residual']<1e-10
                        assert metrics['eigenvector_orthogonality_error']<1e-10
                        if precision=='fp64' and centering!='raw':
                            assert metrics['covariance_relative_error']<1e-10,metrics
                        fresh,_=p.compute()
                        np.testing.assert_array_equal(fresh[1],outputs[1])
                        p.load(np.ascontiguousarray(-source))
                        changed,_=p.compute(witness=True)
                        np.testing.assert_allclose(changed[0],-outputs[0],rtol=0,atol=1e-12)
                        np.testing.assert_allclose(changed[1],outputs[1],rtol=1e-5,atol=1e-8)
                        count+=1
                    finally:p.close()
        print('PROFILE_ALL_PRECISIONS_PASS',shift,flush=True)
    # Tail padding and FP32 accumulation are separate controls, not silently dropped rows.
    for precision in ('fp4','fp32','fp64'):
        method=dict(precision=precision,centering='merge',scaling=True,accumulation='fp32',chunk=256)
        p=PcaPlan(lib,data[:1153].copy(),method)
        try:
            outputs,stages=p.compute(witness=True)
            assert len(stages)==5 and stages[-1]['count']==129
            metrics=quality(*outputs,stable_reference(data[:1153]),rank=16)
            assert metrics['finite'] and metrics['eigensolver_relative_residual']<1e-4
            count+=1
        finally:p.close()
    print('PCA_NATIVE_PREFLIGHT_PASS',count,flush=True)


if __name__=='__main__':main()
