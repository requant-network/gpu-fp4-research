"""GPU preflight for untiled full-matrix FP32/FP64 covariance/PCA controls."""
import numpy as np


def main():
    import pca_native
    assert hasattr(pca_native,'PcaBatch'),'Full-matrix strong PCA references are missing'
    from pca_cases import inputs
    from pca_reference import quality,stable_reference
    count=0
    for profile in ('uniform','shifted','dependent','sift'):
        data,_=inputs(1153,128,profile,'a')
        target=stable_reference(data)
        for precision in ('fp32','fp64'):
            plan=pca_native.PcaBatch(data,precision)
            try:
                outputs,stages=plan.compute(witness=True)
                metrics=quality(*outputs,target,rank=16)
                assert metrics['finite'] and len(stages)==1
                assert metrics['eigensolver_relative_residual']<1e-10
                if precision=='fp64':assert metrics['covariance_relative_error']<1e-10,metrics
                plan.load(np.ascontiguousarray(-data))
                changed,_=plan.compute(witness=True)
                np.testing.assert_allclose(changed[0],-outputs[0],rtol=0,atol=1e-12)
                np.testing.assert_allclose(changed[1],outputs[1],rtol=1e-6,atol=1e-10)
                count+=1
            finally:plan.close()
    print('PCA_FULL_MATRIX_CONTROL_PASS',count,flush=True)


if __name__=='__main__':main()
