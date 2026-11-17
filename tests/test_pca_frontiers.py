"""PCA frontiers use only timed outputs and explicitly charge first-use costs."""
import importlib.util
import unittest


class PcaFrontierTests(unittest.TestCase):
    def test_only_timed_quality_and_all_named_setup_costs_are_used(self):
        self.assertIsNotNone(importlib.util.find_spec('pca_frontiers'),'PCA frontier extraction is missing')
        from pca_frontiers import effective_ms,feasible,routes
        good=dict(finite=True,covariance_relative_error=.01,maximum_scaled_eigenvalue_error=.02,
                  eigenvalues_relative_error=.01,mean_standard_deviation_relative_error=0.,subspace_projection_error=.015,
                  negative_eigenvalues=0,target_relative_eigengap=.03)
        bad={**good,'maximum_scaled_eigenvalue_error':100.,'negative_eigenvalues':4}
        case=dict(available=True,successfully_executed=True,arithmetic_valid=True,setup_wall_ms=3.,first_operation_wall_ms=2.,
                  outputs={'timed':{'quality':good},'untimed':{'quality':bad}},
                  metadata={'owned_device_bytes':100,'memory_scope':'Named buffers; not physical peak'},
                  boundaries=[dict(name='resident',observations=[dict(cuda_ms=x,wall_ms=x+.1,output_key='timed') for x in (1.,2.,3.)]),
                              dict(name='host-service',observations=[dict(cuda_ms=None,wall_ms=x,output_key='timed') for x in (2.,3.,4.)])])
        result=routes(case,bootstrap_ms=7.)
        self.assertEqual(result['resident']['median_ms'],2.)
        self.assertEqual(result['host-service']['median_ms'],3.)
        self.assertEqual(result['resident']['setup_ms'],5.)
        self.assertEqual(result['resident']['common_bootstrap_ms'],7.)
        self.assertEqual(result['resident']['quality_error'],.02)
        self.assertTrue(feasible(result['resident'],.02))
        self.assertFalse(feasible(result['resident'],.01))
        self.assertEqual(effective_ms(result['resident'],10,include_bootstrap=False),2.5)
        self.assertEqual(effective_ms(result['resident'],10,include_bootstrap=True),3.2)
    def test_all_strong_reference_families_and_reuse_are_compared(self):
        import pca_frontiers
        self.assertTrue(hasattr(pca_frontiers,'best'),'PCA constrained selection is missing')
        from pca_frontiers import best
        def row(name,precision,time,setup):
            return dict(case_id=name,method={'precision':precision},boundary='resident',median_ms=time,setup_ms=setup,
                        common_bootstrap_ms=0.,finite=True,negative_eigenvalues=0,quality_error=.01)
        candidates=[row('native-fp4','fp4',.5,2.),row('batch-fp32','fp32',2.,0.),
                    row('stream-fp16','fp16',1.,0.),row('batch-fp64','fp64',.25,10.)]
        self.assertEqual(best(candidates,.02,1,False,fp4=False)['case_id'],'stream-fp16')
        self.assertEqual(best(candidates,.02,100,False,fp4=False)['case_id'],'batch-fp64')
        self.assertEqual(best(candidates,.02,1,False,fp4=True)['case_id'],'native-fp4')
        candidates[0]['negative_eigenvalues']=1
        self.assertIsNone(best(candidates,.02,1,False,fp4=True))


if __name__=='__main__':unittest.main()
