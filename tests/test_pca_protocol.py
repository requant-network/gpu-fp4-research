"""PCA retained-output identity and lossless stage sampling contracts."""
import importlib.util
import unittest

import numpy as np


class PcaProtocolTests(unittest.TestCase):
    def test_output_binding_keeps_covariance_spectrum_and_leading_subspace(self):
        self.assertIsNotNone(importlib.util.find_spec('pca_protocol'),'PCA artifact protocol is missing')
        from pca_protocol import retain_output,validate_binding
        outputs=(np.arange(3,dtype=np.float64),np.diag([1.,2.,3.]),np.array([1.,2.,3.]),np.eye(3))
        arrays,binding,key=retain_output(outputs,rank=1)
        self.assertEqual(set(arrays),{'mean','covariance','eigenvalues','leading'})
        self.assertEqual(arrays['leading'].shape,(3,1))
        validate_binding(arrays,binding)
        self.assertEqual(len(key),64)
        outputs[0][0]=99
        self.assertEqual(arrays['mean'][0],0.)
        arrays['covariance'][0,0]=42
        with self.assertRaises(AssertionError):validate_binding(arrays,binding)
    def test_stage_retention_binds_real_samples_without_claiming_full_matrix(self):
        import pca_protocol
        self.assertTrue(hasattr(pca_protocol,'retain_stage'),'Stage-retention protocol is missing')
        from reference import digest
        source=np.arange(12,dtype=np.float32).reshape(3,4)
        product=source@source.T
        stage=dict(begin=0,count=4,source=source,product=product,center=np.zeros(3),scale=np.ones(3,np.float32),
                   operand_sha256=digest(source),product_sha256=digest(product),encoded_sha256={},maximum_product_bound_ratio=0.)
        arrays,record=pca_protocol.retain_stage(stage,full=False)
        self.assertNotIn('full_product',arrays)
        self.assertFalse(record['full_product_retained'])
        np.testing.assert_array_equal(arrays['product_samples'],product.ravel()[arrays['product_indices']])
        pca_protocol.validate_binding(arrays,record['array_bindings'])
        complete,metadata=pca_protocol.retain_stage(stage,full=True)
        np.testing.assert_array_equal(complete['full_product'],product)
        self.assertTrue(metadata['full_product_retained'])


if __name__=='__main__':unittest.main()
