"""Portable PCA cohorts, shifted/weak spectra and the declared streaming grid."""
import importlib.util
import unittest

import numpy as np


class PcaCasesTests(unittest.TestCase):
    def test_cohorts_and_weak_components_are_real_inputs(self):
        self.assertIsNotNone(importlib.util.find_spec('pca_cases'),'PCA input protocol is missing')
        from pca_cases import inputs,methods
        first,_=inputs(256,128,'dependent','a')
        second,_=inputs(256,128,'dependent','b')
        self.assertEqual(first.dtype,np.float32)
        self.assertTrue(first.flags.c_contiguous and np.isfinite(first).all())
        self.assertFalse(np.array_equal(first,second))
        np.testing.assert_array_equal(first,inputs(256,128,'dependent','a')[0])
        spectrum=np.linalg.eigvalsh(np.cov(first.astype(np.float64),rowvar=False))
        self.assertGreater(spectrum[0],0.)
        self.assertLess(spectrum[0]/spectrum[-1],1e-5)
        uniform,_=inputs(256,128,'uniform','a')
        shifted,_=inputs(256,128,'shifted','a')
        self.assertGreater(float(shifted.mean()),30.)
        self.assertGreater(float(uniform.var()),float(shifted.var()))
        grid=methods()
        self.assertEqual(len(grid),84)
        self.assertEqual({m['precision'] for m in grid},{'fp4','fp8','fp16','bf16','fp32','fp64'})
        self.assertEqual({m['centering'] for m in grid},{'raw','global','merge'})
        self.assertEqual({m['chunk'] for m in grid},{256,2048,8192})


if __name__=='__main__':unittest.main()
