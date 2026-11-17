"""Stable covariance/PCA reference and quality contracts."""
import importlib.util
import unittest

import numpy as np


class PcaReferenceTests(unittest.TestCase):
    def test_stable_shifted_covariance_and_independent_merge(self):
        self.assertIsNotNone(importlib.util.find_spec('pca_reference'),'Stable PCA reference is not implemented')
        from pca_reference import stable_reference,streaming_reference
        data=np.array([[0,0],[1,2],[2,-1],[3,1]],dtype=np.float32)
        shifted=data+np.float32(2**20)
        expected=np.array([[5/3,0],[0,5/3]],np.float64)
        reference=stable_reference(shifted,chunk=3)
        np.testing.assert_array_equal(reference['mean'],np.array([2**20+1.5,2**20+.5]))
        np.testing.assert_allclose(reference['covariance'],expected,rtol=0,atol=1e-14)
        mean,cov=streaming_reference(shifted,chunk=3)
        np.testing.assert_array_equal(mean,reference['mean'])
        np.testing.assert_allclose(cov,expected,rtol=0,atol=1e-10)
    def test_subspace_is_sign_invariant_but_weak_eigenvalues_are_not_hidden(self):
        import pca_reference
        self.assertTrue(hasattr(pca_reference,'quality'),'PCA accuracy metrics are missing')
        covariance=np.diag([1e-6,1.,10.])
        values=np.array([1e-6,1.,10.])
        vectors=np.eye(3)
        reference=dict(mean=np.zeros(3),covariance=covariance,eigenvalues=values,eigenvectors=vectors)
        exact=pca_reference.quality(np.zeros(3),covariance,values,-vectors,reference,rank=1)
        self.assertEqual(exact['subspace_projection_error'],0.)
        self.assertEqual(exact['maximum_scaled_eigenvalue_error'],0.)
        bad=values.copy()
        bad[0]=0.
        metrics=pca_reference.quality(np.zeros(3),np.diag(bad),bad,vectors,reference,rank=1)
        self.assertLess(metrics['covariance_relative_error'],1e-6)
        self.assertEqual(metrics['maximum_scaled_eigenvalue_error'],1.)
        self.assertEqual(metrics['subspace_projection_error'],0.)
        self.assertGreater(metrics['target_relative_eigengap'],.8)
    def test_retained_leading_vectors_have_an_explicit_quality_scope(self):
        import pca_reference
        self.assertTrue(hasattr(pca_reference,'quality_leading'),'Retained leading-subspace quality is missing')
        covariance=np.diag([1e-6,1.,10.])
        values=np.array([1e-6,1.,10.])
        reference=dict(mean=np.zeros(3),covariance=covariance,eigenvalues=values,eigenvectors=np.eye(3))
        good=pca_reference.quality_leading(np.zeros(3),covariance,values,np.eye(3)[:,-1:],reference)
        self.assertEqual(good['subspace_projection_error'],0.)
        self.assertEqual(good['leading_eigensolver_relative_residual'],0.)
        self.assertEqual(good['leading_orthogonality_error'],0.)
        wrong=pca_reference.quality_leading(np.zeros(3),covariance,values,np.eye(3)[:,:1],reference)
        self.assertGreater(wrong['subspace_projection_error'],.9)
        self.assertGreater(wrong['leading_eigensolver_relative_residual'],.9)
        weak=values.copy();weak[0]=0.
        failed=pca_reference.quality_leading(np.zeros(3),np.diag(weak),weak,np.eye(3)[:,-1:],reference)
        self.assertEqual(failed['maximum_scaled_eigenvalue_error'],1.)


if __name__=='__main__':unittest.main()
