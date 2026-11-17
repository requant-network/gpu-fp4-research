"""Original high-order nodal math, independent of GPU packing/backends."""
import importlib.util
import unittest

import numpy as np


class PdeReferenceTests(unittest.TestCase):
    def test_gll_differentiates_nodal_polynomials_and_integrates_its_exact_range(self):
        self.assertIsNotNone(importlib.util.find_spec('pde_reference'),'High-order physical reference is missing')
        from pde_reference import gll
        for q in (4,8):
            nodes,weights,derivative=gll(q)
            self.assertEqual(nodes.shape,(q,))
            np.testing.assert_array_equal(nodes[[0,-1]],[-1.,1.])
            self.assertTrue(np.all(np.diff(nodes)>0) and np.all(weights>0))
            for degree in range(q):
                expected=degree*nodes**(degree-1) if degree else np.zeros(q)
                np.testing.assert_allclose(derivative@nodes**degree,expected,rtol=2e-13,atol=2e-13)
            for degree in range(2*q-2):
                expected=0. if degree%2 else 2./(degree+1)
                self.assertAlmostEqual(float(weights@nodes**degree),expected,places=13)


    def test_conforming_operator_matches_kronecker_assembly_and_dirichlet_identity(self):
        from pde_reference import Mesh,apply_operator,dense_operator
        mesh=Mesh(elements=2,q=4)
        self.assertAlmostEqual(float(mesh.mass.sum()),1.,places=13)
        self.assertFalse(mesh.indices.flags.writeable)
        values=np.random.default_rng(1202).normal(size=mesh.size)
        original=values.copy()
        matrix=dense_operator(mesh,reaction=3.)
        np.testing.assert_allclose(matrix,matrix.T,rtol=0,atol=2e-14)
        self.assertGreater(float(np.linalg.eigvalsh(matrix).min()),0.)
        actual=apply_operator(mesh,values,reaction=3.)
        np.testing.assert_allclose(actual,matrix@values,rtol=2e-13,atol=2e-13)
        np.testing.assert_array_equal(actual[mesh.boundary],values[mesh.boundary])
        np.testing.assert_array_equal(values,original)


    def test_manufactured_poisson_has_independently_checked_spatial_convergence(self):
        from pde_reference import Mesh,apply_operator,dense_operator,manufactured
        errors=[]
        for elements in (1,2):
            mesh=Mesh(elements=elements,q=4)
            problem=manufactured(mesh,'a',reaction=0.)
            solution=np.linalg.solve(dense_operator(mesh),problem['rhs'])
            residual=apply_operator(mesh,solution)-problem['rhs']
            self.assertLess(float(np.linalg.norm(residual)/np.linalg.norm(problem['rhs'])),1e-12)
            difference=solution-problem['exact']
            errors.append(float(np.sqrt(np.dot(mesh.mass,difference**2)/np.dot(mesh.mass,problem['exact']**2))))
        self.assertLess(errors[1],errors[0]/4)


    def test_implicit_heat_has_true_step_residuals_and_temporal_convergence(self):
        from pde_reference import Mesh,apply_operator,implicit_euler_control,manufactured
        mesh=Mesh(elements=1,q=8)
        final_time=1./32
        exact=manufactured(mesh,'a',time=final_time)['exact']
        errors=[]
        for steps in (1,2,4):
            result=implicit_euler_control(mesh,'a',steps,final_time)
            self.assertEqual(len(result['history']),steps)
            for row in result['history']:
                residual=apply_operator(mesh,row['state'],reaction=1./result['dt'])-row['rhs']
                self.assertLess(float(np.linalg.norm(residual)/np.linalg.norm(row['rhs'])),2e-12)
            difference=result['state']-exact
            errors.append(float(np.sqrt(np.dot(mesh.mass,difference**2)/np.dot(mesh.mass,exact**2))))
        self.assertLess(errors[1],errors[0]*.7)
        self.assertLess(errors[2],errors[1]*.7)


if __name__=='__main__':unittest.main()
