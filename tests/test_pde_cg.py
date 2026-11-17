"""Physical diagonal and complete solver controls; no GPU arithmetic oracle reuse."""
import importlib.util
import unittest
import numpy as np
from pde_reference import Mesh,dense_operator,manufactured


class PdeCgTests(unittest.TestCase):
    def test_physical_spd_solve_and_independent_diagonal(self):
        self.assertIsNotNone(importlib.util.find_spec('pde_cg'),'Physical PCG controller is missing')
        from pde_cg import jacobi_diagonal,solve
        for elements in (1,2):
            mesh=Mesh(elements,4)
            matrix=dense_operator(mesh,reaction=2.)
            diagonal=jacobi_diagonal(mesh,reaction=2.)
            np.testing.assert_allclose(diagonal,np.diag(matrix),rtol=3e-15,atol=3e-15)
            exact=manufactured(mesh,'b')['exact']
            rhs=matrix@exact;rhs.setflags(write=False)
            before=rhs.copy()
            result=solve(lambda x:matrix@x,rhs,diagonal,xp=np,tolerance=1e-12,max_iterations=100)
            np.testing.assert_allclose(result['x'],exact,rtol=1e-10,atol=1e-10)
            self.assertLess(np.linalg.norm(matrix@result['x']-rhs)/np.linalg.norm(rhs),1e-12)
            self.assertEqual(result['status'],'internal-converged')
            np.testing.assert_array_equal(rhs,before)

    def test_reliable_correction_rejects_false_convergence_and_negative_curvature(self):
        import pde_cg
        self.assertTrue(hasattr(pde_cg,'solve_reliable'),'Original-residual correction/fallback is missing')
        matrix=np.array([[5.,1.,.5],[1.,4.,.25],[.5,.25,3.]])
        exact=np.array([.25,-.5,1.])
        rhs=matrix@exact;rhs.setflags(write=False)
        diagonal=np.diag(matrix).copy()
        false=pde_cg.solve(lambda x:2*(matrix@x),rhs,diagonal,xp=np,tolerance=1e-12,max_iterations=32)
        self.assertEqual(false['status'],'internal-converged')
        self.assertGreater(np.linalg.norm(rhs-matrix@false['x'])/np.linalg.norm(rhs),.4)
        for sign in (-1.,2.):
            result=pde_cg.solve_reliable(lambda x:sign*(matrix@x),lambda x:matrix@x,rhs,diagonal,
                                         xp=np,tolerance=1e-11,max_iterations=64,low_budget=8,segment_iterations=4)
            self.assertEqual(result['status'],'original-converged')
            self.assertTrue(result['fallback'])
            np.testing.assert_allclose(result['x'],exact,rtol=1e-10,atol=1e-10)
            self.assertLess(np.linalg.norm(rhs-matrix@result['x'])/np.linalg.norm(rhs),1e-11)


if __name__=='__main__':unittest.main()
