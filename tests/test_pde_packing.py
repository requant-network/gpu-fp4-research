"""All-axis grouped GEMM layout must equal the original weak operator in FP64."""
import importlib.util
import unittest

import numpy as np

from pde_reference import Mesh,apply_operator


class PdePackingTests(unittest.TestCase):
    def test_grouped_contractions_recover_the_original_operator_and_boundaries(self):
        self.assertIsNotNone(importlib.util.find_spec('pde_packing'),'Grouped physical contraction layout is missing')
        from pde_packing import layout,pack_operator,unpack_product
        for elements in (1,2):
            for q in (4,8,16):
                mesh=Mesh(elements,q)
                values=np.random.default_rng(1220+elements*100+q).integers(-4096,4097,mesh.size).astype(np.float64)/8192
                original=values.copy()
                metadata=layout(mesh)
                left,right=pack_operator(mesh,values,np.float64)
                self.assertEqual(left.shape,(128,128))
                self.assertEqual(right.shape,(metadata['columns'],128))
                self.assertEqual(metadata['columns']%128,0)
                actual=unpack_product(mesh,left@right.T,values,reaction=3.)
                expected=apply_operator(mesh,values,reaction=3.)
                np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
                np.testing.assert_array_equal(actual[mesh.boundary],values[mesh.boundary])
                np.testing.assert_array_equal(values,original)


if __name__=='__main__':unittest.main()
