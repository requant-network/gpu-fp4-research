"""Independent CPU arithmetic models must reject coherently forged outputs."""
import importlib.util
import unittest
import numpy as np
from pde_reference import Mesh,apply_operator


class PdeModelTests(unittest.TestCase):
    def test_original_direct_and_vendor_models_and_forged_native_result(self):
        self.assertIsNotNone(importlib.util.find_spec('pde_models'),'Independent full arithmetic models are missing')
        from pde_models import operator_model,check_output
        for q in (4,8,16):
            mesh=Mesh(1,q)
            values=np.random.default_rng(1290+q).integers(-1024,1025,mesh.size).astype(np.float64)/2048
            physical=apply_operator(mesh,values)
            for family in ('cuda','ceed'):
                model=operator_model(mesh,values,dict(family=family,precision='fp64'))
                check_output(model,physical,mesh,values)
            for family in ('tensor','small'):
                model=operator_model(mesh,values,dict(family=family,precision='fp16'))
                check_output(model,model['expected'],mesh,values)
                forged=np.zeros_like(values);forged[mesh.boundary]=values[mesh.boundary]
                with self.assertRaisesRegex(AssertionError,'original-data arithmetic'):
                    check_output(model,forged,mesh,values)


if __name__=='__main__':unittest.main()
