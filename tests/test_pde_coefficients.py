"""Frozen GLL bytes prevent platform-specific root solvers changing observations."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


class PdeCoefficientTests(unittest.TestCase):
    def test_high_precision_fixture_is_byte_bound_and_rejects_forged_derivatives(self):
        self.assertIsNotNone(importlib.util.find_spec('pde_coefficients'),'Canonical physical coefficients are missing')
        from pde_coefficients import canonical_gll,validate_coefficients
        from prepare_pde_coefficients import prepare
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as temporary:
            directory=Path(temporary)
            prepare(directory)
            for q in (4,8,16):
                nodes,weights,derivative,stiffness=canonical_gll(q,directory)
                self.assertFalse(nodes.flags.writeable or stiffness.flags.writeable)
                validate_coefficients(nodes,weights,derivative,stiffness)
                forged=derivative.copy();forged[0,1]+=.25
                with self.assertRaises(AssertionError):validate_coefficients(nodes,weights,forged,stiffness)
            path=directory/'pde-gll.npz'
            data=bytearray(path.read_bytes());data[len(data)//2]^=1;path.write_bytes(data)
            with self.assertRaises(AssertionError):canonical_gll(4,directory)

    def test_fixture_validation_also_runs_without_tmpdir_environment(self):
        from unittest.mock import patch
        with patch.dict(os.environ):
            os.environ.pop('TMPDIR', None)
            self.test_high_precision_fixture_is_byte_bound_and_rejects_forged_derivatives()


if __name__=='__main__':unittest.main()
