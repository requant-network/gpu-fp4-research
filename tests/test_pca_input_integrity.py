"""Host input integrity is byte-based and rejects mutations without widening bounds."""
import importlib.util
import unittest

import numpy as np

from reference import digest


class PcaInputIntegrityTests(unittest.TestCase):
    def test_frozen_original_input_accepts_exact_data_and_rejects_a_bit_flip(self):
        self.assertIsNotNone(importlib.util.find_spec('pca_input_integrity'),'PCA original-input guard is missing')
        from pca_input_integrity import freeze_input,verify_input
        values=np.arange(12,dtype=np.float32).reshape(4,3).copy()
        expected=digest(values)
        freeze_input(values,expected)
        self.assertFalse(values.flags.writeable)
        self.assertEqual(verify_input(values,expected),expected)
        values.setflags(write=True)
        values.view(np.uint32)[2,1]^=np.uint32(1<<21)
        values.setflags(write=False)
        with self.assertRaisesRegex(AssertionError,'input bytes changed'):
            verify_input(values,expected)


if __name__=='__main__':unittest.main()
