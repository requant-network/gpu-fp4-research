import unittest

import numpy as np

from spmm_reference import csr, partition, product, reconstruct


class SpmmReferenceTests(unittest.TestCase):
    def test_lossless_partition_and_product(self):
        rng = np.random.Generator(np.random.PCG64(382))
        for shape in ((7, 11), (16, 16), (1, 9)):
            a = rng.integers(-3, 4, shape).astype(np.float32)
            a[rng.integers(0, 4, shape) == 0] = 0
            b = rng.integers(-3, 4, (shape[1], 5)).astype(np.float32)
            for block in (2, 4, 16):
                for fill in (.25, .75, 1.):
                    parts = partition(a, block, fill)
                    np.testing.assert_array_equal(reconstruct(parts), a)
                    np.testing.assert_array_equal(product(parts, b), a.astype(np.float64) @ b)
                    self.assertEqual(int(parts['block_rowptr'][-1]), len(parts['block_values']))
                    nnz = len(parts['values']) + np.count_nonzero(parts['block_values'])
                    self.assertEqual(nnz, np.count_nonzero(a))

    def test_threshold_charges_padding_and_zero_blocks(self):
        a = np.ones((3, 3), dtype=np.float32)
        parts = partition(a, 4, .75)
        self.assertEqual(len(parts['block_values']), 0)
        parts = partition(a, 4, .5)
        self.assertEqual(len(parts['block_values']), 1)
        self.assertEqual(len(parts['values']), 0)
        np.testing.assert_array_equal(reconstruct(parts), a)
        parts = partition(np.zeros((3, 3), dtype=np.float32), 4, .25)
        self.assertEqual(len(parts['block_values']), 0)
        np.testing.assert_array_equal(product(parts, np.ones((3, 2))), 0)

    def test_validation_and_csr_order(self):
        a = np.array([[0, 3, -2], [0, 0, 0]], dtype=np.float32)
        parts = csr(a)
        np.testing.assert_array_equal(parts['rowptr'], [0, 2, 2])
        np.testing.assert_array_equal(parts['columns'], [1, 2])
        for matrix, block, fill in ((a.astype(np.float64), 4, .5), (a, 3, .5), (a, 4, 0), (a, 4, 2),
                                    (np.array([[np.inf]], dtype=np.float32), 4, .5)):
            with self.assertRaises(ValueError):
                partition(matrix, block, fill)


if __name__ == '__main__':
    unittest.main()
