import unittest

import numpy as np

from spmm_formats import prepare
from spmm_reference import csr
from spmm_validation import expected_buffers, hybrid_product, metrics, sparse_product, whole_product


class SpmmValidationTests(unittest.TestCase):
    def test_sample_oracle_matches_full_products(self):
        from verify_spmm import samples
        rng = np.random.Generator(np.random.PCG64(534))
        a = rng.integers(-1024, 1025, (23, 29)).astype(np.float32) / 1024
        b = rng.integers(-1024, 1025, (29, 7)).astype(np.float32) / 1024
        source = csr(a)
        coords = np.array([[0, 0], [22, 6], [11, 3], [4, 2]])
        f = prepare(source, a.shape, {"kind": 1, "block": 16, "fill": .25}, 'scrambled', 534)
        for precision in (1, 2):
            for hybrid in (False, True):
                gold = hybrid_product(f, b, precision) if hybrid else whole_product(source, a.shape, b, precision)
                selected, bound = samples(source, a.shape, f, b, precision, coords, hybrid)
                np.testing.assert_allclose(selected, gold[coords[:, 0], coords[:, 1]], rtol=0, atol=2e-13)
                self.assertTrue(np.all(bound >= 0))
                self.assertTrue(np.all(abs(selected + bound * 100 + .1 - selected) > bound))

    def test_sparse_and_hybrid_rounding_products(self):
        rng = np.random.Generator(np.random.PCG64(533))
        a = rng.integers(-16383, 16384, (21, 27)).astype(np.float32) / 16384
        a[rng.integers(0, 4, a.shape) == 0] = 0
        b = rng.integers(-16383, 16384, (27, 9)).astype(np.float32) / 16384
        b[:, 0] = 0
        source = csr(a)
        np.testing.assert_allclose(sparse_product(source, a.shape, b), a.astype(np.float64) @ b, rtol=0, atol=0)
        for order in ("natural", "degree", "scrambled"):
            f = prepare(source, a.shape, {"kind": 1, "block": 16, "fill": .25}, order, 533)
            for precision in (1, 2):
                expected = hybrid_product(f, b, precision)
                result = metrics(expected.astype(np.float32), sparse_product(source, a.shape, b), expected)
                self.assertTrue(result['arithmetic_model_checked'])
                self.assertEqual(result['exact_zero_columns_checked'], 1)
                buffers = expected_buffers(f, b, precision, 0)
                self.assertEqual(buffers[0].size, f['block_count'] * 16**2 * 2)
                self.assertEqual(buffers[1].size, f['block_count'] * 16 * b.shape[1] * 2)
        np.testing.assert_array_equal(whole_product(source, a.shape, b, 0), sparse_product(source, a.shape, b))

    def test_arithmetic_and_zero_column_failures(self):
        original = np.ones((4, 3), dtype=np.float64)
        original[:, 0] = 0
        wrong = original.astype(np.float32)
        wrong[1, 2] += 1
        with self.assertRaises(AssertionError):
            metrics(wrong, original, original)
        opaque = metrics(wrong, original, original, opaque_fp32_ell=True)
        self.assertFalse(opaque['arithmetic_model_checked'])
        self.assertGreater(opaque['relative_frobenius_error'], .1)
        wrong[0, 0] = 1
        with self.assertRaises(AssertionError):
            metrics(wrong, original, original, opaque_fp32_ell=True)


if __name__ == '__main__':
    unittest.main()
