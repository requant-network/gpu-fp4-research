import unittest

import numpy as np

from spmm_formats import prepare, reconstruct_format, split, validate
from spmm_reference import csr, partition, reconstruct


class SpmmFormatTests(unittest.TestCase):
    def test_direct_csr_matches_independent_dense_partition(self):
        rng = np.random.Generator(np.random.PCG64(531))
        for shape in ((7, 11), (32, 32), (1, 1)):
            a = rng.integers(-7, 8, shape).astype(np.float32)
            a[rng.integers(0, 3, shape) == 0] = 0
            for block in (2, 4, 16):
                for fill in (.25, .75, 1):
                    fast = split(csr(a), shape, block, fill)
                    oracle = partition(a, block, fill)
                    for key in ("rowptr", "columns", "values", "block_rowptr", "block_columns", "block_values"):
                        np.testing.assert_array_equal(fast[key], oracle[key])
                    np.testing.assert_array_equal(reconstruct(fast), a)

    def test_all_formats_orders_padding_and_zero(self):
        rng = np.random.Generator(np.random.PCG64(532))
        for a in (rng.integers(-2, 3, (19, 27)).astype(np.float32), np.zeros((3, 5), dtype=np.float32)):
            for kind in range(5):
                for order in ("natural", "degree", "scrambled"):
                    f = prepare(csr(a), a.shape, {"kind": kind, "block": 16 if kind else 1, "fill": .25}, order, 532)
                    np.testing.assert_array_equal(reconstruct_format(f), a)

    def test_rejects_bad_csr(self):
        source = csr(np.eye(3, dtype=np.float32))
        for changed in ({"rowptr": np.array([0, 2, 1, 3], dtype=np.int32)},
                        {"columns": np.array([0, 1, 3], dtype=np.int32)},
                        {"values": np.array([1, np.nan, 1], dtype=np.float32)},
                        {"rowptr": np.array([0, 2, 2, 3], dtype=np.int32), "columns": np.array([0, 0, 2], dtype=np.int32)}):
            with self.assertRaises(ValueError):
                validate(dict(source, **changed), (3, 3))


if __name__ == "__main__":
    unittest.main()
