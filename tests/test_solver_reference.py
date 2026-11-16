import math
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solver_reference import (accuracy, controlled, dyadic_float, exact_residual_rows,
                              read_matrix_market, real_system)


class SolverReferenceTests(unittest.TestCase):
    def test_replay_rejects_fabricated_accuracy(self):
        from verify_solver import check_accuracy
        a, b, truth = controlled(16, 12, 1, 56)
        quality = accuracy(a, b, truth, truth)
        check_accuracy(a, b, truth, truth, quality)
        wrong = dict(quality, residual_per_rhs=[.1], max_relative_residual=.1)
        with self.assertRaises(AssertionError):
            check_accuracy(a, b, truth, truth, wrong)

    def test_exact_manufactured_system(self):
        for exponent in (0, 12, 40):
            a, b, truth = controlled(16, exponent, 4, 52)
            self.assertTrue(np.array_equal(a, a.T))
            for row in (0, 5, 15):
                for column in range(4):
                    exact = sum(Fraction(float(a[row, k])) * Fraction(float(truth[k, column])) for k in range(16))
                    self.assertEqual(exact, Fraction(float(b[row, column])))
            np.testing.assert_array_equal(exact_residual_rows(a, b, truth, [0, 5, 15]), 0)
        a, b, truth = controlled(16, 12, 1, 52)
        self.assertAlmostEqual(np.linalg.cond(a), 4096, places=5)
        a2, b2, _ = controlled(16, 12, 1, 52, 1)
        np.testing.assert_array_equal(a, a2)
        self.assertFalse(np.array_equal(b, b2))
        self.assertTrue(accuracy(a, b, truth, truth)["target_pass"])

    def test_forward_error_and_each_rhs_are_required(self):
        a = np.diag([1., 2.**-40, 1., 1.])
        truth = np.ones((4, 4))
        b = a @ truth
        wrong = truth.copy()
        wrong[1, 3] = 0
        result = accuracy(a, b, truth, wrong)
        self.assertLess(result["max_relative_residual"], 1e-12)
        self.assertGreater(result["max_forward_error"], .1)
        self.assertFalse(result["target_pass"])

    def test_integer_oracle_and_rounding(self):
        large = 2.**27
        a = np.array([[large + 1, large]])
        x = np.array([[large + 1], [-(large + 2)]])
        np.testing.assert_array_equal(exact_residual_rows(a, np.zeros((1, 1)), x, [0]), [[1.]])
        for integer, exponent in ((1, 1075), (3, 1075), (2**54 + 1, 54), (2**54 + 3, 54), (-(2**54 + 3), 54)):
            self.assertEqual(dyadic_float(integer, exponent), float(Fraction(integer, 2**exponent)))
        self.assertEqual(dyadic_float(1 << 5000, 4999), 2.)
        self.assertTrue(math.isinf(dyadic_float(1 << 5000, 0)))

    def test_matrix_market_and_real_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "matrix.mtx"
            path.write_text("%%MatrixMarket matrix coordinate real symmetric\n% comment\n2 2 3\n1 1 2\n2 1 .25\n2 2 1\n", encoding="ascii")
            matrix = read_matrix_market(path)
            a, b, truth = real_system(matrix, 1, 42, scale_exponent=-20)
            np.testing.assert_array_equal(a @ truth, b)
            path.write_text("%%MatrixMarket matrix coordinate real symmetric\n2 2 3\n1 1 2\n2 1 .25\n1 2 .25\n", encoding="ascii")
            with self.assertRaises(ValueError):
                read_matrix_market(path)


if __name__ == "__main__":
    unittest.main()
