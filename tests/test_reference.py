import unittest
import numpy as np
from reference import (FP4_LEVELS, FP8_LEVELS, check_output, decode_fp8, digest, encode_fp8,
                       inputs, nearest_codes, quantize)


class ReferenceTests(unittest.TestCase):
    def test_all_finite_fp8_codes_and_ties(self):
        codes = np.arange(127, dtype=np.uint8)
        np.testing.assert_array_equal(encode_fp8(decode_fp8(codes)), codes)
        np.testing.assert_array_equal(encode_fp8(decode_fp8(codes | 128)), codes | 128)
        self.assertEqual(FP8_LEVELS[-1], 448)
        self.assertEqual(FP8_LEVELS[1], 1 / 512)
        np.testing.assert_array_equal(encode_fp8(np.array([1.0625, 1.1875], np.float32)), [56, 58])

    def test_fp4_ties_and_representable_values(self):
        np.testing.assert_array_equal(nearest_codes(FP4_LEVELS, FP4_LEVELS), np.arange(8))
        ties = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5], np.float32)
        np.testing.assert_array_equal(nearest_codes(ties, FP4_LEVELS), [0, 2, 2, 4, 4, 6, 6])

    def test_fp4_block_scales_and_packing(self):
        x = np.tile(np.array([0, .5, 1, 1.5, 2, 3, 4, 6] * 2, np.float32), (2, 1))
        x[1] *= -2
        raw, scales, decoded = quantize(x, "fp4")
        np.testing.assert_array_equal(raw[0], [16, 50, 84, 118] * 2)
        np.testing.assert_array_equal(scales.ravel(), [56, 64])
        np.testing.assert_array_equal(decoded, x)

    def test_all_modes_zero_and_known_values(self):
        x = np.zeros((2, 16), np.float32)
        x[1] = np.arange(-8, 8, dtype=np.float32)
        for mode in ("fp4", "fp8", "fp16", "bf16", "fp32"):
            _, _, q = quantize(x, mode)
            np.testing.assert_array_equal(q[0], x[0])
            self.assertTrue(np.isfinite(q).all())
        for mode in ("fp16", "bf16", "fp32"):
            np.testing.assert_array_equal(quantize(x, mode)[2], x)

    def test_invalid_inputs(self):
        for x, mode in ((np.array([[np.nan]], np.float32), "fp8"),
                        (np.array([[np.inf]], np.float32), "fp16"),
                        (np.zeros((1, 15)), "fp4"), (np.zeros((1, 16)), "invalid")):
            with self.assertRaises(ValueError):
                quantize(x, mode)

    def test_input_reproducibility_and_arithmetic_rejection(self):
        a, b = inputs(1, (4, 5, 128), "outliers")
        self.assertEqual(digest(a), "2f327e76751ff3a90a1cab87e5474d5e62233431e5f0fc19000c195eeeb1bc9c")
        self.assertEqual(digest(b), "9d748bbc720daa2a5586b044eeac3ac92c9af2beaffdea9b45a70fd02f23886a")
        np.testing.assert_array_equal(a, inputs(1, (4, 5, 128), "outliers")[0])
        good = (a.astype(np.float64) @ b.astype(np.float64).T).astype(np.float32)
        self.assertLess(check_output(good, a, b), 1)
        bad = good.copy()
        bad[0, 0] = 1
        with self.assertRaises(ValueError):
            check_output(bad, a, b)


if __name__ == "__main__":
    unittest.main()
