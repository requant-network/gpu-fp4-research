import unittest

import numpy as np

from fft_coefficients import coefficients
from fft_inputs import inputs
from fft_native import decode_spectrum, require
from fft_reference import pipeline_reference
from fft_validation import check_output, quality


class FftProtocolTests(unittest.TestCase):
    def test_portable_cohorts_and_nominal_dynamic_range(self):
        c = coefficients(128)
        for profile, db in (("weak72", 72.24719895935549), ("weak120", 120.41199826559249), ("highrange", 96.32959861247399)):
            x, w, h, params = inputs(128, 2, profile, 123, 0, c)
            repeated = inputs(128, 2, profile, 123, 0, c)[0]
            different = inputs(128, 2, profile, 124, 0, c)[0]
            np.testing.assert_array_equal(x, repeated)
            self.assertFalse(np.array_equal(x, different))
            self.assertAlmostEqual(params["nominal_strong_to_weak_db"], db)
            spectrum = pipeline_reference(x, w, h)[0]
            self.assertLess(abs(np.mean(abs(spectrum[:, params["weak_bin"]]))/(128*params["weak_amplitude"])-1), .03)

    def test_weak_error_cannot_hide_under_strong_input(self):
        c = coefficients(128)
        x, w, h, params = inputs(128, 1, "weak120", 123, 0, c)
        ref = pipeline_reference(x, w, h)
        metrics = quality(np.zeros_like(x), np.zeros_like(ref[0]), np.zeros_like(ref[1]), ref, params, np.linalg.norm(x))
        self.assertGreater(metrics["output_error_over_weak"], .9)
        self.assertGreater(metrics["weak_bin_relative_error"], .9)
        self.assertLess(metrics["output_error_over_windowed_input"], 2e-6)

    def test_nonfinite_spectra_are_not_changed_by_unit_scale(self):
        pairs = np.array([np.inf, 2, -np.inf, -3], dtype=np.float16)
        with np.errstate(all="raise"):
            result = decode_spectrum(pairs.view(np.uint8), 2, (1, 2))
        self.assertTrue(np.isposinf(result.real[0, 0]))
        self.assertEqual(result.imag[0, 0], 2)
        self.assertEqual(result.imag[0, 1], -3)

    def test_only_real_half_transform_overflow_is_a_retained_failure(self):
        y = np.array([[complex(np.inf, 0)]], dtype=np.complex64)
        check_output(y, np.ones((1, 1)), dict(backend=0, precision=2), 1)
        for method in (dict(backend=1, precision=1), dict(backend=0, precision=0)):
            with self.assertRaises(AssertionError):
                check_output(y, np.ones((1, 1)), method, 1)

    def test_binding_rejects_wrong_layout_or_dtype(self):
        for value in (np.zeros((2, 3), dtype=np.complex128), np.zeros((3, 2), dtype=np.complex64).T):
            with self.assertRaises(ValueError):
                require(value, np.complex64, (2, 3))


if __name__ == "__main__":
    unittest.main()
