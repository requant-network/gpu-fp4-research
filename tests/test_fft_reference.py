import unittest

import numpy as np

from fft_reference import (augmented_basis, mathematical_fft, permutation, pipeline_reference,
                           radices, rounded_fft, stage_input, stage_output, twiddles)


class FftReferenceTests(unittest.TestCase):
    def test_mixed_radix_fft_and_inverse(self):
        rng = np.random.Generator(np.random.PCG64(933))
        for n in (2, 8, 16, 32, 128, 256, 512, 1024, 4096):
            x = rng.integers(-16, 17, (2, n)) + 1j * rng.integers(-16, 17, (2, n))
            for radix in (8, 16):
                p = permutation(n, radix)
                np.testing.assert_array_equal(np.sort(p), np.arange(n))
                self.assertEqual(int(np.prod(radices(n, radix))), n)
                actual = mathematical_fft(x, radix)
                np.testing.assert_allclose(actual, np.fft.fft(x, axis=-1) / n, rtol=2e-12, atol=2e-12)
                restored = mathematical_fft(actual, radix, inverse=True) * n
                np.testing.assert_allclose(restored, x, rtol=2e-12, atol=2e-12)

    def test_augmented_gemm_and_typed_stages(self):
        rng = np.random.Generator(np.random.PCG64(934))
        for radix in (2, 4, 8, 16):
            for inverse in (False, True):
                x = (rng.integers(-16, 17, (3, radix * 4)) + 1j * rng.integers(-16, 17, (3, radix * 4))).astype(np.complex64)
                phase = twiddles(1, radix).astype(np.complex64)
                packed = stage_input(x, 1, radix, phase, inverse)
                product = augmented_basis(radix, inverse).astype(np.float64) @ packed.astype(np.float64)
                actual = stage_output(product, 3, radix*4, 1, radix)
                expected = (np.fft.ifft(x.reshape(3, 4, radix), axis=-1) if inverse else np.fft.fft(x.reshape(3, 4, radix), axis=-1) / radix).reshape(3, -1)
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        x = (rng.integers(-128, 129, (2, 512)) + 1j * rng.integers(-128, 129, (2, 512))).astype(np.complex64) / 128
        expected = np.fft.fft(x.astype(np.complex128), axis=-1) / 512
        for precision, limit in ((0, 2e-6), (1, .004), (2, .04)):
            actual = rounded_fft(x, 16, precision)
            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected), limit)

    def test_filter_and_validation(self):
        x = np.zeros((2, 128), dtype=np.complex64)
        x[:, 7] = 1
        spectrum, filtered, result = pipeline_reference(x, np.ones(128), np.ones(128))
        np.testing.assert_allclose(result, x, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(filtered, spectrum)
        for n, maximum in ((1000, 16), (0, 8), (128, 4)):
            with self.assertRaises(ValueError):
                radices(n, maximum)


if __name__ == '__main__':
    unittest.main()
