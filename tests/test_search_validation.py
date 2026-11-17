import unittest

import numpy as np

from search_validation import norm_model


class SearchValidationTests(unittest.TestCase):
    def test_norm_oracle_closed_forms_and_exact_dyadic_sums(self):
        for d in (128,256):
            x = np.vstack([np.zeros(d),np.ones(d),np.full(d,2.),np.resize([.5,2.,-4.,1.],d)]).astype(np.float32)
            expected = np.array([0,d,4*d,(.25+4+16+1)*d/4],dtype=np.float32)
            np.testing.assert_array_equal(norm_model(x),expected)


if __name__ == "__main__":
    unittest.main()
