import unittest
import numpy as np

from precision_reference import components,dataset,exact_product,exact_samples,row_statistics


class PrecisionReferenceTests(unittest.TestCase):
    def test_components_and_exact_selection(self):
        a,b=dataset(11,(16,128,128),"normal")
        pieces,exponents,required,decoded=components(a,4)
        np.testing.assert_array_equal(decoded,a.astype(np.float64))
        self.assertTrue(np.all(np.abs(pieces.astype(np.int16))<=127))
        self.assertLessEqual(int(required.max()),4)
        a,b=dataset(12,(16,128,128),"spread")
        self.assertGreater(int(row_statistics(a)[1].max()),8)
        self.assertLess(8*8192*127**2,np.iinfo(np.int32).max)

    def test_exact_cancellation_and_subnormal_products(self):
        a,b=dataset(13,(16,128,128),"cancellation")
        exact=exact_product(a,b)
        np.testing.assert_array_equal(exact,a[:,-1:].astype(np.float64)@b[:,-1:].astype(np.float64).T)
        indices=np.array([[0,1],[1,2],[7,11]],np.int32)
        np.testing.assert_array_equal(exact_samples(a,b,indices),exact[indices[:,0],indices[:,1]])
        a=np.array([[np.nextafter(np.float32(0),np.float32(1)),np.float32(1)]],np.float32)
        b=np.array([[np.float32(1),np.float32(-1)]],np.float32)
        np.testing.assert_array_equal(exact_product(a,b),[[-1.]])
        a,b=dataset(14,(16,128,128),"subnormal")
        np.testing.assert_array_equal(components(a,2)[3],a.astype(np.float64))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):components(np.ones((1,128),np.float32),0)
        with self.assertRaises(ValueError):row_statistics(np.array([[np.inf]],np.float32))


if __name__=="__main__":unittest.main()
