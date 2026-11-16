import unittest

import numpy as np

from outlier_reference import factors,oracle
from reference import inputs


class OutlierReferenceTests(unittest.TestCase):
    def test_column_selection_ties_and_full_rank_correction(self):
        a=np.ones((2,128),np.float32);b=np.ones((128,128),np.float32)
        arrays,metadata,_=factors(a,b,1)
        np.testing.assert_array_equal(metadata["selected_columns"],np.arange(128,dtype=np.int32))
        for x,value in zip(arrays,(1,1,1.03125,1.03125)):
            np.testing.assert_array_equal(x,np.full_like(x,value))
        a,b=inputs(74,(16,128,128),"outliers")
        reference=oracle(a,b,1)
        expected=a.astype(np.float16).astype(np.float64)@b.astype(np.float16).astype(np.float64).T
        np.testing.assert_allclose(reference["expected"],expected,rtol=0,atol=1e-9)

    def test_block_masks_and_residual_reconstruction(self):
        a,b=inputs(75,(16,128,128),"outliers")
        arrays,metadata,quantized=factors(a,b,2,fraction=.75)
        for x,key in ((a,"mask_a"),(b,"mask_b")):
            expected=np.max(np.abs(x.reshape(-1,16)),axis=1)>.75*np.max(np.abs(x))
            np.testing.assert_array_equal(metadata[key],expected)
        full=oracle(a,b,3)
        plain=quantized[0].astype(np.float64)@quantized[1].astype(np.float64).T
        self.assertLess(np.linalg.norm(full["expected"]-full["original"]),np.linalg.norm(plain-full["original"])/10)

    def test_invalid_inputs(self):
        a=np.zeros((1,128),np.float32);b=np.zeros((128,128),np.float32)
        for options in ({"method":0},{"method":1,"budget":1},{"method":2,"fraction":0}):
            with self.assertRaises(ValueError):factors(a,b,**options)
        a[0,0]=np.nan
        with self.assertRaises(ValueError):factors(a,b,1)


if __name__=="__main__":unittest.main()
