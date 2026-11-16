import unittest

import numpy as np

from selector_reference import choose,feature,sample_indices


class SelectorReferenceTests(unittest.TestCase):
    def test_zero_cancellation_and_ranking(self):
        zeros=np.zeros((64,3));self.assertEqual(feature(zeros,128),0)
        nonzero=zeros.copy();nonzero[:,1]=1;self.assertEqual(feature(nonzero,128),1e300)
        policy={"multipliers":{mode:1 for mode in ("fp4","fp8","fp16","bf16","fp32")},
                "costs":[{"shape":[128,128,128],"prepacked_ms":{"fp4":1,"fp8":2,"fp16":3,"bf16":4,"fp32":5,"fp64":10}}]}
        probes={mode:zeros for mode in policy["multipliers"]}
        self.assertEqual(choose(policy,(128,128,128),probes,.001)[0],"fp4")
        probes={mode:nonzero for mode in policy["multipliers"]}
        self.assertEqual(choose(policy,(128,128,128),probes,.001)[0],"fp64")

    def test_indices_and_validation(self):
        indices=sample_indices(3,(16,256,128));np.testing.assert_array_equal(indices,sample_indices(3,(16,256,128)))
        self.assertTrue(np.all(indices>=0));self.assertTrue(np.all(indices[:,0]<16));self.assertTrue(np.all(indices[:,1]<256))
        with self.assertRaises(ValueError):feature(np.zeros((63,3)),128)
        with self.assertRaises(ValueError):feature(np.full((64,3),np.nan),128)


if __name__=="__main__":unittest.main()
