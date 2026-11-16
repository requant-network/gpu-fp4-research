import unittest

import numpy as np

from attention_reference import allowed,attention,arrays,encoding,padded,probabilities


class AttentionReferenceTests(unittest.TestCase):
    def test_padding_and_independent_heads(self):
        q,k,v=arrays(12,(1,2,193,48));qp,kp,vt=padded(q,k,v)
        self.assertEqual(qp.shape,(2,256,128));np.testing.assert_array_equal(qp[:,:193,:48],q.reshape(2,193,48))
        self.assertFalse(np.any(qp[:,193:]));self.assertFalse(np.any(qp[:,:,48:]));self.assertFalse(np.any(vt[:,:,193:]))
        raw,scales,decoded=encoding(qp,"fp4");self.assertEqual(raw.size,qp.size//2);self.assertEqual(scales.size,qp.size//16)
        self.assertFalse(np.any(decoded[:,193:]));self.assertFalse(np.array_equal(decoded[0],decoded[1]))

    def test_masks_zero_rows_and_causality(self):
        q,k,v=arrays(7,(1,1,8,16));result=attention(q,k,v,"padding");np.testing.assert_array_equal(result[:,:,0],0)
        p=probabilities(np.zeros((1,8,8)),16,allowed(8,"causal"));np.testing.assert_array_equal(np.triu(p,1),0)
        np.testing.assert_allclose(p.sum(axis=-1),1);np.testing.assert_allclose(attention(q,k,v,"causal")[:,:,0],v[:,:,0])
        with self.assertRaises(ValueError):allowed(8,"unknown")
        with self.assertRaises(ValueError):padded(q,k[:,:,:7],v)


if __name__=="__main__":unittest.main()
