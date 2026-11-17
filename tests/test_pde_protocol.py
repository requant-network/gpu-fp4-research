"""Lossless full-output retention: no sampled-vector correctness substitution."""
import importlib.util
import unittest
import numpy as np


class PdeProtocolTests(unittest.TestCase):
    def test_full_ieee_output_archive_roundtrip_and_identity_rejection(self):
        self.assertIsNotNone(importlib.util.find_spec('pde_protocol'),'Full-output observation archive is missing')
        from pde_protocol import OutputArchive,decode_output
        archive=OutputArchive()
        first=np.linspace(-2,2,1001,dtype=np.float64)
        first[0]=-0.
        changed=first.copy();changed.view(np.uint64)[333]^=np.uint64(1)
        a=archive.add(first,'variant0','1'*64)
        b=archive.add(changed,'variant0','1'*64)
        self.assertEqual(archive.add(changed.copy(),'variant0','1'*64),b)
        self.assertNotEqual(a,b)
        self.assertEqual(archive.records[b]['encoding'],'xor64')
        for key,value in ((a,first),(b,changed)):
            actual=decode_output(archive.records[key],archive.arrays,'1'*64)
            np.testing.assert_array_equal(actual.view(np.uint64),value.view(np.uint64))
        with self.assertRaisesRegex(AssertionError,'input identity'):
            decode_output(archive.records[b],archive.arrays,'2'*64)
        archive.arrays[archive.records[b]['key']][0]^=np.uint64(1)
        with self.assertRaisesRegex(AssertionError,'output bytes'):
            decode_output(archive.records[b],archive.arrays,'1'*64)


if __name__=='__main__':unittest.main()
