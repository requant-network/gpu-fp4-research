"""Portable query/filter/grid contracts, including real-data row partitioning."""
import unittest
from unittest.mock import patch

import numpy as np

import search_cases as cases


class SearchCasesTests(unittest.TestCase):
    def test_masks_and_candidate_populations(self):
        for cohort in cases.COHORTS:
            for percent in cases.FILTERS:
                mask = cases.allow_mask(16384,percent,cohort)
                self.assertEqual(mask.dtype,np.bool_)
                grid = cases.methods(16384,int(mask.sum()),percent)
                self.assertEqual(len(grid),39 if percent == 100 else 43)
                native = [m for m in grid if m["family"] == "native"]
                self.assertEqual({m["precision"] for m in native},set(range(6)))
                self.assertTrue(all(10 <= m["candidates"] <= mask.sum() for m in native))
                self.assertTrue(any(m["candidates"] == min(512,int(mask.sum())) for m in native))

    def test_independent_cohorts_and_ties(self):
        original = cases.DATASETS["uniform-small"]
        with patch.dict(cases.DATASETS,{"uniform-small":(256,128,"uniform")}):
            a,qa,_ = cases.inputs("uniform-small","a",2,4)
            b,qb,_ = cases.inputs("uniform-small","b",2,4)
            self.assertFalse(np.array_equal(a,b))
            self.assertFalse(np.array_equal(qa,qb))
            np.testing.assert_array_equal(a[:16],np.broadcast_to(a[32],(16,128)))
            np.testing.assert_array_equal(qa[0,0],a[32])
            self.assertFalse(np.array_equal(qa[0,0],qa[1,0]))
            np.testing.assert_array_equal(a,cases.inputs("uniform-small","a",2,4)[0])
        self.assertEqual(cases.DATASETS["uniform-small"],original)


if __name__ == "__main__":
    unittest.main()
