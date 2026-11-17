"""Parallel receipts must form one exact disjoint source-compatible partition."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class PdeParallelTests(unittest.TestCase):
    def test_disjoint_source_compatible_receipt_union(self):
        self.assertIsNotNone(importlib.util.find_spec('verify_pde'),'Whole parallel replay driver is missing')
        from verify_pde import merge_replays
        part=json.loads((ROOT/'results/pde-smoke-replay.json').read_text())
        expected={part['collection_path']}
        result=merge_replays([part],expected)
        self.assertEqual(result['collections'],1)
        self.assertEqual(result['counts']['operator_observations'],128)
        with self.assertRaisesRegex(AssertionError,'Overlapping'):
            merge_replays([part,part],expected)
        wrong=copy.deepcopy(part);wrong['source_sha256']['pde_study.py']='0'*64
        wrong['collection_path']='results/pde-full-a/e1-q4/collection.json'
        with self.assertRaisesRegex(AssertionError,'Incompatible'):
            merge_replays([part,wrong],expected|{wrong['collection_path']})
        with self.assertRaisesRegex(AssertionError,'declared collection set'):
            merge_replays([part],expected|{'missing/collection.json'})


if __name__=='__main__':unittest.main()
