"""Tamper checks against executed GPU PCA artifacts, not synthetic timings."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from pca_cases import inputs
from pca_protocol import array_binding
from reference import digest

ROOT=Path(__file__).resolve().parents[1]


class PcaStageReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results/pca-smoke-a/uniform-n8192-d128-m000.json'
        cls.record=json.loads(path.read_text())
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as archive:
            stage=cls.record['stage_witnesses'][0]
            cls.stage=stage
            cls.arrays={key:archive[stage['prefix']+'_'+key].copy() for key in stage['array_bindings']}
        cls.data=inputs(8192,128,'uniform','a')[0]

    def replay(self,stage,arrays):
        self.assertIsNotNone(importlib.util.find_spec('pca_replay'),'Independent PCA stage replay is missing')
        from pca_replay import replay_stage
        return replay_stage(self.data,self.record['method'],stage,arrays)

    def test_actual_encoded_full_gram_is_replayed(self):
        result=self.replay(self.stage,self.arrays)
        self.assertEqual(result['full_product_entries'],128*128)
        self.assertEqual(result['encoded_buffers'],4)
        self.assertGreater(result['source_entries'],0)

    def test_rebound_numerically_wrong_gram_is_rejected(self):
        stage,arrays=copy.deepcopy(self.stage),{k:a.copy() for k,a in self.arrays.items()}
        arrays['full_product'][0,0]+=1024
        arrays['product_samples'][arrays['product_indices']==0]=arrays['full_product'][0,0]
        stage['product_sha256']=digest(arrays['full_product'])
        stage['array_bindings']={k:array_binding(a) for k,a in arrays.items()}
        self.assertIsNotNone(importlib.util.find_spec('pca_replay'),'Independent PCA stage replay is missing')
        with self.assertRaises(AssertionError):self.replay(stage,arrays)


if __name__=='__main__':unittest.main()
