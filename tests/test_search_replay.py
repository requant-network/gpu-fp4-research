"""Tamper-resistant independent search replay contracts."""
import importlib.util
import unittest

import numpy as np


class ReplayTests(unittest.TestCase):
    def test_independent_oracle_ties_filters_and_missing(self):
        self.assertIsNotNone(importlib.util.find_spec('search_replay'),'Independent search replay is not implemented')
        from search_replay import independent_topk
        data=np.array([[0,0],[1,0],[-1,0],[0,2],[0,0],[9,9]],dtype=np.float32)
        query=np.array([[0,0],[1,0]],dtype=np.float32)
        allowed=np.array([True,False,True,False,True,False])
        distances,ids=independent_topk(data,query,5,allowed,chunk=2)
        np.testing.assert_array_equal(ids,[[0,4,2,-1,-1],[0,4,2,-1,-1]])
        np.testing.assert_array_equal(distances,[[0,0,1,np.inf,np.inf],[1,1,4,np.inf,np.inf]])
    def test_timing_rejects_missing_observation_and_wrong_variant(self):
        import search_replay
        self.assertTrue(hasattr(search_replay,'validate_timing'),'Timing/output binding replay is missing')
        from copy import deepcopy
        case=dict(available=True,valid=True,graph_available=False,setup_wall_ms=1.,first_run_wall_ms=.1,
                  outputs={'v0-x':{'variant':0},'v1-y':{'variant':1}},
                  boundaries=[dict(name=name,observations=[dict(variant=i%2,cuda_ms=None if name=='host-service' else .01,wall_ms=.1,output_key='v0-x' if i%2==0 else 'v1-y') for i in range(4)]) for name in ('resident-uncaptured','host-service')])
        search_replay.validate_timing(case,2,4)
        broken=deepcopy(case)
        broken['boundaries'][0]['observations'].pop()
        with self.assertRaises(AssertionError):search_replay.validate_timing(broken,2,4)
        broken=deepcopy(case)
        broken['boundaries'][0]['observations'][0]['output_key']='v1-y'
        with self.assertRaises(AssertionError):search_replay.validate_timing(broken,2,4)
        broken=deepcopy(case)
        broken['boundaries'][1]['observations'][0]['wall_ms']=-1
        with self.assertRaises(AssertionError):search_replay.validate_timing(broken,2,4)
    def test_independent_quality_counts_actual_distance_and_ties(self):
        import search_replay
        self.assertTrue(hasattr(search_replay,'output_metrics'),'Independent output replay is missing')
        data=np.array([[0,0],[0,0],[1,0],[2,0]],np.float32)
        query=np.array([[0,0]],np.float32)
        allowed=np.array([True,True,True,False])
        truth=np.array([[0.,0.]],np.float64)
        gold=np.array([[0,1]],np.int64)
        reported=np.array([[0.,0.]],np.float32)
        metrics,actual=search_replay.output_metrics(data,query,reported,np.array([[1,1]],np.int64),truth,gold,allowed)
        self.assertEqual(metrics['duplicate_ids'],1)
        self.assertEqual(metrics['recall'],.5)
        self.assertEqual(metrics['tie_aware_recall'],.5)
        self.assertFalse(metrics['ids_valid'])
        np.testing.assert_array_equal(actual,reported)
        metrics,actual=search_replay.output_metrics(data,query,np.array([[7.,np.inf]],np.float32),np.array([[2,-1]],np.int64),truth,gold,allowed)
        self.assertEqual(metrics['missing_ids'],1)
        self.assertEqual(metrics['max_absolute_distance_error'],6.)
        self.assertEqual(metrics['recall'],0.)
        self.assertEqual(metrics['nonfinite_reported_valid_ids'],0)
    def test_actual_stage_replay_rejects_dot_and_global_id_corruption(self):
        import search_replay
        self.assertTrue(hasattr(search_replay,'validate_stage'),'Independent stage replay is missing')
        import json
        from pathlib import Path
        from search_cases import inputs,allow_mask
        root=Path(__file__).resolve().parents[1]
        record=json.loads((root/'results/search-smoke-a/uniform-small-b1-f100.json').read_text())
        case=record['cases'][0]
        with np.load(root/'results/search-smoke-a'/record['witness_file'],allow_pickle=False) as witness:
            prefix=case['stage_witness']['prefix']+'_'
            stage={k[len(prefix):]:witness[k].copy() for k in witness.files if k.startswith(prefix)}
            first=next(v for v in case['outputs'].values() if v['variant']==0)
            final=(witness[first['prefix']+'_distances'],witness[first['prefix']+'_ids'])
        data,queries,_=inputs('uniform-small','a',2,4)
        allowed=allow_mask(len(data),100,'a')
        search_replay.validate_stage(stage,case['stage_witness'],case['method'],data,queries[0,:1],allowed,final)
        broken={k:v.copy() for k,v in stage.items()}
        broken['dot'][0]+=1000
        with self.assertRaises(AssertionError):search_replay.validate_stage(broken,case['stage_witness'],case['method'],data,queries[0,:1],allowed,final)
        broken={k:v.copy() for k,v in stage.items()}
        broken['candidate_global_ids'][0,0]=len(data)+1
        with self.assertRaises(AssertionError):search_replay.validate_stage(broken,case['stage_witness'],case['method'],data,queries[0,:1],allowed,final)


if __name__=='__main__':
    unittest.main()
