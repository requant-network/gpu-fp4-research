"""Whole smoke replay and coherent numerical forgeries, not just stale hashes."""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class PdeReplayTests(unittest.TestCase):
    def test_complete_smoke_and_rehashed_forged_gemm_output(self):
        self.assertIsNotNone(importlib.util.find_spec('pde_replay'),'Whole original-data physical replay is missing')
        from pde_replay import replay_collection,load_bound_json,load_payloads,load_truth,verify_operator_case
        from pde_protocol import array_hash
        from pde_reference import Mesh
        path=ROOT/'results/pde-smoke-a/e2-q4/collection.json'
        result=replay_collection(path)
        self.assertEqual(result['cases'],16)
        self.assertEqual(result['operator_observations'],128)
        self.assertEqual(result['poisson_solves'],64)
        self.assertEqual(result['heat_trajectories'],64)
        self.assertEqual(result['heat_steps'],128)
        import json
        manifest=json.loads(path.read_text())
        truth=load_truth(manifest)
        case=next(load_bound_json(item) for item in manifest['cases'] if item['path'].endswith('case-07.json'))
        arrays=load_payloads(case['payloads'])
        fake=copy.deepcopy(case);arrays={k:v.copy() for k,v in arrays.items()}
        stage=fake['operator_stages']['0'];witness=stage['witness']
        arrays[witness['product_key']].fill(0)
        witness['product_sha256']=array_hash(arrays[witness['product_key']])
        output=np.zeros_like(truth['operator0']);mesh=Mesh(2,4)
        output[mesh.boundary]=truth['operator0'][mesh.boundary]
        arrays['forged-full-output']=output
        record=fake['outputs'][stage['output']]
        record.update(encoding='full',base=None,key='forged-full-output',output_sha256=array_hash(output))
        with self.assertRaisesRegex(AssertionError,'original-data arithmetic|decoded arithmetic'):
            verify_operator_case(mesh,truth,fake,arrays,manifest['protocol'])

    def test_rehashed_false_original_residual_and_missing_observation_rejected(self):
        import json
        from pde_replay import load_bound_json,load_payloads,load_truth,verify_solution,verify_operator_case
        from pde_reference import Mesh
        manifest=json.loads((ROOT/'results/pde-smoke-a/e2-q4/collection.json').read_text())
        truth=load_truth(manifest);mesh=Mesh(2,4)
        case=load_bound_json(manifest['cases'][0]);arrays=load_payloads(case['payloads'])
        row=case['poisson_runs'][0]
        control=copy.deepcopy(row['controller']);control['gpu_original_relative']=0.;control['original_target_met']=True
        with self.assertRaisesRegex(AssertionError,'Claimed physical residual'):
            verify_solution(mesh,np.zeros(mesh.size),truth['poisson_rhs'],0.,control,row['target'],row['policy'],truth['initial'])
        fake=copy.deepcopy(case);fake['operator_observations'].pop()
        with self.assertRaisesRegex(AssertionError,'observation coverage'):
            verify_operator_case(mesh,truth,fake,arrays,manifest['protocol'])


if __name__=='__main__':unittest.main()
