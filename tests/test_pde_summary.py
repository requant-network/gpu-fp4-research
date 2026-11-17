"""Analysis must consume the complete measured collection and replay receipts."""
import importlib.util
import unittest
from pde_cases import campaign_cases,configuration_specs


class PdeSummaryTests(unittest.TestCase):
    def test_complete_attempt_matrix_and_strong_reference_frontiers(self):
        self.assertIsNotNone(importlib.util.find_spec('summarize_pde'),'Complete physical findings generator is missing')
        from summarize_pde import build_tables
        tables,summary=build_tables()
        self.assertEqual(len(tables['operator']),len(campaign_cases())*len(configuration_specs())*2)
        self.assertEqual(len(tables['poisson']),len(campaign_cases())*len(configuration_specs())*2*2*2)
        self.assertEqual(len(tables['heat']),sum(c['heat'] for c in campaign_cases())*len(configuration_specs())*2*2*2*2)
        self.assertEqual(len(tables['operator_frontiers']),len(campaign_cases())*2*3*3)
        self.assertEqual(summary['full_collections'],24)
        self.assertEqual(summary['numeric_counts']['cases'],400)
        self.assertTrue(all(row['reference'] is not None for row in tables['operator_frontiers']))
        self.assertTrue(all(row['n_observations']>=1 for row in tables['operator'] if row['available']))

    def test_heat_frontier_residual_uses_retained_state_not_claimed_scalar(self):
        import copy,json
        from pathlib import Path
        import summarize_pde
        self.assertTrue(hasattr(summarize_pde,'step_original_residuals'),'Independent per-step frontier metrics are missing')
        from pde_replay import ROOT,load_payloads
        from pde_reference import Mesh
        case=json.loads((ROOT/'results/pde-smoke-a/e2-q4/case-07.json').read_text())
        heat=case['heat'][0];run=heat['runs'][0];arrays=load_payloads(heat['payloads'])
        original=summarize_pde.step_original_residuals(Mesh(2,4),heat,run,arrays)
        forged=copy.deepcopy(run)
        for step in forged['step_records']:step['controller']['gpu_original_relative']=0.
        self.assertEqual(summarize_pde.step_original_residuals(Mesh(2,4),heat,forged,arrays),original)
        self.assertTrue(any(value>0 for value in original))


if __name__=='__main__':unittest.main()
