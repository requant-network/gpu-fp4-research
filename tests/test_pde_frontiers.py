"""Synthetic unit fixtures: no row here is a measured GPU result."""
import importlib.util
import unittest


class PdeFrontierTests(unittest.TestCase):
    def test_worst_timed_quality_strongest_reference_and_setup_are_not_dropped(self):
        self.assertIsNotNone(importlib.util.find_spec('pde_frontiers'),'Error-constrained full-cost frontier is missing')
        from pde_frontiers import select_frontier
        rows=[dict(id='fast-but-bad',precision='fp4',available=True,wall_ms=[.001,.001],errors=[.0001,.02],setup_ms=0.),
              dict(id='valid-fp4',precision='fp4',available=True,wall_ms=[.005,.005],errors=[.0001,.0001],setup_ms=1.),
              dict(id='slow-double',precision='fp64',available=True,wall_ms=[.05,.05],errors=[1e-14,1e-14],setup_ms=0.),
              dict(id='strong-short-half',precision='fp16',available=True,wall_ms=[.01,.01],errors=[.0002,.0002],setup_ms=0.)]
        cold=select_frontier(rows,.001,1)
        self.assertEqual(cold['fp4'],'valid-fp4')
        self.assertEqual(cold['reference'],'strong-short-half')
        self.assertFalse(cold['fp4_wins'])
        reused=select_frontier(rows,.001,10000)
        self.assertTrue(reused['fp4_wins'])
        self.assertEqual(reused['eligible_fp4'],1)
        self.assertEqual(reused['eligible_references'],2)
        strict=select_frontier(rows,1e-8,1)
        self.assertIsNone(strict['fp4'])
        self.assertEqual(strict['reference'],'slow-double')


if __name__=='__main__':unittest.main()
