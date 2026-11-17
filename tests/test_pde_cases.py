"""The declared campaign must not silently drop expensive/unavailable shapes."""
import importlib.util
import itertools
import unittest


class PdeCaseTests(unittest.TestCase):
    def test_full_geometry_cohort_and_temporal_matrix_is_explicit(self):
        self.assertIsNotNone(importlib.util.find_spec('pde_cases'),'Declared complete campaign matrix is missing')
        from pde_cases import campaign_cases,protocol,configuration_specs
        cases=campaign_cases()
        actual={(c['cohort'],c['elements'],c['q']) for c in cases}
        self.assertEqual(actual,set(itertools.product(('a','b'),(1,2,4,8),(4,8,16))))
        self.assertEqual(len(cases),len(actual))
        self.assertEqual(sum(c['heat'] for c in cases),12)
        settings=protocol(False)
        self.assertEqual(settings['targets'],[1e-3,1e-8])
        self.assertEqual(settings['heat_steps'],[4,8])
        self.assertEqual(len(configuration_specs()),16)
        self.assertGreaterEqual(settings['resident_observations'],16)
        self.assertGreaterEqual(settings['host_observations'],16)


if __name__=='__main__':unittest.main()
