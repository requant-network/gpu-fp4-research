"""Quality/latency frontiers must include all strong baselines and timed outputs."""
import importlib.util
import unittest


class FrontierTests(unittest.TestCase):
    def test_route_ignores_rejected_graph_output_but_charges_setup(self):
        self.assertIsNotNone(importlib.util.find_spec('search_frontiers'),'Search frontier extraction is missing')
        from search_frontiers import routes
        good=dict(recall=1.,tie_aware_recall=1.,ids_valid=True,missing_ids=0,nonfinite_reported_valid_ids=0,max_absolute_distance_error=0.)
        bad={**good,'recall':0.,'tie_aware_recall':0.}
        case=dict(available=True,valid=True,graph_available=False,setup_wall_ms=3.,first_run_wall_ms=1.,graph_capture_to_error_wall_ms=2.,
                  method={'family':'cagra'},outputs={'good':{'quality':good},'discarded':{'quality':bad}},metadata={},
                  boundaries=[{'name':'resident-uncaptured','observations':[dict(cuda_ms=x,wall_ms=x+.1,output_key='good') for x in (1.,2.,3.)]},
                              {'name':'host-service','observations':[dict(cuda_ms=None,wall_ms=x,output_key='good') for x in (2.,3.,4.)]}])
        result=routes(case)
        self.assertEqual(result['resident-uncaptured']['median_ms'],2.)
        self.assertEqual(result['resident-uncaptured']['recall'],1.)
        self.assertEqual(result['host-service']['median_ms'],3.)
        self.assertEqual(result['best-resident']['boundary'],'resident-uncaptured')
        self.assertEqual(result['best-resident']['setup_ms'],6.)


if __name__=='__main__':unittest.main()
