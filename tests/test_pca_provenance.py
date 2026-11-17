"""Original and guarded producer epochs must stay explicit and exact."""
import importlib.util
import unittest


class PcaProvenanceTests(unittest.TestCase):
    def test_only_the_two_exact_source_epochs_are_accepted(self):
        self.assertIsNotNone(importlib.util.find_spec('pca_provenance'),'PCA source-epoch validation is missing')
        from pca_provenance import producer_sources,validate_sources
        original=producer_sources(False)
        guarded=producer_sources(True)
        self.assertFalse(validate_sources(original))
        self.assertTrue(validate_sources(guarded))
        self.assertEqual({key:guarded[key] for key in original},original)
        corrupted=dict(guarded)
        corrupted['pca_study_guarded.py']='0'*64
        with self.assertRaises(AssertionError):validate_sources(corrupted)
        extra={**original,'unexpected.py':'0'*64}
        with self.assertRaises(AssertionError):validate_sources(extra)


if __name__=='__main__':unittest.main()
