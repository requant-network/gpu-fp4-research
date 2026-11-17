"""Cross-platform receipt comparison keeps identities/counts strict."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class PdeReceiptCompareTests(unittest.TestCase):
    def test_only_derived_quality_scalars_have_cross_platform_rounding_allowance(self):
        self.assertIsNotNone(importlib.util.find_spec('compare_pde_receipts'),'Strict semantic CI receipt comparison is missing')
        from compare_pde_receipts import compare
        original=json.loads((ROOT/'results/pde-smoke-replay.json').read_text())
        changed=copy.deepcopy(original)
        changed['qualities']['0']['operator'][next(iter(changed['qualities']['0']['operator']))]['relative_original_operator_error']+=1e-13
        compare(original,changed)
        changed=copy.deepcopy(original);changed['source_sha256']['pde_study.py']='0'*64
        with self.assertRaisesRegex(AssertionError,'source_sha256'):compare(original,changed)
        changed=copy.deepcopy(original);changed['cases']+=1
        with self.assertRaisesRegex(AssertionError,'cases'):compare(original,changed)
        changed=copy.deepcopy(original)
        changed['qualities']['0']['poisson'][0]['original_relative']+=.1
        with self.assertRaisesRegex(AssertionError,'original_relative'):compare(original,changed)


if __name__=='__main__':unittest.main()
