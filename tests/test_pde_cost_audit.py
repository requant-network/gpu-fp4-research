"""Independent named allocation/tuning checks must catch coherent metadata edits."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from pde_reference import Mesh

ROOT=Path(__file__).resolve().parents[1]


class PdeCostTests(unittest.TestCase):
    def test_all_smoke_allocation_geometries_and_native_size_corruption(self):
        self.assertIsNotNone(importlib.util.find_spec('pde_cost_audit'),'Independent named cost audit is missing')
        from pde_cost_audit import audit_metadata,audit_collection
        manifest=json.loads((ROOT/'results/pde-smoke-a/e2-q4/collection.json').read_text())
        result=audit_collection(ROOT/'results/pde-smoke-a/e2-q4/collection.json')
        self.assertEqual(result['cases'],16)
        mesh=Mesh(2,4)
        case=json.loads((ROOT/'results/pde-smoke-a/e2-q4/case-07.json').read_text())
        corrupted=copy.deepcopy(case['method_metadata']);corrupted['native_owned_device_bytes']+=16
        corrupted['named_owned_device_bytes']+=16
        with self.assertRaisesRegex(AssertionError,'Native allocation geometry'):
            audit_metadata(mesh,case['configuration'],corrupted,manifest['binary_sha256'])
        case=json.loads((ROOT/'results/pde-smoke-a/e2-q4/case-05.json').read_text())
        corrupted=copy.deepcopy(case['method_metadata']);corrupted['libceed_requested_qdata_bytes']+=8
        with self.assertRaisesRegex(AssertionError,'qdata geometry'):
            audit_metadata(mesh,case['configuration'],corrupted,manifest['binary_sha256'])


if __name__=='__main__':unittest.main()
