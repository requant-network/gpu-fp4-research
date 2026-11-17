"""Independent named-memory and calibration checks on real PCA records."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class PcaCostTests(unittest.TestCase):
    def test_recorded_memory_is_checked_against_allocation_geometry(self):
        self.assertIsNotNone(importlib.util.find_spec('pca_cost_audit'),'Independent PCA allocation audit is missing')
        from pca_cost_audit import audit_case
        record=json.loads((ROOT/'results/pca-smoke-a/uniform-n8192-d128-m000.json').read_text())
        checked=audit_case(record)
        self.assertEqual(checked['cases'],1)
        self.assertEqual(checked['native_owned_device_bytes'],record['metadata']['native_owned_device_bytes'])
        record['metadata']['python_owned_device_bytes']+=16
        record['metadata']['owned_device_bytes']+=16
        with self.assertRaisesRegex(AssertionError,'geometry'):
            audit_case(record)


if __name__=='__main__':unittest.main()
