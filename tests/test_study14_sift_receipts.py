"""The retained SIFT receipts can be checked without the private collection."""
import json
from pathlib import Path
import tempfile
import unittest

from verify_study14_sift_receipts import audit, check_epoch, load_pinned

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / 'results/sift-full-accounting-receipts'


class SiftReceiptTests(unittest.TestCase):
    def test_real_receipts(self):
        result = audit(RECEIPTS, ROOT)
        self.assertEqual(result['cases'], 400)
        self.assertEqual(result['observations'], 163680)
        self.assertEqual([r['strict_wins'] for r in result['epochs']], [[0, 0, 0]] * 2)
        self.assertFalse(result['raw_fingerprint_replay'])

    def test_false_strict_win_rejected_by_arithmetic(self):
        data = json.loads((RECEIPTS / 'sift-frontier.json').read_bytes())['epochs'][0]
        data['cells'][0]['frontier']['1']['strict_win'] = True
        with self.assertRaisesRegex(ValueError, 'strict-win arithmetic'):
            check_epoch(data)

    def test_duplicated_cell_rejected(self):
        data = json.loads((RECEIPTS / 'sift-frontier.json').read_bytes())['epochs'][0]
        data['cells'][-1] = data['cells'][0]
        with self.assertRaisesRegex(ValueError, 'cell census'):
            check_epoch(data)

    def test_modified_receipt_cannot_reuse_original_pin(self):
        source = RECEIPTS / 'sift-frontier.json'
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / source.name
            changed.write_bytes(source.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'receipt digest mismatch'):
                load_pinned(changed)


if __name__ == '__main__':
    unittest.main()
