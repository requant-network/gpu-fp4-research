"""Public Study 13 dispatch replay: exact data, independent geometry, no GPU."""
from hashlib import sha256
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'verify_study13_dispatch.py'
BUNDLE = ROOT / 'results/study13-matched-dispatch'


class DispatchReplayTests(unittest.TestCase):
    def run_replay(self, bundle):
        return subprocess.run([sys.executable, '-B', str(SCRIPT), str(bundle)],
                              cwd=ROOT, capture_output=True, text=True, check=False)

    def test_exact_geometry_and_retained_boundary_differences(self):
        run = self.run_replay(BUNDLE)
        self.assertEqual(run.returncode, 0, run.stderr)
        data = json.loads(run.stdout)
        self.assertEqual(data['regular_exact_words'], 33280)
        self.assertEqual(data['regular_exact_canaries'], 504)
        self.assertEqual(data['boundary_checked_words'], 128)
        self.assertEqual(data['boundary_mismatch_lanes'], {'cuda': [4], 'vulkan': [4, 5]})
        self.assertFalse(data['timing_evidence'])
        self.assertFalse(data['hardware_attestation'])

    def test_changed_raw_readback_and_unlisted_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / 'bundle'
            shutil.copytree(BUNDLE, dst, ignore=shutil.ignore_patterns('__pycache__'))
            raw = dst / 'spread.cuda.raw'
            data = raw.read_bytes()
            raw.write_bytes(bytes([data[0] ^ 1]) + data[1:])
            run = self.run_replay(dst)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertIn('digest mismatch: spread.cuda.raw', run.stderr)
            raw.write_bytes(data)
            (dst / 'extra.txt').write_bytes(b'not on allowlist')
            run = self.run_replay(dst)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertIn('bundle inventory differs', run.stderr)

    def test_rewritten_manifest_does_not_authorize_changed_readback(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / 'bundle'
            shutil.copytree(BUNDLE, dst)
            raw = dst / 'spread.cuda.raw'
            original = raw.read_bytes()
            altered = bytes([original[0] ^ 1]) + original[1:]
            raw.write_bytes(altered)
            manifest = dst / 'SHA256SUMS'
            manifest.write_bytes(manifest.read_bytes().replace(sha256(original).hexdigest().encode(),
                                                               sha256(altered).hexdigest().encode()))
            run = self.run_replay(dst)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertIn('bundle manifest digest mismatch', run.stderr)


if __name__ == '__main__':
    unittest.main()
