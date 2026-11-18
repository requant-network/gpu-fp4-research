"""Check the two public Study 13 projections without a GPU or private archive."""
import subprocess
import sys
import unittest
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / 'results'
COST_PIN = 'b65e7185a4ffcdf955db3a0178e8886028100d79ac02f9ac9f8d9fa0f48f0c60'
GEOMETRY_PIN = '0cfed9098f220a0b9f18c2dcde813ce9bfa968304ba53366e5fe45aa2c9660a5'


class PublicProjectionTests(unittest.TestCase):
    def run_replay(self, directory, *args):
        result = subprocess.run([sys.executable, '-I', '-B', *args], cwd=directory,
                                capture_output=True, check=False, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        self.assertEqual(result.stderr, b'')
        return result.stdout

    def test_cost_receipt(self):
        directory = RESULTS / 'study13-cost-projection'
        stdout = self.run_replay(directory, 'replay_cost.py', 'evidence', COST_PIN)
        self.assertEqual(stdout, (directory / 'replay-receipt.json').read_bytes())

    def test_report_matches_pinned_receipts(self):
        stdout = self.run_replay(RESULTS.parent, 'render_study13_report.py', '--check')
        self.assertIn(b'RT_VISIBILITY_REPORT.md:', stdout)

    def test_geometry_and_quartet_receipt(self):
        directory = RESULTS / 'study13-geometry-projection'
        stdout = self.run_replay(directory, 'replay.py', '--manifest-sha256', GEOMETRY_PIN, '--check')
        self.assertIn(b'PASS: 5120 words, 4720 canaries; receipt matches', stdout)


if __name__ == '__main__':
    unittest.main()
