"""CPU checks for the compact Study 14 and 15 publication receipts."""

import json
from pathlib import Path
import unittest


RESULTS = Path(__file__).resolve().parents[1] / 'results'


class CompletionEvidenceTests(unittest.TestCase):
    def test_streamed_cost_scope(self):
        report = json.loads((RESULTS / 'study14-streamed-replay.json').read_bytes())
        self.assertEqual(report['logical_bytes'], 320 * (64 << 20))
        self.assertEqual(report['frames_replayed'], 3 * report['unique_digest_files'])
        self.assertEqual(report['unique_digest_files'], 320)
        plain, ndzip, xbp = report['routes']
        self.assertEqual([row['codec'] for row in (plain, ndzip, xbp)], ['plain', 'ndzip', 'xbp'])
        self.assertEqual(plain['wire_bytes'], report['logical_bytes'])
        for route in (plain, ndzip, xbp):
            self.assertEqual(route['source_pinned_bytes'], 128 << 20)
            named = sum(route[stage + '_ns'] for stage in
                        ('prepare', 'service', 'setup', 'first_use', 'other'))
            self.assertEqual(named, route['wall_ns'])
        self.assertLess(sum(row['wall_ns'] for row in report['routes']), report['owned_wall_ns'])

    def test_kernel_receipt_selection(self):
        report = json.loads((RESULTS / 'study15-kernel-replay-summary.json').read_bytes())
        variants = {'uint-tree', 'uint-warp', 'fp64-warp'}
        for pattern in ('steady', 'irregular'):
            train, held = (report['attempts'][f'{pattern}-{i}'] for i in (1, 2))
            self.assertEqual(set(train), variants)
            self.assertEqual(set(held), variants)
            for rows in (train, held):
                for attempt in rows.values():
                    if not attempt['accepted']:
                        self.assertTrue(attempt['reason'])
                        continue
                    self.assertEqual(set(attempt['arms']), {'singleton', 'batch8'})
                    for arm in attempt['arms'].values():
                        self.assertEqual(arm['deliveries'], 1024)
                        self.assertGreater(arm['measurement_mJ'], 0)
                        self.assertAlmostEqual(arm['joules_per_delivery'], arm['measurement_mJ'] / 1_024_000)
            for cutoff in (2_000_000, 12_000_000):
                choice = report['held_out_rules'][pattern][str(cutoff)]
                if not all(record['accepted'] for record in train.values()):
                    self.assertFalse(choice['available'])
                    continue
                eligible = [
                    (metrics['measurement_mJ'], kernel, arm)
                    for kernel, record in train.items()
                    for arm, metrics in record['arms'].items()
                    if metrics['p99_offered_latency_ns'] <= cutoff
                ]
                if not eligible:
                    self.assertIsNone(choice['selected'])
                    continue
                _, kernel, arm = min(eligible)
                self.assertEqual(choice['selected'], {'kernel': kernel, 'arm': arm})
                if held[kernel]['accepted']:
                    actual = held[kernel]['arms'][arm]
                    self.assertTrue(choice['available'])
                    self.assertEqual(choice['held_board_energy_mJ'], actual['measurement_mJ'])
                    self.assertEqual(choice['held_meets_limit'], actual['p99_offered_latency_ns'] <= cutoff)
                else:
                    self.assertFalse(choice['available'])


if __name__ == '__main__':
    unittest.main()
