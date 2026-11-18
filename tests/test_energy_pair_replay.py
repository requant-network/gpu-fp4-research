"""Synthetic traces for software tests, not measured GPU evidence."""
import hashlib
import unittest

from verify_energy_pair import TraceError, replay_pair


def fixture():
    plan = {'evidence_kind': 'synthetic-software-test', 'gpu_uuid': 'synthetic-only', 'arms': ['a', 'b'],
            'window_offsets_ns': [0, 100, 300, 400], 'boundary_slack_ns': 4,
            'arrival_slack_ns': 2, 'cutoff_offset_ns': 290,
            'schedule': [{'id': 'first', 'arrival_offset_ns': 120},
                         {'id': 'last', 'arrival_offset_ns': 220}],
            'truth': {key: hashlib.sha256(bytes([i])).hexdigest()
                      for i, key in enumerate(('first', 'last'))}}
    runs = []
    for i, arm in enumerate(plan['arms']):
        start = (i + 1) * 1000
        runs.append({'arm': arm, 'gpu_uuid': 'synthetic-only', 'status': 'complete',
                     'origin_ns': start, 'epoch': 'synthetic-epoch',
                     'events': [{'id': key, 'arrival_ns': start + offset,
                                 'delivery_ns': start + offset + latency,
                                 'status': 'delivered', 'output_hex': f'{j:02x}'}
                                for j, (key, offset, latency) in enumerate(
                                    [('first', 120, 15), ('last', 220, 30 + i)])],
                     'samples': [{'start_ns': start + offset - 1, 'end_ns': start + offset + 1,
                                  'epoch': 'synthetic-epoch', 'status': 'success', 'mJ': total + 10000*i,
                                  'gpu_uuid': 'synthetic-only'}
                                 for offset, total in zip(plan['window_offsets_ns'],
                                     [1100, 1200, 5200 - 1000*i, 5300 - 1000*i])],
                     'boundary_sample_indices': [0, 1, 2, 3]})
    return plan, runs


class EnergyReplayTests(unittest.TestCase):
    def test_recomputes_per_run_energy_latency_and_idle_without_subtraction(self):
        plan, runs = fixture()
        summary = replay_pair(plan, runs)
        self.assertEqual([row['measurement_mJ'] for row in summary['runs']], [4000, 3000])
        self.assertEqual([row['joules_per_delivery'] for row in summary['runs']], [2.0, 1.5])
        self.assertEqual([row['p99_latency_ns'] for row in summary['runs']], [30, 31])
        self.assertEqual([row['pre_idle_mJ'] for row in summary['runs']], [100, 100])
        self.assertEqual([row['post_idle_mJ'] for row in summary['runs']], [100, 100])
        self.assertEqual(summary['evidence_kind'], 'synthetic-software-test')
        self.assertEqual(summary['delta_second_minus_first_mJ'], -1000)

    def test_decrease_or_changed_counter_epoch_is_rejected(self):
        for key, value in (('mJ', 1), ('epoch', 'reloaded'), ('status', 'not_supported')):
            with self.subTest(key=key):
                plan, runs = fixture()
                runs[0]['samples'][2][key] = value
                with self.assertRaisesRegex(TraceError, 'counter'):
                    replay_pair(plan, runs)

    def test_valid_stale_output_bytes_rejected(self):
        plan, runs = fixture()
        runs[1]['events'][1]['output_hex'] = runs[1]['events'][0]['output_hex']
        with self.assertRaisesRegex(TraceError, 'output witness'):
            replay_pair(plan, runs)

    def test_missing_or_late_delivery_cannot_win_pair(self):
        for missing in (True, False):
            plan, runs = fixture()
            if missing:
                runs[1]['events'].pop()
            else:
                runs[1]['events'][1]['delivery_ns'] = runs[1]['origin_ns'] + 291
            with self.assertRaisesRegex(TraceError, 'deliver'):
                replay_pair(plan, runs)

    def test_delayed_arrival_even_with_correct_output_is_not_paired(self):
        plan, runs = fixture()
        runs[1]['events'][0]['arrival_ns'] += 3
        with self.assertRaisesRegex(TraceError, 'arrival'):
            replay_pair(plan, runs)

    def test_boundary_aperture_rejects_a_recent_but_nonbracketing_read(self):
        plan, runs = fixture()
        runs[0]['samples'][2]['end_ns'] -= 2
        with self.assertRaisesRegex(TraceError, 'boundary'):
            replay_pair(plan, runs)

    def test_flat_measurement_counter_cannot_be_scored_as_an_energy_win(self):
        plan, runs = fixture()
        for sample in runs[0]['samples']:
            sample['mJ'] = 1100
        with self.assertRaisesRegex(TraceError, 'unresolved energy'):
            replay_pair(plan, runs)

    def test_noninteger_counter_value_is_rejected(self):
        plan, runs = fixture()
        runs[0]['samples'][1]['mJ'] = 1200.0
        with self.assertRaisesRegex(TraceError, 'counter'):
            replay_pair(plan, runs)
