"""Replay matched energy/latency traces against a separately trusted exact-output plan."""
import argparse
import hashlib
import json
from pathlib import Path
import re


class TraceError(ValueError):
    """This pair is not eligible for a quantitative comparison."""


def need(condition, reason):
    if not condition:
        raise TraceError(reason)


def uint(value):
    return type(value) is int and value >= 0


def name(value):
    return isinstance(value, str) and bool(value)


def sha256(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def replay_pair(plan, runs):
    """Return metrics only for a fully delivered, counter-consistent pair."""
    try:
        return _replay_pair(plan, runs)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        raise TraceError('malformed pair or plan') from error


def _replay_pair(plan, runs):
    evidence_kind = plan['evidence_kind']
    need(evidence_kind in ('synthetic-software-test', 'measured'), 'invalid evidence label')
    gpu, arms = plan['gpu_uuid'], plan['arms']
    need(name(gpu) and isinstance(arms, list) and len(arms) == 2 and all(map(name, arms))
         and arms[0] != arms[1], 'invalid GPU or arm plan')
    offsets = plan['window_offsets_ns']
    boundary_slack, arrival_slack = plan['boundary_slack_ns'], plan['arrival_slack_ns']
    cutoff = plan['cutoff_offset_ns']
    need(isinstance(offsets, list) and len(offsets) == 4 and all(map(uint, offsets))
         and offsets[0] == 0 and all(a < b for a, b in zip(offsets, offsets[1:])), 'invalid window offsets')
    need(uint(boundary_slack) and uint(arrival_slack) and uint(cutoff)
         and offsets[1] <= cutoff < offsets[2], 'invalid timing plan')
    schedule = {}
    previous = -1
    for row in plan['schedule']:
        key, offset = row['id'], row['arrival_offset_ns']
        need(name(key) and key not in schedule and uint(offset) and previous <= offset
             and offsets[1] <= offset <= cutoff, 'invalid planned arrival')
        schedule[key] = previous = offset
    truth = plan['truth']
    need(bool(schedule) and isinstance(truth, dict) and set(schedule) == set(truth)
         and all(map(sha256, truth.values())), 'independent output truth is incomplete')
    need(isinstance(runs, list) and len(runs) == 2, 'pair does not contain both runs')
    summaries = []
    last_end, last_mJ, epoch = None, None, None
    for arm, run in zip(arms, runs):
        need(run['arm'] == arm and run['gpu_uuid'] == gpu and run['status'] == 'complete',
             'run order, GPU identity or completion differs from plan')
        origin = run['origin_ns']
        need(uint(origin) and name(run['epoch']) and epoch in (None, run['epoch']), 'counter epoch changed')
        epoch = run['epoch']
        samples = run['samples']
        indices = run['boundary_sample_indices']
        need(isinstance(indices, list) and len(indices) == 4 and all(map(uint, indices))
             and all(a < b for a, b in zip(indices, indices[1:])) and indices[-1] < len(samples),
             'invalid boundary sample index set')
        for row in samples:
            start, end, milli = row['start_ns'], row['end_ns'], row['mJ']
            need(uint(start) and uint(end) and start <= end and (last_end is None or last_end < start),
                 'counter sample timestamp/order invalid')
            need(row['gpu_uuid'] == gpu and row['epoch'] == epoch and row['status'] == 'success'
                 and uint(milli) and milli < 2**64 and (last_mJ is None or last_mJ <= milli),
                 'counter failure, decrease, invalid value, or epoch/identity mismatch')
            last_end, last_mJ = end, milli
        boundaries = [samples[index] for index in indices]
        for offset, row in zip(offsets, boundaries):
            stamp = origin + offset
            need(stamp - boundary_slack <= row['start_ns'] <= stamp <= row['end_ns'] <= stamp + boundary_slack,
                 'counter boundary is not bracketed within the read aperture')
        seen, latencies = set(), []
        for row in run['events']:
            key, arrival, delivery = row['id'], row['arrival_ns'], row['delivery_ns']
            need(name(key) and key in schedule and key not in seen and row['status'] == 'delivered',
                 'unexpected, duplicate or failed delivery')
            need(uint(arrival) and abs(arrival - origin - schedule[key]) <= arrival_slack
                 and arrival >= boundaries[1]['end_ns'], 'actual arrival differs from the schedule or phase')
            need(uint(delivery) and arrival <= delivery <= min(origin + cutoff, boundaries[2]['start_ns']),
                 'late or out-of-order delivery')
            text = row['output_hex']
            try:
                data = bytes.fromhex(text)
            except (TypeError, ValueError) as error:
                raise TraceError('invalid output witness') from error
            need(data.hex() == text and hashlib.sha256(data).hexdigest() == truth[key], 'output witness differs from truth')
            seen.add(key)
            latencies.append(delivery - arrival)
        need(seen == set(schedule), 'missing delivery')
        readings = [row['mJ'] for row in boundaries]
        pre, active, post = [b - a for a, b in zip(readings, readings[1:])]
        need(active > 0, 'unresolved energy: active counter did not advance')
        latencies.sort()
        n = len(latencies)
        summaries.append({'arm': arm, 'measurement_mJ': active, 'pre_idle_mJ': pre, 'post_idle_mJ': post,
                          'deliveries': n, 'joules_per_delivery': active / (1000*n),
                          'p99_latency_ns': latencies[(99*n + 99)//100 - 1]})
    return {'schema': 'paired-energy-trace-replay-v1', 'evidence_kind': evidence_kind, 'runs': summaries,
            'delta_second_minus_first_mJ': summaries[1]['measurement_mJ'] - summaries[0]['measurement_mJ'],
            'scope': 'GPU-device counter; no idle subtraction or whole-system attribution'}


def read_json(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            need(key not in out, f'duplicate JSON key: {key}')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=unique)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('runs', type=Path)
    parser.add_argument('--plan-sha256', required=True, help='trusted plan pin recorded outside the measured run')
    args = parser.parse_args()
    try:
        plan, runs = args.plan.read_bytes(), args.runs.read_bytes()
        digest = hashlib.sha256(plan).hexdigest()
        need(digest == args.plan_sha256, 'independent plan pin differs')
        result = replay_pair(read_json(plan), read_json(runs))
    except (OSError, ValueError) as error:
        parser.exit(1, f'NOT COMPARABLE: {error}\n')
    result.update(plan_sha256=digest, runs_sha256=hashlib.sha256(runs).hexdigest(),
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
