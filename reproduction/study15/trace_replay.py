"""Replay Study 15 raw workload-trace completeness against a pinned plan (CPU only)."""
import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
from pathlib import Path
import re


CAPACITY = {'singleton': 1, 'batch8': 8}
PHASES = ('pre-idle-start', 'active-start', 'active-end', 'post-idle-end')


class TraceError(ValueError):
    """The supplied bytes do not establish a complete, consistent trace."""


def require(condition, message):
    if not condition:
        raise TraceError(message)


def uint(value):
    return type(value) is int and value >= 0


def text(value):
    return isinstance(value, str) and bool(value)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'duplicate JSON key: {key!r}')
            result[key] = value
        return result

    def invalid_constant(value):
        raise TraceError(f'invalid JSON constant: {value}')

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)


def offsets_between(values):
    return [right - left for left, right in zip(values, values[1:])]


def check_samples(run, plan, previous):
    samples, indices = run.get('samples'), run.get('boundary_sample_indices')
    require(isinstance(samples, list) and len(samples) == len(PHASES)
            and isinstance(indices, list) and all(map(uint, indices))
            and indices == list(range(len(PHASES))), 'invalid raw NVML boundary coverage')
    last_end, last_mj = previous
    for phase, offset, sample in zip(PHASES, plan['window_offsets_ns'], samples):
        code = sample.get('nvml_status')
        require(type(code) is int and code == 0 and sample.get('status') == 'success',
                f'invalid or contradictory raw NVML status at {phase}')
        require(sample['phase'] == phase, f'boundary phase differs: expected {phase}')
        target, slack = run['origin_ns'] + offset, plan['boundary_slack_ns']
        require(uint(sample['target_ns']) and sample['target_ns'] == target,
                f'boundary target differs at {phase}')
        require(sample['epoch'] == run['epoch'] and sample['gpu_uuid'] == plan['gpu_uuid'],
                f'NVML identity or epoch differs at {phase}')
        start, end, mj = sample['start_ns'], sample['end_ns'], sample['mJ']
        require(uint(start) and uint(end) and start <= end and (last_end is None or last_end < start),
                f'invalid NVML call interval/order at {phase}')
        require(target - slack <= start <= target <= end <= target + slack,
                f'raw NVML read does not bracket target within boundary aperture at {phase}')
        require(uint(mj) and mj < 2**64 and (last_mj is None or last_mj <= mj),
                f'invalid or decreasing raw NVML counter at {phase}')
        last_end, last_mj = end, mj
    require(samples[2]['mJ'] > samples[1]['mJ'], 'active counter did not advance')
    return (last_end, last_mj)


def check_requests(run, plan, ids, offsets):
    arrivals, events = run.get('arrivals'), run.get('events')
    for rows, label in ((arrivals, 'raw arrival coverage'), (events, 'delivery coverage')):
        require(isinstance(rows, list) and [row['id'] for row in rows] == ids,
                f'{label} differs from exact schedule order')
    origin, tolerance = run['origin_ns'], plan['arrival_slack_ns']
    cutoff = origin + plan['cutoff_offset_ns']
    opened, closing = run['samples'][1]['end_ns'], run['samples'][2]['start_ns']
    stamps = []
    for key, offset, arrival, event in zip(ids, offsets, arrivals, events):
        planned = origin + offset
        require(uint(arrival['planned_ns']) and arrival['planned_ns'] == planned,
                f'raw arrival planned timestamp differs for {key}')
        require(arrival['status'] == 'queued', f'invalid raw arrival status for {key}')
        stamp = arrival['arrival_ns']
        require(uint(stamp) and 0 <= stamp - planned <= tolerance and opened <= stamp < cutoff
                and (not stamps or stamps[-1] <= stamp), f'arrival timing differs for {key}')
        require(uint(event['arrival_ns']) and event['arrival_ns'] == stamp,
                f'delivery arrival binding differs for {key}')
        delivered = event['delivery_ns']
        require(event['status'] == 'delivered' and uint(delivered) and stamp <= delivered <= min(cutoff, closing),
                f'invalid delivery timing/status for {key}')
        witness = event['output_hex']
        require(isinstance(witness, str) and re.fullmatch(r'[0-9a-f]{16}', witness) is not None
                and sha256(bytes.fromhex(witness)) == plan['truth'][key], f'output witness differs for {key}')
        stamps.append(stamp)
    return stamps


def check_dispatch(run, plan, ids, stamps):
    batches = run.get('batches')
    require(isinstance(batches, list) and bool(batches)
            and all(isinstance(b['ids'], list) and bool(b['ids']) for b in batches)
            and [key for batch in batches for key in batch['ids']] == ids,
            'dispatch coverage differs from exact schedule order')
    capacity = CAPACITY[run['arm']]
    cutoff = run['origin_ns'] + plan['cutoff_offset_ns']
    head, previous_delivery = 0, run['samples'][1]['end_ns']
    histogram = {}
    for seq, batch in enumerate(batches):
        require(uint(batch['sequence']) and batch['sequence'] == seq, f'invalid batch sequence at {seq}')
        require(batch['status'] == 'delivered', f'invalid batch status at {seq}')
        size, dispatch, delivery = len(batch['ids']), batch['dispatch_ns'], batch['delivery_ns']
        require(1 <= size <= capacity, f'invalid batch size at {seq}')
        require(uint(dispatch) and previous_delivery <= dispatch < cutoff,
                f'overlapping or out-of-window dispatch at batch {seq}')
        require(stamps[head + size - 1] <= dispatch, f'request not queued at dispatch in batch {seq}')
        # All earlier arrivals must be pending, but an equal clock tick alone
        # does not establish whether an arrival preceded this dispatch.
        earliest = bisect_left(stamps, dispatch, lo=head) - head
        latest = bisect_right(stamps, dispatch, lo=head) - head
        require(min(earliest, capacity) <= size <= min(latest, capacity),
                f'batch does not drain the available FIFO prefix at {seq}')
        reason = 'singleton' if capacity == 1 else 'full' if size == capacity else 'deadline'
        require(batch['reason'] == reason, f'invalid batch reason at {seq}')
        if size < capacity:
            require(dispatch >= stamps[head] + plan['batch_wait_ns'], f'batch deadline missed policy at {seq}')
        require(uint(delivery) and dispatch <= delivery
                and all(event['delivery_ns'] == delivery for event in run['events'][head:head + size]),
                f'batch delivery binding differs at {seq}')
        previous_delivery = delivery
        head += size
        histogram[size] = histogram.get(size, 0) + 1
    return histogram


def replay(plan, runs):
    """Return a completeness receipt (not eligibility, process, or native attestation)."""
    try:
        return _replay(plan, runs)
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise TraceError(f'malformed Study 15 plan or raw trace: {error}') from error


def _replay(plan, runs):
    require(plan['schema'] == 'study15-request-plan-v1', 'unsupported plan schema')
    evidence, arms, gpu = plan['evidence_kind'], plan['arms'], plan['gpu_uuid']
    require(evidence in ('synthetic-software-test', 'measured') and text(gpu)
            and isinstance(arms, list) and sorted(arms) == sorted(CAPACITY), 'invalid evidence label, arms or GPU')
    windows, aperture, tolerance, cutoff, wait = (plan[key] for key in (
        'window_offsets_ns', 'boundary_slack_ns', 'arrival_slack_ns', 'cutoff_offset_ns', 'batch_wait_ns'))
    require(isinstance(windows, list) and len(windows) == len(PHASES) and all(map(uint, windows))
            and windows[0] == 0 and all(a < b for a, b in zip(windows, windows[1:]))
            and all(map(uint, (aperture, tolerance, cutoff, wait))) and wait > 0
            and windows[1] <= cutoff < windows[2], 'invalid timing plan')
    schedule = plan['schedule']
    require(isinstance(schedule, list) and bool(schedule), 'invalid schedule')
    ids = [row['id'] for row in schedule]
    offsets = [row['arrival_offset_ns'] for row in schedule]
    truth = plan['truth']
    require(all(map(text, ids)) and len(set(ids)) == len(ids) and all(map(uint, offsets))
            and windows[1] <= offsets[0] <= offsets[-1] <= cutoff and offsets == sorted(offsets),
            'invalid schedule IDs or timestamps')
    require(isinstance(truth, dict) and set(ids) == set(truth)
            and all(isinstance(d, str) and re.fullmatch(r'[0-9a-f]{64}', d) for d in truth.values()),
            'invalid output truth bindings')
    require(isinstance(runs, list) and len(runs) == len(arms), 'pair does not contain both runs')
    previous = (None, None)
    epoch = None
    summaries, actual_gaps = [], []
    planned_gaps = offsets_between(offsets)
    for arm, run in zip(arms, runs):
        require(run['arm'] == arm and run['gpu_uuid'] == gpu and run['status'] == 'complete',
                'run order, identity or completion differs from plan')
        require(uint(run['origin_ns']) and text(run['epoch']) and epoch in (None, run['epoch']),
                'invalid run origin or counter epoch differs')
        epoch = run['epoch']
        previous = check_samples(run, plan, previous)
        stamps = check_requests(run, plan, ids, offsets)
        histogram = check_dispatch(run, plan, ids, stamps)
        gaps = offsets_between(stamps)
        actual_gaps.append(gaps)
        max_error = max((abs(a - b) for a, b in zip(gaps, planned_gaps)), default=0)
        require(max_error <= tolerance, f'{arm}: realized gaps differ from plan')
        summaries.append({'arm': arm, 'arrivals': len(stamps), 'deliveries': len(ids),
                          'batch_count': len(run['batches']), 'batch_size_histogram': histogram,
                          'boundary_sample_count': len(run['samples']), 'max_gap_error_ns': max_error})
    paired_error = max((abs(a - b) for a, b in zip(*actual_gaps)), default=0)
    require(paired_error <= tolerance, 'paired realized gaps differ between arms')
    return {'schema': 'study15-trace-completeness-replay-v1', 'evidence_kind': evidence,
            'verdict': 'software-test-pass' if evidence == 'synthetic-software-test' else 'trace-check-pass',
            'native_attestation': False, 'scope': 'pinned plan and raw runs only',
            'max_paired_gap_difference_ns': paired_error, 'runs': summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('runs', type=Path)
    parser.add_argument('--plan-sha256', required=True, help='trusted plan pin from outside the attempt')
    args = parser.parse_args()
    try:
        raw_plan, raw_runs = args.plan.read_bytes(), args.runs.read_bytes()
        plan_sha = sha256(raw_plan)
        require(plan_sha == args.plan_sha256, 'independent plan pin differs')
        receipt = replay(read_json(raw_plan), read_json(raw_runs))
        receipt.update(plan_sha256=plan_sha, runs_sha256=sha256(raw_runs),
                       verifier_sha256=sha256(Path(__file__).read_bytes()))
    except (OSError, ValueError, RecursionError) as error:
        parser.exit(1, f'TRACE REJECTED: {error}\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
