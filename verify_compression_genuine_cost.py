"""Replay named timing and allocation accounting for both full SIFT runs."""
import argparse
import json
import math
from pathlib import Path

from compression_pair_models import ndzip_bound_bytes
from verify_compression_genuine_output import expected_routes
from verify_compression_genuine_preparation import ReplayError, require, sha

STAGES = ('setup_ns', 'prepare_ns', 'first_use_ns', 'control_replay_ns', 'post_guard_ns', 'allocator_cleanup_ns')
PREP = ('prepare_input_reference_ns', 'prepare_encoding_ns', 'prepare_cache_ns', 'prepare_other_ns')
EVENTS = ('start_ms', 'ingest_end_ms', 'hash_end_ms', 'result_end_ms')
PINNED = {
    'ordered-copy': ('9b036a1e0a1ce5c679aca567eaa717f440e8060088b44990b3f6e4856cf8c916',
                     '8fb6a4d9dc38f05779d5b77586dd902660bde3dcc6946df010bb56b05e8d70f1'),
    'stock-serial': ('b8522143eeeea14c985b342e2c85c29ef3f04d31947995fd827dbb191888030f',
                     '1174efb682d5765ba20480f67477a757d3591f8102ec76cdea9ccd7ffb819ee3'),
}


def nonnegative(value):
    return type(value) is int and value >= 0


def size_of_digest(n):
    return ((n + 4095) // 4096) * 32


def audit_case(case):
    route, frames, controls = (case[k] for k in ('route', 'frames', 'encoded_controls'))
    codec, policy, depth = (route[k] for k in ('codec', 'policy', 'queue_depth'))
    require(case['dtype'] in ('<f4', '<f8') and codec in ('plain', 'ndzip', 'xbp')
            and depth in (1, 2) and policy in ('online', 'precompressed')
            and (codec != 'plain' or policy == 'online'), 'Bad route identity')
    require(bool(frames) and len(frames) == len(controls), 'Missing frame control')
    require(all(nonnegative(case[k]) for k in STAGES + PREP), 'Bad host timing')
    require(case['prepare_ns'] == sum(case[k] for k in PREP)
            and (case['prepare_encoding_ns'] > 0) == (policy == 'precompressed'), 'Bad preparation accounting')
    width = 4 if case['dtype'] == '<f4' else 8
    sizes = [f['bytes'] for f in frames]
    require(all(nonnegative(n) and n > 0 for n in sizes) and sum(sizes) == case['total_bytes'], 'Bad original-byte census')
    require(all(f['frame'] == i and len(f['shape']) == 2 and f['shape'][1] == 128
                and math.prod(f['shape']) * width == sizes[i] for i, f in enumerate(frames)), 'Bad frame geometry')
    largest = max(sizes)
    status = 4 if codec == 'xbp' else 0
    if codec == 'xbp':
        candidate = 32 + 16 * ((largest // width + 255) // 256) + 8 * ((largest + 7) // 8)
    elif codec == 'ndzip':
        candidate = max(ndzip_bound_bytes(f['shape'], case['dtype']) for f in frames)
    else:
        candidate = 0
    wire = [c['bytes'] for c in controls]
    require(all(nonnegative(n) and n > 0 for n in wire) and
            (wire == sizes if codec == 'plain' else max(wire) <= candidate), 'Packet size outside named allocation')
    cached = wire if policy == 'precompressed' else sizes
    require(cached == [f['cache_bytes'] for f in frames], 'Frame cache sizes differ')
    table = sum(size_of_digest(n) for n in sizes)
    slot = {'requested_device_bytes': largest + candidate + size_of_digest(largest) + status,
            'requested_pinned_slot_bytes': candidate + size_of_digest(largest) + status,
            'candidate_bytes': candidate, 'opaque_library_cpu_workspace_measured': False}
    require(case['memory'] == {'slots': [slot] * depth, 'requested_pinned_source_bytes': sum(cached),
                               'expected_digest_bytes': table, 'physical_peak_measured': False},
            'Named memory accounting differs')
    intervals = case['nonoverlapping_host_intervals']
    require(len(intervals) == len(frames) and intervals[0]['begin'] == 0
            and sorted(i['bytes'] for i in intervals) == sorted(cached)
            and all(nonnegative(i['begin']) for i in intervals)
            and all(a['begin'] + a['bytes'] <= b['begin'] for a, b in zip(intervals, intervals[1:])),
            'Overlapping or incomplete pinned source intervals')
    trials = case['trials']
    require(len(trials) == 2, 'Wrong repeat count')
    for index, trial in enumerate(trials):
        records = trial['observations']
        host = trial['host_ns']
        require(nonnegative(host) and host > 0 and trial['repeat'] == index and len(records) == len(frames)
                and trial['wire_bytes'] == sum(wire), 'Bad trial identity or accounting')
        encodes = [r['cpu_encode_validate_ns'] for r in records]
        require(all(nonnegative(n) and (n > 0) == (codec != 'plain' and policy == 'online') for n in encodes)
                and sum(encodes) <= host, 'Bad online encode timing')
        for j, row in enumerate(records):
            times = [row[k] for k in EVENTS]
            require(row['frame'] == j and row['wire_bytes'] == wire[j] and row['device_error_bits'] == 0,
                    'Bad timed frame or device status')
            require(all(math.isfinite(t) and t >= 0 for t in times) and times == sorted(times), 'Bad CUDA event interval')
            if j >= depth:
                require(records[j-depth]['result_end_ms'] <= times[0], 'Premature slot reuse')
    return {'named_case_ns': sum(case[k] for k in STAGES) + sum(t['host_ns'] for t in trials),
            'observation_count': len(frames) * len(trials),
            'restored_byte_observations': sum(sizes) * len(trials),
            'wire_bytes_observed': sum(wire) * len(trials), 'expected_digest_bytes_observed': table * len(trials),
            'requested_device_bytes': depth * slot['requested_device_bytes'],
            'requested_pinned_bytes': sum(cached) + depth * slot['requested_pinned_slot_bytes']}


def close_wall(launch, named):
    wall, other, envelope = (launch[k] for k in ('process_wall_ns', 'other_child_ns', 'envelope_wall_ns'))
    require(all(nonnegative(n) for n in (wall, other, envelope)) and wall > 0 and envelope >= wall
            and launch['named_case_ns'] == named and named + other == wall, 'Child wall accounting differs')


def audit(directory):
    paths = [directory / 'manifest.json', directory.with_name(directory.name + '-launch') / 'manifest.json',
             directory.with_name(directory.name + '-output-replay.json')]
    raw = [p.read_bytes() for p in paths]
    hashes = [sha(x) for x in raw]
    record, launch, replay = [json.loads(x) for x in raw]
    reference = record['ndzip_reference']
    require(reference in PINNED and (hashes[0], hashes[2]) == PINNED[reference], 'Unknown SIFT evidence identity')
    require(record['schema'] == 'compression-genuine-pair-suite-v1' and record['suite'] == 'full'
            and record['complete'] is True and replay['suite'] == 'full' and replay['ndzip_reference'] == reference
            and replay['collection_manifest_sha256'] == hashes[0], 'Output replay is not bound to collection')
    require(launch['schema'] == 'compression-genuine-launch-v1' and launch['sealed'] is True
            and launch['accepted'] is True and launch['exit_code'] == 0
            and launch['collection_manifest_sha256'] == hashes[0] and launch['closure_valid'] is True
            and launch['timing_valid'] is True and launch['unhandled_error_marker'] is False, 'Launch is not eligible')
    child_log = paths[1].parent / 'child.log'
    require(sha(child_log.read_bytes()) == launch['child_log_sha256'], 'Child log hash differs')
    schedule = expected_routes('full', reference)
    require(len(record['cases']) == len(schedule) == launch['case_count'] == launch['expected_case_count'] == replay['cases'],
            'Missing full route declaration')
    rows = []
    for i, (case, (identity, route)) in enumerate(zip(record['cases'], schedule, strict=True)):
        require(case['index'] == i and tuple(case[k] for k in ('cohort', 'dtype', 'frame_bytes', 'seed')) == identity
                and case['route'] == route, 'Run ordering differs')
        rows.append(audit_case(case))
    totals = {key: sum(r[key] for r in rows) for key in ('named_case_ns', 'observation_count',
              'restored_byte_observations', 'wire_bytes_observed', 'expected_digest_bytes_observed')}
    require(totals['observation_count'] == replay['observations']
            and totals['restored_byte_observations'] == replay['restored_byte_observations'], 'Replay census differs')
    close_wall(launch, totals['named_case_ns'])
    require([sha(p.read_bytes()) for p in paths] == hashes, 'Evidence changed during audit')
    return {'schema': 'compression-genuine-accounting-replay-v1', 'reference': reference, 'cases': len(rows), **totals,
            'process_wall_ns': launch['process_wall_ns'], 'other_child_ns': launch['other_child_ns'],
            'max_named_device_request_bytes': max(r['requested_device_bytes'] for r in rows),
            'max_named_pinned_request_bytes': max(r['requested_pinned_bytes'] for r in rows),
            'collection_sha256': hashes[0], 'launch_sha256': hashes[1], 'output_replay_sha256': hashes[2],
            'verifiers': {name: sha(Path(__file__).with_name(name).read_bytes()) for name in (
                Path(__file__).name, 'compression_pair_models.py', 'verify_compression_genuine_output.py',
                'verify_compression_genuine_preparation.py')},
            'scope': 'Named accounting only; not physical peak, opaque workspaces, per-method cold wall or packet replay'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    text = json.dumps(result, sort_keys=True) + '\n'
    with args.summary.open('x', encoding='utf-8', newline='\n') as output:
        output.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
