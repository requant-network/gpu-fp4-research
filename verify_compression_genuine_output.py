"""Independently reconstruct the SIFT input and replay every timed output witness."""
import argparse
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path
import struct

import numpy as np

from verify_compression_genuine_preparation import (
    BUDGETS, COHORTS, FIXTURE_SHA, ROOT, ReplayError, load_cohorts, require, sha,
)

PREP_PIN = 'a5335cb2103f293e002f3bb040beaf8c836e1380d11f038a6c8a132a0cd64a8e'
PREP_RECEIPT_PIN = 'b5d5ff7b78dc321336d6f52f649b708ee238fb7b652a3b5206a162db5afa1f2e'


def expected_routes(suite, reference):
    require(suite in ('smoke', 'full') and reference in ('ordered-copy', 'stock-serial'),
            'Unknown suite or reference')
    result = []
    for cohort, seed, _, _ in COHORTS:
        routes = [dict(codec=codec, policy=policy, queue_depth=depth,
                       cpu_threads=1 if codec == 'plain' or (codec == 'ndzip' and reference == 'stock-serial') else 4)
                  for codec in ('plain', 'ndzip', 'xbp')
                  for policy in (('online',) if codec == 'plain' else ('online', 'precompressed'))
                  for depth in (1, 2)]
        if suite == 'smoke':
            return [((cohort, '<f4', BUDGETS[0], seed), r) for r in routes
                    if r['policy'] == 'online' and r['queue_depth'] == 1]
        if cohort == 'b':
            routes = routes[::-1]
        result.extend(((cohort, dtype, budget, seed), r)
                      for dtype in ('<f4', '<f8') for budget in BUDGETS for r in routes)
    return result


def check_witness(directory, binding, expected, checked):
    expected_hash = sha(expected)
    relative = f'witnesses/{expected_hash}.npy'
    count = len(expected) // 32
    require(binding['file'] == relative and binding['array_sha256'] == expected_hash
            and binding['array_bytes'] == len(expected) and binding['shape'] == [count, 32],
            'Output witness does not bind the expected frame')
    key = relative, binding['sha256'], expected_hash
    if key in checked:
        return
    file = directory / relative
    require(not file.is_symlink() and file.resolve().is_relative_to(directory.resolve()), 'Unsafe witness path')
    raw = file.read_bytes()
    require(sha(raw) == binding['sha256'], 'Output witness file hash differs')
    table = np.load(BytesIO(raw), allow_pickle=False)
    require(table.dtype == np.uint8 and table.shape == (count, 32) and table.tobytes() == expected,
            'Output witness differs from original bytes')
    checked.add(key)


def read_pin(path, pin):
    raw = path.read_bytes()
    require(sha(raw) == pin, f'Pinned record differs: {path.name}')
    return json.loads(raw)


def replay(directory, root=ROOT):
    directory = Path(directory)
    manifest = directory / 'manifest.json'
    raw = manifest.read_bytes()
    record = json.loads(raw)
    require(record['schema'] == 'compression-genuine-pair-suite-v1' and record['complete'] is True
            and record['active_case'] is None and record['stage'] == 'complete', 'Unsealed collection')
    require(record['device']['name'] == 'NVIDIA GeForce RTX 5060 Ti'
            and (record['device']['major'], record['device']['minor']) == (12, 0), 'Wrong recorded target')
    schedule = expected_routes(record['suite'], record['ndzip_reference'])
    require(len(record['cases']) == len(schedule), 'Wrong route coverage')
    prep = read_pin(root / 'results/compression-genuine-preparation/manifest.json', PREP_PIN)
    receipt = read_pin(root / 'results/compression-genuine-preparation-replay.json', PREP_RECEIPT_PIN)
    require(receipt['preparation_manifest_sha256'] == PREP_PIN and record['input'] == {
        'preparation_manifest_sha256': PREP_PIN, 'preparation_replay_sha256': PREP_RECEIPT_PIN,
        'preparation_verifier_sha256': receipt['verifier_sha256'], 'fixture': prep['fixture'],
    }, 'Wrong pinned preparation binding')
    require(record['fixture_before'] == record['fixture_after'] == prep['fixture'], 'Fixture closure differs')
    for name, digest in record['sources'].items():
        source = root / name
        require(not source.is_symlink() and source.resolve().is_relative_to(root.resolve())
                and sha(source.read_bytes()) == digest, f'Bound source differs: {name}')
    fixture = root / 'fixtures/search-sift'
    fixture_doc = read_pin(fixture / 'manifest.json', FIXTURE_SHA)
    windows, bindings = load_cohorts(fixture, [f for f in fixture_doc['files'] if f['kind'] == 'database'])
    require(bindings == prep['fixture']['files'], 'Fixture file bindings differ')
    originals = {cohort: window.ravel() for (cohort, _, _, _), window in zip(COHORTS, windows, strict=True)}
    prepared = {(p['cohort'], p['dtype'], p['frame_bytes']): p for p in prep['cases']}
    groups = defaultdict(list)
    for index, (case, (key, route)) in enumerate(zip(record['cases'], schedule, strict=True)):
        cohort, dtype, budget, seed = key
        p = prepared[cohort, dtype, budget]
        require((case['index'], case['cohort'], case['dtype'], case['frame_bytes'], case['seed'], case['route'])
                == (index, cohort, dtype, budget, seed, route), f'Wrong route identity at {index}')
        require(case['ndzip_reference'] == record['ndzip_reference'] and case['profile'] == 'sift'
                and case['preparation_case_index'] == p['index'] and case['logical_sha256'] == p['logical_sha256']
                and case['total_bytes'] == p['total_bytes'], 'Wrong preparation case identity')
        frames = len(p['frames'])
        require(len(case['frames']) == len(case['encoded_controls']) == frames and len(case['trials']) == 2,
                'Wrong route frame or repeat coverage')
        for repeat, trial in enumerate(case['trials']):
            observations = trial['observations']
            require(trial['repeat'] == repeat and [o['frame'] for o in observations] == list(range(frames)),
                    'Wrong timed frame coverage')
            require(type(trial['host_ns']) is int and trial['host_ns'] > 0
                    and trial['wire_bytes'] == sum(o['wire_bytes'] for o in observations), 'Invalid trial accounting')
        groups[key].append(case)
    checked = set()
    observations = observed_bytes = 0
    for (cohort, dtype, budget, _), cases in groups.items():
        p = prepared[cohort, dtype, budget]
        fmt, uint = ('<f', '<u4') if dtype == '<f4' else ('<d', '<u8')
        lut = np.frombuffer(b''.join(struct.pack(fmt, x) for x in range(256)), dtype=uint)
        original = originals[cohort]
        stream = np.asarray(lut[original]).view(np.uint8)
        require(stream.nbytes == p['total_bytes'] and sha(stream) == p['logical_sha256'],
                'Independent SIFT stream identity differs')
        for frame, start in enumerate(range(0, stream.nbytes, budget)):
            data = memoryview(stream[start:start + budget])
            n = len(data)
            spec = {'frame': frame, 'shape': [n // (128 * lut.itemsize), 128],
                    'offset_words': start // lut.itemsize, 'bytes': n, 'input_sha256': sha(data)}
            require(p['frames'][frame] == spec, 'Preparation frame identity differs')
            table = b''.join(bytes.fromhex(sha(data[i:i+4096])) for i in range(0, n, 4096))
            table_hash = sha(table)
            for case in cases:
                desc = case['frames'][frame]
                require(all(desc[k] == v for k, v in spec.items()) and desc['expected_digest_sha256'] == table_hash,
                        'Prepared frame commitment differs')
                control = case['encoded_controls'][frame]
                expected_cache = control['sha256'] if case['route']['policy'] == 'precompressed' else spec['input_sha256']
                require(desc['cache_sha256'] == expected_cache, 'Wrong cached-input identity')
                if case['route']['codec'] == 'plain':
                    require(control == {'bytes': n, 'sha256': spec['input_sha256'], 'retained_file': None},
                            'Wrong plain wire control')
                for trial in case['trials']:
                    obs = trial['observations'][frame]
                    require(obs['wire_bytes'] == control['bytes'] and obs['device_error_bits'] == 0,
                            'Wrong transfer accounting or device status')
                    check_witness(directory, obs['witness'], table, checked)
                    observations += 1
                    observed_bytes += n
        del data, stream
    require(manifest.read_bytes() == raw, 'Collection changed during replay')
    return {'schema': 'compression-genuine-output-replay-v1', 'collection_manifest_sha256': sha(raw),
            'verifier_sha256': sha(Path(__file__).read_bytes()), 'prep_verifier_sha256': receipt['verifier_sha256'],
            'ndzip_reference': record['ndzip_reference'], 'suite': record['suite'], 'cases': len(schedule),
            'observations': observations, 'restored_byte_observations': observed_bytes,
            'unique_witnesses': len(checked),
            'scope': 'Independent original words and timed output fingerprints; not encoded packet, launch, or full cost audit'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.directory)
    with args.summary.open('x', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
