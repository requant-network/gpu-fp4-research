"""Replay all SIFT preparation digests without importing any producer code."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

ROOT = Path(__file__).resolve().parent
FIXTURE_SHA = '4938d01f6ab7e872d362dffc8bb5d43fec95a636dd99c1c5e5e65ff11bb2601f'
SOURCE_SHA = '21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816'
SOURCE_FILES = ('check_compression_genuine.py', 'compression_genuine_source.py',
                'compression_pair_cases.py', 'pca_cases.py', 'reference.py',
                'verify_search_sift.py', 'prepare_search_sift.py')
COHORTS = (('a', 71, 0, 262144), ('b', 171, 737856, 1000000))
BUDGETS = tuple(1 << shift for shift in (18, 20, 22, 24, 26))
COLS = 128
CACHE_LIMIT = 2 * 65536 * COLS


class ReplayError(ValueError):
    """A retained preparation record does not match independently derived input."""


def require(condition, message):
    if not condition:
        raise ReplayError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def equal_json(a, b):
    # The serial form distinguishes bools and integral floats from JSON integers.
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def load_cohorts(fixture, parts):
    windows = [np.empty((hi - lo, COLS), dtype=np.uint8) for _, _, lo, hi in COHORTS]
    cursor = 0
    bindings = {}
    for part in parts:
        lo, hi, name = part['start'], part['stop'], part['file']
        require(lo == cursor < hi <= 1000000 and name not in bindings, 'fixture shard coverage')
        path = fixture / name
        require(path.name == name and not path.is_symlink(), 'fixture shard path')
        packet = path.read_bytes()
        digest = sha(packet)
        require(digest == part['sha256'], 'fixture shard file digest')
        from io import BytesIO
        with np.load(BytesIO(packet), allow_pickle=False) as archive:
            values = archive['vectors']
        require(values.dtype == np.uint8 and values.shape == (hi - lo, COLS)
                and list(values.shape) == part['shape'], 'fixture shard layout')
        require(sha(values.tobytes(order='C')) == part['vectors_sha256'], 'fixture shard vectors digest')
        for (_, _, first, last), window in zip(COHORTS, windows):
            left, right = max(first, lo), min(last, hi)
            if left < right:
                window[left-first:right-first] = values[left-lo:right-lo]
        cursor = hi
        bindings[name] = digest
    require(cursor == 1000000, 'fixture database row coverage')
    return windows, bindings


def replay(record, root):
    root = Path(root)
    require(record.get('schema') == 'compression-genuine-preparation-v1'
            and record.get('complete') is True, 'preparation record is not complete')
    cases = record['cases']
    require(isinstance(cases, list) and len(cases) == len(COHORTS)*2*len(BUDGETS), 'case count mismatch')
    def sources():
        return {name: sha((root / name).read_bytes()) for name in SOURCE_FILES}
    require(record.get('sources') == sources(), 'preparation source bindings do not match')
    fixture = root / 'fixtures/search-sift'
    manifest = (fixture / 'manifest.json').read_bytes()
    require(sha(manifest) == FIXTURE_SHA, 'pinned fixture manifest digest')
    declared = json.loads(manifest)
    meta = declared['inputs']['database']
    require(meta['sha256'] == SOURCE_SHA and meta['dimension'] == COLS and meta['source_rows'] == 1000000,
            'original fixture database identity')
    parts = [part for part in declared['files'] if part['kind'] == 'database']
    windows, bindings = load_cohorts(fixture, parts)
    require(equal_json(record.get('fixture'), {
        'fixture_manifest_sha256': FIXTURE_SHA, 'database_source_sha256': SOURCE_SHA,
        'files': bindings, 'cohort_rows': {str(seed): [lo, hi] for _, seed, lo, hi in COHORTS},
        'max_decoded_cache_bytes': CACHE_LIMIT,
    }), 'fixture bindings or cache contract')
    index = frame_count = observed = logical = 0
    identities = {}
    for (cohort, seed, lo, hi), window in zip(COHORTS, windows):
        for dtype, fmt, word in (('<f4', '<f', '<u4'), ('<f8', '<d', '<u8')):
            # Build IEEE bytes with stdlib scalar packing, then independently look up
            # each original uint8 code; no frame generator or float cast is shared.
            packed = b''.join(struct.pack(fmt, value) for value in range(256))
            lut = np.frombuffer(packed, dtype=word)
            words = lut[window.ravel(order='C')]
            payload = memoryview(words.view(np.uint8))
            size = len(payload)
            logical_hash = sha(payload)
            logical += size
            identities[f'{cohort}:{dtype}'] = logical_hash
            for budget in BUDGETS:
                case = cases[index]
                expected = {'index': index, 'cohort': cohort, 'selector': seed, 'dtype': dtype,
                            'source_rows': [lo, hi], 'total_bytes': size, 'frame_bytes': budget,
                            'requested_decoded_cache_limit_bytes': CACHE_LIMIT}
                require(equal_json({key: case.get(key) for key in expected}, expected),
                        f'case geometry mismatch at {index}')
                require(case.get('logical_sha256') == logical_hash, f'logical input digest mismatch at {index}')
                frames = case['frames']
                require(len(frames) == (size + budget - 1) // budget, f'frame count mismatch at {index}')
                for number, begin in enumerate(range(0, size, budget)):
                    stop = min(begin + budget, size)
                    geometry = {'frame': number, 'offset_words': begin // lut.itemsize,
                                'shape': [(stop-begin) // (COLS*lut.itemsize), COLS], 'bytes': stop-begin}
                    frame = frames[number]
                    require(equal_json({key: frame.get(key) for key in geometry}, geometry),
                            f'frame geometry mismatch at {index}:{number}')
                    require(frame.get('input_sha256') == sha(payload[begin:stop]),
                            f'frame input digest mismatch at {index}:{number}')
                    frame_count += 1
                    observed += stop - begin
                index += 1
            del payload, words
    require(record['sources'] == sources(), 'preparation sources changed during replay')
    require(sha((fixture / 'manifest.json').read_bytes()) == FIXTURE_SHA, 'fixture manifest changed during replay')
    for name, digest in bindings.items():
        require(sha((fixture / name).read_bytes()) == digest, 'fixture shard changed during replay')
    return {'schema': 'compression-genuine-preparation-replay-v1', 'cases': index,
            'frames': frame_count, 'logical_input_identities': len(identities),
            'logical_sha256': identities, 'logical_input_bytes': logical,
            'frame_byte_observations': observed, 'fixture_manifest_sha256': FIXTURE_SHA,
            'database_shards_checked': len(bindings),
            'scope': 'CPU-only prepared inputs, not codec or GPU evidence'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    verifier = Path(__file__)
    verifier_hash = sha(verifier.read_bytes())
    receipt = replay(json.loads(raw), ROOT)
    require(args.manifest.read_bytes() == raw, 'preparation manifest changed during replay')
    require(sha(verifier.read_bytes()) == verifier_hash, 'verifier source changed during replay')
    receipt.update(preparation_manifest_sha256=sha(raw), verifier_sha256=verifier_hash)
    with args.summary.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
