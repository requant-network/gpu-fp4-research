"""Audit retained bounded-stream outputs against the pinned CPU-replayed reference."""

import hashlib
import json
from pathlib import Path
import sys


PIN = '44cf4bb43382e2c631836a59b2be9d92cff95c3b1fcd2722ff08ece4289ff756'


def hash_file(path):
    if path.is_symlink():
        raise ValueError(f'Unexpected symlink {path.name}')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(collection, prior_manifest, wait_path, source_pin):
    assert hash_file(prior_manifest) == PIN
    old = json.loads(prior_manifest.read_bytes())
    manifest = collection / 'manifest.json'
    new = json.loads(manifest.read_bytes())
    wait = json.loads(wait_path.read_bytes())
    assert wait['schema'] == 'compression-owned-wrapper-wait-v1' and wait['returncode'] == 0
    assert wait['same_boot'] and wait['monotonic_started_ns'] <= wait['started_ns'] < wait['ended_ns'] <= wait['monotonic_ended_ns']
    assert wait['argv'][-2:] == [str(collection), new['mode']]
    assert new['schema'] == 'study14-bounded-stream-v1' and new['complete'] is True
    assert new['original_collection_sha256'] == PIN and new['source_sha256'] == source_pin
    mode = new['mode']
    assert mode in ('smoke', 'full')
    count = 4 if mode == 'smoke' else 320
    assert new['frame_bytes'] == 64 << 20 and new['logical_bytes'] == count * new['frame_bytes']
    assert new['framebuffer_physical_bytes'] == 17_103_323_136
    assert mode == 'smoke' or new['logical_bytes'] > new['framebuffer_physical_bytes']
    frames = old['cases'][0]['frames']
    expected = old['cases'][0]['trials'][0]['observations']
    rows, checked = [], set()
    for original, case in zip(old['cases'], new['cases'], strict=True):
        assert case['route'] == original['route'] and case['requested_pinned_source_bytes'] == 2 * new['frame_bytes']
        assert len(case['slot_requests']) == 2
        groups = case['groups']
        assert len(groups) == count // 2 and all(len(group['frames']) == 2 for group in groups)
        assert case['first_use_ns'] > 0 and case['setup_ns'] > 0
        service = prep = wire = 0
        for group in groups:
            assert 0 < group['prepare_ns'] + group['service_ns'] <= group['group_wall_ns']
            assert group['service_ns'] > 0
            prep += group['prepare_ns']
            service += group['service_ns']
            assert sum(frame['encode_validate_ns'] for frame in group['frames']) <= group['service_ns']
        for index, frame in enumerate(f for group in groups for f in group['frames']):
            assert frame['frame'] == index and frame['input_sha256'] == frames[index]['input_sha256']
            assert frame['wire_bytes'] > 0 and frame['encode_validate_ns'] >= 0
            assert frame['witness'] == expected[index]['witness']
            digest = frame['witness']['array_sha256']
            if digest not in checked:
                path = collection / 'witnesses' / (digest + '.npy')
                assert hash_file(path) == frame['witness']['sha256']
                checked.add(digest)
            wire += frame['wire_bytes']
        if case['route']['codec'] == 'plain':
            assert wire == new['logical_bytes']
        other = case['wall_ns'] - prep - service - case['setup_ns'] - case['first_use_ns']
        assert other >= 0 and case['wall_ns'] >= sum(group['group_wall_ns'] for group in groups)
        rows.append(dict(codec=case['route']['codec'], available_host_bytes=new['available_host_bytes'],
                         wire_bytes=wire, prepare_ns=prep, service_ns=service, wall_ns=case['wall_ns'],
                         setup_ns=case['setup_ns'], first_use_ns=case['first_use_ns'], other_ns=other,
                         source_pinned_bytes=case['requested_pinned_source_bytes'],
                         slot_pinned_bytes=sum(s['requested_pinned_slot_bytes'] for s in case['slot_requests']),
                         slot_device_bytes=sum(s['requested_device_bytes'] for s in case['slot_requests'])))
    assert [row['codec'] for row in rows] == ['plain', 'ndzip', 'xbp']
    owned_wall = wait['ended_ns'] - wait['started_ns']
    assert sum(row['wall_ns'] for row in rows) <= owned_wall
    assert {p.name for p in (collection / 'witnesses').iterdir()} == {sha + '.npy' for sha in checked}
    return dict(schema='study14-bounded-stream-replay-v1', mode=mode, logical_bytes=new['logical_bytes'],
                frames_replayed=3 * count, unique_digest_files=len(checked), owned_wall_ns=owned_wall,
                collection_sha256=hash_file(manifest), wait_sha256=hash_file(wait_path), routes=rows)


if __name__ == '__main__':
    collection, prior, wait, source = sys.argv[1:]
    print(json.dumps(verify(Path(collection), Path(prior), Path(wait), source), indent=2, sort_keys=True))
