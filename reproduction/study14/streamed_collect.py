"""Shared-host streaming comparison over the committed 20 GiB dataset."""

import hashlib
import json
from pathlib import Path
import sys
from time import get_clock_info, perf_counter_ns

import cupy as cp
import numpy as np

from check_compression_reference_pair import SOURCES, digest, prerequisites
from compression_integrity import DeviceIntegrity
from compression_pair_cases import input_words
from compression_pair_protocol import WitnessStore
from compression_reference_engine import Slot, loaded_libraries, pinned


ROOT = Path.cwd()
ORIGINAL = ROOT / 'results/compression-reference-stock-above-repeat'
COLLECTION_PIN = '44cf4bb43382e2c631836a59b2be9d92cff95c3b1fcd2722ff08ece4289ff756'
PHYSICAL_VRAM = 17_103_323_136
FRAME_BYTES = 64 << 20
DTYPE = np.dtype('<f4')
WORDS_CHUNK = 1 << 20


def expect(ok, why):
    if not ok:
        raise ValueError(why)


def capture_case(route, specs, expected, store):
    integrity = DeviceIntegrity(4096)
    began = perf_counter_ns()
    slots = []
    for _ in range(2):
        slots.append(Slot(route, specs[:1], DTYPE, integrity))
    inputs = [pinned(specs[0]['shape'], DTYPE) for _ in slots]
    events = [[cp.cuda.Event(disable_timing=True) for _ in range(4)] for _ in slots]
    setup_ns = perf_counter_ns() - began

    def prepare(index, target):
        spec, witness = specs[index], expected[index]['witness']
        flat = target.reshape(-1)
        for begin in range(0, flat.size, WORDS_CHUNK):
            end = min(begin + WORDS_CHUNK, flat.size)
            flat[begin:end] = input_words('smooth', DTYPE, spec['offset_words'] + begin, end - begin, 71)
        expect(hashlib.sha256(target).hexdigest() == spec['input_sha256'],
               f'Frame {index} input differs from its pinned identity')
        old = ORIGINAL / 'witnesses' / (witness['array_sha256'] + '.npy')
        expect(not old.is_symlink() and digest(old) == witness['sha256'], f'Frame {index} CPU-verified reference differs')
        return np.load(old, allow_pickle=False)

    groups, first_use_ns = [], None
    try:
        for begin in range(0, len(specs), 2):
            indices = list(range(begin, min(begin + 2, len(specs))))
            group_start = perf_counter_ns()
            truth = [prepare(i, inputs[j]) for j, i in enumerate(indices)]
            prep_end = perf_counter_ns()
            if first_use_ns is None:
                for j, i in enumerate(indices):
                    slots[j].enqueue(0, specs[i], inputs[j], None, events[j])
                    slots[j].finish([truth[j]])
                first_use_ns = perf_counter_ns() - prep_end
            service_start = perf_counter_ns()
            for j, i in enumerate(indices):
                slots[j].enqueue(0, specs[i], inputs[j], None, events[j])
            checked = [slots[j].finish([truth[j]]) for j in range(len(indices))]
            service_ns = perf_counter_ns() - service_start
            results = [dict(frame=i, input_sha256=specs[i]['input_sha256'],
                            wire_bytes=row['wire_bytes'], encode_validate_ns=row['cpu_encode_validate_ns'],
                            witness=store.put(table))
                       for i, (row, table) in zip(indices, checked, strict=True)]
            groups.append(dict(prepare_ns=prep_end - group_start, service_ns=service_ns,
                               frames=results, group_wall_ns=perf_counter_ns() - group_start))
            if not begin % 64:
                print(route['codec'], begin + len(indices), flush=True)
    finally:
        for slot in slots:
            slot.close()
    return dict(route=route, setup_ns=setup_ns, first_use_ns=first_use_ns,
                wall_ns=perf_counter_ns() - began, groups=groups,
                requested_pinned_source_bytes=sum(a.nbytes for a in inputs),
                slot_requests=[s.memory() for s in slots])


def main():
    output, mode = Path(sys.argv[1]), sys.argv[2]
    expect(mode in ('smoke', 'full'), 'Unknown fixed scope')
    expect(get_clock_info('perf_counter').implementation == 'clock_gettime(CLOCK_MONOTONIC)',
           'Expected a common host monotonic clock')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    expect(available >= 2 << 30, 'Insufficient 2 GiB host-memory margin')
    manifest = ORIGINAL / 'manifest.json'
    expect(digest(manifest) == COLLECTION_PIN, 'Original collection pin differs')
    old = json.loads(manifest.read_bytes())
    expect(prerequisites() == old['dependencies'] and all(digest(ROOT / s) == old['sources'][s] for s in SOURCES),
           'Native dependency/source closure differs')
    count = 4 if mode == 'smoke' else 320
    specs = old['cases'][0]['frames'][:count]
    expected = old['cases'][0]['trials'][0]['observations'][:count]
    expect(len(specs) == len(expected) == count and all(
        s['bytes'] == FRAME_BYTES and s['offset_words'] * DTYPE.itemsize == i * FRAME_BYTES
        for i, s in enumerate(specs)), 'Source stream not unique/contiguous')
    expect(mode == 'smoke' or count * FRAME_BYTES > PHYSICAL_VRAM, 'Full input must exceed physical VRAM')
    pci = cp.cuda.runtime.deviceGetPCIBusId(0)
    if isinstance(pci, bytes):
        pci = pci.decode('ascii')
    expect(cp.cuda.runtime.getDeviceCount() == 1 and pci in ('0000:01:00.0', '00000000:01:00.0'),
           'Expected the approved physical target')
    output.mkdir()
    store = WitnessStore(output)
    result = dict(schema='study14-bounded-stream-v1', mode=mode, original_collection_sha256=COLLECTION_PIN,
                  source_sha256=digest(Path(__file__)), logical_bytes=count * FRAME_BYTES,
                  framebuffer_physical_bytes=PHYSICAL_VRAM, available_host_bytes=available,
                  frame_bytes=FRAME_BYTES, complete=False, cases=[])

    def write():
        temp = output / 'manifest.tmp'
        temp.write_text(json.dumps(result, sort_keys=True))
        temp.replace(output / 'manifest.json')

    write()
    for item in old['cases']:
        result['cases'].append(capture_case(item['route'], specs, expected, store))
        write()
    result['loaded_libraries'] = loaded_libraries()
    result['complete'] = True
    write()


if __name__ == '__main__':
    main()
