"""Freeze Study 15 inputs, arrivals and independent output hashes before acquisition."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

N = 4096
REQUEST_BYTES = N * 4
ARMS = ('singleton', 'batch8')
WORKLOAD = 'u16-dot4096-u64le-v1'
GPU_UUID = re.compile(r'GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}')


def build_plan(gpu_uuid, *, count=1024, seed=15, pattern='steady', reverse=False, measured=False):
    if not isinstance(gpu_uuid, str) or not gpu_uuid or (measured and not GPU_UUID.fullmatch(gpu_uuid)):
        raise ValueError('a measured plan requires a physical GPU UUID')
    if type(count) is not int or not 1 <= count <= 4096:
        raise ValueError('count must be 1..4096')
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('seed must be a uint64')
    if pattern not in ('steady', 'irregular'):
        raise ValueError('unknown arrival pattern')
    pre, lead = 1_000_000_000, 30_000_000
    active = lead + count * 250_000 + 200_000_000
    end = pre + active
    schedule, truth, records, unique = [], {}, [], set()
    for i in range(count):
        key = f'r{i:06d}'
        rel = i * 250_000 if pattern == 'steady' else i // 8 * 2_000_000 + i % 8 * 100_000
        schedule.append({'id': key, 'arrival_offset_ns': pre + lead + rel})
        record = hashlib.shake_256(b'Study15/u16-dot/v1\0' + struct.pack('<QI', seed, i)).digest(REQUEST_BYTES)
        words = struct.unpack(f'<{2*N}H', record)
        result = sum(a * b for a, b in zip(words[:N], words[N:]))
        if result in unique:
            raise ValueError('non-unique output; choose another seed before freezing')
        unique.add(result)
        records.append(record)
        truth[key] = hashlib.sha256(struct.pack('<Q', result)).hexdigest()
    inputs = b''.join(records)
    return {
        'schema': 'study15-request-plan-v1',
        'evidence_kind': 'measured' if measured else 'synthetic-software-test',
        'gpu_uuid': gpu_uuid,
        'arms': list(reversed(ARMS)) if reverse else list(ARMS),
        'window_offsets_ns': [0, pre, end, end + pre],
        'boundary_slack_ns': 2_000_000,
        'arrival_slack_ns': 40_000,
        'cutoff_offset_ns': end - 50_000_000,
        'batch_wait_ns': 2_000_000,
        'arrival_pattern': pattern,
        'schedule': schedule,
        'truth': truth,
        'workload': {'id': WORKLOAD, 'inputs_sha256': hashlib.sha256(inputs).hexdigest(),
                     'input_bytes_per_request': REQUEST_BYTES, 'output_bytes_per_request': 8,
                     'input_order': 'schedule-order, A[4096] then B[4096], uint16 little-endian',
                     'seed': seed},
    }, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outdir', type=Path, help='must not already exist')
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--count', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=15)
    parser.add_argument('--pattern', choices=['steady', 'irregular'], default='steady')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--measured', action='store_true', help='prepare a plan for real acquisition; does not measure')
    args = parser.parse_args()
    try:
        plan, inputs = build_plan(args.gpu_uuid, count=args.count, seed=args.seed,
                                  pattern=args.pattern, reverse=args.reverse, measured=args.measured)
        raw = (json.dumps(plan, sort_keys=True, indent=2) + '\n').encode('ascii')
        args.outdir.mkdir(exist_ok=False)
        (args.outdir / 'inputs.bin').write_bytes(inputs)
        (args.outdir / 'plan.json').write_bytes(raw)
    except (ValueError, OSError) as error:
        parser.exit(1, f'plan not written completely: {error}\n')
    print(json.dumps({'plan_sha256': hashlib.sha256(raw).hexdigest(),
                      'inputs_sha256': plan['workload']['inputs_sha256'], 'requests': args.count}, sort_keys=True))


if __name__ == '__main__':
    main()
