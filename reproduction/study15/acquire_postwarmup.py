"""Study 15 prospective pilot with fail-closed workload admission."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
from statistics import median_low
import sys
import time
import uuid

import batch_core
import native_backends as native
import request_plan
import verify_energy_pair as verifier

CALIBRATION_READS = 9
ORIGIN_LEAD_NS = 20_000_000
POLL_NS = 10_000
SPIN_NS = 500_000
PHASES = ('pre-idle-start', 'active-start', 'active-end', 'post-idle-end')
PROTOCOL_V1_OFFSETS = [0, 4_000_000_000, 8_000_000_000, 12_000_000_000]
PROTOCOL_V1_WIDTH_CAP_NS = 16_000_000
PCI = re.compile(r'(?:[0-9a-fA-F]{4}|[0-9a-fA-F]{8}):[0-9a-fA-F]{2}:[01][0-9a-fA-F]\.[0-7]')


@dataclass(frozen=True)
class Inputs:
    plan: Path
    fixture: Path
    plan_sha256: str
    out: Path
    library: Path | None = None
    library_sha256: str | None = None
    kernel_source: Path | None = None
    activity_note: str = ''
    boundary_receipt: Path | None = None
    boundary_receipt_sha256: str | None = None
    gpu_uuid: str | None = None
    pci_bus_id: str | None = None


class HostClock:
    now = staticmethod(time.perf_counter_ns)

    def wait_until(self, target):
        while (remaining := target - self.now()) > 0:
            if remaining > SPIN_NS:
                time.sleep((remaining - SPIN_NS) / 1_000_000_000)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    raw = (json.dumps(data, sort_keys=True, indent=2) + '\n').encode('utf-8')
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(raw)
    tmp.replace(path)
    return sha(raw)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def same_uuid(observed, expected):
    return isinstance(observed, str) and observed.casefold() == expected.casefold()


def pci_identity(value):
    if isinstance(value, str) and PCI.fullmatch(value):
        domain, bus, device = value.lower().split(':')
        return int(domain, 16), bus, device
    return None


def check_boundary_receipt(raw, args, plan, evidence):
    """Consume an independently pinned replay result, not an acquisition self-report."""
    require(sha(raw) == args.boundary_receipt_sha256, 'boundary receipt pin differs')
    receipt = verifier.read_json(raw)
    require(isinstance(receipt, dict) and receipt.get('schema') == 'study15-boundary-cpu-replay-v1',
            'invalid boundary replay receipt schema')
    native_scope = evidence == 'measured'
    require((receipt.get('evidence_kind'), receipt.get('verdict')) == (
        ('native-boundary-feasibility', 'native-boundary-feasibility-pass') if native_scope else
        ('synthetic-software-test', 'software-test-pass')), 'boundary receipt evidence or verdict is not admissible')
    require(type(receipt.get('process_exit')) is int and receipt['process_exit'] == 0 and
            type(receipt.get('boundary_row_count')) is int and receipt['boundary_row_count'] == 8 and
            receipt.get('schedule') == receipt.get('workload') == 'not-run' and
            receipt.get('acquisition_stage') == 'boundary' and
            receipt.get('acquisition_outcome') == receipt.get('trace_outcome') == 'boundary-feasibility-pass' and
            receipt.get('cleanup_outcome') == 'ok' and receipt.get('failures') == [],
            'boundary receipt does not certify a complete passing acquisition and cleanup')
    pilot = plan.get('prospective_pilot')
    require(isinstance(pilot, dict) and pilot.get('family') == 'study15-prospective-pilot-v1' and
            pilot.get('lane') == 'exact-output-pair' and plan['window_offsets_ns'] == PROTOCOL_V1_OFFSETS and
            plan['boundary_slack_ns'] == PROTOCOL_V1_WIDTH_CAP_NS,
            'workload plan is not compatible with the prospective boundary protocol')
    index = pilot.get('pair_index')
    require(type(index) is int and index > 0 and type(receipt.get('pair_index')) is int and
            receipt['pair_index'] == index, 'boundary receipt pair index differs from workload plan')
    protocol = pilot.get('feasibility_sha256')
    require(verifier.sha256(protocol) and receipt.get('protocol_sha256') == protocol,
            'boundary receipt protocol digest differs from workload plan')
    require(isinstance(args.gpu_uuid, str) and request_plan.GPU_UUID.fullmatch(args.gpu_uuid) and
            same_uuid(plan['gpu_uuid'], args.gpu_uuid) and same_uuid(receipt.get('gpu_uuid'), args.gpu_uuid),
            'boundary receipt or workload GPU UUID differs from independent pin')
    require(pci_identity(args.pci_bus_id) is not None and
            pci_identity(receipt.get('pci_bus_id')) == pci_identity(args.pci_bus_id),
            'boundary receipt PCI address differs from independent pin')
    require(not native_scope or verifier.sha256(receipt.get('library_image_sha256')),
            'native boundary receipt lacks NVML library identity')
    return {key: receipt[key] for key in ('verdict', 'protocol_sha256', 'pair_index', 'gpu_uuid', 'pci_bus_id')}


def freeze_inputs(args, evidence):
    raw_plan = args.plan.read_bytes()
    (args.out / 'plan.json').write_bytes(raw_plan)
    require(sha(raw_plan) == args.plan_sha256, 'independent plan pin differs')
    plan = verifier.read_json(raw_plan)
    require(plan['schema'] == 'study15-request-plan-v1' and plan['evidence_kind'] == evidence,
            'plan schema/evidence does not match this acquisition path')
    gpu = plan['gpu_uuid']
    require(verifier.name(gpu) and (evidence != 'measured' or request_plan.GPU_UUID.fullmatch(gpu)),
            'measured plan requires a physical GPU UUID')
    require(plan['arms'] in (list(request_plan.ARMS), list(reversed(request_plan.ARMS))), 'invalid arm order')
    workload = plan['workload']
    width = request_plan.REQUEST_BYTES
    require(workload['id'] == request_plan.WORKLOAD and workload['input_bytes_per_request'] == width
            and workload['output_bytes_per_request'] == 8
            and workload['input_order'] == 'schedule-order, A[4096] then B[4096], uint16 little-endian',
            'incorrect workload geometry or format')
    offsets, slack, arrive_slack, cutoff, wait = (plan[k] for k in
        ('window_offsets_ns', 'boundary_slack_ns', 'arrival_slack_ns', 'cutoff_offset_ns', 'batch_wait_ns'))
    require(isinstance(offsets, list) and len(offsets) == 4 and all(map(verifier.uint, offsets))
            and offsets[0] == 0 and all(a < b for a, b in zip(offsets, offsets[1:]))
            and all(map(verifier.uint, (slack, arrive_slack, cutoff, wait))) and wait > 0
            and offsets[1] + slack < cutoff < offsets[2] - slack
            and min(b - a for a, b in zip(offsets, offsets[1:])) > 2 * slack, 'invalid phase plan')
    schedule = plan['schedule']
    require(isinstance(schedule, list) and 1 <= len(schedule) <= 4096, 'invalid request count')
    ids = [r['id'] for r in schedule]
    times = [r['arrival_offset_ns'] for r in schedule]
    require(all(map(verifier.name, ids)) and len(set(ids)) == len(ids) and all(map(verifier.uint, times))
            and all(a <= b for a, b in zip(times, times[1:]))
            and offsets[1] + slack + arrive_slack < times[0] <= times[-1] <= cutoff - arrive_slack,
            'invalid request order or arrivals outside guarded active window')
    truth = plan['truth']
    require(isinstance(truth, dict) and set(truth) == set(ids) and all(map(verifier.sha256, truth.values()))
            and len(set(truth.values())) == len(ids), 'invalid exact-output truth bindings')
    raw_inputs = args.fixture.read_bytes()
    (args.out / 'inputs.bin').write_bytes(raw_inputs)
    require(len(raw_inputs) == width * len(ids) and sha(raw_inputs) == workload['inputs_sha256'],
            'input pin/byte count differs')
    records = {key: raw_inputs[i*width:(i+1)*width] for i, key in enumerate(ids)}
    return plan, records


def sample(counter, clock, target, phase, gpu, epoch):
    row = counter.energy(clock.now)
    row.update(target_ns=target, phase=phase, gpu_uuid=gpu, epoch=epoch)
    return row


def check_nvml(row):
    require(row['nvml_status'] == 0, f"NVML energy returned status {row['nvml_status']} in {row['phase']}")


def checked_read_width(row):
    require(type(row.get('nvml_status')) is int and row['nvml_status'] == 0 and
            row.get('status') == 'success' and verifier.uint(row.get('mJ')) and row['mJ'] < 2**64,
            f"invalid successful NVML counter read in {row['phase']}")
    start, end = row.get('start_ns'), row.get('end_ns')
    require(verifier.uint(start) and verifier.uint(end) and start <= end,
            f"invalid NVML counter interval in {row['phase']}")
    return end - start


def calibrate(reads, counter, clock, gpu, epoch, slack, timing, lead_key):
    widths = []
    for _ in range(CALIBRATION_READS):
        reads.append(sample(counter, clock, None, 'calibration', gpu, epoch))
        check_nvml(reads[-1])
        widths.append(checked_read_width(reads[-1]))
        if len(reads) > 1:
            require(reads[-2]['end_ns'] <= reads[-1]['start_ns'] and reads[-2]['mJ'] <= reads[-1]['mJ'],
                    'NVML calibration intervals or counter values are not chronological')
    lead = median_low(widths) // 2
    timing[lead_key] = lead
    require(all(width <= PROTOCOL_V1_WIDTH_CAP_NS for width in widths),
            'NVML calibration read width exceeds the prospective per-read cap')
    require(lead <= slack, 'counter calibration wider than declared boundary aperture')
    return lead


def collect_arm(run, plan, records, clock, counter, cuda, lead):
    origin = run['origin_ns'] = clock.now() + ORIGIN_LEAD_NS + lead
    schedule = [(origin + r['arrival_offset_ns'], r['id']) for r in plan['schedule']]
    times = [origin + t for t in plan['window_offsets_ns']]
    cutoff = origin + plan['cutoff_offset_ns']
    batcher = batch_core.Batcher(run['arm'], wait_ns=plan['batch_wait_ns'])
    batch = batch_row = None
    admitted = 0
    idx = 0

    def boundary(i):
        clock.wait_until(times[i] - lead)
        row = sample(counter, clock, times[i], PHASES[i], run['gpu_uuid'], run['epoch'])
        run['boundary_sample_indices'].append(len(run['samples']))
        run['samples'].append(row)
        check_nvml(row)
        return row

    # Gate on the actual unshifted intervals, after appending each raw sample.
    # This is before the first scheduled arrival or submit for this arm.
    for i in (0, 1):
        row = boundary(i)
        checked_read_width(row)
        target, slack = times[i], plan['boundary_slack_ns']
        require(target - slack <= row['start_ns'] <= target <= row['end_ns'] <= target + slack,
                f"opening counter boundary {row['phase']} is not bracketed within the read aperture")
    active_end_call = times[2] - lead
    while (now := clock.now()) < active_end_call:
        while idx < len(schedule) and schedule[idx][0] <= now:
            planned, key = schedule[idx]
            stamp = clock.now()
            arrival = {'id': key, 'planned_ns': planned, 'arrival_ns': stamp,
                       'status': 'queued' if stamp < cutoff else 'cutoff-miss'}
            run['arrivals'].append(arrival)
            if stamp < cutoff:
                batcher.arrive(key, stamp)
                admitted += 1
            idx += 1
        if batch is not None:
            try:
                outputs = cuda.poll()
                if outputs is not None:
                    stamp = clock.now()
                    run['events'].extend(batcher.complete(batch, stamp, outputs))
                    batch_row.update(delivery_ns=stamp, status='late' if stamp > cutoff else 'delivered')
                    batch = batch_row = None
            except Exception as error:
                fail_batch(run, batcher, batch, batch_row, clock, error)
                raise
        now = clock.now()
        if now < cutoff and batch is None and batcher.ready_at_ns() is not None:
            batch = batcher.dispatch(now)
            if batch is not None:
                batch_row = {'sequence': batch.sequence, 'ids': list(batch.ids), 'reason': batch.reason,
                             'dispatch_ns': batch.dispatch_ns, 'status': 'pending'}
                run['batches'].append(batch_row)
                try:
                    cuda.submit(tuple(records[key] for key in batch.ids))
                except Exception as error:
                    fail_batch(run, batcher, batch, batch_row, clock, error)
                    raise
        if idx == len(schedule) and batch is None and batcher.ready_at_ns() is None:
            break
        deadlines = [active_end_call]
        if batch is not None:
            deadlines.append(now + POLL_NS)
        if idx < len(schedule):
            deadlines.append(schedule[idx][0])
        if now < cutoff:
            deadlines.append(cutoff)
            ready = batcher.ready_at_ns()
            if ready is not None:
                deadlines.append(ready)
        target = min(deadlines)
        # The owner may have become runnable after calculating a deadline.
        if target > clock.now():
            clock.wait_until(target)
    boundary(2)
    require(batch is None, 'CUDA work still in flight at active-end read; not a post-idle window')
    boundary(3)
    require(idx == admitted == len(schedule) == len(run['events']), 'missing arrival/delivery at fixed cutoff')
    run['status'] = 'complete'  # All transitions acquired; eligibility is decided separately.


def fail_batch(run, batcher, batch, row, clock, error):
    stamp = clock.now()
    detail = f'{type(error).__name__}: {error}'
    run['events'].extend(batcher.fail(batch, stamp, detail))
    row.update(status='backend-error', observed_ns=stamp, error=detail)


def timing_diagnostics(plan, runs):
    schedule = [r['arrival_offset_ns'] for r in plan['schedule']]
    planned = [b-a for a, b in zip(schedule, schedule[1:])]
    diagnostics = {'planned_interarrival_ns': planned, 'arms': []}
    all_gaps = []
    for run in runs:
        stamps = [r['arrival_ns'] for r in run['arrivals']]
        gaps = [b-a for a, b in zip(stamps, stamps[1:])]
        all_gaps.append(gaps)
        phase_bounds = []
        for left, right in zip(run['samples'], run['samples'][1:]):
            # A malformed rejected read must not prevent the raw trace checkpoint.
            if not all(verifier.uint(row.get(key)) for row in (left, right)
                       for key in ('start_ns', 'end_ns')):
                continue
            phase_bounds.append({'from': left['phase'], 'to': right['phase'],
                                 'planned_ns': right['target_ns'] - left['target_ns'],
                                 'min_ns': right['start_ns'] - left['end_ns'],
                                 'max_ns': right['end_ns'] - left['start_ns']})
        diagnostics['arms'].append({'arm': run['arm'], 'interarrival_ns': gaps,
            'max_gap_error_ns': max((abs(a-b) for a,b in zip(gaps, planned)), default=None),
            'exposure_bounds_ns': phase_bounds})
    if len(all_gaps) == 2:
        diagnostics['max_paired_gap_difference_ns'] = max(
            (abs(a-b) for a,b in zip(*all_gaps)), default=None)
    return diagnostics


def pilot_timing_gate(plan, runs, calibration):
    reads = [*calibration, *(s for r in runs for s in (*r['postwarmup_calibration'], *r['samples']))]
    verifier.need(all(a['mJ'] <= b['mJ'] for a, b in zip(reads, reads[1:])),
                  'observed NVML counter decrease across calibration/measurement reads')
    diag = timing_diagnostics(plan, runs)
    tol = plan['arrival_slack_ns']
    verifier.need(all((arm['max_gap_error_ns'] or 0) <= tol for arm in diag['arms']) and
                  (diag['max_paired_gap_difference_ns'] or 0) <= tol,
                  'realized interarrival gaps differ from plan or between arms')
    slack = plan['boundary_slack_ns']
    # Also bound the exposure compatible with the raw host-call intervals.
    for arm in diag['arms']:
        for phase in arm['exposure_bounds_ns']:
            expected = phase['planned_ns']
            verifier.need(expected - 2*slack <= phase['min_ns'] <= expected <= phase['max_ns'] <=
                          expected + 2*slack, 'phase exposure outside the boundary plan')


def run_attempt(args, *, injected=None):
    """Run one attempt (never retry or resume). Injection is software-only."""
    evidence = 'measured' if injected is None else 'synthetic-software-test'
    args.out.mkdir(exist_ok=False)
    runs, originals = [], {}
    envelope = {'schema': 'study15-postwarmup-collection-attempt-v1', 'evidence_kind': evidence,
                'status': 'incomplete', 'stage': 'plan-and-input', 'errors': [], 'calibration': [],
                'plan_pin_sha256': args.plan_sha256, 'activity_note': args.activity_note,
                'boundary_receipt_pin_sha256': args.boundary_receipt_sha256,
                'started_utc_ns': time.time_ns(), 'pid': os.getpid(), 'platform': platform.platform(),
                'python': sys.version, 'python_executable': sys.executable,
                'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
                'timing': {'clock': 'perf_counter_ns' if injected is None else 'injected', 'poll_ns': POLL_NS,
                           'spin_ns': SPIN_NS, 'calibration_reads': CALIBRATION_READS,
                           'origin_lead_ns': ORIGIN_LEAD_NS}}
    counter = cuda = result = None

    def checkpoint():
        envelope['runs_sha256'] = write_json(args.out / 'runs.json', runs)
        write_json(args.out / 'attempt.json', envelope)

    def archive(path, name, pin=None):
        original = Path(path).resolve(strict=True)
        data = original.read_bytes()
        digest = sha(data)
        if pin is not None:
            require(digest == pin, f'{name} pin differs')
        (args.out / name).write_bytes(data)
        originals[name] = (original, digest)
        envelope.setdefault('provenance', {})[name] = {'original_path': str(original), 'sha256': digest}
        return original

    def error(stage, exc):
        envelope['errors'].append({'stage': stage, 'type': type(exc).__name__, 'message': str(exc),
                                   'notes': getattr(exc, '__notes__', [])})

    checkpoint()
    try:
        plan, records = freeze_inputs(args, evidence)
        envelope.update(gpu_uuid=plan['gpu_uuid'], input_sha256=plan['workload']['inputs_sha256'],
                        requests=len(records), planned_arms=plan['arms'])
        envelope['stage'] = 'boundary-qualification'
        require(args.boundary_receipt is not None and verifier.sha256(args.boundary_receipt_sha256),
                'an independently pinned boundary replay receipt is required')
        receipt_name = 'boundary-replay.json'
        archive(args.boundary_receipt, receipt_name, args.boundary_receipt_sha256)
        admission = check_boundary_receipt((args.out / receipt_name).read_bytes(), args, plan, evidence)
        envelope['boundary_admission'] = dict(admission, receipt_sha256=args.boundary_receipt_sha256)
        epoch = ('nvml-session:' if injected is None else 'synthetic-session:') + uuid.uuid4().hex
        envelope['epoch'] = {'label': epoch, 'scope': 'one process/session label, not a hardware reset identifier'}
        envelope['stage'] = 'runtime-provenance'
        for module in (sys.modules[__name__], batch_core, request_plan, verifier, native):
            source = Path(module.__file__)
            archive(source, source.name)
        if injected is None:
            require(args.activity_note.strip(), 'record the device activity/ownership note before acquisition')
            require(args.library is not None and verifier.sha256(args.library_sha256) and args.kernel_source is not None,
                    'measured path requires a pinned CUDA library and the kernel source')
            library = archive(args.library, 'kernel-binary' + args.library.suffix, args.library_sha256)
            archive(args.kernel_source, 'kernel-source.cu')
            runtime = (HostClock(), native.NativeNvml, lambda gpu: native.CudaDot(gpu, library))
        else:
            runtime = injected
        clock, open_counter, open_cuda = runtime
        gpu = plan['gpu_uuid']
        envelope['stage'] = 'nvml-identity-and-calibration'
        checkpoint()
        counter = open_counter(gpu)
        envelope['nvml_before'] = nvml_before = counter.snapshot()
        require(same_uuid(nvml_before['gpu_uuid'], gpu), 'NVML UUID does not match plan')
        reads = envelope['calibration']
        envelope['timing']['calibration_max_width_ns'] = PROTOCOL_V1_WIDTH_CAP_NS
        calibrate(reads, counter, clock, gpu, epoch, plan['boundary_slack_ns'],
                  envelope['timing'], 'boundary_lead_ns')
        envelope['stage'] = 'cuda-identity-and-setup'
        checkpoint()
        cuda = open_cuda(gpu)
        envelope['cuda'] = dict(cuda.snapshot())
        cuda_pci = envelope['cuda'].get('pci_bus_id')
        if isinstance(cuda_pci, bytes):
            cuda_pci = envelope['cuda']['pci_bus_id'] = cuda_pci.decode('ascii', errors='backslashreplace')
        require(same_uuid(envelope['cuda']['gpu_uuid'], gpu), 'CUDA UUID does not match plan')
        require(pci_identity(cuda_pci) == pci_identity(args.pci_bus_id), 'CUDA PCI address does not match pin')
        schedule = plan['schedule']
        warm_keys = [row['id'] for row in schedule[:8]]
        for arm in plan['arms']:
            envelope['stage'] = f'warmup:{arm}'
            run = {'arm': arm, 'gpu_uuid': gpu, 'epoch': epoch, 'origin_ns': None, 'status': 'incomplete',
                   'arrivals': [], 'events': [], 'batches': [], 'samples': [], 'boundary_sample_indices': []}
            runs.append(run)
            checkpoint()
            warm_start = clock.now()
            warm = cuda.warmup(tuple(records[key] for key in warm_keys))
            warm_end = clock.now()
            run['warmup'] = {'ids': warm_keys, 'elapsed_ns': warm_end - warm_start,
                             'output_hex': [data.hex() for data in warm]}
            require([sha(data) for data in warm] == [plan['truth'][key] for key in warm_keys],
                    'untimed warmup differs from independent output truth')
            envelope['stage'] = f'postwarmup-calibration:{arm}'
            run['postwarmup_calibration'] = []
            lead = calibrate(run['postwarmup_calibration'], counter, clock, gpu, epoch,
                             plan['boundary_slack_ns'], run, 'boundary_lead_ns')
            envelope['stage'] = f'collect:{arm}'
            checkpoint()
            collect_arm(run, plan, records, clock, counter, cuda, lead)
            checkpoint()
        envelope['stage'] = 'post-acquisition-provenance'
        envelope['nvml_after'] = nvml_after = counter.snapshot()
        for key in ('gpu_uuid', 'driver_version', 'nvml_version'):
            require(nvml_before[key] == nvml_after[key], f'NVML post-acquisition identity changed: {key}')
        for name, (source, digest) in originals.items():
            require(sha(source.read_bytes()) == sha((args.out / name).read_bytes()) == digest,
                    f'provenance bytes changed during attempt: {name}')
        require(sha(args.plan.read_bytes()) == sha((args.out / 'plan.json').read_bytes()) == args.plan_sha256
                and sha(args.fixture.read_bytes()) == sha((args.out / 'inputs.bin').read_bytes()) == envelope['input_sha256'],
                'plan/input bytes changed during attempt')
        envelope['stage'] = 'replay'
        metrics = verifier.replay_pair(plan, runs)
        pilot_timing_gate(plan, runs, reads)
        result = metrics
    except verifier.TraceError as exc:
        # read_json may raise a TraceError as well; only completed acquisition can be ineligible.
        if envelope['stage'] == 'replay':
            envelope['status'] = 'ineligible'
        error(envelope['stage'], exc)
    except (Exception, KeyboardInterrupt) as exc:
        error(envelope['stage'], exc)
    finally:
        for run in runs:
            delivered = {row['id'] for row in run['events'] if row['status'] == 'delivered'}
            run['unserved_ids'] = [row['id'] for row in plan['schedule'] if row['id'] not in delivered]
            run['late_ids'] = [row['id'] for row in run['events'] if row['status'] == 'delivered'
                               and row['delivery_ns'] > run['origin_ns'] + plan['cutoff_offset_ns']]
        envelope['timing_diagnostics'] = timing_diagnostics(plan, runs) if runs else None
        # Preserve the trace before a failed async CUDA context can block teardown.
        checkpoint()
        for name, resource in (('cuda', cuda), ('nvml', counter)):
            if resource is not None:
                try:
                    resource.close()
                except (Exception, KeyboardInterrupt) as exc:
                    error(f'cleanup:{name}', exc)
                    envelope['status'] = 'incomplete'
                    checkpoint()
        if result is not None and not envelope['errors']:
            result.update(plan_sha256=args.plan_sha256, runs_sha256=envelope['runs_sha256'],
                          verifier_sha256=originals[Path(verifier.__file__).name][1])
            envelope['replay_sha256'] = write_json(args.out / 'replay.json', result)
            envelope['status'] = 'eligible-pilot'
        envelope['ended_utc_ns'] = time.time_ns()
        envelope['exit_code_intent'] = 0 if envelope['status'] == 'eligible-pilot' else 1
        checkpoint()
    return envelope['status']


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', required=True, type=Path, help='must not already exist')
    parser.add_argument('--boundary-receipt', required=True, type=Path, help='accepted independent boundary replay JSON')
    parser.add_argument('--boundary-receipt-sha256', required=True, help='out-of-band pin of the exact receipt bytes')
    parser.add_argument('--gpu-uuid', required=True, help='independent physical GPU UUID')
    parser.add_argument('--pci-bus-id', required=True, help='independent full PCI address')
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--library-sha256', required=True)
    parser.add_argument('--kernel-source', required=True, type=Path)
    parser.add_argument('--activity-note', required=True, help='operator observation of device ownership and other activity')
    a = parser.parse_args()
    try:
        status = run_attempt(Inputs(a.plan, a.inputs, a.plan_sha256, a.out, a.library,
                                    a.library_sha256, a.kernel_source, a.activity_note,
                                    a.boundary_receipt, a.boundary_receipt_sha256, a.gpu_uuid, a.pci_bus_id))
    except OSError as error:
        parser.exit(1, f'attempt storage failed: {error}\n')
    print(json.dumps({'attempt': str(a.out), 'status': status}, sort_keys=True))
    return 0 if status == 'eligible-pilot' else 1


if __name__ == '__main__':
    sys.exit(main())
