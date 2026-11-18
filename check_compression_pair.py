"""Execute fixed three-way lossless transfer suites; incomplete broader study remains explicit."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from compression_pair_cases import suite_spec

ROOT = Path(__file__).resolve().parent
SOURCES = ('check_compression_pair.py', 'compression_pair_engine.py', 'compression_pair_cases.py',
           'compression_pair_protocol.py', 'compression_native.py', 'compression_integrity.py',
           'native/compression_ndzip_bridge.cpp', 'native/compression_integrity.cu',
           'xbp_native.py', 'xbp_reference.py', 'native/xbp_cpu.cpp', 'native/xbp_decode.cu',
           'build_xbp.sh', 'build_compression_ndzip.sh')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def prerequisites():
    proofs = {}
    for name, binary in (('compression-native-preflight', 'libcompression_ndzip.so'),
                         ('xbp-native-preflight', 'libxbp_cpu.so')):
        path = ROOT / 'results' / name / 'manifest.json'; record = json.loads(path.read_text())
        for source, expected in record['sources'].items():
            assert digest(ROOT / source) == expected, source
        assert digest(ROOT / 'build' / binary) == record['binary_sha256'], binary
        proofs[str(path.relative_to(ROOT))] = digest(path)
    path = ROOT / 'results/compression-integrity-preflight/manifest.json'
    for source, expected in json.loads(path.read_text())['sources'].items():
        assert digest(ROOT / source) == expected, source
    proofs[str(path.relative_to(ROOT))] = digest(path)
    path = ROOT / 'results/compression-ndzip-cli-preflight/manifest.json'
    record = json.loads(path.read_text()); upstream = Path(os.environ['NDZIP_ROOT']).resolve()
    assert record['upstream_commit'] == 'ff4e6702bf0abb86d4aeef8249bd30344dfeef75'
    for source, expected in record['vendor_sources'].items():
        assert digest(upstream / source) == expected, source
    for binary, expected in record['binaries'].items():
        assert digest(upstream / 'build-sm120-module-boost' / binary) == expected, binary
    proofs[str(path.relative_to(ROOT))] = digest(path)
    return {'proofs': proofs, 'vendor_sources': record['vendor_sources'],
            'vendor_binaries': record['binaries'], 'vendor_commit': record['upstream_commit']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('smoke', 'paired', 'over-vram', 'above-probe'), default='smoke')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from compression_pair_engine import run_route, loaded_libraries
    import cupy as cp
    from compression_pair_protocol import WitnessStore
    began = time.perf_counter_ns(); cp.cuda.Device(0).use()
    props = cp.cuda.runtime.getDeviceProperties(0)
    device = {'name': props['name'].decode() if isinstance(props['name'], bytes) else props['name'],
              'total_memory_bytes': int(props['totalGlobalMem']), 'major': int(props['major']),
              'minor': int(props['minor']), 'runtime_api': cp.cuda.runtime.runtimeGetVersion(),
              'driver_api': cp.cuda.runtime.driverGetVersion()}
    context_setup_ns = time.perf_counter_ns() - began
    config = suite_spec(args.suite)
    args.output.mkdir(parents=True, exist_ok=False)
    source_map = {name: digest(ROOT / name) for name in SOURCES}
    dependencies = prerequisites()
    record = {'schema': 'compression-pair-suite-v1', 'suite': args.suite, 'config': config,
              'scope': 'fixed synthetic three-way suites; genuine-data/final study and complete vendor memory audit still pending',
              'sources': source_map, 'dependencies': dependencies, 'device': device,
              'numpy': np.__version__, 'cupy': cp.__version__, 'context_setup_ns': context_setup_ns,
              'cases': [], 'complete': False}
    manifest = args.output / 'manifest.json'; manifest.write_text(json.dumps(record, indent=2))
    store = WitnessStore(args.output)
    for cohort, dtype, profile in config['datasets']:
        selected = list(config['routes'])
        if cohort == 'b':
            selected.reverse()
        for route in selected:
            index = len(record['cases'])
            case = run_route(args.output, store, cohort, dtype, profile, route, config,
                             device['total_memory_bytes'])
            case['index'] = index
            began = time.perf_counter_ns()
            gc.collect(); cp.get_default_pinned_memory_pool().free_all_blocks(); cp.get_default_memory_pool().free_all_blocks()
            case['allocator_cleanup_ns'] = time.perf_counter_ns() - began
            record['cases'].append(case)
            assert source_map == {name: digest(ROOT / name) for name in SOURCES}
            manifest.write_text(json.dumps(record, indent=2))
            print('THREE_WAY_PAIR_ROUTE_COMPLETE', args.suite, index + 1, cohort, dtype, profile, route,
                  [trial['host_ns'] for trial in case['trials']], flush=True)
    assert dependencies == prerequisites(), 'Dependency bytes changed during suite'
    record['loaded_libraries'] = loaded_libraries()
    record['complete'] = True
    manifest.write_text(json.dumps(record, indent=2))
    print('THREE_WAY_PAIR_SUITE_PROCESS_COMPLETE', args.suite, len(record['cases']), flush=True)
