"""Replay the public Study 13 dispatch evidence without Vulkan or CUDA."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ORACLE = ROOT / 'rt_visibility_oracle.py'
ORACLE_SHA256 = '573de4d69c51402665ed81484ae83c89245ac5628811252c57a6a22a9f26718a'
MANIFEST_SHA256 = '6c56d357b6fcd0c033f7f552d1d0f819733521ca9caa8079629748abca1e8e55'
SCENES = ('spread', 'clustered', 'boundary')
BACKENDS = ('cuda', 'vulkan')
FILES = {'verify.py', 'wire.py'} | {
    f'{scene}.{suffix}' for scene in SCENES for suffix in ('s13', 'replay.json')
} | {
    f'{scene}.{backend}.{suffix}' for scene in SCENES for backend in BACKENDS
    for suffix in ('exit', 'raw', 's13out', 'stderr')
}


def check_bundle(bundle):
    names = {p.name for p in bundle.iterdir()}
    if names != FILES | {'SHA256SUMS'}:
        raise ValueError('bundle inventory differs')
    if any(p.is_symlink() or not p.is_file() for p in bundle.iterdir()):
        raise ValueError('bundle must contain only regular files')
    listing = (bundle / 'SHA256SUMS').read_bytes()
    if sha256(listing).hexdigest() != MANIFEST_SHA256:
        raise ValueError('bundle manifest digest mismatch')
    rows = [line.split('  ', 1) for line in listing.decode('ascii').splitlines()]
    if len(rows) != len(FILES) or {name for _, name in rows} != FILES:
        raise ValueError('manifest inventory differs')
    for digest, name in rows:
        if sha256((bundle / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'digest mismatch: {name}')
    if sha256(ORACLE.read_bytes()).hexdigest() != ORACLE_SHA256:
        raise ValueError('oracle digest mismatch')


def replay(bundle):
    check_bundle(bundle)
    exact_words = exact_canaries = boundary_words = 0
    boundary_lanes = {}
    for scene in SCENES:
        fixture = bundle / f'{scene}.s13'
        command = [sys.executable, '-E', '-s', '-B', str(bundle / 'verify.py'),
                   'check', str(fixture), '--oracle', str(ORACLE)]
        for backend in BACKENDS:
            prefix = bundle / f'{scene}.{backend}'
            if (Path(f'{prefix}.exit').read_bytes() != b'0\n'
                    or Path(f'{prefix}.stderr').read_bytes()):
                raise ValueError(f'{scene}/{backend}: unsuccessful archived dispatch')
            bound = Path(f'{prefix}.s13out')
            if bound.read_bytes() != (b'S13OUT1\0' + sha256(fixture.read_bytes()).digest()
                                      + Path(f'{prefix}.raw').read_bytes()):
                raise ValueError(f'{scene}/{backend}: raw and bound readbacks differ')
            command += ['--readback', backend, str(bound)]
        run = subprocess.run(command, capture_output=True, check=False)
        if run.returncode or run.stderr:
            raise ValueError(f'{scene}: independent verifier failed: {run.stderr!r}')
        expected = (bundle / f'{scene}.replay.json').read_bytes()
        if run.stdout.replace(b'\r\n', b'\n') != expected:
            raise ValueError(f'{scene}: replay receipt differs')
        report = json.loads(expected)
        for backend, result in report['readbacks'].items():
            boundary = scene == 'boundary'
            status = 'diagnostic_only' if boundary else 'pass'
            if result['status'] != status or result['invalid_words']:
                raise ValueError(f'{scene}/{backend}: wrong replay status')
            if boundary:
                boundary_words += result['words_checked']
                boundary_lanes[backend] = [item['lane'] for item in result['examples']]
            else:
                exact_words += result['words_checked']
                exact_canaries += result['words_checked'] - report['active_total']
    return dict(schema='study13-matched-dispatch-replay-v1', bundle_sha256=MANIFEST_SHA256,
                regular_exact_words=exact_words, regular_exact_canaries=exact_canaries,
                boundary_checked_words=boundary_words, boundary_mismatch_lanes=boundary_lanes,
                timing_evidence=False, hardware_attestation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    args = parser.parse_args()
    try:
        payload = replay(args.bundle.resolve())
    except (ValueError, OSError) as error:
        parser.exit(2, f'error: {error}\n')
    print(json.dumps(payload, sort_keys=True))


if __name__ == '__main__':
    main()
