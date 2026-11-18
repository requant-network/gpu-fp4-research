"""Independently verify padded visibility readbacks against serialized FP32 inputs."""
import argparse
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
import json
from pathlib import Path
from struct import Struct
import sys

from wire import POISON, WORD, decode

RESULT_HEADER = Struct('<8s32s')
RESULT_MAGIC = b'S13OUT1\0'
BIN_WIDTH = 16
MAX_BOX_BINS = 64


def load_oracle(path):
    """Load the trusted external oracle from the same source snapshot we hash."""
    path = Path(path).resolve()
    source = path.read_bytes()
    digest = sha256(source).hexdigest()
    name = f'_study13_oracle_{digest}'

    class Loader(SourceFileLoader):
        def get_code(self, fullname):
            return self.source_to_code(source, str(path))

    spec = spec_from_file_location(name, path, loader=Loader(name, str(path)))
    module = module_from_spec(spec)
    prior = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if prior is None:
            del sys.modules[name]
        else:
            sys.modules[name] = prior
    return module.first_hit, digest


def _box(points):
    xyz = tuple(zip(*points))
    return tuple(min(c) for c in xyz), tuple(max(c) for c in xyz)


def _bin_ranges(box):
    lo, hi = box
    return tuple(range(lo[i] // BIN_WIDTH, hi[i] // BIN_WIDTH + 1) for i in (0, 1))


def _small(ranges):
    # Avoid len(range) overflow even for arbitrary finite binary32 magnitudes.
    x, y = ranges
    return (x.stop - x.start) * (y.stop - y.start) <= MAX_BOX_BINS


class _ExactIndex:
    """Conservative exact XY grid + 3D closed AABB; the oracle tests all survivors."""
    def __init__(self, triangles):
        self.triangles = triangles
        self.boxes = tuple(map(_box, triangles))
        self.bins = {}
        self.wide = []
        for i, box in enumerate(self.boxes):
            ranges = _bin_ranges(box)
            if not _small(ranges):
                self.wide.append(i)
                continue
            for cell in product(*ranges):
                self.bins.setdefault(cell, []).append(i)

    def candidates(self, start, end):
        low, high = box = _box((start, end))
        ranges = _bin_ranges(box)
        if _small(ranges):
            possible = set(self.wide)
            for cell in product(*ranges):
                possible.update(self.bins.get(cell, ()))
            ids = sorted(possible)
        else:
            ids = range(len(self.triangles))
        return tuple(self.triangles[i] for i in ids
                     if all(a <= d and c <= b for a, b, c, d in zip(low, high, *self.boxes[i])))


def _endpoints(ray):
    *p, t_min, dx, dy, dz, t_max = map(Fraction, ray)
    return tuple(tuple(o + t * v for o, v in zip(p, (dx, dy, dz))) for t in (t_min, t_max))


@dataclass(frozen=True)
class Reference:
    words: tuple[int, ...]
    candidates: int
    max_candidates: int


def expected(fixture, first_hit, *, indexed=True):
    """Use the open segment between exact o+tMin*d and o+tMax*d.

    No seed, generator labels, producer arithmetic, or output digest is used
    to decide hits. The broad phase cannot remove a true hit: each possible
    contact lies in both closed AABBs, including bin/endpoint boundaries.
    """
    tris = tuple(tuple(tuple(map(Fraction, p)) for p in t) for t in fixture.triangles)
    index = _ExactIndex(tris) if indexed else None
    words = []
    tested = max_tested = 0
    for batch in fixture.batches:
        for ray in batch[:fixture.active]:
            p, q = _endpoints(ray)
            candidates = index.candidates(p, q) if index is not None else tris
            tested += len(candidates)
            max_tested = max(max_tested, len(candidates))
            words.append(int(bool(candidates) and first_hit(p, q, candidates, include_endpoints=False) is not None))
        words.extend([POISON] * (fixture.lanes - fixture.active))
    return Reference(tuple(words), tested, max_tested)


def pack_words(words):
    return b''.join(WORD.pack(w) for w in words)


def bind(fixture, raw):
    """Bind raw bytes to a fixture identity, not to GPU execution provenance."""
    if len(raw) != WORD.size * fixture.lanes * len(fixture.batches):
        raise ValueError('raw readback length differs from fixture geometry')
    return RESULT_HEADER.pack(RESULT_MAGIC, fixture.digest) + raw


def compare(fixture, words, bound, *, example_limit=12):
    """Compare every padded slot. Boundary differences cannot certify a backend."""
    total = fixture.lanes * len(fixture.batches)
    if len(words) != total or len(bound) != RESULT_HEADER.size + total * WORD.size:
        raise ValueError('bound readback/reference length differs from fixture geometry')
    magic, digest = RESULT_HEADER.unpack_from(bound)
    if magic != RESULT_MAGIC or digest != fixture.digest:
        raise ValueError('readback magic or fixture SHA-256 mismatch')
    count, invalid, examples = 0, 0, []
    for i, (want, (got,)) in enumerate(zip(words, WORD.iter_unpack(memoryview(bound)[RESULT_HEADER.size:]), strict=True)):
        if want == got:
            continue
        batch, lane = divmod(i, fixture.lanes)
        kind = 'tail' if lane >= fixture.active else 'active' if got in (0, 1) else 'nonbinary'
        invalid += kind != 'active'
        count += 1
        if len(examples) < example_limit:
            examples.append(dict(batch=batch, lane=lane, expected=want, observed=got, kind=kind))
    status = ('fail' if invalid else 'diagnostic_only') if fixture.mode == 'boundary' else ('fail' if count else 'pass')
    return dict(status=status, words_checked=total, mismatches=count, invalid_words=invalid,
                readback_sha256=sha256(bound).hexdigest(), examples=examples)


def audit(fixture, reference):
    n = fixture.active
    active_words = [reference.words[i * fixture.lanes:i * fixture.lanes + n] for i in range(len(fixture.batches))]
    if fixture.mode == 'performance' and (any(x == y for x, y in zip(*active_words, strict=True)) or
            any(a == b for a, b in zip(fixture.batches[0][:n], fixture.batches[1][:n], strict=True))):
        raise ValueError('performance input does not flip every active ray answer')
    return dict(mode=fixture.mode, profile=fixture.profile, seed=fixture.seed,
                fixture_sha256=fixture.digest.hex(), triangles=len(fixture.triangles), active_per_batch=n,
                padded_per_batch=fixture.lanes, blocked_by_batch=list(map(sum, active_words)),
                active_total=n * len(fixture.batches),
                exact_candidate_pairs=reference.candidates, max_candidates_per_ray=reference.max_candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('audit', 'check', 'bind'):
        p = sub.add_parser(command)
        p.add_argument('fixture', type=Path)
        if command == 'bind':
            p.add_argument('raw', type=Path)
            p.add_argument('out', type=Path)
        else:
            p.add_argument('--oracle', type=Path, required=True, help='trusted rt_visibility_oracle.py source')
        if command == 'check':
            p.add_argument('--readback', action='append', nargs=2, required=True, metavar=('NAME', 'BOUND_FILE'))
    args = parser.parse_args()
    try:
        fixture = decode(args.fixture.read_bytes())
        if args.command == 'bind':
            bound = bind(fixture, args.raw.read_bytes())
            with args.out.open('xb') as f:
                f.write(bound)
            return 0
        first_hit, oracle_digest = load_oracle(args.oracle)
        reference = expected(fixture, first_hit)
        report = audit(fixture, reference)
        report['oracle_sha256'] = oracle_digest
        if args.command == 'audit':
            print(json.dumps(report, sort_keys=True))
            return 0
        reports = {}
        for name, path in args.readback:
            if not name or name in reports:
                raise ValueError('readback names must be nonempty and unique')
            reports[name] = compare(fixture, reference.words, Path(path).read_bytes())
        report['readbacks'] = reports
        print(json.dumps(report, sort_keys=True))
        return int(any(r['status'] == 'fail' for r in reports.values()))
    except (OSError, ValueError) as e:
        parser.exit(2, f'error: {e}\n')


if __name__ == '__main__':
    sys.exit(main())
