"""Trusted Study 13 wire decoder and rational CPU oracle; no loaders or device calls."""
from fractions import Fraction
from itertools import product

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from struct import Struct

MAGIC = b'S13VIS1\0'
HEADER = Struct('<8s6IQ')
TRIANGLE = Struct('<9f')
RAY = Struct('<8f')
WORD = Struct('<I')
LOCAL_SIZE = 64
POISON = 0xfeedface
MAX_TRIANGLES = 65_536
MAX_ACTIVE = 262_144
MODES = ('performance', 'boundary')
PROFILES = ('spread', 'clustered', 'boundary')
Point = tuple[float, float, float]
Triangle = tuple[Point, Point, Point]
Ray = tuple[float, float, float, float, float, float, float, float]


@dataclass(frozen=True)
class Fixture:
    mode: str
    profile: str
    seed: int
    triangles: tuple[Triangle, ...]
    active: int
    lanes: int
    batches: tuple[tuple[Ray, ...], ...]
    digest: bytes


def lanes_for(active):
    return ((active + LOCAL_SIZE - 1) // LOCAL_SIZE) * LOCAL_SIZE


def _valid_ray(ray):
    return (all(map(isfinite, ray)) and 0 <= ray[3] < ray[7]
            and any(ray[4:7]))


def decode(data):
    """Validate the declared workload and parse a single immutable byte snapshot."""
    data = bytes(data)
    if len(data) < HEADER.size:
        raise ValueError('truncated fixture header')
    magic, mode, profile, n, active, batches, lanes, seed = HEADER.unpack_from(data)
    if magic != MAGIC or (mode, profile, batches) not in ((0, 0, 2), (0, 1, 2), (1, 2, 1)):
        raise ValueError('invalid fixture magic, mode, profile or batch count')
    if not 1 <= n <= MAX_TRIANGLES or not 1 <= active <= MAX_ACTIVE or lanes != lanes_for(active):
        raise ValueError('invalid triangle, active or padded count')
    if mode == 1 and seed != 0:
        raise ValueError('boundary fixtures have seed zero')
    tri_end = HEADER.size + TRIANGLE.size * n
    batch_bytes = RAY.size * lanes
    if len(data) != tri_end + batches * batch_bytes:
        raise ValueError('fixture byte length does not match header')
    view = memoryview(data)
    triangles = []
    for t in TRIANGLE.iter_unpack(view[HEADER.size:tri_end]):
        if not all(map(isfinite, t)):
            raise ValueError(f'nonfinite triangle {len(triangles)}')
        triangles.append((t[:3], t[3:6], t[6:9]))
    rays = []
    for batch in range(batches):
        offset = tri_end + batch * batch_bytes
        rows = tuple(RAY.iter_unpack(view[offset:offset + batch_bytes]))
        for i, r in enumerate(rows):
            if not _valid_ray(r):
                raise ValueError(f'invalid ray: batch {batch}, lane {i}')
        rays.append(rows)
    return Fixture(MODES[mode], PROFILES[profile], seed, tuple(triangles), active,
                   lanes, tuple(rays), sha256(data).digest())

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

Scalar = int | float | Fraction
Point = Sequence[Scalar]
Triangle = Sequence[Point]
_Vec = tuple[Fraction, Fraction, Fraction]
_ZERO, _ONE = Fraction(0), Fraction(1)


@dataclass(frozen=True)
class Hit:
    """Exact entry parameter and all tied zero-based input triangle indices."""

    entry_t: Fraction
    triangle_indices: tuple[int, ...]


def _point(point: Point) -> _Vec:
    x, y, z = point
    return Fraction(x), Fraction(y), Fraction(z)


def _sub(a: _Vec, b: _Vec) -> _Vec:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _dot(a: _Vec, b: _Vec) -> Fraction:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: _Vec, b: _Vec) -> _Vec:
    x, y, z = a
    u, v, w = b
    return y * w - z * v, z * u - x * w, x * v - y * u


def _closed_interval(
    start: _Vec, end: _Vec, triangle: Triangle,
) -> tuple[Fraction, Fraction] | None:
    """Clip [0, 1] to the triangle plane and its three closed half-spaces."""
    a, b, c = map(_point, triangle)
    normal = _cross(_sub(b, a), _sub(c, a))
    if not any(normal):
        return None

    d0 = _dot(normal, _sub(start, a))
    d1 = _dot(normal, _sub(end, a))
    lo, hi = _ZERO, _ONE
    if d0 == d1:
        if d0 != 0:
            return None
    else:
        t = d0 / (d0 - d1)
        if not lo <= t <= hi:
            return None
        lo = hi = t

    # The inward edge test is affine in t: s0 + t * (s1 - s0) >= 0.
    for u, v in ((a, b), (b, c), (c, a)):
        edge = _sub(v, u)
        s0 = _dot(normal, _cross(edge, _sub(start, u)))
        s1 = _dot(normal, _cross(edge, _sub(end, u)))
        slope = s1 - s0
        if slope > 0:
            lo = max(lo, -s0 / slope)
        elif slope < 0:
            hi = min(hi, -s0 / slope)
        elif s0 < 0:
            return None
        if lo > hi:
            return None
    return lo, hi


def first_hit(
    start: Point, end: Point, triangles: Iterable[Triangle], *,
    include_endpoints: bool = False,
) -> Hit | None:
    """Return the earliest blocker along start + t*(end - start), or None.

    Triangles are two-sided with closed edges and vertices. Zero-area triangles
    never block. Coplanar overlap and single-point tangencies block if any
    contact is in the queried segment.

    By default only 0 < t < 1 is queried and a zero-length segment is empty.
    Set include_endpoints=True to query 0 <= t <= 1 and to treat a zero-length
    segment as a point (a hit has t=0). entry_t is the infimum of the blocking
    parameters, so an open segment overlapping a face just after its start
    can report t=0 even though the start itself is excluded.

    All triangles with the same earliest entry parameter are returned in
    input order, without merging shared edges, vertices, or duplicate faces.
    """
    p, q = _point(start), _point(end)
    if p == q and not include_endpoints:
        return None

    best: Fraction | None = None
    indices: list[int] = []
    for index, triangle in enumerate(triangles):
        interval = _closed_interval(p, q, triangle)
        if interval is None:
            continue
        t, exit_t = interval
        if not include_endpoints and (exit_t == _ZERO or t == _ONE):
            continue
        if best is None or t < best:
            best, indices = t, [index]
        elif t == best:
            indices.append(index)
    return None if best is None else Hit(best, tuple(indices))


def is_visible(
    start: Point, end: Point, triangles: Iterable[Triangle], *,
    include_endpoints: bool = False,
) -> bool:
    """True iff no triangle blocks the segment under first_hit's semantics."""
    return first_hit(start, end, triangles, include_endpoints=include_endpoints) is None

RESULT_HEADER = Struct('<8s32s')
RESULT_MAGIC = b'S13OUT1\0'
BIN_WIDTH = 16
MAX_BOX_BINS = 64


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
