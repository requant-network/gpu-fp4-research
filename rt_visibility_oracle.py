"""Independent CPU visibility reference for small scenes; no epsilon or BVH.

Coordinates are triples of ints, Fractions, or finite floats. A float is
interpreted as its exact stored binary value, not as a decimal approximation.
Use Fraction values directly for intentional non-binary rationals.
"""

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
