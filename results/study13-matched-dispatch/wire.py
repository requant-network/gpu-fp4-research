"""Study 13 v1 little-endian fixture ABI; shared by generator and verifier."""
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
