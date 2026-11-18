"""Chunk-invariant logical input generation and explicit three-way pipeline routes."""
import numpy as np


def input_words(profile, dtype, offset, count, seed):
    dtype = np.dtype(dtype)
    assert dtype in (np.dtype('<f4'), np.dtype('<f8'))
    assert isinstance(offset, int) and isinstance(count, int) and offset >= 0 and count >= 0
    assert offset + count < 2**63 and isinstance(seed, int) and 0 <= seed < 2**32
    index = np.arange(offset, offset + count, dtype=np.uint64)
    if profile == 'constant':
        result = np.full(count, (seed % 1024) / 64, dtype)
    elif profile == 'smooth':
        integer = ((index // 128) % 1024 + 2 * (index % 128)) * 256
        integer += ((index // (128 * 1024)) + np.uint64(seed)) % 65536
        result = integer.astype(dtype) / dtype.type(16384)
    elif profile == 'random-bits':
        with np.errstate(over='ignore'):
            mixed = index + np.uint64(seed) + np.uint64(0x9e3779b97f4a7c15)
            mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
            mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
            mixed ^= mixed >> np.uint64(31)
        words = (mixed ^ (mixed >> np.uint64(32))).astype('<u4') if dtype.itemsize == 4 else mixed
        result = words.view(dtype)
    else:
        raise ValueError(f'Unknown compression source profile: {profile}')
    result.flags.writeable = False
    return result


def routes(threads=(1, 4), depths=(1, 2)):
    result = []
    for codec in ('plain', 'ndzip', 'xbp'):
        for policy in (('online',) if codec == 'plain' else ('online', 'precompressed')):
            for cpu_threads in ((1,) if codec == 'plain' else threads):
                for depth in depths:
                    assert cpu_threads > 0 and depth in (1, 2)
                    result.append({'codec': codec, 'policy': policy, 'cpu_threads': cpu_threads, 'queue_depth': depth})
    return result


def frame_specs(dtype, total_bytes, frame_bytes, columns=128):
    dtype = np.dtype(dtype)
    assert dtype in (np.dtype('<f4'), np.dtype('<f8'))
    row_bytes = columns * dtype.itemsize
    assert total_bytes > 0 and frame_bytes > 0 and total_bytes % row_bytes == frame_bytes % row_bytes == 0
    frames = []
    for begin in range(0, total_bytes, frame_bytes):
        count = min(frame_bytes, total_bytes - begin)
        frames.append({'frame': len(frames), 'offset_words': begin // dtype.itemsize,
                       'shape': [count // row_bytes, columns], 'bytes': count})
    return frames


def suite_spec(name):
    if name == 'smoke':
        return {'datasets': [('a', '<f4', 'smooth')], 'routes': routes(),
                'total_bytes': 1024**2, 'frame_bytes': 256 * 1024, 'repeats': 1}
    if name == 'paired':
        return {'datasets': [(cohort, dtype, profile) for cohort in ('a', 'b')
                             for dtype in ('<f4', '<f8') for profile in ('constant', 'smooth', 'random-bits')],
                'routes': routes(), 'total_bytes': 4 * 1024**2,
                'frame_bytes': 1024**2, 'repeats': 2}
    if name == 'over-vram':
        return {'datasets': [(cohort, '<f4', profile) for cohort in ('a', 'b')
                             for profile in ('smooth', 'random-bits')],
                'routes': routes(threads=(4,)), 'total_bytes': 20 * 1024**3,
                'frame_bytes': 64 * 1024**2, 'repeats': 2}
    if name == 'above-probe':
        return {'datasets': [('a', '<f4', 'smooth')],
                'routes': [r for r in routes(threads=(4,), depths=(2,)) if r['policy'] == 'online'],
                'total_bytes': 20 * 1024**3, 'frame_bytes': 64 * 1024**2, 'repeats': 1}
    raise ValueError(f'Unknown fixed compression suite {name}')
