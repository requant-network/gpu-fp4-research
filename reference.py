"""Independent NumPy encodings, input generation and numerical checks."""
import hashlib
import numpy as np

MODES = ("fp4", "fp8", "fp16", "bf16", "fp32")
SHAPES = ((16, 512, 512), (64, 1024, 1024), (128, 128, 128),
          (256, 1024, 1024), (512, 512, 512), (1024, 1024, 1024))
PROFILES = ("normal", "outliers")
FP4_LEVELS = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)


def fp8_positive_levels():
    codes = np.arange(127, dtype=np.int32)
    exponent, fraction = codes >> 3, codes & 7
    return np.where(exponent == 0, fraction * 2.0**-9,
                    (1 + fraction / 8) * np.exp2(exponent - 7)).astype(np.float32)


FP8_LEVELS = fp8_positive_levels()


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def nearest_codes(values, levels):
    """Round to the nearest finite magnitude; midpoint ties use even codes."""
    x = np.abs(values)
    upper = np.searchsorted(levels, x).clip(0, len(levels) - 1)
    lower = (upper - 1).clip(0)
    low_error, high_error = x - levels[lower], levels[upper] - x
    choose_upper = (high_error < low_error) | ((high_error == low_error) & ((upper & 1) == 0))
    return np.where(choose_upper, upper, lower).astype(np.uint8)


def encode_fp8(values):
    codes = nearest_codes(values, FP8_LEVELS)
    return codes | (np.signbit(values).astype(np.uint8) << 7)


def decode_fp8(codes):
    raw = np.asarray(codes, np.uint8)
    if np.any((raw & 127) == 127):
        raise ValueError("Nonfinite E4M3 encoding")
    return FP8_LEVELS[raw & 127] * np.where(raw & 128, -1, 1).astype(np.float32)


def quantize(values, mode):
    x = np.asarray(values, np.float32)
    if mode not in MODES or x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a supported mode and a finite two-dimensional input")
    scales = np.empty(0, np.uint8)
    if mode == "fp4":
        if x.shape[1] % 16:
            raise ValueError("FP4 rows must contain whole 16-value blocks")
        groups = x.reshape(x.shape[0], -1, 16)
        maximum = np.abs(groups).max(axis=-1)
        candidate = np.clip(maximum / np.float32(6), np.float32(1 / 512), np.float32(448))
        candidate = np.where(maximum == 0, np.float32(1), candidate)
        scales = encode_fp8(candidate)
        factors = decode_fp8(scales)
        normalized = groups / factors[..., None]
        codes = nearest_codes(normalized, FP4_LEVELS)
        codes |= np.signbit(normalized).astype(np.uint8) << 3
        codes = codes.reshape(x.shape)
        raw = codes[:, 0::2] | (codes[:, 1::2] << 4)
        decoded = FP4_LEVELS[codes & 7] * np.where(codes & 8, -1, 1).astype(np.float32)
        decoded = (decoded.reshape(groups.shape) * factors[..., None]).reshape(x.shape)
    elif mode == "fp8":
        raw = encode_fp8(x)
        decoded = decode_fp8(raw)
    elif mode == "fp16":
        raw = x.astype(np.float16).view(np.uint16)
        decoded = raw.view(np.float16).astype(np.float32)
    elif mode == "bf16":
        bits = x.view(np.uint32)
        rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & 1)
        raw = (rounded >> 16).astype(np.uint16)
        decoded = (raw.astype(np.uint32) << 16).view(np.float32)
    else:
        decoded = x.copy()
        raw = x.view(np.uint32).copy()
    if not np.isfinite(decoded).all():
        raise ValueError("Input exceeds the finite range of the selected format")
    return np.ascontiguousarray(raw), np.ascontiguousarray(scales), np.ascontiguousarray(decoded)


def inputs(seed, shape, profile):
    if profile not in PROFILES:
        raise ValueError("Unknown input profile")
    m, n, k = shape
    rng = np.random.default_rng(np.random.SeedSequence([seed, m, n, k, PROFILES.index(profile)]))
    def normal_like(rows):
        # Integer-only CLT construction avoids platform-dependent normal tail math.
        raw = rng.bit_generator.random_raw((2, rows, k))
        total = np.zeros((rows, k), np.int32)
        for shift in (0, 8, 16, 24, 32, 40):
            total += ((raw >> shift) & 255).sum(axis=0, dtype=np.int32)
        return (total - 1530).astype(np.float32) / np.float32(256)
    a = normal_like(m)
    b = normal_like(n)
    if profile == "outliers":
        a.ravel()[::97] *= np.float32(32)
        b.ravel()[::89] *= np.float32(32)
    a[0] = 0
    b[0] = 0
    return a, b


def check_output(actual, qa, qb):
    """Check every output against FP64 on independently reconstructed operands."""
    x, y = qa.astype(np.float64), qb.astype(np.float64)
    expected = x @ y.T
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("Invalid output shape or nonfinite output")
    difference = np.abs(actual.astype(np.float64) - expected)
    unit_roundoff = 2.0**-24
    gamma = qa.shape[1] * unit_roundoff / (1 - qa.shape[1] * unit_roundoff)
    bound = 8 * gamma * (np.abs(x) @ np.abs(y).T) + 1e-6
    ratio = float(np.max(difference / bound))
    if ratio > 1:
        raise ValueError(f"GPU arithmetic exceeds the declared FP32 accumulation bound: {ratio}")
    return ratio


def relative_error(actual, expected):
    error = float(np.linalg.norm(actual.astype(np.float64) - expected))
    norm = float(np.linalg.norm(expected))
    return error / norm if norm else error
