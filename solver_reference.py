"""Independent FP64 input construction and accuracy checks for the SPD solver study."""
import gzip
import math
from pathlib import Path

import numpy as np

RESIDUAL_TARGET = 1e-12
FORWARD_TARGET = 1e-8


def controlled(n, exponent, nrhs, seed, variant=0):
    """Exactly stored dyadic SPD system with condition 2**exponent and exact truth.

    A signed/permuted Walsh basis avoids a rounded QR factorization. Integer
    construction bounds guarantee that every matrix entry is exactly FP64.
    RHS variants keep A unchanged and excite the largest/smallest eigenvalues.
    """
    if n < 4 or n > 2048 or n & (n - 1) or not 0 <= exponent <= 40 or nrhs not in (1, 4):
        raise ValueError("Expected power-of-two n in [4,2048], exponent in [0,40], nrhs 1 or 4")
    rng = np.random.Generator(np.random.PCG64(seed))
    powers = (np.arange(n, dtype=np.int64) * exponent) // (n - 1)
    powers = powers[rng.permutation(n)]
    coefficients = np.left_shift(np.int64(1), exponent - powers)
    transformed = coefficients.copy()
    step = 1
    while step < n:
        blocks = transformed.reshape(-1, 2 * step)
        left, right = blocks[:, :step].copy(), blocks[:, step:].copy()
        blocks[:, :step], blocks[:, step:] = left + right, left - right
        step *= 2
    permutation = rng.permutation(n)
    signs = rng.choice(np.array([-1., 1.]), n)
    indices = np.bitwise_xor(permutation[:, None], permutation[None, :])
    a = np.ldexp(transformed[indices].astype(np.float64), -exponent - (n.bit_length() - 1))
    a *= signs[:, None] * signs[None, :]
    rhs_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, variant, 7001])))
    truth, b = np.zeros((n, nrhs)), np.zeros((n, nrhs))
    for column in range(nrhs):
        selected = [int(rhs_rng.choice(np.flatnonzero(abs(powers - value) == abs(powers - value).min())))
                    for value in (0, exponent, exponent // 2)]
        for component, (index, weight) in enumerate(zip(selected, (1., .5, .25))):
            weight *= -1 if (variant >> component) & 1 else 1
            basis = signs * np.array([1. if (int(row) & index).bit_count() % 2 == 0 else -1.
                                      for row in permutation])
            truth[:, column] += weight * basis
            b[:, column] += math.ldexp(weight, -int(powers[index])) * basis
    return np.asfortranarray(a), np.asfortranarray(b), np.asfortranarray(truth)


def read_matrix_market(path):
    """Read a bounded symmetric real coordinate file, rejecting duplicate pairs."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="ascii") as stream:
        if stream.readline().strip().lower() != "%%matrixmarket matrix coordinate real symmetric":
            raise ValueError("Expected symmetric real coordinate Matrix Market data")
        lines = (line.strip() for line in stream if line.strip() and not line.startswith("%"))
        n, m, entries = map(int, next(lines).split())
        if n != m or not 1 <= n <= 4096 or not 1 <= entries <= n * (n + 1) // 2:
            raise ValueError("Invalid dimensions or entry count")
        a, seen = np.zeros((n, n), order="F"), set()
        for line in lines:
            i_text, j_text, value_text = line.split()
            i, j, value = int(i_text) - 1, int(j_text) - 1, float(value_text)
            key = min(i, j), max(i, j)
            if not (0 <= i < n and 0 <= j < n) or key in seen or not math.isfinite(value):
                raise ValueError("Out-of-range, duplicate or nonfinite entry")
            seen.add(key)
            a[i, j] = a[j, i] = value
        if len(seen) != entries or np.any(np.diag(a) <= 0):
            raise ValueError("Entry count mismatch or nonpositive diagonal")
    return a


def real_system(a, nrhs, seed, variant=0, scale_exponent=0):
    """Exact manufactured RHS from actual matrix columns; not physical loading."""
    if nrhs not in (1, 4) or a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) < nrhs:
        raise ValueError("Invalid square matrix or RHS count")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, variant, 7002])))
    columns = rng.choice(len(a), nrhs, replace=False)
    truth = np.zeros((len(a), nrhs), order="F")
    truth[columns, np.arange(nrhs)] = 1.
    a = np.asfortranarray(np.ldexp(a, scale_exponent))
    return a, np.asfortranarray(a[:, columns]), truth


def accuracy(a, b, truth, solution):
    """Maximum per-RHS errors: no averaging away a failed RHS."""
    if solution.shape != b.shape or truth.shape != b.shape or a.shape != (len(b), len(b)):
        raise ValueError("Incompatible shapes")
    if not np.isfinite(solution).all():
        return {"finite": False, "residual_per_rhs": None, "forward_per_rhs": None,
                "max_relative_residual": None, "max_forward_error": None, "target_pass": False}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual = np.linalg.norm(a @ solution - b, axis=0) / np.linalg.norm(b, axis=0)
        forward = np.linalg.norm(solution - truth, axis=0) / np.linalg.norm(truth, axis=0)
    finite = bool(np.isfinite(residual).all() and np.isfinite(forward).all())
    return {"finite": finite, "residual_per_rhs": residual.tolist() if finite else None,
            "forward_per_rhs": forward.tolist() if finite else None,
            "max_relative_residual": float(residual.max()) if finite else None,
            "max_forward_error": float(forward.max()) if finite else None,
            "target_pass": bool(finite and residual.max() <= RESIDUAL_TARGET and forward.max() <= FORWARD_TARGET)}


def scaled_integers(values):
    """Exact common power-of-two denominator for finite binary64 values."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Exact oracle requires finite values")
    ratios = [float(value).as_integer_ratio() for value in values.flat]
    exponents = [denominator.bit_length() - 1 for _, denominator in ratios]
    exponent = max(exponents, default=0)
    integers = np.array([numerator << (exponent - power)
                         for (numerator, _), power in zip(ratios, exponents)], dtype=object)
    return integers.reshape(values.shape), exponent


def dyadic_float(integer, exponent):
    """Round once to binary64, including subnormal/ties-to-even boundaries."""
    integer = int(integer)
    if not integer:
        return 0.
    sign, integer = (-1 if integer < 0 else 1), abs(integer)
    drop = max(0, integer.bit_length() - 53, exponent - 1074)
    quotient, remainder = divmod(integer, 1 << drop)
    if drop and (remainder > 1 << (drop - 1) or (remainder == 1 << (drop - 1) and quotient & 1)):
        quotient += 1
    try:
        return math.ldexp(float(sign * quotient), drop - exponent)
    except OverflowError:
        return math.copysign(math.inf, sign)


def exact_residual_rows(a, b, solution, rows):
    """Integer dot products expose the FP64 residual cancellation floor."""
    ai, ae = scaled_integers(a[rows])
    xi, xe = scaled_integers(solution)
    bi, be = scaled_integers(b[rows])
    exponent = max(ae + xe, be)
    product = (ai @ xi) * (1 << (exponent - ae - xe)) - bi * (1 << (exponent - be))
    return np.array([dyadic_float(value, exponent) for value in product.flat]).reshape(product.shape)
