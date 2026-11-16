"""Independent FFT factorization, typed stage encoding and complete DSP reference."""
import numpy as np

from spmm_preflight import encode

ROOT16 = np.array([1., .9238795042037964, .7071067690849304, .3826834261417389,
                   0., -.3826834261417389, -.7071067690849304, -.9238795042037964,
                   -1., -.9238795042037964, -.7071067690849304, -.3826834261417389,
                   0., .3826834261417389, .7071067690849304, .9238795042037964], dtype=np.float32)


def radices(n, maximum=16):
    if n < 2 or n > 65536 or n & (n - 1) or maximum not in (8, 16):
        raise ValueError("Expected a power-of-two length in [2,65536] and radix 8 or 16")
    result = []
    while n > 1:
        radix = min(n, maximum)
        result.append(radix)
        n //= radix
    return result


def permutation(n, maximum=16):
    rest = np.arange(n, dtype=np.int64)
    output = np.zeros(n, dtype=np.int64)
    divisor = n
    for radix in radices(n, maximum):
        divisor //= radix
        output += (rest % radix) * divisor
        rest //= radix
    return output.astype(np.int32)


def augmented_basis(radix, inverse=False):
    if radix not in (2, 4, 8, 16):
        raise ValueError("Unsupported matrix DFT radix")
    index = np.arange(radix)
    phase = (index[:, None] * index[None, :] * (16 // radix)) % 16
    real = ROOT16[phase]
    imag = ROOT16[(4 - phase) % 16] * (1 if inverse else -1)
    size = max(16, 2 * radix)
    matrix = np.zeros((size, size), dtype=np.float32, order="F")
    matrix[:radix, :radix] = real
    matrix[:radix, radix:2*radix] = -imag
    matrix[radix:2*radix, :radix] = imag
    matrix[radix:2*radix, radix:2*radix] = real
    return matrix


def twiddles(previous, radix):
    return np.exp(-2j * np.pi * np.arange(radix)[:, None] * np.arange(previous)[None, :] / (previous * radix))


def mathematical_fft(values, maximum=16, inverse=False):
    """FP64 mixed-radix DIT, normalized at each stage; independent of NumPy FFT."""
    values = np.asarray(values, dtype=np.complex128)
    if values.ndim != 2:
        raise ValueError("Expected batch-by-length complex inputs")
    batch, n = values.shape
    result = values[:, permutation(n, maximum)].copy()
    previous = 1
    for radix in radices(n, maximum):
        stage = result.reshape(batch, n // (previous * radix), radix, previous)
        phase = twiddles(previous, radix)
        if inverse:
            phase = phase.conj()
        stage = stage * phase[None, None, :, :]
        index = np.arange(radix)
        matrix = np.exp((1 if inverse else -1) * 2j * np.pi * index[:, None] * index[None, :] / radix)
        result = (np.einsum('qt,bgtj->bgqj', matrix, stage) / radix).reshape(batch, n)
        previous *= radix
    return result


def stage_input(values, previous, radix, phase, inverse=False):
    """Exact FP32 multiply/add order and column-major augmented real packing."""
    values = np.asarray(values, dtype=np.complex64)
    batch, n = values.shape
    if n % (previous * radix) or phase.shape != (radix, previous):
        raise ValueError("Invalid stage geometry or twiddle table")
    stage = values.reshape(batch, n // (previous * radix), radix, previous)
    phase = np.asarray(phase, dtype=np.complex64)
    if inverse:
        phase = phase.conj()
    real = (stage.real * phase.real - stage.imag * phase.imag).astype(np.float32)
    imag = (stage.real * phase.imag + stage.imag * phase.real).astype(np.float32)
    columns = batch * n // radix
    padded = (columns + 15) // 16 * 16
    matrix = np.zeros((max(16, 2 * radix), padded), dtype=np.float32, order="F")
    matrix[:radix, :columns] = real.transpose(0, 1, 3, 2).reshape(columns, radix).T
    matrix[radix:2*radix, :columns] = imag.transpose(0, 1, 3, 2).reshape(columns, radix).T
    return matrix


def stage_output(product, batch, n, previous, radix):
    columns = batch * n // radix
    real = np.asarray(product[:radix, :columns], dtype=np.float32) / radix
    imag = np.asarray(product[radix:2*radix, :columns], dtype=np.float32) / radix
    values = real.T + np.complex64(1j) * imag.T
    return values.reshape(batch, n // (previous * radix), previous, radix).transpose(0, 1, 3, 2).reshape(batch, n)


def rounded_fft(values, maximum, precision, inverse=False, phases=None):
    """CPU model of typed real GEMMs, with FP64 products and FP32 stage outputs."""
    values = np.asarray(values, dtype=np.complex64)
    batch, n = values.shape
    result = values[:, permutation(n, maximum)].copy()
    previous = 1
    for index, radix in enumerate(radices(n, maximum)):
        phase = np.asarray(twiddles(previous, radix) if phases is None else phases[index], dtype=np.complex64)
        packed = stage_input(result, previous, radix, phase, inverse)
        matrix = augmented_basis(radix, inverse)
        a = encode(matrix.copy(order="C"), precision)[1].astype(np.float64)
        b = encode(packed.copy(order="C"), precision)[1].astype(np.float64)
        result = stage_output(a @ b, batch, n, previous, radix)
        previous *= radix
    return result


def pipeline_reference(values, window, response):
    """Complete FP64 window/FFT/filter/inverse reference on stored FP32 inputs."""
    values = np.asarray(values, dtype=np.complex128)
    window, response = np.asarray(window, dtype=np.float64), np.asarray(response, dtype=np.complex128)
    if values.ndim != 2 or window.shape != (values.shape[1],) or response.shape != window.shape:
        raise ValueError("Invalid filtering shapes")
    spectrum = np.fft.fft(values * window[None, :], axis=-1)
    filtered = spectrum * response[None, :]
    return spectrum, filtered, np.fft.ifft(filtered, axis=-1)
