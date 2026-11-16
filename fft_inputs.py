"""Portable finite-FP32 signal cohorts and an explicit native comparison grid."""
import numpy as np

PROFILES = ("impulse_identity", "impulse_lowpass", "noise", "offbin", "weak72", "weak120", "highrange")
SHAPES = tuple((n, b) for n in (128, 512, 1000, 1024, 4096, 16384) for b in (1, 16)) + ((1024, 64),)
METHODS = tuple([dict(name=f"cufft-{name}", backend=0, precision=p) for p, name in enumerate(("fp32", "fp64", "fp16", "bf16"))]
                + [dict(name=f"matrix-r{r}-{name}", backend=1, precision=p, radix=r) for r in (8, 16) for p, name in enumerate(("fp32", "fp16", "bf16"))]
                + [dict(name=f"cufftdx-fp32-pack{pack}", backend=2, precision=0, pack=pack) for pack in (1, 4)])


def inputs(n, batch, profile, seed, variant, coefficients):
    if profile not in PROFILES or variant not in range(4):
        raise ValueError("Unknown FFT signal profile or variant")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, n, batch, PROFILES.index(profile), variant])))
    roots = coefficients["roots"]
    index = np.arange(n)[None, :]
    phase = rng.integers(0, 4*n, (batch, 1))
    q = n//8
    window, response = "rect", "identity"
    parameters = {"strong_bin": q, "weak_bin": None, "weak_amplitude": None, "nominal_strong_to_weak_db": None}
    if profile.startswith("impulse"):
        values = np.zeros((batch, n), dtype=np.complex64)
        at = rng.integers(0, n, batch)
        values[np.arange(batch), at] = roots[phase[:, 0]]
        if profile.endswith("lowpass"):
            response = "lowpass"
    elif profile == "noise":
        pairs = rng.integers(-32768, 32768, (batch, n, 2)).astype(np.float32) / np.float32(32768)
        values = pairs.view(np.complex64).reshape(batch, n)
        window, response = "hann", "lowpass"
    elif profile == "offbin":
        values = (np.float32(.5)*roots[((4*q+1)*index+phase) % (4*n)]
                  + np.float32(.125)*roots[((4*(q+11)+3)*index-phase) % (4*n)]).astype(np.complex64)
        window = "hann"
    else:
        amplitude = np.float32(2.**(-13 if profile in ("weak72", "highrange") else -21))
        strong = np.float32(8. if profile == "highrange" else .5)
        weak_phase = rng.integers(0, 4*n, (batch, 1))
        values = (strong*roots[(4*q*index+phase) % (4*n)]
                  + amplitude*roots[(4*(q+7)*index+weak_phase) % (4*n)]).astype(np.complex64)
        response = "notch"
        parameters.update(weak_bin=q+7, weak_amplitude=float(amplitude),
                          nominal_strong_to_weak_db=float(20*np.log10(float(strong/amplitude))))
    parameters.update(window=window, response=response)
    return np.ascontiguousarray(values), coefficients[window], coefficients[response], parameters
