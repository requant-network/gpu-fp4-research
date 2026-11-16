"""Canonical stored FP32 coefficients for portable FFT experiment replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fft_reference import augmented_basis, permutation, radices, twiddles

LENGTHS = (128, 512, 1000, 1024, 4096, 16384)


def coefficients(n):
    index = np.arange(n)
    roots = np.exp(2j * np.pi * np.arange(4*n) / (4*n)).astype(np.complex64)
    roots[np.arange(4)*n] = np.array([1, 1j, -1, -1j], dtype=np.complex64)
    basis = np.zeros(3584, dtype=np.float32)
    for inverse in (False, True):
        for radix, offset in ((2, 0), (4, 256), (8, 512), (16, 768)):
            matrix = augmented_basis(radix, inverse).ravel(order="F")
            start = 1792*inverse + offset
            basis[start:start+matrix.size] = matrix
    frequency = np.minimum(index, n-index) / n
    lowpass = np.where(frequency <= .125, 1., np.where(frequency >= .25, 0., .5*(1+np.cos(np.pi*(frequency-.125)/.125))))
    lowpass = (lowpass * np.exp(-2j*np.pi*index*3/n)).astype(np.complex64)
    notch = np.ones(n, dtype=np.complex64)
    notch[n//8-1:n//8+2] = 0
    result = {"roots": roots, "basis": basis, "rect": np.ones(n, dtype=np.float32),
              "hann": (.5-.5*np.cos(2*np.pi*index/n)).astype(np.float32),
              "identity": np.ones(n, dtype=np.complex64), "lowpass": lowpass, "notch": notch}
    if not n & (n-1):
        for maximum in (8, 16):
            result[f"permutation{maximum}"] = permutation(n, maximum)
            previous, phases = 1, []
            for radix in radices(n, maximum):
                phases.append(twiddles(previous, radix).astype(np.complex64).ravel())
                previous *= radix
            result[f"phases{maximum}"] = np.concatenate(phases)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fixtures/fft"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = {}
    for n in LENGTHS:
        path = args.output / f"coefficients-{n}.npz"
        if path.exists():
            raise FileExistsError(f"Refusing to replace canonical fixture {path}")
        np.savez_compressed(path, **coefficients(n))
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    root = Path(__file__).resolve().parent
    sources = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ("fft_coefficients.py", "fft_reference.py")}
    (args.output / "manifest.json").write_text(json.dumps({"schema_version": 1, "sources_sha256": sources, "files_sha256": files}, indent=2)+"\n", encoding="utf-8")
    print(f"Stored {len(files)} canonical FFT coefficient fixtures")


if __name__ == "__main__":
    main()
