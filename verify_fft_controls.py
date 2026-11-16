"""Replay the cuFFT capability probe and verify the external timer/SDK controls."""
import json
from pathlib import Path

import numpy as np

from fft_preflight import complex_values, decode, inputs, prepare, relative
from reference import digest
from verify_fft import close, sha

ROOT = Path(__file__).resolve().parent


def main():
    record = json.loads((ROOT/"results/fft-preflight.json").read_text())
    assert record["complete"] and len(record["groups"]) == 12
    for name, expected in record["source_sha256"].items():
        assert sha(ROOT/name) == expected
    checked = available = 0
    for group in record["groups"]:
        n, batch = group["n"], group["batch"]
        original = inputs(n, batch)
        assert digest(original) == group["original_sha256"]
        path = ROOT/"results"/group["witness_file"]
        assert sha(path) == group["witness_sha256"]
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["original"], original)
            for case in group["cases"]:
                p = case["precision"]
                encoded, rounded = prepare(original, p)
                assert digest(encoded) == case["encoded_sha256"]
                assert case["available"] == (n != 1000 or p < 2)
                if not case["available"]:
                    assert "16" in case["error"]
                    continue
                available += 1
                forward, inverse = z[f"fft_p{p}"], z[f"inverse_p{p}"]
                assert digest(forward) == case["fft_sha256"] and digest(inverse) == case["inverse_sha256"]
                spectrum = complex_values(decode(forward, p))
                restored = complex_values(decode(inverse, p))/n
                metrics = {"forward_vs_original": relative(spectrum, np.fft.fft(complex_values(original))),
                           "forward_vs_rounded_input": relative(spectrum, np.fft.fft(complex_values(rounded))),
                           "inverse_vs_actual_forward_spectrum": relative(restored, np.fft.ifft(spectrum)),
                           "roundtrip_vs_original": relative(restored, complex_values(original))}
                for name, error in metrics.items():
                    close(error, case["errors"][name])
                    assert error < case["screening_threshold"]
                assert len(case["forward_ms"]) == len(case["inverse_ms"]) == 5
                checked += forward.size+inverse.size
    assert available == 44
    timer = json.loads((ROOT/"results/fft-timer.json").read_text())
    for name, expected in timer["source_sha256"].items():
        assert sha(ROOT/name) == expected
    assert [case["repetitions"] for case in timer["cases"]] == [1, 8]
    for case in timer["cases"]:
        assert len(case["observations"]) == 30
        assert all(row["cuda_ms"] > 0 and row["wall_ms"] > 0 for row in case["observations"])
    dependency = json.loads((ROOT/"results/fft-dx-dependency.json").read_text())
    assert dependency["cufftdx_version"] == 10703 and len(dependency["header_sha256"]) == 1022
    assert dependency["archive_sha256"] == "59a9233db34b75568acbcc5284e6cefe6fad5577ee644f85044971d62eeea353"
    assert dependency["archive_bytes"] == 37377287
    for name, expected in dependency["source_sha256"].items():
        assert sha(ROOT/name) == expected
    print(f"FFT controls: {available}/48 supported, {checked} scalar FFT/inverse entries replayed; timer and pinned SDK verified")


if __name__ == "__main__":
    main()
