"""CPU-only replay of native FFT data, signal quality and retained stage witnesses."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fft_inputs import METHODS, PROFILES, SHAPES, inputs
from fft_reference import augmented_basis, pipeline_reference
from fft_records import load_witness
from fft_validation import check_output, quality
from reference import digest
from spmm_preflight import encode

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    if a is None or b is None or isinstance(a, bool):
        assert a == b, (a, b)
    else:
        assert np.isclose(a, b, rtol=2e-7, atol=2e-8), (a, b)


def stage_replay(case, group, values, coefficients, witness):
    method = case["method"]
    p = method["precision"]
    batch, n = values.shape
    checked = 0
    forward = None
    previous_after = None
    direction = None
    complete_chain = all(item["full_columns_retained"] for item in case["stage_checks"])
    for item in case["stage_checks"]:
        prefix, stage, inverse = item["prefix"], item["geometry"], item["inverse"]
        radix, size, previous = stage["radix"], stage["tile"], stage["previous"]
        columns = witness[prefix+"_columns"]
        before, packed = witness[prefix+"_before"], witness[prefix+"_packed"]
        product, after = witness[prefix+"_product"], witness[prefix+"_after"]
        start = stage["phase_offset"]
        phase = coefficients[f"phases{method['radix']}"][start:start+radix*previous].reshape(radix, previous)[:, columns % previous]
        if inverse:
            phase = phase.conj()
        expected = np.zeros((size, len(columns)), dtype=np.float32)
        expected[:radix] = before.real*phase.real-before.imag*phase.imag
        expected[radix:2*radix] = before.real*phase.imag+before.imag*phase.real
        rounded, b = encode(expected, p)
        np.testing.assert_array_equal(packed, rounded)
        a = encode(np.ascontiguousarray(augmented_basis(radix, inverse)), p)[1].astype(np.float64)
        gold = a@b.astype(np.float64)
        bound = 2*size*np.finfo(np.float32).eps*(abs(a)@abs(b.astype(np.float64)))+1e-35
        assert np.all(abs(product-gold) <= bound), (prefix, "accumulation")
        decoded = np.empty((radix, len(columns)), dtype=np.complex64)
        decoded.real = product[:radix]/radix
        decoded.imag = product[radix:2*radix]/radix
        np.testing.assert_array_equal(after, decoded)
        assert item["checked_product_entries"] == size*stage["padded"]
        assert item["checked_packed_bytes"] == size*stage["padded"]*(4 if p == 0 else 2)
        if p:
            assert stage["lt_numerical_flags"] & 2
        if item["full_columns_retained"]:
            np.testing.assert_array_equal(columns, np.arange(stage["columns"]))
            t = np.arange(radix)[:, None]
            j = columns % previous
            g = (columns//previous) % (n//(radix*previous))
            bi = columns//(n//radix)
            positions = g[None, :]*radix*previous+j[None, :]+t*previous
            before_full = np.empty((batch, n), dtype=np.complex64)
            after_full = np.empty_like(before_full)
            before_full[bi[None, :], positions] = before
            after_full[bi[None, :], positions] = after
            assert digest(before_full) == item["before_sha256"]
            assert digest(after_full) == item["after_sha256"]
            if complete_chain and direction != inverse:
                if not inverse:
                    initial = values.copy()
                    initial.real *= coefficients[group["parameters"]["window"]]
                    initial.imag *= coefficients[group["parameters"]["window"]]
                else:
                    assert forward is not None
                    h = coefficients[group["parameters"]["response"]]
                    initial = np.empty_like(forward)
                    initial.real = forward.real*h.real-forward.imag*h.imag
                    initial.imag = forward.real*h.imag+forward.imag*h.real
                np.testing.assert_array_equal(before_full, initial[:, coefficients[f"permutation{method['radix']}"]])
            elif complete_chain and previous_after is not None:
                np.testing.assert_array_equal(before_full, previous_after)
            direction, previous_after = inverse, after_full
            if not inverse:
                forward = after_full.copy()
            padded_pack = np.zeros((stage["padded"], size), dtype=packed.dtype)
            padded_pack[columns] = packed.T
            padded_product = np.zeros((stage["padded"], size), dtype=np.float32)
            padded_product[columns] = product.T
            assert digest(padded_pack) == item["packed_sha256"]
            assert digest(padded_product) == item["product_sha256"]
        checked += product.size
    if complete_chain and direction is True:
        original = case["outputs"][case["initial_output"]]
        if original["full"]:
            actual = witness[original["prefix"]+"_output"]
            np.testing.assert_array_equal(previous_after*np.float32(n), actual)
    return checked


def verify(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["complete"] and record["study"] == 9
    assert record["environment"]["compute_capability"] == [12, 0]
    assert record["protocol"]["observations"] == 30 and record["protocol"]["host_variants"] == 4
    for name, expected in record["source_sha256"].items():
        assert sha(ROOT/name) == expected, name
    dependency = ROOT/"results"/record["dependency_file"]
    assert sha(dependency) == record["dependency_sha256"]
    shapes = ((128, 1), (1000, 1), (16384, 16)) if record["quick"] else SHAPES
    profiles = ("noise", "weak120", "highrange") if record["quick"] else PROFILES
    assert {(g["n"], g["batch"], g["profile"]) for g in record["groups"]} == {(n, b, p) for n, b in shapes for p in profiles}
    assert len(record["groups"]) == len(shapes)*len(profiles)
    configurations = available = full_entries = samples = stage_entries = observations = 0
    for group in record["groups"]:
        n, batch = group["n"], group["batch"]
        fixture = ROOT/group["coefficient_file"]
        assert sha(fixture) == group["coefficient_sha256"]
        with np.load(fixture, allow_pickle=False) as z:
            coefficients = {key: z[key] for key in z.files}
        generated = [inputs(n, batch, group["profile"], record["seed"], variant, coefficients) for variant in range(4)]
        for variant, (x, window, response, parameters) in enumerate(generated):
            assert digest(x) == group["input_sha256"][variant]
            assert digest(window) == group["window_sha256"] and digest(response) == group["response_sha256"]
            assert parameters == group["parameters"]
        refs = [pipeline_reference(x, w, h) for x, w, h, _ in generated]
        input_norms = [float(np.linalg.norm(x.astype(np.complex128)*w)) for x, w, _, _ in generated]
        witness_path = ROOT/"results"/group["witness_file"]
        assert sorted(c["method"]["name"] for c in group["cases"]) == sorted(m["name"] for m in METHODS)
        with load_witness(witness_path, group["witness_sha256"]) as z:
            indices = z["sample_indices"]
            assert np.all((indices >= 0) & (indices < n*batch))
            for case in group["cases"]:
                configurations += 1
                method = case["method"]
                expected_available = (method["backend"] == 0 and method["precision"] < 2) if n == 1000 else not (n == 16384 and method["backend"] == 2 and method["pack"] == 4)
                assert case["available"] == expected_available, (n, method, case.get("error"))
                assert case["plan_setup_ms"] > 0
                if not case["available"]:
                    assert case["error"]
                    continue
                available += 1
                meta = case["initial_metadata"]
                assert meta["owned_device_bytes"] > 0 and meta["cufft_version"] == 12300
                assert meta["spectrum_scale"] == (n if method["backend"] == 1 else 1)
                if method["backend"] == 2:
                    assert meta["cufftdx_version"] == 10703 and meta["dx_registers"] > 0 and meta["dx_theoretical_blocks_per_sm"] > 0
                expected_parameters = {2: generated[0][0], 3: generated[0][1], 4: generated[0][2]}
                if method["backend"] == 1:
                    expected_parameters.update({5: coefficients[f"permutation{method['radix']}"], 6: coefficients[f"phases{method['radix']}"], 7: coefficients["basis"]})
                    assert meta["workspace_bytes"] == 64 << 20
                    if method["precision"]:
                        assert all(s["lt_numerical_flags"] & 2 for s in meta["stages"])
                for which, expected in expected_parameters.items():
                    assert digest(expected) == case["parameter_snapshots_sha256"][str(which)]
                assert {item["variant"] for item in case["outputs"]} == set(range(4))
                for item in case["outputs"]:
                    variant, prefix = item["variant"], item["prefix"]
                    values = [z[prefix+"_"+key] for key in ("output", "spectrum", "filtered")]
                    if item["full"]:
                        for key, actual in zip(("output", "spectrum", "filtered"), values):
                            assert actual.shape == (batch, n) and digest(actual) == item[key+"_sha256"]
                        check_output(values[0], refs[variant][2], method, input_norms[variant])
                        replay = quality(*values, refs[variant], group["parameters"], input_norms[variant])
                        for key in replay:
                            close(replay[key], item["quality"][key])
                        full_entries += 3*n*batch
                    else:
                        assert all(a.shape == indices.shape for a in values)
                        for key, actual, gold in zip(("output", "spectrum", "filtered"), values, (refs[variant][2], refs[variant][0], refs[variant][1])):
                            maximum = item["quality"][key+"_max_abs_error"]
                            if maximum is not None:
                                assert np.isfinite(actual).all()
                                assert np.max(abs(actual-gold.ravel()[indices])) <= maximum*(1+1e-7)+2e-8
                        samples += 3*len(indices)
                    if not item["quality"]["finite_output"]:
                        assert method["backend"] == 0 and method["precision"] == 2 and group["profile"] == "highrange" and n == 16384
                    assert item["checked_output_entries"] == n*batch
                for boundary, information in case["boundaries"].items():
                    assert boundary in ("captured_complete", "uncaptured_complete") and len(information["observations"]) == 30
                    assert information["actual_output_sha256"] == case["outputs"][information["output"]]["output_sha256"]
                    for observation in information["observations"]:
                        assert observation["cuda_ms"] > 0 and observation["wall_ms"] > 0
                    observations += 30
                assert set(case["boundaries"]) == {"captured_complete", "uncaptured_complete"}
                assert len(case["host_requests"]) == 30
                for i, request in enumerate(case["host_requests"]):
                    output = case["outputs"][request["output"]]
                    assert request["variant"] == i % 4 == output["variant"]
                    assert request["actual_output_sha256"] == output["output_sha256"] and request["wall_ms"] > 0 and request["cuda_ms"] > 0
                observations += 30
                if method["backend"] == 1 and group["profile"] == "noise":
                    stage_entries += stage_replay(case, group, generated[0][0], coefficients, z)
    print(f"{path.name}: {available}/{configurations} available; {observations} observations; {full_entries} full FFT entries, {samples} sampled entries and {stage_entries} retained GEMM entries replayed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="*", type=Path, default=[ROOT/"results"/f"fft-{name}.json" for name in ("quick", "a", "b")])
    args = parser.parse_args()
    manifest = json.loads((ROOT/"fixtures/fft/manifest.json").read_text())
    for name, expected in manifest["sources_sha256"].items():
        assert sha(ROOT/name) == expected, name
    for name, expected in manifest["files_sha256"].items():
        assert sha(ROOT/"fixtures/fft"/name) == expected, name
    for path in args.records:
        verify(path)


if __name__ == "__main__":
    main()
