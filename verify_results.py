"""Replay public numerical witnesses and check campaign coverage/provenance on CPU."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from reference import (MODES, PROFILES, SHAPES, check_output, digest, inputs,
                       quantize, relative_error)

ROOT = Path(__file__).resolve().parent
SOURCES = ("reference.py", "gpu_study.py", "native/bridge.cu", "build.sh")
BOUNDARIES = ("prepacked", "online-a", "online-both")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_timings(timings, count):
    require(set(timings) == set(BOUNDARIES), "Timing boundary coverage changed")
    for boundary, values in timings.items():
        require(isinstance(values, list) and len(values) == count, f"Wrong observation count: {boundary}")
        require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and np.isfinite(value) and value > 0 for value in values), "Invalid duration")


def verify_record(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    require(record.get("schema_version") == 1 and record.get("complete") is True
            and record.get("quick") is False, "Expected a completed full campaign")
    require((record["warmups"], record["samples"], record["graph_repetitions"]) == (5, 30, 32),
            "Protocol observation counts changed")
    require(record["native_invalid_controls"] == 3, "Missing native rejection controls")
    require(record["reverse"] == (record["seed"] == 20261216), "Replication order control changed")
    require(record["environment"]["compute_capability"] == [12, 0], "Wrong GPU architecture")
    require(record["environment"]["gpu_exclusive"] is False, "Shared-system qualification missing")
    require(set(record["source_sha256"]) == set(SOURCES), "Unexpected source manifest")
    for name, expected in record["source_sha256"].items():
        require(sha256(ROOT / name) == expected, f"Executed source changed: {name}")
    require(re.fullmatch(r"[0-9a-f]{64}", record["binary_sha256"]) is not None, "Invalid binary digest")
    require(record["cutlass_revision"] == "b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd", "Unexpected CUTLASS pin")
    expected_cases = {(shape, profile, mode) for shape in SHAPES for profile in PROFILES for mode in MODES}
    seen, witness_count, output_entries = set(), 0, 0
    cached_key, a, b = None, None, None
    for case in record["cases"]:
        key = (tuple(case["shape"]), case["profile"], case["mode"])
        require(key in expected_cases and key not in seen, "Missing, duplicated or unexpected case")
        seen.add(key)
        shape, profile, mode = key
        if (shape, profile) != cached_key:
            a, b = inputs(record["seed"], shape, profile)
            cached_key = (shape, profile)
        require(case["input_sha256"] == [digest(a), digest(b)], "Input regeneration hash mismatch")
        quantized, encoded_hashes = [], {}
        for operand, original in enumerate((a, b)):
            raw, scales, decoded = quantize(original, mode)
            quantized.append(decoded)
            encoded_hashes[f"operand_{operand}_encoded"] = digest(raw)
            encoded_hashes[f"operand_{operand}_scales"] = digest(scales)
        for checks in (case["checks"], case["post_timing_checks"]):
            require(checks["encoded_sha256"] == encoded_hashes, "Encoded operand hash mismatch")
            require(np.isfinite(checks["arithmetic_max_bound_ratio"]) and
                    0 <= checks["arithmetic_max_bound_ratio"] <= 1, "Arithmetic check failed")
            for field in ("relative_frobenius_error", "maximum_absolute_error"):
                require(np.isfinite(checks[field]) and checks[field] >= 0, "Invalid error metric")
            require(checks["checked_output_entries"] == shape[0] * shape[1], "Output coverage mismatch")
            require(re.fullmatch(r"[0-9a-f]{64}", checks["output_sha256"]) is not None, "Invalid output digest")
        require(case["output_bitwise_repeatable"] ==
                (case["checks"]["output_sha256"] == case["post_timing_checks"]["output_sha256"]),
                "Repeatability flag mismatch")
        validate_timings(case["timings_ms"], record["samples"])
        require(1 <= len(case["tuning"]) <= 8, "Invalid candidate search coverage")
        require(case["algorithm_candidate"] in [row["candidate"] for row in case["tuning"]], "Missing selected algorithm")
        for row in case["tuning"]:
            require(len(row["ms"]) == 3 and all(np.isfinite(v) and v > 0 for v in row["ms"]), "Invalid tuning observations")
        if shape == (128, 128, 128):
            witness = case["witness"]
            require(witness["file"] == f"witness-{record['seed']}-{profile}-{mode}.npz", "Unexpected witness path")
            witness_path = path.parent / witness["file"]
            require(sha256(witness_path) == witness["sha256"], "Witness hash mismatch")
            with np.load(witness_path, allow_pickle=False) as arrays:
                require(set(arrays.files) == {"a", "b", "output"}, "Unexpected witness arrays")
                np.testing.assert_array_equal(arrays["a"], a)
                np.testing.assert_array_equal(arrays["b"], b)
                actual = arrays["output"]
                require(digest(actual) == case["checks"]["output_sha256"], "Witness output digest mismatch")
                check_output(actual, *quantized)
                expected = a.astype(np.float64) @ b.astype(np.float64).T
                require(np.isclose(relative_error(actual, expected),
                                   case["checks"]["relative_frobenius_error"], rtol=2e-6, atol=1e-12),
                        "Witness error metric mismatch")
            witness_count += 1
        output_entries += case["checks"]["checked_output_entries"]
    require(seen == expected_cases, "Campaign case coverage incomplete")
    require(witness_count == 10, "Small numerical witness coverage incomplete")
    return record, output_entries, witness_count


def main():
    paths = sorted((ROOT / "results").glob("pilot-*.json"))
    require(len(paths) == 2, "Expected exactly two retained pilot cohorts")
    seeds, entries, witnesses, cases = set(), 0, 0, 0
    manifests, binaries = [], set()
    for path in paths:
        record, count, witness_count = verify_record(path)
        seeds.add(record["seed"])
        manifests.append(record["source_sha256"])
        binaries.add(record["binary_sha256"])
        entries += count
        witnesses += witness_count
        cases += len(record["cases"])
    require(seeds == {20261116, 20261216}, "Independent cohort seeds missing")
    require(manifests[0] == manifests[1] and len(binaries) == 1, "Campaign source/binary identity changed")
    build_path = ROOT / "results/build.json"
    build = json.loads(build_path.read_text(encoding="utf-8"))
    require(build["binary_sha256"] in binaries, "Build binary binding mismatch")
    require(build["native_source_sha256"] == sha256(ROOT / "native/bridge.cu"), "Build source binding mismatch")
    require(build["cutlass_revision"] == "b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd"
            and build["cutlass_tracked_changes"] == "", "Upstream source was not the clean pin")
    print(f"Verified {cases} mode/shape/profile cases, {entries:,} runtime-checked outputs, "
          f"{witnesses} CPU-replayed numerical witnesses and all source/build bindings")


if __name__ == "__main__":
    main()
