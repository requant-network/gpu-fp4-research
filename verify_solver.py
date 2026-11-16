"""CPU-only replay of every retained solver output and exact residual witness."""
import gzip
import json
from pathlib import Path

import numpy as np

from reference import digest
from solver_reference import accuracy, exact_residual_rows
from solver_study import METHODS, ROOT, SOURCES, groups, sha, system


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_accuracy(a, b, truth, x, recorded):
    quality = accuracy(a, b, truth, x)
    for flag in ("finite", "target_pass"):
        require(quality[flag] == recorded[flag], f"Numerical classification changed: {flag}")
    if not quality["finite"]:
        require(all(recorded[key] is None for key in ("residual_per_rhs", "forward_per_rhs", "max_relative_residual", "max_forward_error")),
                "Nonfinite numerical outcome has finite metrics")
        return
    # CPU BLAS reduction order differs across platforms, especially for failed
    # solutions with enormous cancellation. Bound both FP64 dot roundoffs.
    gamma = len(a) * np.finfo(np.float64).eps / (1 - len(a) * np.finfo(np.float64).eps)
    bound = 4 * gamma * np.linalg.norm(np.abs(a) @ np.abs(x), axis=0) / np.linalg.norm(b, axis=0)
    for metric in ("residual_per_rhs", "max_relative_residual"):
        tolerance = bound if metric == "residual_per_rhs" else float(bound.max())
        difference = abs(np.asarray(quality[metric]) - np.asarray(recorded[metric]))
        require(np.all(difference <= tolerance + 3e-15), "Residual differs beyond FP64 reduction bound")
    for metric in ("forward_per_rhs", "max_forward_error"):
        np.testing.assert_allclose(quality[metric], recorded[metric], rtol=2e-10, atol=3e-15)


def verify_data():
    manifest = json.loads((ROOT / "fixtures/solver-matrices.json").read_text(encoding="utf-8"))
    for matrix in manifest["matrices"]:
        file = ROOT / matrix["file"]
        require(sha(file) == matrix["sha256"], "Compressed real matrix changed")
        import hashlib
        require(hashlib.sha256(gzip.decompress(file.read_bytes())).hexdigest() == matrix["decompressed_sha256"],
                "Decompressed real matrix changed")
        require(matrix["source_url"].startswith("https://math.nist.gov/pub/MatrixMarket2/"), "Unexpected data source")


def verify_preflight():
    record = json.loads((ROOT / "results/solver-preflight.json").read_text(encoding="utf-8"))
    require(record["complete"] and len(record["cases"]) == 60, "Incomplete capability probe")
    for file, expected in record["source_sha256"].items():
        require(sha(ROOT / file) == expected, f"Probe source changed: {file}")
    n = record["n"]
    truth = np.array([1. if i % 2 == 0 else -1. for i in range(n)])[:, None]
    for case in record["cases"]:
        diagonal = np.ldexp(np.ones(n), -(np.arange(n) // 4)) if case["profile"] else np.ones(n)
        a = np.full((n, n), 1. / 1024.) + np.diag(diagonal)
        b = diagonal[:, None] * truth
        x = np.array(case["solution"], dtype=np.float64)[:, None]
        result = accuracy(a, b, truth, x)
        require(case["status"] == 0 and case["device_info"] == 0 and result["finite"], "Probe status/finite-array mismatch")
        check_accuracy(a, b, truth, x, {"finite": True, "target_pass": result["target_pass"],
                       "residual_per_rhs": [case["relative_residual"]], "max_relative_residual": case["relative_residual"],
                       "forward_per_rhs": [case["forward_error"]], "max_forward_error": case["forward_error"]})
    return len(record["cases"]) * n


def verify_record(filename):
    record = json.loads((ROOT / "results" / filename).read_text(encoding="utf-8"))
    require(record["complete"] and record["study"] == 7, f"Incomplete solver campaign {filename}")
    config = record["configuration"]
    require(set(record["source_sha256"]) == set(SOURCES), "Source binding coverage")
    for field in ("source_sha256", "data_sha256"):
        for file, expected in record[field].items():
            require(sha(ROOT / file) == expected, f"Bound file changed: {file}")
    require(record["native_runtime"]["compute_capability"] == [12, 0] and record["gpu_index"] == 0, "Wrong target GPU")
    require(config["residual_target"] == 1e-12 and config["forward_target"] == 1e-8, "Accuracy contract changed")
    require(config["observations"] == (3 if config["quick"] else 30), "Timing observation count")
    require(config["warmups"] == (1 if config["quick"] else 5), "Warmup count")
    expected_groups = groups(config["quick"])
    if config["reverse"]:
        expected_groups.reverse()
    require([g["specification"] for g in record["groups"]] == expected_groups, "Matrix/RHS/scale grid coverage")
    require(len(record["controls"]) == 7, "Native control coverage")
    for control in record["controls"]:
        require(control.get("rejected") or control.get("passed"), "Failed negative/lifecycle control")
    require(len(record["controls"][-1]["solves"]) == 4, "IRS reuse regression coverage")
    count = {"cases": 0, "available": 0, "solutions": 0, "entries": 0, "exact_entries": 0, "observations": 0}
    expected_methods = list(reversed(METHODS)) if config["reverse"] else METHODS
    for gi, group in enumerate(record["groups"]):
        inputs = [system(group["specification"], config["seed"], variant) for variant in range(4)]
        a, b, truth = inputs[0]
        n, nrhs = b.shape
        require((n, nrhs) == (group["n"], group["nrhs"]), "Recorded input dimensions")
        for variant, (aa, bb, xx) in enumerate(inputs):
            require(group["input_sha256"][variant] == {"a": digest(aa), "b": digest(bb), "truth": digest(xx)}, "Input reconstruction hash")
            np.testing.assert_array_equal(exact_residual_rows(aa, bb, xx, group["exact_rows"]), 0)
        path = ROOT / "results" / group["witness_file"]
        require(path.name == group["witness_file"] and sha(path) == group["witness_sha256"], "Witness name/hash")
        with np.load(path, allow_pickle=False) as arrays:
            require(set(arrays.files) == set(group["solutions"]), "Full output witness coverage")
            for key, witness in group["solutions"].items():
                x = arrays[key]
                aa, bb, xx = inputs[witness["variant"]]
                require(x.shape == bb.shape and x.dtype == np.float64 and digest(x) == witness["sha256"], "Full solution bytes")
                check_accuracy(aa, bb, xx, x, witness["accuracy"])
                if np.isfinite(x).all():
                    exact = exact_residual_rows(aa, bb, x, group["exact_rows"])
                    if np.isfinite(exact).all():
                        np.testing.assert_array_equal(exact, witness["exact_residual_samples"])
                        count["exact_entries"] += exact.size
                    else:
                        require(witness["exact_residual_samples"] is None, "Overflow sample classification")
                count["solutions"] += 1
                count["entries"] += x.size
        require([case["method"] for case in group["cases"]] == expected_methods, "Solver method grid")
        for case in group["cases"]:
            count["cases"] += 1
            if not case["available"]:
                require(nrhs == 4 and case["method"]["name"] == "gmres_fp16" and "one RHS" in case["setup_error"],
                        "Unexplained missing required solver")
                continue
            count["available"] += 1
            preparation = case["preparation"]
            require(preparation["verified_entries"] == n*n + n*nrhs + n, "Full runtime preparation check coverage")
            for key in ("a_sha256", "b_sha256", "scale_sha256"):
                require(len(preparation[key]) == 64, "Prepared input hash missing")
            for boundary in ("resident_observations", "host_observations"):
                observations = case[boundary]
                require(len(observations) == config["observations"], "Incomplete timing series")
                for oi, observation in enumerate(observations):
                    require(observation["solution"] in group["solutions"], "Actual timed output missing")
                    variant = 0 if boundary == "resident_observations" else oi % 4
                    require(group["solutions"][observation["solution"]]["variant"] == variant, "Changing RHS witness mismatch")
                    for timer in (("cuda_ms", "wall_ms") if boundary == "resident_observations" else ("cuda_ms", "wall_ms", "host_ms")):
                        require(np.isfinite(observation[timer]) and observation[timer] > 0, "Nonpositive/nonfinite timing")
                    trace = observation["trace"]
                    require(1 <= len(trace) <= 3 and all(len(s) == 6 for s in trace), "Missing numerical/iteration trace")
                    for precision, status, info, iterations, accepted, residual in trace:
                        require(isinstance(iterations, int), "Raw iteration code missing")
                        require(not accepted or (status == 0 and info == 0 and residual is not None and residual <= 1e-12), "Unsafe recorded residual gate")
                    if case["method"].get("mode") == 2:
                        require(trace[0][0] == 1205, "Adaptive first precision")
                        require(len(trace) < 2 or trace[1][0] == 1204, "Adaptive second precision")
                        require(len(trace) < 3 or trace[2][0] == 1203, "Adaptive native fallback")
                        require(all(not s[4] for s in trace[:-1]), "Adaptive ignored a passing gate")
                    count["observations"] += 1
            for checkpoint in ("before", "after"):
                sample = case[checkpoint]
                require(sample["solution"] in group["solutions"] and group["solutions"][sample["solution"]]["variant"] == 0,
                        "Pre/post solution missing")
                for stage in sample["metadata"]["stages"]:
                    if "valid_history" in stage:
                        history = np.asarray(stage["valid_history"], dtype=np.float64)
                        require(stage["iterations"] > 0 and history.shape == (stage["reported_outer_iterations"] + 1, 2), "Residual history extent")
                        require(history[0, 0] == 0 and np.all(np.diff(history[:, 0]) >= 0), "Residual history iteration counts")
                        require(np.isfinite(history).all() and np.all(history[:, 1] >= 0), "Invalid residual history")
            if case["method"].get("mode") == 3:
                require(len(case["factor_observations"]) == config["observations"], "Separate reuse setup timing")
                require(all(o["trace"][0][1:3] == [0, 0] for o in case["factor_observations"]), "Native reuse factorization failed")
        print(f"{filename}: replayed group {gi+1}/{len(record['groups'])}", flush=True)
    return count


def main():
    verify_data()
    preflight = verify_preflight()
    counts = {}
    for file in ("solver-quick.json", "solver-a.json", "solver-b.json"):
        counts[file] = verify_record(file)
    print(json.dumps({"preflight_entries": preflight, "campaigns": counts}, indent=2))


if __name__ == "__main__":
    main()
