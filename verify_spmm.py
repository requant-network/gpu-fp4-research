"""CPU replay of SpMM layouts, encoded-buffer hashes, samples and full witnesses."""
import json

import numpy as np

from reference import digest
from spmm_formats import prepare, reconstruct_format
from spmm_preflight import SOURCES as PROBE_SOURCES, encode, inputs as probe_inputs
from spmm_reference import csr
from spmm_records import load_record
from spmm_study import METHODS, ROOT, SOURCES, coordinates, groups, inputs, keep_full, sha
from spmm_validation import expected_buffers, hybrid_product, metrics, sparse_product, whole_product


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def samples(source, shape, formatted, right, precision, coords, hybrid):
    """Independent row-dot samples and absolute-product rounding envelopes."""
    rounded_b = encode(right, precision)[1].astype(np.float64)
    rounded_values = encode(source["values"], precision)[1].astype(np.float64)
    if hybrid:
        inverse = np.empty(shape[0], dtype=np.int32)
        inverse[formatted["permutation"]] = np.arange(shape[0], dtype=np.int32)
        block = formatted["block"]
        tiles = encode(formatted["block_values"], precision)[1].reshape(-1, block, block).astype(np.float64)
    values, bounds = [], []
    for row, column in coords:
        if not hybrid:
            start, end = source["rowptr"][row:row+2]
            av = rounded_values[start:end]
            bv = rounded_b[source["columns"][start:end], column]
            value, magnitude, terms = float(av @ bv), float(abs(av) @ abs(bv)), end - start
        else:
            reordered = int(inverse[row])
            start, end = formatted["rowptr"][reordered:reordered+2]
            av = formatted["values"][start:end].astype(np.float64)
            bv = right[formatted["columns"][start:end], column].astype(np.float64)
            value, magnitude, terms = float(av @ bv), float(abs(av) @ abs(bv)), end - start
            for index in range(formatted["block_rowptr"][reordered//block], formatted["block_rowptr"][reordered//block+1]):
                begin = int(formatted["block_columns"][index]) * block
                count = min(block, shape[1] - begin)
                av = tiles[index, reordered % block, :count]
                bv = rounded_b[begin:begin+count, column]
                value += float(av @ bv)
                magnitude += float(abs(av) @ abs(bv))
                terms += count + 1
        gamma = max(1, int(terms)) * np.finfo(np.float32).eps / (1 - max(1, int(terms)) * np.finfo(np.float32).eps)
        values.append(value)
        bounds.append(8 * gamma * magnitude + 4 * np.finfo(np.float32).eps * abs(value) + 1e-30)
    return np.array(values), np.array(bounds)


def verify_probe():
    record = json.loads((ROOT / "results/spmm-preflight.json").read_text(encoding="utf-8"))
    require(record["complete"] and len(record["groups"]) == 6, "Incomplete CSR capability probe")
    require(set(record["source_sha256"]) == set(PROBE_SOURCES), "Probe source coverage")
    for field in ("source_sha256", "data_sha256"):
        for file, expected in record[field].items():
            require(sha(ROOT / file) == expected, f"Probe bound file changed: {file}")
    entries = cases = 0
    for group in record["groups"]:
        a, b = probe_inputs(group["matrix"], group["width"])
        require(digest(a) == group["a_sha256"] and digest(b) == group["b_sha256"], "Probe input reconstruction")
        path = ROOT / "results" / group["witness_file"]
        require(path.name == group["witness_file"] and sha(path) == group["witness_sha256"], "Probe witness hash/name")
        source = csr(a)
        golds = {p: whole_product(source, a.shape, b, p) for p in (0, 1, 2)}
        require(len(group["cases"]) == 36, "Probe algorithm/layout/preprocessing grid")
        with np.load(path, allow_pickle=False) as arrays:
            require(set(arrays.files) == set(group["outputs"]), "Probe witness coverage")
            for key, witness in group["outputs"].items():
                out = arrays[key]
                require(out.dtype == np.float32 and out.shape == (len(a), b.shape[1]) and digest(out) == witness["sha256"], "Probe full output bytes")
                expected = golds[witness["precision"]] * (-1 if witness["variant"] else 1)
                error = float(np.linalg.norm(out.astype(np.float64) - expected))
                require(error <= witness["arithmetic_bound"], "Probe rounded-input arithmetic failure")
                np.testing.assert_allclose(error, witness["error_quantized_gold"], rtol=1e-6, atol=2e-12)
                entries += out.size
            for case in group["cases"]:
                require(case["available"] and len(case["observations"]) == 5, "Missing available CSR probe")
                expected = golds[case["precision"]]
                for observation in case["observations"]:
                    out = arrays[observation["output"]]
                    require(np.linalg.norm(out.astype(np.float64) - expected) <= 2e-5 * max(1, np.linalg.norm(expected)), "Probe timed output arithmetic")
                out = arrays[case["changed_b_output"]]
                require(np.linalg.norm(out.astype(np.float64) + expected) <= 2e-5 * max(1, np.linalg.norm(expected)), "Probe changed B arithmetic")
                cases += 1
    return {"cases": cases, "full_entries": entries}


def verify_record(filename):
    record = load_record(ROOT / "results" / filename)
    require(record["complete"] and record["study"] == 8, "Incomplete full/hybrid campaign")
    config = record["configuration"]
    require(set(record["source_sha256"]) == set(SOURCES), "Executed source coverage")
    for field in ("source_sha256", "data_sha256"):
        for file, expected in record[field].items():
            require(sha(ROOT / file) == expected, f"SpMM bound file changed: {file}")
    require(record["native_runtime"]["compute_capability"] == [12, 0] and record["gpu_index"] == 0, "Wrong target architecture")
    require(config["error_budgets"] == [1e-5, 1e-3, 1e-2], "Quality budgets changed")
    require(config["observations"] == (3 if config["quick"] else 30) and config["host_observations"] == (4 if config["quick"] else 30), "Observation protocol changed")
    expected_groups = groups(config["quick"])
    if config["reverse"]:
        expected_groups.reverse()
    require([g["specification"] for g in record["groups"]] == expected_groups, "Shape/profile/order coverage")
    require(len(record["controls"]) == 4 and all(c.get("zero_matrix_graphs_pass") or c.get("null_plan_rejected") for c in record["controls"]), "Native zero/invalid controls")
    expected_methods = list(reversed(METHODS)) if config["reverse"] else METHODS
    totals = {"cases": 0, "available": 0, "full_entries": 0, "sample_entries": 0, "runtime_checked_entries": 0, "observations": 0}
    for gi, group in enumerate(record["groups"]):
        spec = group["specification"]
        a, rights = inputs(spec, config["seed"])
        source, shape = csr(a), a.shape
        require(list(shape) == group["shape"] and digest(a) == group["a_sha256"] and [digest(b) for b in rights] == group["b_sha256"], "Full input reconstruction")
        require({key: digest(value) for key, value in source.items()} == group["csr_sha256"] and len(source["values"]) == group["nnz"], "Canonical source CSR")
        coords = coordinates((len(a), spec["width"]))
        np.testing.assert_array_equal(coords, group["coordinates"])
        require([case["method"] for case in group["cases"]] == expected_methods, "Method/grid order coverage")
        path = ROOT / "results" / group["witness_file"]
        require(path.name == group["witness_file"] and sha(path) == group["witness_sha256"], "Full witness file hash/name")
        formats, whole_samples, full_originals, full_models, used_witnesses = {}, {}, {}, {}, set()
        with np.load(path, allow_pickle=False) as arrays:
            for case in group["cases"]:
                totals["cases"] += 1
                method = case["method"]
                if case["format"] not in formats:
                    f = prepare(source, shape, method, spec["order"], config["seed"])
                    np.testing.assert_array_equal(reconstruct_format(f), a)
                    data = group["formats"][case["format"]]
                    require(data["array_sha256"] == {key: digest(value) for key, value in f.items() if isinstance(value, np.ndarray)}, "Format source bytes")
                    require(data["host_format_bytes"] == sum(value.nbytes for value in f.values() if isinstance(value, np.ndarray)), "CPU format memory accounting")
                    require(data["conversion_wall_ms"] > 0 and data["lossless_entries_checked"] == a.size, "CPU conversion/losslessness evidence")
                    formats[case["format"]] = f
                f = formats[case["format"]]
                if not case["available"]:
                    require("setup_error" in case and ("status" in case["setup_error"] or "No usable cuBLASLt candidate" in case["setup_error"]), "Unexplained unavailable method")
                    continue
                totals["available"] += 1
                require(case["plan_setup_wall_ms"] > 0 and case["graph_setup_wall_ms"] > 0, "Setup measurements missing")
                native = case["native"]
                require(native["owned_device_bytes"] > 0, "Owned memory missing")
                direct = bool(np.array_equal(f["permutation"], np.arange(len(a))) and method.get("layout", 0) == 0
                              and (method["kind"] != 1 or f["block_count"] == 0))
                require(native["direct_output"] == direct and native["packing"] == "shared_transpose32", "Unexpected output/packing implementation")
                repeats = 2 if config["quick"] else (1 if int(shape[0])*shape[1]*spec["width"] >= 2**30 else 8)
                require(native["graph_repetitions"] == repeats, "Graph operation count")
                if method["kind"] == 4 or method["kind"] == 1 and f["block_count"]:
                    require(native["lt_selected_index"] >= 0, "Dense implementation not selected")
                    if method["precision"]:
                        require(native["lt_numerical_flags"] & 2, "Tensor Core implementation flag missing")
                for stage in ("prepared_before", "prepared_after_device", "prepared_after_host"):
                    preparation = case[stage]
                    wanted = expected_buffers(f, rights[preparation["variant"]], method["precision"], method.get("layout", 0))
                    require(preparation["bytes_checked"] == sum(v.size for v in wanted) and preparation["buffer_sha256"] == [digest(v) for v in wanted], "GPU prepared/source buffer hash differs from independent CPU encoding")
                require(case["prepared_before"]["variant"] == 0 and case["prepared_after_device"]["variant"] == 0 and case["prepared_after_host"]["variant"] == (config["host_observations"]-1) % 4, "Prepared changing-input boundary")
                opaque = method["kind"] == 3 and method["precision"] == 0
                sampled = {}
                for key, check in case["checks"].items():
                    variant = check["variant"]
                    observed = np.array(check["samples"], dtype=np.float32)
                    require(observed.shape == (64,) and np.isfinite(observed).all(), "Missing/nonfinite actual samples")
                    sample_key = (method["precision"], variant)
                    cache = sampled if method["kind"] == 1 else whole_samples
                    if sample_key not in cache:
                        cache[sample_key] = samples(source, shape, f, rights[variant], method["precision"], coords, method["kind"] == 1)
                    expected, bound = cache[sample_key]
                    if not opaque:
                        require(np.all(abs(observed.astype(np.float64) - expected) <= bound), "Actual output samples exceed FP32 dot-rounding envelope")
                    else:
                        require(not check["metrics"]["arithmetic_model_checked"], "Opaque ELL mislabeled as exact rounded arithmetic")
                    zero_indices = coords[:, 1] == 0
                    require(np.all(observed[zero_indices] == 0), "Exact-zero sampled column corrupted")
                    require(check["full_output_entries_checked"] == len(a)*spec["width"] and len(check["output_sha256"]) == 64, "Full runtime output check extent/hash")
                    require(check["metrics"]["arithmetic_model_checked"] == (not opaque) and check["metrics"]["exact_zero_columns_checked"] == 1, "Arithmetic/zero contract mislabeled")
                    for name in ("relative_frobenius_error", "max_relative_column_error", "relative_error_vs_product_model"):
                        require(np.isfinite(check["metrics"][name]) and check["metrics"][name] >= 0, "Nonfinite/negative full error metric")
                    if keep_full(spec, method):
                        witness_key = check["witness_key"]
                        output = arrays[witness_key]
                        used_witnesses.add(witness_key)
                        require(output.dtype == np.float32 and output.shape == (len(a), spec["width"]) and digest(output) == check["output_sha256"], "Complete retained output bytes")
                        np.testing.assert_array_equal(output[coords[:, 0], coords[:, 1]], observed)
                        if variant not in full_originals:
                            full_originals[variant] = sparse_product(source, shape, rights[variant])
                        if method["kind"] == 1:
                            model = hybrid_product(f, rights[variant], method["precision"])
                        else:
                            if sample_key not in full_models:
                                full_models[sample_key] = whole_product(source, shape, rights[variant], method["precision"])
                            model = full_models[sample_key]
                        quality = metrics(output, full_originals[variant], model, opaque)
                        for name in ("relative_frobenius_error", "max_relative_column_error", "relative_error_vs_product_model"):
                            np.testing.assert_allclose(quality[name], check["metrics"][name], rtol=2e-6, atol=3e-12)
                        totals["full_entries"] += output.size
                    else:
                        require("witness_key" not in check, "Unexpected undeclared full witness")
                    totals["runtime_checked_entries"] += check["full_output_entries_checked"]
                    totals["sample_entries"] += observed.size
                for boundary in ("prepared", "online-b", "online-both"):
                    require(len(case["timings_ms"][boundary]) == config["observations"] and len(case["graph_wall_ms"][boundary]) == config["observations"], "Incomplete device/host graph timing observations")
                    require(all(np.isfinite(x) and x > 0 for x in case["timings_ms"][boundary] + case["graph_wall_ms"][boundary]), "Invalid timing")
                    require(case["after_device"][boundary] in case["checks"] and case["checks"][case["after_device"][boundary]]["variant"] == 0, "Last actual timed result missing")
                    totals["observations"] += config["observations"]
                require(len(case["host_observations"]) == config["host_observations"], "Incomplete host series")
                for oi, observation in enumerate(case["host_observations"]):
                    require(observation["variant"] == oi % 4 and case["checks"][observation["check"]]["variant"] == oi % 4, "Changed input/output witness mismatch")
                    require(observation["wall_ms"] > 0 and np.isfinite(observation["wall_ms"]), "Invalid host latency")
                    totals["observations"] += 1
            require(set(arrays.files) == used_witnesses, "Orphan/missing complete witnesses")
        require(any(c["available"] and c["method"]["kind"] == 1 for c in group["cases"]), "Proposed hybrid never ran")
        for family in (0, 2, 3, 4):
            require(any(c["available"] and c["method"]["kind"] == family for c in group["cases"]), "Required native reference family never ran")
        print(f"{filename}: verified {gi+1}/{len(record['groups'])} groups", flush=True)
    return totals


def main():
    print("CSR probe:", verify_probe(), flush=True)
    for file in ("spmm-quick.json", "spmm-a.json", "spmm-b.json"):
        print(file, verify_record(file), flush=True)


if __name__ == "__main__":
    main()
