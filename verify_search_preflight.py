"""CPU replay of the actual Faiss/cuVS capability and filter probe."""
import hashlib
import json
from pathlib import Path

import numpy as np

from reference import digest
from search_reference import evaluate, reference_search, refine

ROOT = Path(__file__).resolve().parent


def main():
    record = json.loads((ROOT/"results/search-preflight.json").read_text())
    assert record["complete"] and record["study"] == 10 and len(record["cases"]) == 23
    assert record["compute_capability"] == [12, 0]
    for name, expected in record["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    path = ROOT/"results"/record["witness_file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["witness_sha256"]
    checked = available = 0
    rng = np.random.Generator(np.random.PCG64(20261127))
    expected_data = rng.integers(-2048, 2049, (8192, 128)).astype(np.float32)/4096
    expected_queries = rng.integers(-2048, 2049, (4, 128)).astype(np.float32)/4096
    with np.load(path,allow_pickle=False) as witness:
        data, queries = witness["database"], witness["queries"]
        np.testing.assert_array_equal(data,expected_data)
        np.testing.assert_array_equal(queries,expected_queries)
        assert digest(data) == record["database_sha256"] and digest(queries) == record["queries_sha256"]
        for case in record["cases"]:
            prefix = case["prefix"]
            allowed = witness[prefix+"_allowed"]
            np.testing.assert_array_equal(allowed,np.arange(len(data)) % 10 < round(case["allowed_fraction"]*10))
            gold_distances,gold_ids = reference_search(data,queries,10,allowed,chunk=1024)
            np.testing.assert_array_equal(gold_ids,witness[prefix+"_truth_ids"])
            np.testing.assert_array_equal(gold_distances,witness[prefix+"_truth_distances"])
            unsupported = case["family"] == "cuvs-ivf-pq" and case["allowed_fraction"] != 1.
            assert case["available"] != unsupported
            if unsupported:
                assert "unexpected keyword argument 'filter'" in case["error"]
                continue
            available += 1
            ids = witness[prefix+"_ids"]
            metrics,distances = evaluate(data,queries,ids,gold_distances,gold_ids,allowed)
            assert metrics == case["candidate_set_quality"]
            np.testing.assert_array_equal(distances,witness[prefix+"_actual_distances"])
            refined_distances,refined_ids = refine(data,queries,ids,10,allowed)
            np.testing.assert_array_equal(refined_ids,witness[prefix+"_refined_ids"])
            np.testing.assert_array_equal(refined_distances,witness[prefix+"_refined_distances"])
            refined_metrics,_ = evaluate(data,queries,refined_ids,gold_distances,gold_ids,allowed)
            assert refined_metrics == case["refined_quality"]
            reported = witness[prefix+"_reported_distances"]
            assert np.isfinite(reported).all()
            if case["family"] != "cuvs-ivf-pq":
                error = np.linalg.norm(reported.astype(np.float64)-distances)/np.linalg.norm(distances)
                assert error < (5e-3 if case["precision"] == "fp16" else 1e-4)
            assert digest(ids) == case["ids_sha256"]
            assert digest(refined_ids) == case["refined_ids_sha256"]
            assert digest(reported) == case["reported_distances_sha256"]
            assert len(case["api_wall_ms"]) == 3 and all(ms > 0 for ms in case["api_wall_ms"])
            checked += ids.size+refined_ids.size
    assert available == 19
    print(f"Search capability probe: {available}/23 available; {checked} candidate/refined IDs, complete FP64 distances, masks, recalls and source/witness hashes replayed")


if __name__ == "__main__":
    main()
