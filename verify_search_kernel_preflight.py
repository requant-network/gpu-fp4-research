"""CPU replay of native search stages, ranking envelopes and filtered corner cases."""
import hashlib
import json
from pathlib import Path

import numpy as np

from reference import MODES,digest,quantize
from search_kernel_preflight import inputs
from search_reference import evaluate,reference_search
from search_validation import norm_model

ROOT = Path(__file__).resolve().parent


def main():
    record = json.loads((ROOT/"results/search-kernel-preflight.json").read_text())
    assert record["complete"] and len(record["groups"]) == 36 and record["study"] == 10
    for name,expected in record["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected,name
    assert hashlib.sha256((ROOT/"results"/record["environment_file"]).read_bytes()).hexdigest() == record["environment_sha256"]
    data_cache,quant_cache,dot_cache,truth_cache = {},{},{},{}
    available = sampled = full = output_ids = 0
    for group in record["groups"]:
        geometry = group["n"],group["d"],group["batch"]
        n,d,batch = geometry
        if geometry not in data_cache:
            data_cache[geometry] = inputs(*geometry)
        database,queries = data_cache[geometry]
        assert digest(database) == group["database_sha256"] and digest(queries) == group["queries_sha256"]
        mask = group["mask"]
        allowed = np.arange(n)%10<3 if mask == "fraction30" else np.arange(n)<(4 if mask == "four" else 0)
        assert digest(allowed) == group["allowed_sha256"]
        assert group["available"],group.get("error")
        available += 1
        precision = group["precision"]
        quant_key = (*geometry,precision%5)
        if quant_key not in quant_cache:
            raw,scales,b = quantize(database,MODES[precision%5])
            quant_cache[quant_key] = (digest(raw),digest(scales),b,norm_model(b))
        raw_b_sha,scale_b_sha,b,norm_b = quant_cache[quant_key]
        original_a = quantize(queries,MODES[precision%5])[2]
        if quant_key not in dot_cache:
            exact = original_a.astype(np.float64)@b.astype(np.float64).T
            magnitude = abs(original_a.astype(np.float64))@abs(b.astype(np.float64)).T
            gamma = d*2.**-24/(1-d*2.**-24)
            dot_cache[quant_key] = exact,8*gamma*magnitude+1e-6
        exact,arithmetic_bound = dot_cache[quant_key]
        path = ROOT/"results"/group["witness_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == group["witness_sha256"]
        meta = group["metadata"]
        assert meta["fp32_input_alias"] == (precision == 4)
        assert meta["selection_initial_workspace_pool_bytes"] == 64 << 20
        assert meta["owned_device_bytes"] > 0 and group["stale_output_rejected"]
        flags = meta["lt_candidate_numerical_flags"]
        if precision in (2,3):
            assert flags[meta["lt_selected_index"]] & 2
        if flags:
            assert len(flags) == len(meta["lt_candidate_ms"]) == len(meta["lt_candidate_states"])
            assert meta["lt_candidate_ms"][meta["lt_selected_index"]] > 0
        with np.load(path,allow_pickle=False) as witness:
            for variant,sign in (("original",1),("changed",-1)):
                q = np.ascontiguousarray(queries*sign)
                raw_a,scale_a,a = quantize(q,MODES[precision%5])
                item = group[variant]
                hashes = {"0":digest(database),"1":digest(q),"2":digest(raw_a),"3":raw_b_sha,"15":digest(allowed)}
                if precision in (0,5):
                    hashes.update({"4":digest(scale_a),"5":scale_b_sha})
                assert hashes == item["parameter_sha256"]
                dn,qn = witness[variant+"_database_norm"],witness[variant+"_query_norm"]
                np.testing.assert_array_equal(dn,norm_b if group["norm_mode"] else norm_model(database))
                np.testing.assert_array_equal(qn,norm_model(a if group["norm_mode"] else q))
                actual_dot,score = witness[variant+"_dot"],witness[variant+"_score"]
                if group["full_dot_score_witnesses"]:
                    assert digest(actual_dot) == item["dot_sha256"] and digest(score) == item["score_sha256"]
                    assert np.all(abs(actual_dot-exact*sign) <= arithmetic_bound)
                    expected = (qn[:,None]+dn[None,:])-np.float32(2)*actual_dot
                    expected[:,~allowed] = np.inf
                    np.testing.assert_array_equal(score,expected)
                    full += actual_dot.size+score.size
                else:
                    indices = witness["dot_indices"]
                    assert np.all(abs(actual_dot-(exact*sign).ravel()[indices]) <= arithmetic_bound.ravel()[indices])
                    qi,di = indices//n,indices % n
                    expected = (qn[qi]+dn[di])-np.float32(2)*actual_dot
                    expected[~allowed[di]] = np.inf
                    np.testing.assert_array_equal(score,expected)
                    sampled += 2*len(indices)
                candidates = witness[variant+"_candidates"]
                candidate_distances = witness[variant+"_candidate_distances"]
                ids,distances = witness[variant+"_output_ids"],witness[variant+"_output_distances"]
                assert digest(candidates) == item["candidate_ids_sha256"] and digest(ids) == item["output_ids_sha256"]
                truth_key = (*geometry,mask,sign)
                if truth_key not in truth_cache:
                    truth_cache[truth_key] = reference_search(database,q,10,allowed,chunk=1024)
                gold,truth_ids = truth_cache[truth_key]
                metrics,true_distances = evaluate(database,q,ids,gold,truth_ids,allowed)
                assert metrics == item["output_quality"]
                candidate_metrics,_ = evaluate(database,q,candidates,gold,truth_ids,allowed)
                assert candidate_metrics == item["candidate_quality"]
                model = (qn[:,None]+dn[None,:]).astype(np.float64)-2*exact*sign
                score_bound = 2*arithmetic_bound+np.finfo(np.float32).eps*abs(model)+1e-30
                model[:,~allowed] = np.inf
                for qi,row in enumerate(candidates):
                    valid = row >= 0
                    assert np.count_nonzero(valid) == min(allowed.sum(),group["candidates"])
                    assert np.all(abs(candidate_distances[qi,valid]-model[qi,row[valid]]) <= score_bound[qi,row[valid]])
                    if group["full_dot_score_witnesses"]:
                        np.testing.assert_array_equal(candidate_distances[qi,valid],score[qi,row[valid]])
                        cutoff = np.partition(score[qi],group["candidates"]-1)[group["candidates"]-1]
                    else:
                        upper = model[qi]+score_bound[qi]
                        cutoff = np.partition(upper,group["candidates"]-1)[group["candidates"]-1]
                    assert np.all(candidate_distances[qi] <= cutoff)
                if group["refine"]:
                    refined = witness[variant+"_refined_distances"]
                    for qi,row in enumerate(candidates):
                        valid = row >= 0
                        difference = database[row[valid]].astype(np.float64)-q[qi].astype(np.float64)
                        expected = np.einsum("ij,ij->i",difference,difference)
                        np.testing.assert_allclose(refined[qi,valid],expected,rtol=2e-6,atol=1e-30)
                        assert np.isinf(refined[qi,~valid]).all()
                        np.testing.assert_array_equal(distances[qi],np.sort(refined[qi])[:10])
                        for position in np.flatnonzero(ids[qi] >= 0):
                            index = np.flatnonzero(row == ids[qi,position])
                            assert len(index) == 1 and refined[qi,index[0]] == distances[qi,position]
                    np.testing.assert_allclose(distances,true_distances,rtol=2e-6,atol=1e-30)
                else:
                    np.testing.assert_array_equal(ids,candidates)
                    np.testing.assert_array_equal(distances,candidate_distances)
                output_ids += ids.size
            assert len(group["timings"]) == 2
            for boundary in group["timings"]:
                assert len(boundary["observations"]) == 5
                assert all(o["cuda_ms"] > 0 and o["wall_ms"] > 0 for o in boundary["observations"])
                assert boundary["actual_ids_sha256"] == digest(witness["original_output_ids"])
                assert boundary["actual_distances_sha256"] == digest(witness["original_output_distances"])
    print(f"Native search probe: {available}/36 available; {full} full and {sampled} sampled dot/score entries; {output_ids} final IDs, global ranking envelopes, norms, encodings, refinement and filtered controls replayed")


if __name__ == "__main__":
    main()
