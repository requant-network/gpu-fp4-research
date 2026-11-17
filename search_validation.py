"""Independent stage checks for native quantized candidate search."""
import numpy as np

from reference import MODES, check_output, digest, quantize
from search_reference import evaluate


def norm_model(values):
    """Exact FP32 order of the 32-lane squared-norm reduction."""
    chunks = values.reshape(len(values),-1,32)
    work = np.zeros((len(values),32),dtype=np.float32)
    for part in range(chunks.shape[1]):
        work += chunks[:,part]*chunks[:,part]
    for step in (16,8,4,2,1):
        work[:,:step] += work[:,step:2*step]
    return work[:,0]


def check_native(plan,database,queries,allowed,precision,norm_mode,truth_distances,truth_ids,rerank,arrays,prefix):
    batch,n = len(queries),len(database)
    raw_a,scale_a,a = quantize(queries,MODES[precision%5])
    raw_b,scale_b,b = quantize(database,MODES[precision%5])
    expected = {0:database,1:queries,2:raw_a,3:raw_b,15:allowed}
    if precision in (0,5):
        expected.update({4:scale_a,5:scale_b})
    hashes = {}
    for which,value in expected.items():
        actual = plan.snapshot(which)
        if actual.tobytes() != value.tobytes():
            raise AssertionError(f"Native input/encoding mismatch in buffer {which}")
        hashes[str(which)] = digest(actual)
    qn = plan.snapshot(8).view(np.float32)
    dn = plan.snapshot(7).view(np.float32)
    np.testing.assert_array_equal(qn,norm_model(a if norm_mode else queries))
    np.testing.assert_array_equal(dn,norm_model(b if norm_mode else database))
    dot = plan.snapshot(6).view(np.float32).reshape(batch,n)
    ratio = check_output(dot,a,b)
    score = plan.snapshot(9).view(np.float32).reshape(batch,n)
    target = (qn[:,None]+dn[None,:])-np.float32(2)*dot
    target[:,~allowed] = np.inf
    np.testing.assert_array_equal(score,target)
    candidates = plan.snapshot(10).view(np.int64).reshape(batch,plan.count)
    candidate_distances = plan.snapshot(11).view(np.float32).reshape(candidates.shape)
    for qi,row in enumerate(candidates):
        valid = row >= 0
        assert np.all(row[valid] < n) and allowed[row[valid]].all()
        assert len(np.unique(row[valid])) == np.count_nonzero(valid)
        assert np.count_nonzero(valid) == min(int(allowed.sum()),plan.count)
        np.testing.assert_array_equal(candidate_distances[qi,valid],score[qi,row[valid]])
        assert np.isinf(candidate_distances[qi,~valid]).all()
        threshold = np.partition(score[qi],plan.count-1)[plan.count-1]
        assert np.all(candidate_distances[qi] <= threshold)
    if rerank:
        gpu_refined = plan.snapshot(12).view(np.float32).reshape(candidates.shape)
        reference_refined = np.full(candidates.shape,np.inf,dtype=np.float64)
        for qi,row in enumerate(candidates):
            valid = row >= 0
            difference = database[row[valid]].astype(np.float64)-queries[qi].astype(np.float64)
            reference_refined[qi,valid] = np.einsum("ij,ij->i",difference,difference)
        np.testing.assert_allclose(gpu_refined,reference_refined,rtol=2e-6,atol=1e-30)
    output_distances,ids = plan.result()
    metrics,true_output_distances = evaluate(database,queries,ids,truth_distances,truth_ids,allowed)
    assert metrics["invalid_or_filtered_ids"] == metrics["duplicate_ids"] == 0
    candidate_metrics,_ = evaluate(database,queries,candidates,truth_distances,truth_ids,allowed)
    if rerank:
        np.testing.assert_allclose(output_distances,true_output_distances,rtol=2e-6,atol=1e-30)
        for qi,row in enumerate(ids):
            valid = row >= 0
            best = np.sort(gpu_refined[qi])[:plan.k]
            np.testing.assert_array_equal(output_distances[qi],best)
            for position in np.flatnonzero(valid):
                index = np.flatnonzero(candidates[qi] == row[position])
                assert len(index) == 1 and gpu_refined[qi,index[0]] == output_distances[qi,position]
    else:
        np.testing.assert_array_equal(ids,candidates)
        np.testing.assert_array_equal(output_distances,candidate_distances)
    for name,value in (("dot",dot),("database_norm",dn),("query_norm",qn),("score",score),("candidates",candidates),
                       ("candidate_distances",candidate_distances),("output_ids",ids),("output_distances",output_distances)):
        arrays[prefix+"_"+name] = value.copy()
    if rerank:
        arrays[prefix+"_refined_distances"] = gpu_refined.copy()
    return {"parameter_sha256":hashes,"full_dot_entries_checked":dot.size,"arithmetic_bound_ratio":ratio,
            "output_quality":metrics,"candidate_quality":candidate_metrics,"output_ids_sha256":digest(ids),
            "dot_sha256":digest(dot),"score_sha256":digest(score),"candidate_ids_sha256":digest(candidates)}
