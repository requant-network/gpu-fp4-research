"""Bounded stage witnesses for the complete native search, including global IDs."""
import numpy as np

from reference import MODES,quantize
from search_validation import norm_model


def capture_stage(plan,queries):
    base = plan.base
    database,allowed = plan.local_database,plan.local_allowed
    n,d = database.shape
    mode = plan.method["precision"]%5
    raw_q,scales_q,q = quantize(queries,MODES[mode])
    # Read current computed buffers before any load/re-execution.
    arrays = {"candidate_local_ids":base.snapshot(10).view(np.int64).reshape(len(queries),base.count),
              "candidate_scores":base.snapshot(11).view(np.float32).reshape(len(queries),base.count),
              "query_norm":base.snapshot(8).view(np.float32),
              "query_source":base.snapshot(1).view(np.float32).reshape(queries.shape),
              "query_encoded":base.snapshot(2)}
    np.testing.assert_array_equal(arrays["query_source"],queries)
    np.testing.assert_array_equal(arrays["query_encoded"],raw_q.view(np.uint8).ravel())
    if mode == 0:
        arrays["query_scales"] = base.snapshot(4)
        np.testing.assert_array_equal(arrays["query_scales"],scales_q.ravel())
    norm_mode = plan.method.get("norm_mode",1)
    np.testing.assert_array_equal(arrays["query_norm"],norm_model(q if norm_mode else queries))
    rng = np.random.Generator(np.random.PCG64(1701))
    pair_queries = np.concatenate((rng.integers(0,len(queries),128),np.arange(min(32,len(queries)))))
    pair_rows = np.concatenate((rng.integers(0,n,128),arrays["candidate_local_ids"][:min(32,len(queries)),0].clip(0)))
    pairs = pair_queries*n+pair_rows
    rows = np.unique(np.concatenate((pair_rows,np.array([0,n-1],dtype=np.int64))))
    raw_b,scales_b,b = quantize(database[rows],MODES[mode])
    arrays.update(sample_rows=rows,pair_queries=pair_queries,pair_rows=pair_rows)
    elements = (rows[:,None]*d+np.arange(d)).ravel()
    arrays["database_source"] = plan.gather(0,elements,np.float32).reshape(len(rows),d)
    np.testing.assert_array_equal(arrays["database_source"],database[rows])
    dtype = raw_b.dtype
    encoded_indices = (rows[:,None]*raw_b.shape[1]+np.arange(raw_b.shape[1])).ravel()
    arrays["database_encoded"] = plan.gather(3,encoded_indices,dtype).reshape(raw_b.shape)
    np.testing.assert_array_equal(arrays["database_encoded"],raw_b)
    if mode == 0:
        scale_indices = (rows[:,None]*(d//16)+np.arange(d//16)).ravel()
        arrays["database_scales"] = plan.gather(5,scale_indices,np.uint8).reshape(scales_b.shape)
        np.testing.assert_array_equal(arrays["database_scales"],scales_b)
    arrays["database_norm"] = plan.gather(7,rows,np.float32)
    np.testing.assert_array_equal(arrays["database_norm"],norm_model(b if norm_mode else database[rows]))
    arrays["allowed_rows"] = plan.gather(15,rows,np.uint8)
    np.testing.assert_array_equal(arrays["allowed_rows"],allowed[rows].astype(np.uint8))
    arrays["dot"] = plan.gather(6,pairs,np.float32)
    arrays["score"] = plan.gather(9,pairs,np.float32)
    selected = np.searchsorted(rows,pair_rows)
    lhs,rhs = q[pair_queries].astype(np.float64),b[selected].astype(np.float64)
    expected = np.einsum("ij,ij->i",lhs,rhs)
    magnitude = np.einsum("ij,ij->i",abs(lhs),abs(rhs))
    gamma = d*2.**-24/(1-d*2.**-24)
    bound = 8*gamma*magnitude+1e-6
    if not np.all(abs(arrays["dot"]-expected) <= bound):
        raise AssertionError("Sampled native DOT exceeds its FP32 accumulation envelope")
    expected_score = (arrays["query_norm"][pair_queries]+arrays["database_norm"][selected])-np.float32(2)*arrays["dot"]
    expected_score[~allowed[pair_rows]] = np.inf
    np.testing.assert_array_equal(arrays["score"],expected_score)
    local = arrays["candidate_local_ids"]
    valid = local >= 0
    if not np.all(local[valid] < n) or not allowed[local[valid]].all():
        raise AssertionError("Invalid candidate before global-ID mapping")
    global_ids = local.copy()
    if plan.device_mapping is not None:
        global_ids[valid] = plan.device_mapping[local[valid]]
    arrays["candidate_global_ids"] = global_ids
    if plan.method.get("refine",True):
        arrays["refined_scores"] = base.snapshot(12).view(np.float32).reshape(local.shape)
        for qi,row in enumerate(local):
            good = row >= 0
            delta = database[row[good]].astype(np.float64)-queries[qi].astype(np.float64)
            exact = np.einsum("ij,ij->i",delta,delta)
            np.testing.assert_allclose(arrays["refined_scores"][qi,good],exact,rtol=2e-6,atol=1e-30)
    distances,ids = plan.result()
    for qi,row in enumerate(ids):
        if len(np.unique(local[qi,valid[qi]])) != np.count_nonzero(valid[qi]):
            raise AssertionError("Duplicate native candidates")
        for j,id_ in enumerate(row):
            positions = np.flatnonzero(global_ids[qi] == id_) if id_ >= 0 else np.empty(0,dtype=int)
            if id_ >= 0:
                if len(positions) != 1:
                    raise AssertionError("Final global ID does not belong to the candidate set")
                source = arrays["refined_scores"] if plan.method.get("refine",True) else arrays["candidate_scores"]
                if distances[qi,j] != source[qi,positions[0]]:
                    raise AssertionError("Final distance does not belong to its candidate")
        if plan.method.get("refine",True):
            np.testing.assert_array_equal(distances[qi],np.sort(arrays["refined_scores"][qi])[:base.k])
        else:
            np.testing.assert_array_equal(ids[qi],global_ids[qi])
    return arrays,{"sampled_dot_entries":len(pairs),"database_rows_checked":len(rows),
                   "max_sampled_dot_bound_ratio":float(np.max(abs(arrays["dot"]-expected)/bound)),
                   "ranking_scope":"Complete candidate/final membership and refinement; sampled competitors and arithmetic. Full-score cutoffs are independently checked in the separate native kernel preflight."}
