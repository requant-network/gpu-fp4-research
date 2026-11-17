"""CPU-only independent replay primitives; no GPU/producer imports."""
import numpy as np


def independent_topk(database,queries,k,allowed,chunk=32768):
    """Direct FP64 differences, partitioned local top-k and explicit ID ties.

    Unlike the producer's lexsort over every streamed candidate, reduce each
    chunk with partition before a small lexicographic merge. Distances never
    use the cancellation-prone norm/dot identity.
    """
    if database.dtype!=np.float32 or queries.dtype!=np.float32 or database.ndim!=2 or queries.ndim!=2:
        raise ValueError('Expected FP32 vector matrices')
    if database.shape[1]!=queries.shape[1] or allowed.dtype!=np.bool_ or allowed.shape!=(len(database),) or k<1 or chunk<1:
        raise ValueError('Invalid geometry, mask or top-k')
    if not np.isfinite(database).all() or not np.isfinite(queries).all():
        raise ValueError('Finite vectors are required')
    out_ids=np.full((len(queries),k),-1,dtype=np.int64)
    out_distances=np.full((len(queries),k),np.inf,dtype=np.float64)
    for qi,query in enumerate(queries):
        best_ids=np.empty(0,dtype=np.int64)
        best_distances=np.empty(0,dtype=np.float64)
        for begin in range(0,len(database),chunk):
            ids=np.flatnonzero(allowed[begin:begin+chunk])+begin
            delta=database[ids].astype(np.float64)-query.astype(np.float64)
            distances=np.sum(delta*delta,axis=1,dtype=np.float64)
            if len(ids)>k:
                cutoff=np.partition(distances,k-1)[k-1]
                keep=distances<=cutoff
                ids,distances=ids[keep],distances[keep]
            order=np.lexsort((ids,distances))[:k]
            ids=np.concatenate((best_ids,ids[order]))
            distances=np.concatenate((best_distances,distances[order]))
            order=np.lexsort((ids,distances))[:k]
            best_ids,best_distances=ids[order],distances[order]
        out_ids[qi,:len(best_ids)]=best_ids
        out_distances[qi,:len(best_distances)]=best_distances
    return out_distances,out_ids


def validate_timing(case,variants,observations):
    """Check complete boundaries and each observation's query/output binding."""
    if not case['available']:
        assert not case['valid'] and not case['boundaries'] and 'error' in case
        return
    assert np.isfinite(case['setup_wall_ms']) and case['setup_wall_ms']>=0
    assert np.isfinite(case['first_run_wall_ms']) and case['first_run_wall_ms']>=0
    expected=['resident-uncaptured','host-service']
    if case['graph_available']:
        expected.insert(1,'resident-captured')
    assert [b['name'] for b in case['boundaries']]==expected
    for boundary in case['boundaries']:
        rows=boundary['observations']
        assert len(rows)==observations
        for i,row in enumerate(rows):
            assert row['variant']==i%variants
            assert row['output_key'] in case['outputs']
            assert case['outputs'][row['output_key']]['variant']==row['variant']
            assert np.isfinite(row['wall_ms']) and row['wall_ms']>0
            if boundary['name']=='host-service':
                assert row['cuda_ms'] is None
            else:
                assert np.isfinite(row['cuda_ms']) and row['cuda_ms']>0


def output_metrics(database,queries,reported,ids,gold_distances,gold_ids,allowed):
    """Recompute geometric errors/recall without the producer's evaluator."""
    assert ids.dtype==np.int64 and reported.dtype==np.float32
    assert ids.shape==reported.shape==gold_ids.shape==gold_distances.shape
    assert len(ids)==len(queries)
    actual=np.full(ids.shape,np.inf,dtype=np.float64)
    strict,tie,counts=[],[],[]
    duplicates=invalid=0
    for qi,row in enumerate(ids):
        good=(row>=0)&(row<len(database))
        positions=np.flatnonzero(good)
        good[positions]&=allowed[row[positions]]
        invalid+=int(np.sum((row!=-1)&~good))
        selected=row[good]
        unique=set(map(int,selected))
        duplicates+=len(selected)-len(unique)
        difference=database[selected].astype(np.float64)-queries[qi].astype(np.float64)
        actual[qi,good]=np.sum(difference*difference,axis=1,dtype=np.float64)
        truth=set(map(int,gold_ids[qi,gold_ids[qi]>=0]))
        count=len(truth)
        counts.append(count)
        strict.append(len(unique&truth)/count if count else 1.)
        cutoff=gold_distances[qi,count-1] if count else -1.
        accepted=set(map(int,row[good&(actual[qi]<=cutoff)]))
        tie.append(min(len(accepted),count)/count if count else 1.)
    in_range=(ids>=0)&(ids<len(database))
    finite=np.isfinite(reported)&in_range
    metrics=dict(recall=float(np.mean(strict)),tie_aware_recall=float(np.mean(tie)),duplicate_ids=duplicates,
                 invalid_or_filtered_ids=invalid,truth_counts=counts,missing_ids=int(np.sum(ids==-1)),
                 nonfinite_reported_valid_ids=int(np.sum(in_range&~np.isfinite(reported))),
                 max_absolute_distance_error=float(np.max(abs(reported[finite].astype(np.float64)-actual[finite]))) if finite.any() else None,
                 ids_valid=duplicates==invalid==0)
    return metrics,actual


def replay_norm(values):
    """Model the native lane-strided FP32 squared norm and shuffle tree."""
    lanes=np.zeros((len(values),32),np.float32)
    for offset in range(0,values.shape[1],32):
        lanes=np.add(lanes,np.multiply(values[:,offset:offset+32],values[:,offset:offset+32],dtype=np.float32),dtype=np.float32)
    width=16
    while width:
        lanes[:,:width]=np.add(lanes[:,:width],lanes[:,width:2*width],dtype=np.float32)
        width//=2
    return lanes[:,0]


def validate_stage(stage,record,method,database,queries,allowed,final):
    """Replay bytes, sampled arithmetic, candidate mapping and full refinement.

    Large full-score matrices are not retained: this verifies their sampled
    competitors and all selected-candidate score envelopes, not all large
    unselected ranks. Separate kernel-preflight records prove full small ranks.
    """
    from reference import MODES,quantize
    compact=method.get('compact',False)
    mapping=np.flatnonzero(allowed) if compact else None
    population=len(mapping) if compact else len(database)
    padded=max(128,(population+127)//128*128)
    rows=stage['sample_rows']
    assert rows.dtype==np.int64 and rows.ndim==1 and (rows>=0).all() and (rows<padded).all()
    def original(indices):
        valid=indices<population
        values=np.zeros((len(indices),database.shape[1]),np.float32)
        values[valid]=database[mapping[indices[valid]] if compact else indices[valid]]
        return values
    def permitted(indices):
        valid=indices<population
        result=np.zeros(len(indices),bool)
        result[valid]=True if compact else allowed[indices[valid]]
        return result
    mode=MODES[method['precision']%5]
    encoded_q,scale_q,q=quantize(queries,mode)
    source_b=original(rows)
    encoded_b,scale_b,b=quantize(source_b,mode)
    for key,expected in (('query_source',queries),('query_encoded',encoded_q.view(np.uint8).ravel()),('database_source',source_b),('database_encoded',encoded_b)):
        np.testing.assert_array_equal(stage[key],expected,err_msg=key)
    if method['precision'] in (0,5):
        np.testing.assert_array_equal(stage['query_scales'],scale_q.ravel())
        np.testing.assert_array_equal(stage['database_scales'],scale_b)
    use_encoded=method.get('norm_mode',1)!=0
    query_norm=replay_norm(q if use_encoded else queries)
    db_norm=replay_norm(b if use_encoded else source_b)
    np.testing.assert_array_equal(stage['query_norm'],query_norm)
    np.testing.assert_array_equal(stage['database_norm'],db_norm)
    np.testing.assert_array_equal(stage['allowed_rows'],permitted(rows).astype(np.uint8))
    candidates=stage['candidate_local_ids']
    assert candidates.dtype==np.int64 and candidates.shape==(len(queries),method['candidates'])
    good=candidates>=0
    assert (candidates[good]<padded).all() and (candidates>=-1).all()
    assert permitted(candidates[good]).all()
    assert (good.sum(axis=1)==min(method['candidates'],int(allowed.sum()))).all()
    global_ids=candidates.copy()
    if compact:
        global_ids[good]=mapping[candidates[good]]
    np.testing.assert_array_equal(stage['candidate_global_ids'],global_ids)
    rng=np.random.Generator(np.random.PCG64(1701))
    pair_q=np.concatenate((rng.integers(0,len(queries),128),np.arange(min(32,len(queries)))))
    pair_rows=np.concatenate((rng.integers(0,padded,128),candidates[:min(32,len(queries)),0].clip(0)))
    np.testing.assert_array_equal(stage['pair_queries'],pair_q)
    np.testing.assert_array_equal(stage['pair_rows'],pair_rows)
    np.testing.assert_array_equal(rows,np.unique(np.concatenate((pair_rows,np.array([0,padded-1],np.int64)))))
    selected=np.searchsorted(rows,pair_rows)
    lhs,rhs=q[pair_q].astype(np.float64),b[selected].astype(np.float64)
    product=np.sum(lhs*rhs,axis=1,dtype=np.float64)
    dimension=database.shape[1]
    gamma=dimension*2.**-24/(1-dimension*2.**-24)
    bound=8*gamma*np.sum(abs(lhs)*abs(rhs),axis=1,dtype=np.float64)+1e-6
    assert np.all(abs(stage['dot']-product)<=bound),'Sampled DOT outside its accumulation envelope'
    expected=np.float32(query_norm[pair_q]+db_norm[selected])-np.float32(2)*stage['dot']
    expected[~permitted(pair_rows)]=np.inf
    np.testing.assert_array_equal(stage['score'],expected)
    ratio=float(np.max(abs(stage['dot']-product)/bound))
    assert abs(ratio-record['max_sampled_dot_bound_ratio'])<=1e-8
    assert record['sampled_dot_entries']==len(pair_q) and record['database_rows_checked']==len(rows)
    distances,ids=final
    assert ids.shape==distances.shape==(len(queries),10)
    candidate_scores=stage['candidate_scores']
    assert candidate_scores.shape==candidates.shape
    assert np.all(candidate_scores[:,1:]>=candidate_scores[:,:-1])
    assert np.isinf(candidate_scores[~good]).all()
    for qi,row in enumerate(candidates):
        valid=good[qi]
        selected_rows=row[valid]
        assert len(np.unique(selected_rows))==len(selected_rows)
        originals=original(selected_rows)
        _,_,encoded=quantize(originals,mode)
        cn=replay_norm(encoded if use_encoded else originals)
        l=q[qi].astype(np.float64)
        r=encoded.astype(np.float64)
        dot=np.sum(r*l,axis=1,dtype=np.float64)
        dot_bound=8*gamma*np.sum(abs(r)*abs(l),axis=1,dtype=np.float64)+1e-6
        norm_sum=np.float32(query_norm[qi]+cn)
        predicted=norm_sum.astype(np.float64)-2*dot
        rounding=4*2.**-24*(abs(norm_sum.astype(np.float64))+2*abs(dot))+1e-6
        score_bound=2*dot_bound+rounding
        assert np.all(abs(candidate_scores[qi,valid].astype(np.float64)-predicted)<=score_bound)
        if method.get('refine',True):
            values=stage['refined_scores'][qi]
            delta=originals.astype(np.float64)-queries[qi].astype(np.float64)
            exact=np.sum(delta*delta,axis=1,dtype=np.float64)
            np.testing.assert_allclose(values[valid],exact,rtol=2e-6,atol=1e-30)
            assert np.isinf(values[~valid]).all()
            np.testing.assert_array_equal(distances[qi],np.sort(values)[:10])
        else:
            values=candidate_scores[qi]
            np.testing.assert_array_equal(ids[qi],global_ids[qi])
        for position,id_ in enumerate(ids[qi]):
            if id_<0:
                assert id_==-1 and np.isinf(distances[qi,position])
                continue
            positions=np.flatnonzero(global_ids[qi]==id_)
            assert len(positions)==1
            assert distances[qi,position]==values[positions[0]]
    return dict(sampled_dots=len(pair_q),candidate_ids=candidates.size,final_ids=ids.size)
