"""PCA hindsight quality/cost frontiers, never a future-input error guarantee."""
import numpy as np

QUALITY_METRICS=('covariance_relative_error','maximum_scaled_eigenvalue_error','eigenvalues_relative_error',
                 'mean_standard_deviation_relative_error','subspace_projection_error')


def routes(case,bootstrap_ms):
    if not (case['available'] and case['successfully_executed'] and case['arithmetic_valid']):return {}
    setup=float(case['setup_wall_ms']+case['first_operation_wall_ms'])
    assert np.isfinite(setup) and setup>0 and np.isfinite(bootstrap_ms) and bootstrap_ms>=0
    result={}
    for boundary in case['boundaries']:
        name=boundary['name']
        assert name in ('resident','host-service') and name not in result
        observations=boundary['observations']
        qualities=[case['outputs'][key]['quality'] for key in sorted({row['output_key'] for row in observations})]
        samples=np.array([row['wall_ms'] if name=='host-service' else row['cuda_ms'] for row in observations],np.float64)
        assert len(samples)>0 and np.isfinite(samples).all() and (samples>0).all()
        finite=bool(qualities) and all(quality['finite'] for quality in qualities)
        errors={metric:max(quality[metric] for quality in qualities) for metric in QUALITY_METRICS} if finite else {}
        if finite:assert all(np.isfinite(value) and value>=0 for value in errors.values())
        result[name]=dict(boundary=name,median_ms=float(np.median(samples)),p25_ms=float(np.percentile(samples,25)),p75_ms=float(np.percentile(samples,75)),
                          setup_ms=setup,common_bootstrap_ms=float(bootstrap_ms),finite=finite,errors=errors,
                          quality_error=max(errors.values()) if finite else None,
                          negative_eigenvalues=max(quality['negative_eigenvalues'] for quality in qualities) if finite else None,
                          minimum_relative_eigengap=min(quality['target_relative_eigengap'] for quality in qualities) if finite else None,
                          named_owned_device_bytes=case['metadata']['owned_device_bytes'],memory_scope=case['metadata']['memory_scope'],
                          timed_output_count=len(qualities))
    return result


def feasible(route,budget):
    assert np.isfinite(budget) and budget>0
    return bool(route['finite'] and route['negative_eigenvalues']==0 and route['quality_error']<=budget)


def effective_ms(route,reuse,include_bootstrap):
    assert type(reuse) is int and reuse>0 and type(include_bootstrap) is bool
    return route['median_ms']+(route['setup_ms']+(route['common_bootstrap_ms'] if include_bootstrap else 0.))/reuse


def best(rows,budget,reuse,include_bootstrap,fp4):
    assert type(fp4) is bool
    assert len({row['boundary'] for row in rows})<=1
    eligible=[row for row in rows if (row['method']['precision']=='fp4')==fp4 and feasible(row,budget)]
    if not eligible:return None
    selected=min(eligible,key=lambda row:(effective_ms(row,reuse,include_bootstrap),row['case_id']))
    return {**selected,'effective_ms':effective_ms(selected,reuse,include_bootstrap)}
