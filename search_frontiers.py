"""CPU-only ex-post frontiers; measured quality is not a predictive guarantee."""
import numpy as np


def routes(case):
    if not case['available'] or not case['valid']:
        return {}
    setup=case['setup_wall_ms']+case['first_run_wall_ms']+case.get('graph_capture_wall_ms',case.get('graph_capture_to_error_wall_ms',0.))
    result={}
    for boundary in case['boundaries']:
        name=boundary['name']
        if name=='resident-captured' and not case['graph_available']:
            continue
        outputs=[case['outputs'][key]['quality'] for key in sorted({row['output_key'] for row in boundary['observations']})]
        values=np.array([row['wall_ms'] if name=='host-service' else row['cuda_ms'] for row in boundary['observations']],np.float64)
        assert len(values) and np.isfinite(values).all() and (values>0).all()
        valid=all(o['ids_valid'] and not o['missing_ids'] and not o['nonfinite_reported_valid_ids'] for o in outputs)
        result[name]=dict(boundary=name,median_ms=float(np.median(values)),p25_ms=float(np.percentile(values,25)),p75_ms=float(np.percentile(values,75)),
                          recall=min(o['recall'] for o in outputs),tie_aware_recall=min(o['tie_aware_recall'] for o in outputs),
                          finite_complete_ids=valid,setup_ms=setup)
    resident=[row for name,row in result.items() if name.startswith('resident-')]
    if resident:
        result['best-resident']=min(resident,key=lambda row:row['median_ms']).copy()
    return result
