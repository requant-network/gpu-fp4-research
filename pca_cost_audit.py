"""Independent named allocation geometry and heuristic-selection audit.

This validates requests represented by the fixed PCA adapter, not physical peak,
opaque cuSOLVER/cuBLAS internals, allocator cache, temporaries or instruction traces.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from pca_cases import DATASETS,methods
from verify_pca import PRODUCERS
from pca_provenance import GUARDED_ADDITIONS,validate_sources,verify_guarded_observations

ROOT=Path(__file__).resolve().parent


def audit_case(record):
    assert record['available'] and record['successfully_executed'] and record['arithmetic_valid']
    n,d=record['shape']
    method,metadata=record['method'],record['metadata']
    assert d in (128,256) and n>=2 and method['family'] in ('stream','batch')
    assert method['precision'] in ('fp4','fp8','fp16','bf16','fp32','fp64')
    assert method['accumulation'] in ('fp32','fp64')
    chunk=method['chunk']
    assert type(chunk) is int and (chunk==n if method['family']=='batch' else chunk in (256,512,1024,2048,4096,8192))
    source_element=8 if method['precision']=='fp64' else 4
    accumulated_element=8 if method['family']=='batch' or method['accumulation']=='fp64' else 4
    # Eight distinct live named arrays in metadata(): original FP32 input,
    # transposed/padded source, Gram, FP64 mean, moment, covariance, values, vectors.
    python=4*n*d+source_element*d*chunk+source_element*d*d+8*d+3*accumulated_element*d*d+accumulated_element*d
    if method['family']=='batch' or method['precision']=='fp64':native=0
    elif method['precision']=='fp4':
        # Both packed operands, both logical flat SF arrays, both physical
        # swizzled SF arrays. Fixed aligned SM120 configuration has zero workspace.
        native=5*d*chunk//4
    else:
        native=2*d*chunk*{'fp8':1,'fp16':2,'bf16':2,'fp32':4}[method['precision']]+(32<<20)
    assert metadata['python_owned_device_bytes']==python,'Python allocation geometry mismatch'
    assert metadata['native_owned_device_bytes']==native,'Native allocation geometry mismatch'
    assert metadata['owned_device_bytes']==python+native,'Combined allocation geometry mismatch'
    assert metadata['host_original_bytes']==4*n*d
    assert metadata['memory_scope'].endswith('not physical peak') and 'excludes' in metadata['memory_scope']
    for field in ('setup_wall_ms','first_operation_wall_ms'):assert math.isfinite(record[field]) and record[field]>0
    calibration=0
    tuning=metadata['tuning']
    if native==0:
        assert tuning==[] and metadata['selected_candidate']==0
    else:
        assert tuning and len({row['candidate'] for row in tuning})==len(tuning)
        available=[]
        for row in tuning:
            assert type(row['available']) is bool and type(row['candidate']) is int and type(row['flags']) is int
            if row['available']:
                assert len(row['cuda_ms'])==3 and all(math.isfinite(value) and value>0 for value in row['cuda_ms'])
                if method['precision'] in ('fp16','bf16'):assert row['flags']&2
                available.append(row)
                calibration+=len(row['cuda_ms'])
            else:
                assert method['precision'] in ('fp16','bf16') and not row['flags']&2
        assert available
        selected=min(available,key=lambda row:statistics.median(row['cuda_ms']))
        assert selected['candidate']==metadata['selected_candidate'],'Selected heuristic is not the measured eligible minimum'
    return dict(cases=1,python_owned_device_bytes=python,native_owned_device_bytes=native,
                named_owned_device_bytes=python+native,calibration_observations=calibration)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--summary',type=Path)
    args=parser.parse_args()
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in PRODUCERS}
    counts=Counter()
    geometries=set()
    named_sizes=[]
    collections={}
    epochs={}
    for scope,cohort in (('smoke','a'),('full','a'),('full','b')):
        directory=ROOT/'results'/f'pca-{scope}-{cohort}'
        datasets=[next(iter(DATASETS))] if scope=='smoke' else list(DATASETS)
        assert {path.name for path in directory.glob('*-collection.json')}=={name+'-collection.json' for name in datasets}
        for name in datasets:
            path=directory/(name+'-collection.json')
            manifest=json.loads(path.read_text())
            assert manifest['complete']
            guarded=validate_sources(manifest['source_sha256'])
            assert len(manifest['cases'])==len(methods())+2
            collections[directory.name+'/'+path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            epochs[directory.name+'/'+path.name]='guarded' if guarded else 'original'
            for item in manifest['cases']:
                case_path=directory/item['file']
                assert hashlib.sha256(case_path.read_bytes()).hexdigest()==item['sha256']
                record=json.loads(case_path.read_text())
                assert record['source_sha256']==manifest['source_sha256']
                counts['guarded_cases']+=int(guarded)
                counts['guarded_input_observations']+=verify_guarded_observations(record,guarded)
                result=audit_case(record)
                counts['cases']+=1
                counts['calibration_observations']+=result['calibration_observations']
                counts['full_cases' if scope=='full' else 'smoke_cases']+=1
                if record['method']['precision'] in ('fp16','bf16'):counts['half_bfloat_capability_filtered_cases']+=1
                geometries.add((record['method']['family'],record['method']['precision'],*record['shape'],record['method']['chunk'],record['method']['accumulation']))
                named_sizes.append(result['named_owned_device_bytes'])
    assert counts['full_cases']==2*len(DATASETS)*(len(methods())+2)
    assert counts['smoke_cases']==len(methods())+2
    result=dict(scope='Complete declared PCA named-allocation/recorded-heuristic audit; not numerical replay or physical peak measurement',
                counts=dict(sorted(counts.items())),allocation_geometries=len(geometries),
                minimum_named_owned_device_bytes=min(named_sizes),maximum_named_owned_device_bytes=max(named_sizes),
                producer_source_sha256=sources,
                additional_producer_source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in GUARDED_ADDITIONS} if counts['guarded_cases'] else {},
                auditor_source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('pca_cost_audit.py','pca_provenance.py')},
                producer_epoch_collections=dict(sorted(epochs.items())),
                collection_sha256=dict(sorted(collections.items())),
                limits='Only fixed-adapter live named allocation requests and recorded calibration decisions. No opaque library/cache/temporary peak, resource-counter or executed instruction trace certificate.')
    if args.summary:args.summary.write_text(json.dumps(result,indent=2)+'\n')
    print('PCA_NAMED_COST_AUDIT_PASS',json.dumps(result['counts'],sort_keys=True),flush=True)


if __name__=='__main__':main()
