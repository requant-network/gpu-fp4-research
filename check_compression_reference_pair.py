"""Run source-isolated ordered-copy or unmodified serial ndzip through one measured pipeline."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
from check_compression_pair import SOURCES as ORIGINAL_SOURCES,prerequisites as original_prerequisites,digest,ROOT
from check_ndzip_ordered_native import SOURCES as NATIVE_SOURCES
from compression_reference_epoch import reference_config,REFERENCES

SOURCES=tuple(dict.fromkeys((*ORIGINAL_SOURCES,*NATIVE_SOURCES,
    'check_compression_reference_pair.py','compression_reference_engine.py','compression_reference_epoch.py',
    'capture_ndzip_ordered_gpu.py')))
REFERENCE_FILES=tuple('results/ndzip-ordered-build-evidence/'+name for name in
    ('manifest.json','cpu_codec.inl','ordered-copy.patch','LICENSE','environment.json'))+(
    'results/ndzip-ordered-native/manifest.json','results/ndzip-ordered-native-replay.json',
    'results/ndzip-cpu-causal-replay.json')


def prerequisites():
    result=original_prerequisites()
    native=json.loads((ROOT/'results/ndzip-ordered-native/manifest.json').read_text())
    replay=json.loads((ROOT/'results/ndzip-ordered-native-replay.json').read_text())
    assert native['complete'] is True and len(native['cases'])==32
    assert replay['manifest_sha256']==digest(ROOT/'results/ndzip-ordered-native/manifest.json')
    for name,expected in native['sources'].items():assert digest(ROOT/name)==expected,name
    for name,expected in replay['verifiers'].items():assert digest(ROOT/name)==expected,name
    meta=json.loads((ROOT/'results/ndzip-ordered-build-evidence/manifest.json').read_text())
    assert native['build']==meta and native['build_manifest_sha256']==digest(ROOT/'results/ndzip-ordered-build-evidence/manifest.json')
    assert meta['complete'] is True and meta['reference']=='ndzip-ff4e6702+ordered-copy'
    actual=ROOT/'build/ndzip-ordered-gpu-v1'
    assert digest(actual/'manifest.json')==native['build_manifest_sha256']
    for name,expected in meta['sources'].items():assert digest(actual/'source'/name)==expected,name
    for name,expected in meta['binaries'].items():assert digest(actual/name)==expected,name
    result['ordered_reference']={'proofs':{name:digest(ROOT/name) for name in REFERENCE_FILES},'build':meta}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=('smoke','paired','over-vram','above-probe'),default='smoke')
    parser.add_argument('--ndzip-reference',choices=REFERENCES,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    from compression_reference_engine import run_route,loaded_libraries
    from compression_native import NdzipPlan
    from compression_native_ordered import OrderedNdzipPlan
    from compression_pair_protocol import WitnessStore
    import cupy as cp
    factory=OrderedNdzipPlan if args.ndzip_reference=='ordered-copy' else NdzipPlan
    began=time.perf_counter_ns();cp.cuda.Device(0).use();props=cp.cuda.runtime.getDeviceProperties(0)
    device={'name':props['name'].decode() if isinstance(props['name'],bytes) else props['name'],
            'total_memory_bytes':int(props['totalGlobalMem']),'major':int(props['major']),'minor':int(props['minor']),
            'runtime_api':cp.cuda.runtime.runtimeGetVersion(),'driver_api':cp.cuda.runtime.driverGetVersion()}
    context_setup_ns=time.perf_counter_ns()-began
    config=reference_config(args.suite,args.ndzip_reference)
    args.output.mkdir(parents=True,exist_ok=False)
    source_map={name:digest(ROOT/name) for name in SOURCES};dependencies=prerequisites()
    record={'schema':'compression-pair-reference-suite-v2','suite':args.suite,
            'ndzip_reference':args.ndzip_reference,'config':config,
            'scope':'Explicit isolated reference epoch; synthetic suite, not genuine-data/full-study or physical-memory closure',
            'sources':source_map,'dependencies':dependencies,'device':device,'numpy':np.__version__,'cupy':cp.__version__,
            'context_setup_ns':context_setup_ns,'cases':[],'complete':False}
    manifest=args.output/'manifest.json';manifest.write_text(json.dumps(record,indent=2));store=WitnessStore(args.output)
    for cohort,dtype,profile in config['datasets']:
        selected=list(config['routes'])
        if cohort=='b':selected.reverse()
        for route in selected:
            case=run_route(args.output,store,cohort,dtype,profile,route,config,device['total_memory_bytes'],ndzip_factory=factory)
            case['index']=len(record['cases']);case['ndzip_reference']=args.ndzip_reference
            began=time.perf_counter_ns()
            gc.collect();cp.get_default_pinned_memory_pool().free_all_blocks();cp.get_default_memory_pool().free_all_blocks()
            case['allocator_cleanup_ns']=time.perf_counter_ns()-began
            record['cases'].append(case)
            assert source_map=={name:digest(ROOT/name) for name in SOURCES}
            manifest.write_text(json.dumps(record,indent=2))
            print('REFERENCE_PAIR_ROUTE_COMPLETE',args.ndzip_reference,args.suite,case['index']+1,route,
                  [trial['host_ns'] for trial in case['trials']],flush=True)
    assert dependencies==prerequisites(),'Dependency bytes changed during reference epoch'
    record['loaded_libraries']=loaded_libraries();record['complete']=True
    manifest.write_text(json.dumps(record,indent=2))
    print('REFERENCE_PAIR_SUITE_PROCESS_COMPLETE',args.ndzip_reference,args.suite,len(record['cases']),flush=True)
