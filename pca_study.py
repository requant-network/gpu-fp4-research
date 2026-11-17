"""Complete streaming covariance/PCA GPU collections on one authorized SM120.

Full protocol: both portable cohorts, all declared synthetic/real/tail inputs,
84 stream configurations plus untiled FP32/FP64 library controls, two changing
input variants, complete covariance/spectrum/leading-16 quality, preparation,
resident and full host-service costs. --smoke keeps the same cases/code paths but
uses five observations and two warmups; it is never full study completion.
"""
import argparse
import ctypes as ct
import datetime
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import cupy as cp
import numpy as np

from pca_cases import COHORTS,DATASETS,inputs,methods
from pca_native import PcaBatch,PcaPlan,library
from pca_protocol import array_binding,retain_output,retain_stage
from pca_reference import quality_leading,stable_reference,streaming_reference
from reference import digest

ROOT=Path(__file__).resolve().parent
SOURCES=('pca_study.py','pca_native.py','pca_cases.py','pca_protocol.py','pca_reference.py','reference.py',
         'native/pca_bridge.cu','native/hardware_bridge.cu','native/application_bridge.cu',
         'native/optimized_bridge.cu','native/bridge.cu','build_pca.sh')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,record):
    temporary=path.with_name(path.name+'.part')
    temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def write_npz(path,arrays):
    temporary=path.with_name(path.name+'.part')
    with temporary.open('wb') as output:np.savez_compressed(output,**arrays)
    temporary.replace(path)


def error_text(error):
    text=str(error).split('Obtained ')[0].strip()
    text=re.sub(r'/(?:home|pyenv|__w)/[^\s:]+','<vendor-path>',text)
    return type(error).__name__+': '+text[:1600]


def environment(lib,bootstrap_ms):
    properties=cp.cuda.runtime.getDeviceProperties(0)
    packages={}
    for name in ('numpy','cupy-cuda13x','cuda-python','cuda-toolkit','nvidia-cuda-runtime','nvidia-nvjitlink','nvidia-cublas','nvidia-cusolver'):
        try:
            distribution=md.distribution(name)
            record=distribution.read_text('RECORD')
            packages[name]=dict(version=distribution.version,record_sha256=hashlib.sha256(record.encode()).hexdigest() if record else None)
        except md.PackageNotFoundError:packages[name]=None
    paths=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        item=line.split()[-1]
        if item.startswith('/') and '.so' in item and any(label in Path(item).name for label in ('cuda','cublas','cusolver','nvrtc','nvJitLink','libpca','cupy')):
            path=Path(item)
            if path.is_file():paths.add(path.resolve())
    loaded=[dict(name=path.name,bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(paths)]
    lib.rq_cublas_version.restype=ct.c_size_t
    return dict(scope='Actual in-process PCA bootstrap/package/loaded-library snapshot; no whole-device peak or exclusive-system claim',
                captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.version,
                gpu_index=0,gpu_exclusive=False,gpu_name=properties['name'].decode(),
                compute_capability=[int(properties['major']),int(properties['minor'])],multiprocessors=int(properties['multiProcessorCount']),
                runtime_api_version=cp.cuda.runtime.runtimeGetVersion(),driver_api_version=cp.cuda.runtime.driverGetVersion(),
                cublas_lt_version=int(lib.rq_cublas_version()),cupy_version=cp.__version__,numpy_version=np.__version__,
                bootstrap_wall_ms=bootstrap_ms,packages=packages,loaded_libraries=loaded,
                nvcc=subprocess.check_output(['/usr/local/cuda-13.3/bin/nvcc','--version'],text=True).strip(),
                nvidia_smi=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip())


def truth_record(directory,name,cohort,variants,provenance,sources):
    path=directory/(name+'-truth.json')
    if path.exists():
        record=json.loads(path.read_text())
        assert record['complete'] and record['source_sha256']==sources
        assert record['input_sha256']==[digest(data) for data in variants]
        assert sha(path.with_suffix('.npz'))==record['witness_sha256']
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as archive:
            targets=[{field:archive[f'v{index}_{field}'] for field in ('mean','covariance','eigenvalues','eigenvectors')} for index in range(2)]
        return record,targets
    arrays,targets,checks={},[],[]
    start=time.perf_counter_ns()
    for index,data in enumerate(variants):
        target=stable_reference(data,chunk=4096)
        merged_mean,merged_covariance=streaming_reference(data,chunk=1531)
        relative=float(np.linalg.norm(merged_covariance-target['covariance'])/max(float(np.linalg.norm(target['covariance'])),np.finfo(np.float64).tiny))
        assert relative<1e-10
        np.testing.assert_allclose(merged_mean,target['mean'],rtol=1e-10,atol=1e-12)
        for field,value in target.items():arrays[f'v{index}_{field}']=value
        targets.append(target)
        checks.append(dict(variant=index,independent_merge_relative_covariance_error=relative))
    npz=path.with_suffix('.npz')
    write_npz(npz,arrays)
    record=dict(complete=True,study=11,dataset=name,cohort=cohort,shape=list(variants[0].shape),provenance=provenance,
                input_sha256=[digest(data) for data in variants],source_sha256=sources,
                truth_scope='All observations, direct two-pass FP64 centered products/eigh, independently checked count/mean/moment merge with different chunk boundaries',
                checks=checks,cpu_truth_wall_ms=(time.perf_counter_ns()-start)/1e6,
                common_original_host_bytes=sum(data.nbytes for data in variants),
                witness_file=npz.name,witness_sha256=sha(npz))
    write_json(path,record)
    return record,targets


def run_case(lib,name,cohort,variants,targets,method,observations,warmups):
    arrays={}
    plan=None
    record=dict(method=method,available=False,arithmetic_valid=False,outputs={},stage_witnesses=[],boundaries=[])
    started=time.perf_counter_ns()
    try:
        plan=PcaBatch(variants[0],method['precision']) if method['family']=='batch' else PcaPlan(lib,variants[0],method)
        cp.cuda.runtime.deviceSynchronize()
        record.update(available=True,setup_wall_ms=(time.perf_counter_ns()-started)/1e6)
        started=time.perf_counter_ns()
        plan.device_compute()
        plan.stream.synchronize()
        record['first_operation_wall_ms']=(time.perf_counter_ns()-started)/1e6

        def remember(variant,outputs=None):
            actual=plan.result() if outputs is None else outputs
            kept,binding,key=retain_output(actual,rank=16)
            output_key=f'v{variant}-'+key
            if output_key not in record['outputs']:
                prefix='o'+str(len(record['outputs']))
                for field,value in kept.items():arrays[prefix+'_'+field]=value
                metrics=quality_leading(kept['mean'],kept['covariance'],kept['eigenvalues'],kept['leading'],targets[variant])
                record['outputs'][output_key]=dict(prefix=prefix,variant=variant,array_bindings=binding,quality=metrics)
            return output_key

        for variant in range(2):
            plan.load(variants[variant])
            actual,stages=plan.compute(witness=True)
            remember(variant,actual)
            for block,stage in enumerate(stages):
                keep_full=name=='uniform-n8192-d128'
                retained,metadata=retain_stage(stage,full=keep_full)
                prefix=f'sv{variant}_b{block}'
                for field,value in retained.items():arrays[prefix+'_'+field]=value
                record['stage_witnesses'].append(dict(prefix=prefix,variant=variant,**metadata))
            del stages
        record['arithmetic_valid']=True
        start_event,end_event=cp.cuda.Event(),cp.cuda.Event()
        for boundary in ('resident','host-service'):
            for _ in range(warmups):
                plan.load(variants[0])
                plan.device_compute()
                plan.result()
            rows=[]
            for observation in range(observations):
                variant=observation%2 if cohort=='a' else 1-observation%2
                if boundary=='resident':
                    plan.load(variants[variant])
                    wall_start=time.perf_counter_ns()
                    start_event.record(plan.stream)
                    plan.device_compute()
                    end_event.record(plan.stream)
                    end_event.synchronize()
                    cuda_ms=float(cp.cuda.get_elapsed_time(start_event,end_event))
                    wall_ms=(time.perf_counter_ns()-wall_start)/1e6
                    actual=plan.result()
                else:
                    wall_start=time.perf_counter_ns()
                    plan.load(variants[variant])
                    plan.device_compute()
                    actual=plan.result()
                    wall_ms=(time.perf_counter_ns()-wall_start)/1e6
                    cuda_ms=None
                rows.append(dict(variant=variant,cuda_ms=cuda_ms,wall_ms=wall_ms,output_key=remember(variant,actual)))
            record['boundaries'].append(dict(name=boundary,observations=rows))
        record['metadata']=plan.metadata()
        record['finite_outputs']=all(output['quality']['finite'] for output in record['outputs'].values())
        record['successfully_executed']=True
    except Exception as error:
        record.update(error=error_text(error),elapsed_to_error_wall_ms=(time.perf_counter_ns()-started)/1e6,successfully_executed=False)
        if record['available']:cp.cuda.runtime.deviceSynchronize()
    finally:
        if plan is not None:plan.close()
    return record,arrays


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cohort',choices=COHORTS,required=True)
    parser.add_argument('--dataset',choices=DATASETS)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    sources={name:sha(ROOT/name) for name in SOURCES}
    binary=sha(ROOT/'build/libpca.so')
    directory=ROOT/'results'/f'pca-{"smoke" if args.smoke else "full"}-{args.cohort}'
    directory.mkdir(exist_ok=True)
    bootstrap_start=time.perf_counter_ns()
    cp.cuda.Device(0).use()
    assert cp.cuda.Device(0).compute_capability=='120'
    lib=library()
    # Common warm-process initialization is measured separately, not presented
    # as a free cold application start or hidden in one particular method.
    tiny=cp.eye(2,dtype=cp.float64)
    cp.matmul(tiny,tiny)
    cp.linalg.eigh(tiny)
    cp.cuda.runtime.deviceSynchronize()
    bootstrap_ms=(time.perf_counter_ns()-bootstrap_start)/1e6
    env=environment(lib,bootstrap_ms)
    env_bytes=(json.dumps(env,indent=2,allow_nan=False)+'\n').encode()
    env_name='environment-'+hashlib.sha256(env_bytes).hexdigest()+'.json'
    (directory/env_name).write_bytes(env_bytes)
    env_sha=sha(directory/env_name)
    names=[args.dataset] if args.dataset else ([next(iter(DATASETS))] if args.smoke else list(DATASETS))
    if args.cohort=='b':names.reverse()
    observations,warmups=(5,2) if args.smoke else (30,5)
    for name in names:
        n,d,profile=DATASETS[name]
        data,provenance=inputs(n,d,profile,args.cohort)
        variants=(data,np.ascontiguousarray(-data))
        truth,targets=truth_record(directory,name,args.cohort,variants,provenance,sources)
        grid=[dict(family='stream',**method) for method in methods()]
        grid.extend(dict(family='batch',precision=precision,centering='global',scaling=False,accumulation='fp64',chunk=n) for precision in ('fp32','fp64'))
        if args.cohort=='b':grid.reverse()
        cases=[]
        for index,method in enumerate(grid):
            path=directory/f'{name}-m{index:03d}.json'
            if path.exists():
                old=json.loads(path.read_text())
                assert old['complete'] and old['source_sha256']==sources and old['binary_sha256']==binary and old['method']==method
                assert old['input_sha256']==truth['input_sha256'] and old['truth_record_sha256']==sha(directory/(name+'-truth.json'))
                assert sha(path.with_suffix('.npz'))==old['witness_sha256']
                cases.append(dict(file=path.name,sha256=sha(path)))
                print('RESUME',path.name,flush=True)
                continue
            record,arrays=run_case(lib,name,args.cohort,variants,targets,method,observations,warmups)
            record.update(study=11,complete=True,smoke=args.smoke,dataset=name,cohort=args.cohort,case_index=index,shape=[n,d],
                          input_sha256=truth['input_sha256'],source_sha256=sources,binary_sha256=binary,
                          environment_file=env_name,environment_sha256=env_sha,truth_file=name+'-truth.json',truth_record_sha256=sha(directory/(name+'-truth.json')),
                          observations_per_boundary=observations,warmups_per_boundary=warmups,input_variants=2,retained_rank=16,gpu_index=0,gpu_exclusive=False,
                          timing_contract=dict(resident='CUDA events around complete GPU centering/scaling/transpose/encoding/products/accumulation/symmetrization/eigh. Current FP32 observations already on device. Python launch gaps and GPU-status checks are inside; output D2H and validation are outside.',
                                               host_service='Python wall time: current FP32 observations copied into the adapter host witness buffer, pageable H2D, all complete uncaptured GPU computation/synchronization and full mean/covariance/eigenvalues/eigenvectors D2H. Hashes/quality/serialization and dataset generation are outside.',
                                               setup='Constructor includes host-original copy, source H2D, named allocations/native workspace and heuristic calibration; first operation/JIT is separate. Common process/library bootstrap is separately recorded in environment, not a cold-start-free claim.',
                                               retention='Complete means, covariances, eigenvalues and leading16 eigenvectors retained and every observation bound. All eigenvectors computed/transferred but no unretained full-basis correctness claim. Full uniform-small Gram witnesses; elsewhere stage source/encoding hashes, all diagonal plus sampled Gram entries and complete final covariance.'),
                          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            npz=path.with_suffix('.npz')
            write_npz(npz,arrays)
            record.update(witness_file=npz.name,witness_sha256=sha(npz))
            write_json(path,record)
            cases.append(dict(file=path.name,sha256=sha(path)))
            print('PCA_CASE',args.cohort,name,index+1,len(grid),json.dumps(method,sort_keys=True),record['available'],record['successfully_executed'],record.get('error',''),flush=True)
        manifest=dict(study=11,complete=True,smoke=args.smoke,dataset=name,cohort=args.cohort,source_sha256=sources,binary_sha256=binary,
                      input_sha256=truth['input_sha256'],truth_file=name+'-truth.json',truth_record_sha256=sha(directory/(name+'-truth.json')),cases=cases)
        write_json(directory/(name+'-collection.json'),manifest)
        print('PCA_DATASET_COLLECTION_COMPLETE',args.cohort,name,len(cases),flush=True)
    print('PCA_COHORT_COLLECTION_COMPLETE',args.cohort,'smoke' if args.smoke else 'full',flush=True)


if __name__=='__main__':main()
