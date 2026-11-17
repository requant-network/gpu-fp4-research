"""Additive guarded PCA producer; the original executed producer stays immutable.

Numerical operations are the original PcaPlan/PcaBatch. Read-only original host
inputs plus byte fingerprints of host/device observations are checked outside
recorded timing boundaries. This is an explicit, separately source-bound producer
epoch, not a rewrite of historical records or a numerical-tolerance relaxation.
"""
import time

import cupy as cp
import numpy as np

import pca_study as base
from pca_cases import DATASETS
from pca_input_integrity import freeze_input,input_sha256,verify_input

_EXPECTED={}
_original_inputs=base.inputs


def guarded_inputs(n,d,profile,cohort):
    data,provenance=_original_inputs(n,d,profile,cohort)
    expected=(input_sha256(data),input_sha256(np.ascontiguousarray(-data)))
    freeze_input(data,expected[0])
    _EXPECTED[(cohort,n,d,profile)]=expected
    return data,provenance


def guarded_run_case(lib,name,cohort,variants,targets,method,observations,warmups):
    n,d,profile=DATASETS[name]
    expected=_EXPECTED[(cohort,n,d,profile)]
    for variant in range(2):freeze_input(variants[variant],expected[variant])
    arrays={}
    plan=None
    record=dict(method=method,available=False,arithmetic_valid=False,outputs={},stage_witnesses=[],boundaries=[])
    guards=dict(host_checks=0,device_checks=0,expected_original_input_sha256=list(expected),original_inputs_readonly=True,
                scope='Full host/device input fingerprints outside service timers; inter-request spacing includes validation, not a cold-system or unchanged-spacing claim')
    def host_guard(variant):
        value=verify_input(variants[variant],expected[variant])
        guards['host_checks']+=1
        return value
    def device_guard(variant):
        plan.stream.synchronize()
        actual=cp.asnumpy(plan.data,stream=plan.stream)
        value=verify_input(actual,expected[variant],require_readonly=False)
        guards['device_checks']+=1
        return value
    started=time.perf_counter_ns()
    try:
        plan=base.PcaBatch(variants[0],method['precision']) if method['family']=='batch' else base.PcaPlan(lib,variants[0],method)
        cp.cuda.runtime.deviceSynchronize()
        record.update(available=True,setup_wall_ms=(time.perf_counter_ns()-started)/1e6)
        started=time.perf_counter_ns()
        plan.device_compute()
        plan.stream.synchronize()
        record['first_operation_wall_ms']=(time.perf_counter_ns()-started)/1e6
        host_guard(0);device_guard(0)
        def remember(variant,outputs):
            kept,binding,key=base.retain_output(outputs,rank=16)
            output_key=f'v{variant}-'+key
            if output_key not in record['outputs']:
                prefix='o'+str(len(record['outputs']))
                for field,value in kept.items():arrays[prefix+'_'+field]=value
                metrics=base.quality_leading(kept['mean'],kept['covariance'],kept['eigenvalues'],kept['leading'],targets[variant])
                record['outputs'][output_key]=dict(prefix=prefix,variant=variant,array_bindings=binding,quality=metrics)
            return output_key
        for variant in range(2):
            host_guard(variant)
            plan.load(variants[variant])
            actual,stages=plan.compute(witness=True)
            host_guard(variant);device_guard(variant)
            remember(variant,actual)
            for block,stage in enumerate(stages):
                retained,metadata=base.retain_stage(stage,full=name=='uniform-n8192-d128')
                prefix=f'sv{variant}_b{block}'
                for field,value in retained.items():arrays[prefix+'_'+field]=value
                record['stage_witnesses'].append(dict(prefix=prefix,variant=variant,**metadata))
            del stages
        record['arithmetic_valid']=True
        start_event,end_event=cp.cuda.Event(),cp.cuda.Event()
        for boundary in ('resident','host-service'):
            for _ in range(warmups):
                host_guard(0)
                plan.load(variants[0])
                plan.device_compute()
                plan.result()
                host_guard(0);device_guard(0)
            rows=[]
            for observation in range(observations):
                variant=observation%2 if cohort=='a' else 1-observation%2
                before=host_guard(variant)
                if boundary=='resident':
                    plan.load(variants[variant])
                    device_guard(variant)
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
                after=host_guard(variant)
                device_after=device_guard(variant)
                rows.append(dict(variant=variant,cuda_ms=cuda_ms,wall_ms=wall_ms,output_key=remember(variant,actual),
                                 original_input_before_sha256=before,original_input_after_sha256=after,device_input_after_sha256=device_after))
            record['boundaries'].append(dict(name=boundary,observations=rows))
        record['metadata']=plan.metadata()
        record['finite_outputs']=all(output['quality']['finite'] for output in record['outputs'].values())
        record['successfully_executed']=True
    except Exception as error:
        record.update(error=base.error_text(error),elapsed_to_error_wall_ms=(time.perf_counter_ns()-started)/1e6,successfully_executed=False)
        if record['available']:cp.cuda.runtime.deviceSynchronize()
    finally:
        if plan is not None:plan.close()
    record['input_integrity']=guards
    return record,arrays


def main():
    base.inputs=guarded_inputs
    base.run_case=guarded_run_case
    base.SOURCES=(*base.SOURCES,'pca_input_integrity.py','pca_study_guarded.py')
    base.main()


if __name__=='__main__':main()
