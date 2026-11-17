"""Source-bound complete physical operator, Poisson and implicit-heat collection.

Artifact hashing/copying outside named resident timers changes spacing. It is
not a continuous saturated-throughput or exclusive-device experiment.
"""
import argparse
import math
import os
from pathlib import Path
import re
import time
import cupy as cp
import numpy as np

from pde_cases import campaign_cases,configuration_specs,protocol
from pde_cg import jacobi_diagonal,solve,solve_reliable
from pde_evidence import ROOT,binaries,dump,environment,file_hash,save_payloads,sources
from pde_methods import configurations,create,DeviceOperator
from pde_protocol import OutputArchive,array_hash
from pde_reference import Mesh,manufactured


def elapsed(start):return (time.perf_counter()-start)*1000


def safe(value):
    if isinstance(value,float) and not math.isfinite(value):return {'nonfinite':str(value)}
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    return value


def expected_unavailable(error,config,mesh):
    message=str(error)
    shared=(config['family']=='ceed' and config['backend']=='ref' and mesh.q==16 and
            'uses too much shared data' in message and '0xc000 max' in message)
    columns=((3*mesh.elements**3*mesh.q**2+128//mesh.q-1)//(128//mesh.q)+127)//128*128
    bounded=(config['family']=='tensor' and columns>8192 and 'dimensions 1..8192' in message)
    if not (shared or bounded):raise error
    return dict(reason='Pinned reference basis static-shared allocation exceeds this kernel limit' if shared else 'Original native bridge N<=8192 implementation guard, not a hardware limit',
                error=re.sub(r'/(?:home|pyenv|__w)/[^\s:]+','<vendor-path>',message))


def truth_inputs(mesh,cohort,settings):
    physical=manufactured(mesh,cohort)
    arrays=dict(initial=physical['exact'],poisson_rhs=physical['rhs'],mass=mesh.mass,
                operator0=physical['exact'].copy(),heat_exact=manufactured(mesh,cohort,time=settings['heat_final_time'])['exact'])
    seed=120000+(0 if cohort=='a' else 10000)+mesh.elements*100+mesh.q
    noise=np.random.Generator(np.random.PCG64(seed)).integers(-2048,2049,mesh.size,dtype=np.int32).astype(np.float64)/2048
    arrays['operator1']=arrays['operator0']+noise/32
    for steps in settings['heat_steps']:
        dt=settings['heat_final_time']/steps
        target=np.zeros(mesh.size,np.float64)
        for (amplitude,frequency),eigenvalue in zip(physical['modes'],physical['laplacian_eigenvalues'],strict=True):
            target+=amplitude*np.prod(np.sin(np.pi*mesh.coordinates*np.asarray(frequency)[None,:]),axis=1)/(1+eigenvalue*dt)**steps
        target[mesh.boundary]=0;arrays['backward_euler_'+str(steps)]=target
    for value in arrays.values():value.setflags(write=False)
    return arrays,dict(seed=seed,modes=physical['modes'],eigenvalues=physical['laplacian_eigenvalues'])


def snapshot(plan,variant,arrays):
    if not hasattr(plan,'product'):return None
    key='gemm'+str(variant)
    arrays[key]=cp.asnumpy(plan.product)
    indices=(0,1,2,3) if plan.mode=='fp4' else (0,1)
    return dict(product_key=key,product_sha256=array_hash(arrays[key]),left_sha256=array_hash(cp.asnumpy(plan.left)),
                right_sha256=array_hash(cp.asnumpy(plan.right)),encoded_sha256={str(i):array_hash(plan.raw_buffer(i)) for i in indices})


def complete_solve(operator,original,rhs,diagonal,settings,target,policy,x0=None):
    before=(operator.calls,original.calls)
    if policy=='fixed':result=solve(operator,rhs,diagonal,xp=cp,tolerance=target,max_iterations=settings['max_iterations'],x0=x0)
    else:result=solve_reliable(operator,original,rhs,diagonal,xp=cp,tolerance=target,max_iterations=settings['max_iterations'],x0=x0,
                               low_budget=settings['low_budget'],segment_iterations=settings['segment_iterations'])
    state=result.pop('x')
    finite=bool(cp.isfinite(state).all())
    relative=None;checks=0
    if finite:
        residual=rhs-original(state);checks=1
        norm=float(cp.linalg.norm(residual).get());denominator=float(cp.linalg.norm(rhs).get())
        if math.isfinite(norm) and denominator>0:relative=norm/denominator
    result.update(physical_finite=finite,gpu_original_relative=relative,original_target_met=relative is not None and relative<=target,
                  endpoint_original_checks=checks,actual_operator_calls=operator.calls-before[0],actual_original_calls=original.calls-before[1])
    return state,safe(result)


def collect_operator(plan,host,device,settings,archive,arrays):
    rows=[];stages={}
    for route,count in (('resident',settings['resident_observations']),('host',settings['host_observations'])):
        for ordinal in range(count):
            variant=ordinal%2;input_sha=array_hash(host[variant])
            start,end=cp.cuda.Event(),cp.cuda.Event()
            wall=time.perf_counter();start.record(plan.stream)
            if route=='host':plan.load(host[variant])
            else:
                with plan.stream:cp.copyto(plan.data,device[variant])
            plan.run();end.record(plan.stream);end.synchronize()
            output=plan.result() if route=='host' else None
            wall_ms=elapsed(wall);cuda_ms=float(cp.cuda.get_elapsed_time(start,end))
            if output is None:output=plan.result()
            assert array_hash(host[variant])==input_sha and array_hash(cp.asnumpy(plan.data))==input_sha,'Original operation input changed'
            identifier=archive.add(output,'operator'+str(variant),input_sha)
            if str(variant) not in stages:
                stage=snapshot(plan,variant,arrays)
                stages[str(variant)]=dict(output=identifier,observation=len(rows),witness=stage)
            rows.append(dict(route=route,ordinal=ordinal,variant=variant,input_sha256=input_sha,output=identifier,wall_ms=wall_ms,cuda_ms=cuda_ms))
    return rows,stages


def collect_poisson(plan,original,g_rhs,host_rhs,diagonal,settings,archive):
    operator=DeviceOperator(plan);rows=[];identity=array_hash(host_rhs)
    for target in settings['targets']:
        for policy in settings['policies']:
            for route in ('resident','host'):
                for repetition in range(settings['poisson_repetitions_per_route']):
                    started=time.perf_counter()
                    rhs=cp.asarray(host_rhs) if route=='host' else g_rhs
                    state,result=complete_solve(operator,original,rhs,diagonal,settings,target,policy)
                    output=cp.asnumpy(state) if route=='host' else None
                    cp.cuda.runtime.deviceSynchronize();wall_ms=elapsed(started)
                    if output is None:output=cp.asnumpy(state)
                    assert array_hash(host_rhs)==identity and array_hash(cp.asnumpy(rhs))==identity,'Original Poisson RHS changed'
                    identifier=archive.add(output,'poisson/'+policy+'/'+str(target),identity)
                    rows.append(dict(route=route,repetition=repetition,target=target,policy=policy,input_sha256=identity,output=identifier,wall_ms=wall_ms,controller=result))
    return rows


def collect_heat(mesh,config,truth,settings,directory,prefix):
    records=[]
    for steps in settings['heat_steps']:
        reaction=steps/settings['heat_final_time']
        started=time.perf_counter()
        plan=create(mesh,config,reaction);original_plan=None
        try:
            original_plan=create(mesh,dict(family='cuda',precision='fp64',threads=256),reaction)
            plan.load(truth['initial'])
            if hasattr(plan,'calibrate'):plan.calibrate()
            original=DeviceOperator(original_plan);operator=DeviceOperator(plan)
            diagonal=cp.asarray(jacobi_diagonal(mesh,reaction));mass=cp.asarray(mesh.mass);initial=cp.asarray(truth['initial'])
            cp.cuda.runtime.deviceSynchronize();setup_ms=elapsed(started)
            archive=OutputArchive();arrays={};runs=[]
            for target in settings['targets']:
                for policy in settings['policies']:
                    for route in ('resident','host'):
                        started=time.perf_counter()
                        state=cp.asarray(truth['initial']) if route=='host' else initial
                        states=[];right_sides=[];controllers=[]
                        for step in range(steps):
                            rhs=mass*state*reaction
                            state,control=complete_solve(operator,original,rhs,diagonal,settings,target,policy,x0=state)
                            states.append(state);right_sides.append(rhs);controllers.append(control)
                            if not control['physical_finite']:break
                        final=cp.asnumpy(state) if route=='host' else None
                        cp.cuda.runtime.deviceSynchronize();wall_ms=elapsed(started)
                        step_records=[]
                        for step,(value,rhs,control) in enumerate(zip(states,right_sides,controllers,strict=True)):
                            host_rhs=cp.asnumpy(rhs);output=cp.asnumpy(value)
                            key='rhs'+str(len(arrays));arrays[key]=host_rhs
                            identity=array_hash(host_rhs)
                            identifier=archive.add(output,'step'+str(step),identity)
                            step_records.append(dict(step=step+1,rhs_key=key,input_sha256=identity,output=identifier,controller=control))
                        if final is not None:assert array_hash(final)==archive.records[step_records[-1]['output']]['output_sha256']
                        assert array_hash(cp.asnumpy(initial))==array_hash(truth['initial']),'Original heat initial state changed'
                        runs.append(dict(route=route,target=target,policy=policy,wall_ms=wall_ms,steps_completed=len(step_records),step_records=step_records))
            arrays.update(archive.arrays)
            payloads=save_payloads(directory,prefix+'-heat'+str(steps),arrays)
            records.append(dict(steps=steps,reaction=reaction,setup_ms=setup_ms,method_metadata=plan.metadata(),original_metadata=original_plan.metadata(),
                                retained_gpu_state_rhs_bytes=steps*2*mesh.size*8,
                                memory_scope='Named plan/vector categories plus retained step references; no temporary/cached/physical-peak claim',
                                outputs=archive.records,payloads=payloads,runs=runs))
        finally:
            if original_plan is not None:original_plan.close()
            plan.close()
    return records


def run(cohort,elements,q,smoke):
    cp.cuda.Device(0).use()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert configurations()==configuration_specs()
    settings=protocol(smoke)
    selected=next(c for c in campaign_cases() if (c['cohort'],c['elements'],c['q'])==(cohort,elements,q))
    directory=ROOT/'results'/('pde-'+settings['phase']+'-'+cohort)/('e'+str(elements)+'-q'+str(q))
    if directory.exists() and any(directory.iterdir()):raise RuntimeError('Never overwrite an executed collection; archive explicitly before a new epoch')
    directory.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();mesh=Mesh(elements,q);truth,input_metadata=truth_inputs(mesh,cohort,settings)
    device=[cp.asarray(truth['operator'+str(i)]) for i in (0,1)]
    high_plan=create(mesh,dict(family='cuda',precision='fp64',threads=256),0.)
    original=DeviceOperator(high_plan)
    diagonal=cp.asarray(jacobi_diagonal(mesh,0.));g_rhs=cp.asarray(truth['poisson_rhs'])
    cp.cuda.runtime.deviceSynchronize();common_preparation_ms=elapsed(started)
    env=environment();source_map=env['source_sha256'];binary_map=env['binary_sha256']
    dump(directory/'environment.json',env)
    truth_bindings={}
    for key,value in truth.items():
        path=directory/('truth-'+key+'.npz');np.savez_compressed(path,value=value)
        truth_bindings[key]=dict(path=path.relative_to(ROOT).as_posix(),sha256=file_hash(path),array_sha256=array_hash(value),shape=list(value.shape),dtype=value.dtype.str)
    case_files=[]
    order=list(range(len(configurations())))
    if cohort=='b':order.reverse()
    try:
        for index in order:
            config=configurations()[index];prefix='case-'+str(index).zfill(2)
            case=dict(configuration=config,configuration_index=index,source_sha256=source_map,binary_sha256=binary_map)
            started=time.perf_counter();plan=None
            try:
                plan=create(mesh,config,0.)
                construction_ms=elapsed(started)
                started=time.perf_counter();plan.load(truth['operator0']);load_ms=elapsed(started)
                started=time.perf_counter()
                if hasattr(plan,'calibrate'):plan.calibrate()
                cp.cuda.runtime.deviceSynchronize();calibration_ms=elapsed(started)
                started=time.perf_counter();plan.run();first_use_ms=elapsed(started)
                archive=OutputArchive();arrays={}
                assert array_hash(cp.asnumpy(plan.data))==array_hash(truth['operator0']),'First-use original input changed'
                first_use_output=archive.add(plan.result(),'operator0',array_hash(truth['operator0']))
                observations,stages=collect_operator(plan,[truth['operator0'],truth['operator1']],device,settings,archive,arrays)
                poisson=collect_poisson(plan,original,g_rhs,truth['poisson_rhs'],diagonal,settings,archive)
                metadata=plan.metadata()
                arrays.update(archive.arrays)
                payloads=save_payloads(directory,prefix,arrays)
                case.update(available=True,construction_ms=construction_ms,first_data_load_ms=load_ms,calibration_ms=calibration_ms,first_complete_use_ms=first_use_ms,
                            first_use_output=first_use_output,first_use_input_sha256=array_hash(truth['operator0']),
                            cuda_event_scope='Input staging through operation acknowledgment; host D2H is included only in wall time',
                            method_metadata=metadata,operator_observations=observations,operator_stages=stages,poisson_runs=poisson,outputs=archive.records,payloads=payloads)
            except RuntimeError as error:
                if plan is not None:raise
                case.update(available=False,failed_setup_ms=elapsed(started),availability=expected_unavailable(error,config,mesh))
            finally:
                if plan is not None:plan.close()
            if case['available'] and selected['heat']:case['heat']=collect_heat(mesh,config,truth,settings,directory,prefix)
            else:case['heat']=[]
            assert sources()==source_map and binaries()==binary_map,'Executed source/binary epoch changed'
            path=directory/(prefix+'.json');dump(path,safe(case));case_files.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=file_hash(path)))
            print('PDE_CASE_SEALED',settings['phase'],cohort,elements,q,index,'available' if case['available'] else 'unavailable',flush=True)
        assert len(case_files)==len(configurations())
        manifest=dict(scope='Complete declared dataset collection; independent whole replay and publication are still required',cohort=cohort,elements=elements,q=q,
                      dofs=mesh.size,protocol=settings,input_metadata=input_metadata,heat_included=selected['heat'],truth=truth_bindings,
                      common_preparation_ms=common_preparation_ms,common_reference_metadata=high_plan.metadata(),
                      common_named_device_bytes=sum(v.nbytes for v in device)+diagonal.nbytes+g_rhs.nbytes+high_plan.metadata()['named_owned_device_bytes'],
                      startup_scope='Python/import/device-discovery startup not measured; warm disk/allocator/JIT caches and diagnostic gaps are not cold-start claims',
                      source_sha256=source_map,binary_sha256=binary_map,environment=dict(path=(directory/'environment.json').relative_to(ROOT).as_posix(),sha256=file_hash(directory/'environment.json')),
                      cases=case_files)
        dump(directory/'collection.json',manifest)
        print('PDE_DATASET_COLLECTION_COMPLETE_NOT_YET_INDEPENDENTLY_REPLAYED',settings['phase'],cohort,elements,q,flush=True)
    finally:high_plan.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cohort',choices=('a','b'),required=True);parser.add_argument('--elements',type=int,required=True)
    parser.add_argument('--q',type=int,choices=(4,8,16),required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args.cohort,args.elements,args.q,args.smoke)
