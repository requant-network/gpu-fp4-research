"""Fresh CPU numerical replay of full retained physical observations and solves.

This certifies retained products/final vectors and original-system residuals,
not every unretained CG intermediate or a best-in-class solver implementation.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path,PurePosixPath
import re
import numpy as np
from pde_cases import campaign_cases,configuration_specs,protocol
from pde_models import check_output,check_stage,operator_model,gamma
from pde_protocol import array_hash,decode_output
from pde_reference import Mesh,apply_operator

ROOT=Path(__file__).resolve().parent
PRODUCERS=('pde_study.py','pde_evidence.py','pde_cases.py','pde_protocol.py','pde_methods.py','pde_cg.py',
           'pde_reference.py','pde_coefficients.py','prepare_pde_coefficients.py','reference.py','pde_cuda.py','native/pde_kernels.cu',
           'pde_ceed.py','native/pde_ceed_bridge.cpp','native/pde_ceed_qfunction.h','build_pde_ceed.sh',
           'pde_tensor.py','native/pde_tensor_kernels.cu','pde_packing.py','pca_native.py','native/pca_bridge.cu',
           'native/hardware_bridge.cu','native/application_bridge.cu','native/optimized_bridge.cu','native/bridge.cu','build_pca.sh',
           'pde_small.py','native/pde_small_bridge.cu','native/pde_small_kernels.cu','build_pde_small.sh')
VERIFIERS=('pde_replay.py','pde_models.py')


def file_hash(path):
    digest=hashlib.sha256()
    with open(path,'rb') as source:
        while block:=source.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def bound_path(item):
    name=item['path'];path=PurePosixPath(name)
    assert isinstance(name,str) and not path.is_absolute() and '..' not in path.parts and '\\' not in name and ':' not in name,'Nonliteral/nonlocal artifact path'
    expected=item['sha256'];assert re.fullmatch('[0-9a-f]{64}',expected),'Malformed artifact hash'
    target=ROOT/name;assert file_hash(target)==expected,('Artifact bytes changed',name)
    return target


def load_bound_json(item):return json.loads(bound_path(item).read_text())


def load_payloads(items):
    result={}
    for item in items:
        with np.load(bound_path(item),allow_pickle=False) as data:
            assert set(data.files)==set(item['keys']),'Payload key declaration changed'
            for key in data.files:
                assert key not in result,'Overlapping payload arrays'
                result[key]=data[key]
    return result


def norm(value):
    if not np.isfinite(value).all():return math.inf
    maximum=float(np.max(abs(value),initial=0.))
    return 0. if maximum==0 else maximum*float(np.sqrt(np.sum((value/maximum)**2)))


def mass_error(mesh,value,exact):return norm(np.sqrt(mesh.mass)*(value-exact))/max(norm(np.sqrt(mesh.mass)*exact),np.finfo(np.float64).tiny)


def load_truth(manifest):
    result={}
    for name,item in manifest['truth'].items():
        with np.load(bound_path(item),allow_pickle=False) as data:
            assert data.files==['value'];value=data['value']
        assert list(value.shape)==item['shape'] and value.dtype.str==item['dtype'] and array_hash(value)==item['array_sha256'],'Original truth array changed'
        result[name]=value
    return result


def check_truth(mesh,manifest,truth):
    settings=manifest['protocol'];cohort=manifest['cohort']
    expected_keys={'initial','poisson_rhs','mass','operator0','operator1','heat_exact'}|{'backward_euler_'+str(s) for s in settings['heat_steps']}
    assert set(truth)==expected_keys
    for value in truth.values():assert value.shape==(mesh.size,) and value.dtype==np.float64 and np.isfinite(value).all()
    assert np.array_equal(truth['mass'],mesh.mass),'Physical mass is not the canonical geometry'
    modes=((1.,(1,1,1)),) if cohort=='a' else ((.75,(1,2,3)),(-.125,(2,1,2)))
    initial=np.zeros(mesh.size);forcing=initial.copy();exact=initial.copy();envelope=initial.copy();be={s:initial.copy() for s in settings['heat_steps']}
    eigenvalues=[]
    for amplitude,frequency in modes:
        eigenvalue=math.pi**2*sum(f*f for f in frequency);eigenvalues.append(eigenvalue)
        mode=amplitude*np.sin(mesh.coordinates[:,0]*math.pi*frequency[0])*np.sin(mesh.coordinates[:,1]*math.pi*frequency[1])*np.sin(mesh.coordinates[:,2]*math.pi*frequency[2])
        initial+=mode;forcing+=eigenvalue*mode;envelope+=abs(mode)
        exact+=math.exp(-eigenvalue*settings['heat_final_time'])*mode
        for steps in be:be[steps]+=mode/(1+eigenvalue*settings['heat_final_time']/steps)**steps
    expected={'initial':initial,'poisson_rhs':mesh.mass*forcing,'heat_exact':exact}
    expected.update({'backward_euler_'+str(s):v for s,v in be.items()})
    for key,value in expected.items():
        value[mesh.boundary]=0
        scale=mesh.mass*sum(eigenvalues)*envelope if key=='poisson_rhs' else envelope
        bound=128*np.finfo(np.float64).eps*scale+np.finfo(np.float64).tiny
        assert np.all(abs(truth[key]-value)<=bound),('Manufactured physical source differs',key)
    assert np.array_equal(truth['operator0'],truth['initial'])
    seed=120000+(0 if cohort=='a' else 10000)+mesh.elements*100+mesh.q
    assert manifest['input_metadata']['seed']==seed
    noise=np.random.Generator(np.random.PCG64(seed)).integers(-2048,2049,mesh.size,dtype=np.int32).astype(np.float64)/2048
    assert np.array_equal(truth['operator1'],truth['operator0']+noise/32),'Changing rough source is not its declared original data'


def finite_time(value):assert isinstance(value,(int,float)) and math.isfinite(value) and value>=0,'Invalid observed time'


def verify_operator_case(mesh,truth,case,arrays,settings):
    expected_pairs={(route,i) for route,key in (('resident','resident_observations'),('host','host_observations')) for i in range(settings[key])}
    observations=case['operator_observations']
    assert len(observations)==len(expected_pairs) and {(r['route'],r['ordinal']) for r in observations}==expected_pairs,'Operator observation coverage changed'
    quality={};models={};decoded={};used=set()
    def validate(identifier,variant):
        values=truth['operator'+str(variant)];identity=array_hash(values)
        record=case['outputs'][identifier]
        assert record['label']=='operator'+str(variant),'Timing output namespace changed'
        if identifier not in decoded:
            output=decode_output(record,arrays,identity)
            if variant not in models:models[variant]=operator_model(mesh,values,case['configuration'])
            check_output(models[variant],output,mesh,values)
            physical=apply_operator(mesh,values)
            quality[identifier]=dict(variant=variant,relative_original_operator_error=norm(output-physical)/norm(physical))
            decoded[identifier]=output
        else:assert record['input_sha256']==identity,'Stale-input observation'
        used.add(identifier);return decoded[identifier]
    assert case['first_use_input_sha256']==array_hash(truth['operator0'])
    validate(case['first_use_output'],0)
    for row in observations:
        variant=row['ordinal']%2;assert row['variant']==variant and row['input_sha256']==array_hash(truth['operator'+str(variant)]),'Stale timing input identity'
        finite_time(row['wall_ms']);finite_time(row['cuda_ms']);validate(row['output'],variant)
    assert set(case['operator_stages'])=={'0','1'}
    for name,stage in case['operator_stages'].items():
        variant=int(name);row=observations[stage['observation']]
        assert row['route']=='resident' and row['variant']==variant and row['output']==stage['output'],'Stage is not bound to its measured output'
        output=validate(stage['output'],variant)
        check_stage(models[variant],stage['witness'],arrays,output,mesh,truth['operator'+name])
    return dict(quality=quality,used=used,observations=len(observations),stages=sum(s['witness'] is not None for s in case['operator_stages'].values()))


def verify_solution(mesh,values,rhs,reaction,control,target,policy,exact):
    assert values.shape==(mesh.size,) and values.dtype==np.float64 and rhs.shape==values.shape
    finite=bool(np.isfinite(values).all());assert control['physical_finite']==finite
    assert control['endpoint_original_checks']==int(finite)
    if policy=='fixed':
        assert control['actual_operator_calls']==control['operator_calls'] and control['actual_original_calls']==control['endpoint_original_checks'],'Fixed solver call accounting changed'
    else:
        assert control['actual_operator_calls']==control['low_operator_calls']
        assert control['actual_original_calls']==control['high_operator_calls']+control['endpoint_original_checks'],'Reliable solver call accounting changed'
        assert control['operator_calls']==control['low_operator_calls']+control['high_operator_calls'] and control['low_operator_calls']<=32
    reported=control['gpu_original_relative']
    assert control['original_target_met']==(reported is not None and reported<=target),'Original-target claim contradicts recorded residual'
    if not finite:
        assert reported is None and not control['original_target_met']
        return dict(finite=False,original_relative=None,continuum_mass_relative=None,target_met=False)
    assert np.all(values[mesh.boundary]==0),'Homogeneous solution boundary changed'
    with np.errstate(over='ignore',invalid='ignore'):
        physical=apply_operator(mesh,values,reaction)
        residual=rhs-physical
        relative=norm(residual)/norm(rhs)
    if reported is not None:
        assert math.isfinite(reported) and reported>=0
        high=operator_model(mesh,values,dict(family='cuda',precision='fp64'),reaction)
        uncertainty=norm(high['bound'])/norm(rhs)+gamma(2*mesh.size+32,np.finfo(np.float64).eps*.5)*(relative+norm(high['bound'])/norm(rhs))
        uncertainty+=16*np.finfo(np.float64).eps*(norm(rhs)+norm(physical))/norm(rhs)
        assert abs(relative-reported)<=uncertainty,'Claimed physical residual is not the original-system final vector'
    return dict(finite=True,original_relative=relative if math.isfinite(relative) else None,
                continuum_mass_relative=mass_error(mesh,values,exact) if math.isfinite(relative) else None,target_met=relative<=target)


def verify_poisson(mesh,truth,case,arrays,settings):
    expected=set(itertools.product(settings['targets'],settings['policies'],('resident','host'),range(settings['poisson_repetitions_per_route'])))
    rows=case['poisson_runs']
    assert len(rows)==len(expected) and {(r['target'],r['policy'],r['route'],r['repetition']) for r in rows}==expected,'Poisson coverage changed'
    used=set();quality=[];identity=array_hash(truth['poisson_rhs'])
    for row in rows:
        assert row['input_sha256']==identity
        record=case['outputs'][row['output']]
        assert record['label']=='poisson/'+row['policy']+'/'+str(row['target'])
        output=decode_output(record,arrays,identity);used.add(row['output']);finite_time(row['wall_ms'])
        metrics=verify_solution(mesh,output,truth['poisson_rhs'],0.,row['controller'],row['target'],row['policy'],truth['initial'])
        quality.append(dict(output=row['output'],route=row['route'],target=row['target'],policy=row['policy'],**metrics))
    return dict(used=used,quality=quality,solves=len(rows))


def verify_heat(mesh,truth,case,settings,included):
    heats=case['heat'];assert len(heats)==(len(settings['heat_steps']) if included else 0),'Heat geometry coverage changed'
    trajectories=0;step_count=0;quality=[];unique=0
    for heat,steps in zip(heats,settings['heat_steps'],strict=False):
        assert heat['steps']==steps and heat['reaction']==steps/settings['heat_final_time']
        arrays=load_payloads(heat['payloads']);used=set();finite_time(heat['setup_ms'])
        expected=set(itertools.product(settings['targets'],settings['policies'],('resident','host')))
        assert len(heat['runs'])==len(expected) and {(r['target'],r['policy'],r['route']) for r in heat['runs']}==expected
        for run in heat['runs']:
            finite_time(run['wall_ms']);previous=truth['initial'];last_metrics=None
            assert run['steps_completed']==len(run['step_records']) and 1<=run['steps_completed']<=steps
            for number,row in enumerate(run['step_records'],1):
                assert row['step']==number
                rhs=arrays[row['rhs_key']];identity=array_hash(rhs)
                assert row['input_sha256']==identity
                expected_rhs=mesh.mass*previous*heat['reaction']
                assert np.array_equal(rhs,expected_rhs),'Heat RHS is not the previous retained physical state'
                record=heat['outputs'][row['output']];assert record['label']=='step'+str(number-1)
                value=decode_output(record,arrays,identity);used.add(row['output'])
                last_metrics=verify_solution(mesh,value,rhs,heat['reaction'],row['controller'],run['target'],run['policy'],truth['heat_exact'])
                previous=value;step_count+=1
            if run['steps_completed']<steps:assert not last_metrics['finite'],'Unexecuted heat steps without numerical failure'
            trajectories+=1
            quality.append(dict(steps=steps,route=run['route'],policy=run['policy'],target=run['target'],completed=run['steps_completed'],
                                continuum_mass_relative=mass_error(mesh,previous,truth['heat_exact']) if last_metrics['finite'] else None,
                                backward_euler_continuum_mass_relative=mass_error(mesh,previous,truth['backward_euler_'+str(steps)]) if last_metrics['finite'] else None,
                                final_step_original_relative=last_metrics['original_relative']))
        assert used==set(heat['outputs']),'Unreferenced heat certificate'
        unique+=len(used)
    return dict(trajectories=trajectories,steps=step_count,quality=quality,outputs=unique)


def replay_collection(path):
    path=Path(path);manifest=json.loads(path.read_text());settings=manifest['protocol']
    assert settings==protocol(settings['phase']=='smoke'),'Declared protocol changed'
    assert set(manifest['source_sha256'])==set(PRODUCERS),'Unexpected producer epoch'
    for name,expected in manifest['source_sha256'].items():assert re.fullmatch('[0-9a-f]{64}',expected) and file_hash(ROOT/name)==expected,('Executed source changed',name)
    assert set(manifest['binary_sha256'])=={'native_core','ceed_bridge','short_q_bridge','libceed'}
    for value in manifest['binary_sha256'].values():assert re.fullmatch('[0-9a-f]{64}',value)
    assert file_hash(ROOT/'fixtures/pde-gll.npz')=='39e043321b929d36c6d53d04dae6cb61b84c1558db804da0e49fa8fe632035a2','Canonical coefficient epoch changed'
    environment=load_bound_json(manifest['environment'])
    assert environment['source_sha256']==manifest['source_sha256'] and environment['binary_sha256']==manifest['binary_sha256']
    selection=next(c for c in campaign_cases() if (c['cohort'],c['elements'],c['q'])==(manifest['cohort'],manifest['elements'],manifest['q']))
    assert selection['heat']==manifest['heat_included']
    mesh=Mesh(manifest['elements'],manifest['q']);assert mesh.size==manifest['dofs']
    truth=load_truth(manifest);check_truth(mesh,manifest,truth)
    specs=configuration_specs();cases=[load_bound_json(item) for item in manifest['cases']]
    assert len(cases)==len(specs) and {c['configuration_index'] for c in cases}==set(range(len(specs)))
    counts=dict(cases=len(cases),available=0,unavailable=0,operator_observations=0,operator_stages=0,poisson_solves=0,heat_trajectories=0,heat_steps=0,outputs=0)
    qualities={}
    for case in cases:
        index=case['configuration_index'];assert case['configuration']==specs[index]
        assert case['source_sha256']==manifest['source_sha256'] and case['binary_sha256']==manifest['binary_sha256']
        if not case['available']:
            config=case['configuration'];shared=config['family']=='ceed' and config['backend']=='ref' and mesh.q==16
            groups=128//mesh.q;columns=((3*mesh.elements**3*mesh.q**2+groups-1)//groups+127)//128*128
            bounded=config['family']=='tensor' and columns>8192
            assert shared or bounded,'Unexpected missing experiment'
            assert 'payloads' not in case and not case['heat'];finite_time(case['failed_setup_ms']);counts['unavailable']+=1;continue
        counts['available']+=1
        for key in ('construction_ms','first_data_load_ms','calibration_ms','first_complete_use_ms'):finite_time(case[key])
        arrays=load_payloads(case['payloads'])
        operators=verify_operator_case(mesh,truth,case,arrays,settings)
        poisson=verify_poisson(mesh,truth,case,arrays,settings)
        assert operators['used']|poisson['used']==set(case['outputs']),'Unreferenced final output certificate'
        heat=verify_heat(mesh,truth,case,settings,manifest['heat_included'])
        counts['operator_observations']+=operators['observations'];counts['operator_stages']+=operators['stages']
        counts['poisson_solves']+=poisson['solves'];counts['heat_trajectories']+=heat['trajectories'];counts['heat_steps']+=heat['steps']
        counts['outputs']+=len(case['outputs'])+heat['outputs']
        qualities[str(index)]=dict(operator=operators['quality'],poisson=poisson['quality'],heat=heat['quality'])
    return dict(**counts,collection_path=path.relative_to(ROOT).as_posix(),collection_sha256=file_hash(path),source_sha256=manifest['source_sha256'],binary_sha256=manifest['binary_sha256'],
                verifier_sha256={name:file_hash(ROOT/name) for name in VERIFIERS},qualities=qualities,
                scope='Complete retained numeric arithmetic/final original-system replay for this collection only; separate cost/full-set/publication gates still required')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--collection',required=True);parser.add_argument('--summary',required=True)
    args=parser.parse_args();result=replay_collection((ROOT/args.collection).resolve())
    Path(args.summary).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('PDE_COLLECTION_INDEPENDENT_NUMERIC_REPLAY_PASS',result['cases'],result['operator_observations'],result['poisson_solves'],result['heat_steps'])
