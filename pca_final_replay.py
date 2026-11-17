"""Independent CPU covariance assembly, spectrum/leading quality and timing replay."""
from collections import Counter
import hashlib
import json

import numpy as np

from pca_protocol import validate_binding
from pca_replay import replay_stage
from reference import digest


def _aggregate(data,method,stages,archive):
    n,d=data.shape
    unit=2.**(-24 if method['accumulation']=='fp32' else -53)
    if method['family']=='batch':unit=2.**-53
    tiny=np.finfo(np.float64).tiny
    total=0
    mean=np.zeros(d,np.float64)
    mean_error=np.zeros(d,np.float64)
    moment=np.zeros((d,d),np.float64)
    error=np.zeros((d,d),np.float64)
    counts=Counter()
    chunk=n if method['family']=='batch' else method['chunk']
    assert [stage['begin'] for stage in stages]==list(range(0,n,chunk)),'Incomplete or reordered stage coverage'
    global_center=None
    for stage in stages:
        arrays={key:archive[stage['prefix']+'_'+key] for key in stage['array_bindings']}
        checked=replay_stage(data,method,stage,arrays)
        for key in ('source_entries','encoded_buffers','full_product_entries','sampled_product_entries'):counts[key]+=checked[key]
        counts['native_stage_records']+=1
        # The independent interval includes reduction variation between timing
        # outputs and untimed Gram witnesses; this is not bitwise re-execution.
        scale=arrays['scale'].astype(np.float64)
        factor=scale[:,None]*scale[None,:]
        gram=checked['expected']*factor
        gram_error=checked['bound']*abs(factor)+8*unit*(abs(gram)+checked['bound']*abs(factor))+tiny
        center=arrays['center']
        if method['centering']=='merge':
            count=stage['count']
            delta=center-mean
            delta_error=mean_error+4*2.**-53*(abs(center)+abs(mean))
            weight=total*count/(total+count)
            correction=np.outer(delta,delta)*weight
            correction_error=(np.outer(abs(delta),delta_error)+np.outer(delta_error,abs(delta))+np.outer(delta_error,delta_error))*abs(weight)
            gram=gram+correction
            gram_error=gram_error+correction_error+16*unit*(abs(gram)+abs(correction))+tiny
            next_mean=(mean*total+center*count)/(total+count)
            mean_error=(mean_error*total)/(total+count)+16*2.**-53*(abs(mean)+abs(center)+abs(next_mean))+tiny
            mean=next_mean
            total+=count
        else:
            if global_center is None:global_center=center.copy()
            else:np.testing.assert_array_equal(center,global_center)
        next_moment=moment+gram
        error=error+gram_error+8*unit*(abs(moment)+abs(gram)+error+gram_error)+tiny
        moment=next_moment
    if method['centering']!='merge':
        mean=global_center
        mean_error=8*n*2.**-53*np.mean(abs(data.astype(np.float64)),axis=0)+tiny
    if method['centering']=='raw':
        arithmetic_dtype=np.float32 if method['accumulation']=='fp32' else np.float64
        typed_mean=mean.astype(arithmetic_dtype).astype(np.float64)
        correction=n*np.outer(typed_mean,typed_mean)
        error=error+16*unit*(abs(moment)+abs(correction))+tiny
        moment=moment-correction
    covariance=(moment+moment.T)*(.5/(n-1))
    envelope=(error+error.T)*(.5/(n-1))+8*unit*abs(covariance)+tiny
    return mean,mean_error,covariance,envelope,counts


def _quality(arrays,target):
    # Do not import or trust the producer's quality implementation/values.
    mean,covariance,eigenvalues,leading=(arrays[key].astype(np.float64) for key in ('mean','covariance','eigenvalues','leading'))
    norm=lambda value:float(np.sqrt(np.sum(value*value,dtype=np.float64)))
    tiny=np.finfo(np.float64).tiny
    rank=leading.shape[1]
    reference=target['covariance']
    spectrum=target['eigenvalues']
    radius=max(float(np.max(abs(spectrum))),tiny)
    floor=radius*1e-12
    gold=target['eigenvectors'][:,-rank:]
    residual=covariance@leading-leading*eigenvalues[None,-rank:]
    return dict(finite=True,
                covariance_relative_error=norm(covariance-reference)/max(norm(reference),tiny),
                maximum_scaled_eigenvalue_error=float(np.max(abs(eigenvalues-spectrum)/np.maximum(abs(spectrum),floor))),
                weak_eigenvalue_absolute_floor=floor,
                eigenvalues_relative_error=norm(eigenvalues-spectrum)/max(norm(spectrum),tiny),
                mean_standard_deviation_relative_error=norm(mean-target['mean'])/max(float(np.sqrt(max(float(np.sum(np.diag(reference))),0.))),tiny),
                subspace_projection_error=norm(leading@leading.T-gold@gold.T)/float(np.sqrt(2*rank)),
                target_relative_eigengap=float((spectrum[-rank]-spectrum[-rank-1])/radius),
                leading_eigensolver_relative_residual=norm(residual)/max(norm(covariance),tiny),
                leading_orthogonality_error=norm(leading.T@leading-np.eye(rank))/float(np.sqrt(rank)),
                covariance_symmetry_error=norm(covariance-covariance.T)/max(norm(covariance),tiny),
                negative_eigenvalues=int(np.count_nonzero(eigenvalues < -floor)))


def replay_case_outputs(data,record,archive,targets):
    n,d=data.shape
    assert record['complete'] and record['available'] and record['successfully_executed'] and record['arithmetic_valid'] and 'error' not in record
    assert record['shape']==[n,d] and record['input_variants']==2 and record['gpu_index']==0 and record['gpu_exclusive'] is False
    rank=record['retained_rank']
    assert rank==16 and rank<d
    variants=(data,np.ascontiguousarray(-data))
    assert record['input_sha256']==[digest(value) for value in variants]
    counts=Counter()
    models={}
    used_arrays=set()
    prefixes=set()
    for variant in range(2):
        stages=[stage for stage in record['stage_witnesses'] if stage['variant']==variant]
        assert stages
        for index,stage in enumerate(stages):
            assert stage['prefix']==f'sv{variant}_b{index}' and stage['prefix'] not in prefixes
            prefixes.add(stage['prefix'])
            used_arrays.update(stage['prefix']+'_'+key for key in stage['array_bindings'])
        models[variant]=_aggregate(variants[variant],record['method'],stages,archive)
        counts.update(models[variant][-1])
    assert sum(sum(s['variant']==v for s in record['stage_witnesses']) for v in (0,1))==len(record['stage_witnesses'])
    assert record['outputs'] and record['finite_outputs'] is True
    for key,output in record['outputs'].items():
        variant=output['variant']
        assert type(variant) is int and variant in (0,1)
        prefix=output['prefix']
        assert prefix not in prefixes
        prefixes.add(prefix)
        bindings=output['array_bindings']
        assert set(bindings)=={'mean','covariance','eigenvalues','leading'}
        arrays={name:archive[prefix+'_'+name] for name in bindings}
        used_arrays.update(prefix+'_'+name for name in bindings)
        validate_binding(arrays,bindings)
        identity=hashlib.sha256(json.dumps(bindings,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert key==f'v{variant}-'+identity,'Output identity/variant is not bound to its retained arrays'
        dtype=np.dtype('float64' if record['method']['family']=='batch' or record['method']['accumulation']=='fp64' else 'float32')
        for name,shape in (('mean',(d,)),('covariance',(d,d)),('eigenvalues',(d,)),('leading',(d,rank))):
            assert arrays[name].shape==shape and np.isfinite(arrays[name]).all()
            assert arrays[name].dtype==(np.dtype('float64') if name=='mean' else dtype)
        mean,mean_error,expected,bound,_=models[variant]
        assert np.all(abs(arrays['mean']-mean)<=mean_error+64*2.**-53*np.maximum(abs(mean),1.)),'Final PCA mean is not its original-data reduction'
        covariance=arrays['covariance'].astype(np.float64)
        assert np.all(abs(covariance-expected)<=bound),'Final covariance is not its independently reconstructed arithmetic'
        epsilon=2.**(-24 if dtype==np.dtype('float32') else -53)
        scale=max(float(np.linalg.norm(covariance)),np.finfo(np.float64).tiny)
        assert np.linalg.norm(covariance-covariance.T)<=64*d*epsilon*scale
        eigenvalues,leading=arrays['eigenvalues'].astype(np.float64),arrays['leading'].astype(np.float64)
        assert np.all(np.diff(eigenvalues)>=0)
        independent=np.linalg.eigvalsh((covariance+covariance.T)*.5)
        assert np.max(abs(eigenvalues-independent))<=64*d*epsilon*scale,'Retained spectrum does not solve the actual covariance'
        residual=covariance@leading-leading*eigenvalues[None,-rank:]
        assert np.linalg.norm(residual)<=64*d*epsilon*scale,'Leading eigenvectors fail the actual covariance residual'
        assert np.linalg.norm(leading.T@leading-np.eye(rank))<=64*d*epsilon*np.sqrt(rank)
        quality=_quality(arrays,targets[variant])
        assert set(quality)==set(output['quality'])
        for metric,value in quality.items():
            stated=output['quality'][metric]
            if type(value) in (bool,int):assert type(stated) is type(value) and stated==value,(key,metric)
            else:
                assert type(stated) in (float,int) and np.isfinite(stated)
                absolute=128*d*2.**-53 if metric in ('subspace_projection_error','leading_eigensolver_relative_residual','leading_orthogonality_error') else 32*2.**-53*abs(value)
                assert abs(stated-value)<=absolute+1e-10*abs(value),(key,metric,stated,value)
        counts['output_records']+=1
        counts['output_covariance_entries']+=d*d
        counts['output_mean_entries']+=d
        counts['output_spectrum_entries']+=d
        counts['output_leading_entries']+=d*rank
        counts['retained_negative_eigenvalue_outputs']+=int(quality['negative_eigenvalues']>0)
    assert set(archive.keys())==used_arrays,'Unbound or missing PCA witness arrays'
    observations,warmups=(5,2) if record['smoke'] else (30,5)
    assert record['observations_per_boundary']==observations and record['warmups_per_boundary']==warmups
    assert record['cohort'] in ('a','b')
    assert len(record['boundaries'])==2 and {b['name'] for b in record['boundaries']}=={'resident','host-service'}
    for boundary in record['boundaries']:
        assert len(boundary['observations'])==observations,'Incomplete PCA timing observations'
        for index,observation in enumerate(boundary['observations']):
            variant=index%2 if record['cohort']=='a' else 1-index%2
            assert observation['variant']==variant and observation['output_key'] in record['outputs']
            assert record['outputs'][observation['output_key']]['variant']==variant,'Timing is bound to a stale input output'
            assert np.isfinite(observation['wall_ms']) and observation['wall_ms']>0
            if boundary['name']=='resident':assert np.isfinite(observation['cuda_ms']) and observation['cuda_ms']>0 and observation['cuda_ms']<=observation['wall_ms']+1.
            else:assert observation['cuda_ms'] is None
            counts['timing_observations']+=1
    for field in ('setup_wall_ms','first_operation_wall_ms'):assert np.isfinite(record[field]) and record[field]>0
    metadata=record['metadata']
    assert metadata['host_original_bytes']==data.nbytes
    for field in ('python_owned_device_bytes','native_owned_device_bytes','owned_device_bytes'):assert type(metadata[field]) is int and metadata[field]>=0
    assert metadata['owned_device_bytes']==metadata['python_owned_device_bytes']+metadata['native_owned_device_bytes']
    assert metadata['precision']==record['method']['precision'] and metadata['centering']==record['method']['centering'] and metadata['scaling']==record['method']['scaling']
    assert metadata['chunk']==record['method']['chunk'] and metadata['accumulation']==('float32' if record['method']['accumulation']=='fp32' else 'float64')
    counts['cases']+=1
    return counts
