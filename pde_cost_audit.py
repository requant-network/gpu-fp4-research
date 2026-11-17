"""Independent declared allocation geometry, tuning and timing-boundary audit.

Request categories are not physical peaks. libCEED opaque allocations remain
unmeasured; do not rank its memory against another family's fuller accounting.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from pde_cases import configuration_specs
from pde_reference import Mesh
from pde_replay import ROOT,file_hash,load_bound_json,finite_time


def audit_metadata(mesh,configuration,metadata,binary_map):
    n,q,local=mesh.size,mesh.q,mesh.elements**3*mesh.q**3
    family,precision=configuration['family'],configuration['precision']
    calibration=0
    if family=='ceed':
        expected={'data':8*n,'masked':8*n,'output':8*n,'boundary':n}
        assert metadata['python_named_device_buffers']==expected
        assert metadata['python_named_owned_device_bytes']==sum(expected.values())
        assert metadata['libceed_requested_qdata_bytes']==32*local,'libCEED qdata geometry mismatch'
        assert metadata['libceed_requested_index_bytes']==4*local,'libCEED restriction geometry mismatch'
        assert metadata['requested_resource']=='/gpu/cuda/'+configuration['backend'] and metadata['root_resource']==metadata['requested_resource']
        assert metadata['source_commit']=='3e98d546622c05ba46cfce6d4ba2e7d0b9ad4b97'
        assert metadata['bridge_binary_sha256']==binary_map['ceed_bridge'] and metadata['libceed_binary_sha256']==binary_map['libceed']
        return dict(python_named_bytes=sum(expected.values()),native_named_bytes=None,libceed_request_qdata_bytes=32*local,libceed_request_index_bytes=4*local,calibration_observations=0)
    size=4 if family=='cuda' and precision=='fp32' else 8
    expected={'data':8*n,'output':8*n,'local':local*size,'indices':4*local,'stiffness':q*q*8,'weights':q*8,'mass':8*n,'boundary':n}
    native=0
    assert metadata['component_precision']==precision and metadata['threads']==configuration['threads']
    if family=='tensor':
        groups=128//q;vectors=3*mesh.elements**3*q*q;columns=((vectors+groups-1)//groups+127)//128*128
        assert columns<=8192,'Successful grouped native shape violates its bound'
        expected.update(left=128*128*4,right=columns*128*4,product=128*columns*4)
        declared=metadata['layout']
        for key,value in dict(rows=128,reduction=128,columns=columns,groups=groups,physical_vectors=vectors,physical_values=vectors*q,padded_values=columns*128).items():assert declared[key]==value
        if precision=='fp4':
            # Fixed M/N multiples128,K128: packed values plus both logical and
            # physical SF requests, zero CUTLASS scheduler workspace.
            native=(128+columns)*128*5//8
            assert metadata['encoder_variant']==12
        else:native=(128+columns)*128*{'fp8':1,'fp16':2,'bf16':2,'fp32':4}[precision]+(32<<20)
        assert metadata['native_binary_sha256']==binary_map['native_core']
    elif family=='small':
        pad=16 if precision=='fp8' else max(8,q) if precision in ('fp16','bf16') else q
        columns=3*mesh.elements**3*q*q
        assert metadata['pad']==pad and metadata['columns']==columns
        expected.update(left=pad*pad*4,right=columns*pad*4,product=pad*columns*4)
        native=(pad*pad+columns*pad)*{'fp8':1,'fp16':2,'bf16':2,'fp32':4}[precision]+(32<<20)
        assert metadata['native_binary_sha256']==binary_map['short_q_bridge']
    else:assert family=='cuda'
    assert metadata['named_device_buffers']==expected,'Python allocation geometry mismatch'
    python=sum(expected.values())
    if family in ('tensor','small'):
        assert metadata['python_named_owned_device_bytes']==python
        assert metadata['native_owned_device_bytes']==native,'Native allocation geometry mismatch'
        assert metadata['product_storage']=='fp32' and metadata['post_arithmetic']=='fp64'
        tuning=metadata['tuning'];assert tuning and len({r['candidate'] for r in tuning})==len(tuning)
        available=[]
        for row in tuning:
            assert type(row['candidate']) is int and row['candidate']>=0 and type(row['flags']) is int and row['flags']>=0
            assert type(row['available']) is bool
            if row['available']:
                assert len(row['cuda_ms'])==3
                for value in row['cuda_ms']:finite_time(value)
                if precision in ('fp16','bf16'):assert row['flags']&2
                available.append(row);calibration+=len(row['cuda_ms'])
            else:assert precision in ('fp16','bf16') and not row['flags']&2 and 'cuda_ms' not in row
        assert available
        selected=min(available,key=lambda r:statistics.median(r['cuda_ms']))
        assert selected['candidate']==metadata['selected_candidate'] and selected['flags']==metadata['selected_flags'],'Selection is not the measured eligible minimum'
    assert metadata['named_owned_device_bytes']==python+native,'Combined named allocation geometry mismatch'
    return dict(python_named_bytes=python,native_named_bytes=native,calibration_observations=calibration)


def audit_collection(path):
    path=Path(path);manifest=json.loads(path.read_text());mesh=Mesh(manifest['elements'],manifest['q'])
    binary=manifest['binary_sha256'];settings=manifest['protocol'];specs=configuration_specs();counts=Counter();geometries={}
    assert math.isfinite(manifest['common_preparation_ms']) and manifest['common_preparation_ms']>=0
    high=dict(family='cuda',precision='fp64',threads=256)
    common=audit_metadata(mesh,high,manifest['common_reference_metadata'],binary)
    assert manifest['common_named_device_bytes']==32*mesh.size+common['python_named_bytes']
    records=[load_bound_json(i) for i in manifest['cases']]
    assert len(records)==len(specs) and {r['configuration_index'] for r in records}==set(range(len(specs)))
    for case in records:
        counts['cases']+=1;index=case['configuration_index'];assert case['configuration']==specs[index]
        if not case['available']:
            availability=case['availability'];config=case['configuration'];error=availability['error']
            if config['family']=='ceed':assert config['backend']=='ref' and mesh.q==16 and 'uses too much shared data' in error and '0xc000 max' in error
            else:
                groups=128//mesh.q;columns=((3*mesh.elements**3*mesh.q**2+groups-1)//groups+127)//128*128
                assert config['family']=='tensor' and columns>8192 and 'dimensions 1..8192' in error
            finite_time(case['failed_setup_ms']);counts['unavailable']+=1;continue
        counts['available']+=1
        result=audit_metadata(mesh,case['configuration'],case['method_metadata'],binary);geometries[str(index)]=result
        counts['metadata_checks']+=1;counts['calibration_observations']+=result['calibration_observations']
        for field in ('construction_ms','first_data_load_ms','calibration_ms','first_complete_use_ms'):finite_time(case[field])
        for row in case['operator_observations']:
            finite_time(row['wall_ms']);finite_time(row['cuda_ms']);counts['operator_timings']+=1
        for row in case['poisson_runs']:finite_time(row['wall_ms']);counts['poisson_timings']+=1
        assert len(case['heat'])==(len(settings['heat_steps']) if manifest['heat_included'] else 0)
        for heat in case['heat']:
            finite_time(heat['setup_ms'])
            for config,meta in ((case['configuration'],heat['method_metadata']),(high,heat['original_metadata'])):
                value=audit_metadata(mesh,config,meta,binary);counts['metadata_checks']+=1;counts['calibration_observations']+=value['calibration_observations']
            assert heat['retained_gpu_state_rhs_bytes']==heat['steps']*2*mesh.size*8
            for run in heat['runs']:finite_time(run['wall_ms']);counts['heat_timings']+=1
    return dict(**dict(counts),geometries=geometries,collection_sha256=file_hash(path),collection_path=path.relative_to(ROOT).as_posix(),
                auditor_sha256=file_hash(ROOT/'pde_cost_audit.py'),
                limits='Declared live/request categories and observed selection only. Opaque CEED/library temporaries, allocator caches, Python CG reduction temporaries and physical peaks excluded. Shared benchmark preparation includes reference-field generation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--collection',required=True);parser.add_argument('--summary',required=True);args=parser.parse_args()
    result=audit_collection((ROOT/args.collection).resolve());Path(args.summary).write_text(json.dumps(result,indent=2)+'\n')
    print('PDE_NAMED_COST_AUDIT_PASS',result['cases'],result['metadata_checks'],result['calibration_observations'])
