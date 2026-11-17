"""Independently replay complete vector-search records, using CPU only.

Default requires every declared group in both full cohorts and the fresh smoke
cohort. --directory/--dataset selects an explicit bounded diagnostic subset;
that mode never certifies the whole study. No GPU imports or recorded quality
values are used as the arithmetic oracle.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from reference import digest
from search_cases import DATASETS,allow_mask,inputs,methods
from search_replay import independent_topk,output_metrics,validate_stage,validate_timing

ROOT=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    tree=ast.parse((ROOT/'search_study.py').read_text())
    names=next(ast.literal_eval(item.value) for item in tree.body if isinstance(item,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='SOURCES' for target in item.targets))
    return {name:sha(ROOT/name) for name in names}


def load_witness(path,record):
    witness=path.parent/record['witness_file']
    assert witness==path.with_suffix('.npz')
    assert sha(witness)==record['witness_sha256'],witness
    with np.load(witness,allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def verify_truth(directory,name,cohort,smoke,source_hashes,totals):
    path=directory/(name+'-truth.json')
    record=json.loads(path.read_text())
    variants,max_batch=(2,4) if smoke else (4,128)
    database,queries,provenance=inputs(name,cohort,variants,max_batch)
    assert record['complete'] and record['source_sha256']==source_hashes
    assert record['dataset']==name and record['cohort']==cohort
    assert record['database_shape']==list(database.shape) and record['queries_shape']==list(queries.shape)
    assert record['database_sha256']==digest(database) and record['queries_sha256']==digest(queries)
    assert record['provenance']==provenance
    arrays=load_witness(path,record)
    assert set(arrays)=={f'f{percent}_{field}' for percent in (100,10,1) for field in ('distances','ids')}
    assert [f['percent'] for f in record['filters']]==[100,10,1]
    for item in record['filters']:
        percent=item['percent']
        allowed=allow_mask(len(database),percent,cohort)
        assert item['allowed_count']==int(allowed.sum()) and item['allowed_sha256']==digest(allowed)
        assert np.isfinite(item['cpu_oracle_ms']) and item['cpu_oracle_ms']>0
        distances,ids=independent_topk(database,queries.reshape(-1,database.shape[1]),10,allowed)
        np.testing.assert_array_equal(arrays[f'f{percent}_ids'],ids.reshape(variants,max_batch,10))
        np.testing.assert_array_equal(arrays[f'f{percent}_distances'],distances.reshape(variants,max_batch,10))
        totals['independent_truth_queries']+=len(distances)
        totals['independent_truth_distances']+=int(allowed.sum())*len(distances)
    totals['truth_records']+=1
    print('INDEPENDENT_TRUTH_PASS',cohort,name,'smoke' if smoke else 'full',flush=True)
    return record,arrays,database,queries


def verify_group(path,database,queries,truth_record,truth_arrays,source_hashes,totals):
    record=json.loads(path.read_text())
    smoke=record['smoke']
    cohort,name,batch,percent=(record[k] for k in ('cohort','dataset','batch','allowed_percent'))
    assert record['study']==10 and record['complete'] and record['source_sha256']==source_hashes
    assert path.name==f'{name}-b{batch}-f{percent}.json'
    assert record['database_sha256']==digest(database) and record['queries_sha256']==digest(queries)
    assert record['truth_file']==name+'-truth.json'
    assert record['truth_record_sha256']==sha(path.parent/record['truth_file'])
    assert record['database_sha256']==truth_record['database_sha256']
    assert record['queries_sha256']==truth_record['queries_sha256']
    assert record['gpu_index']==0 and record['gpu_exclusive'] is False
    assert re.fullmatch('[0-9a-f]{64}',record['binary_sha256'])
    if (ROOT/'build/libsearch-full.so').exists():
        assert sha(ROOT/'build/libsearch-full.so')==record['binary_sha256']
    assert record['environment_file']=='../search-environment.json'
    assert sha(path.parent/record['environment_file'])==record['environment_sha256']
    allowed=allow_mask(len(database),percent,cohort)
    assert record['allowed_sha256']==digest(allowed)
    variants,observations,warmups=(2,5,2) if smoke else (4,30,5)
    assert record['query_variants']==variants and record['observations_per_boundary']==observations and record['warmups_per_boundary']==warmups
    grid=methods(len(database),int(allowed.sum()),percent)
    if cohort=='b':grid.reverse()
    assert [case['method'] for case in record['cases']]==grid
    arrays=load_witness(path,record)
    used=set()
    for index,case in enumerate(record['cases']):
        method=case['method']
        totals['attempted']+=1
        validate_timing(case,variants,observations)
        if not case['available']:
            assert method['family']=='ivf-pq' and percent!=100
            assert 'no filter argument' in case['error']
            assert not case['outputs'] and case['stage_witness'] is None
            totals['unsupported']+=1
            continue
        assert 'error' not in case,(path,index,case.get('error'))
        assert case['valid']==all(o['quality']['ids_valid'] for o in case['outputs'].values())
        assert case['valid'] or method['family']!='native'
        totals['available']+=1
        totals['valid_id_configurations' if case['valid'] else 'retained_invalid_id_configurations']+=1
        output_variants=set()
        first=None
        for key,output in case['outputs'].items():
            variant=output['variant']
            output_variants.add(variant)
            assert variant in range(variants)
            prefix=output['prefix']
            assert prefix.startswith('c'+str(index)+'_o')
            ids_key,dist_key=prefix+'_ids',prefix+'_distances'
            ids,distances=arrays[ids_key],arrays[dist_key]
            used.update((ids_key,dist_key))
            assert output['ids_sha256']==digest(ids) and output['distances_sha256']==digest(distances)
            assert key==f'v{variant}-'+digest(ids)+'-'+digest(distances)
            metrics,actual=output_metrics(database,queries[variant,:batch],distances,ids,truth_arrays[f'f{percent}_distances'][variant,:batch],truth_arrays[f'f{percent}_ids'][variant,:batch],allowed)
            assert metrics==output['quality'],(path,index,key,metrics,output['quality'])
            totals['retained_duplicate_ids']+=metrics['duplicate_ids']
            totals['retained_missing_ids']+=metrics['missing_ids']
            totals['retained_invalid_or_filtered_ids']+=metrics['invalid_or_filtered_ids']
            if variant==0 and first is None:first=(distances,ids)
            totals['output_records']+=1
            totals['output_ids']+=ids.size
        assert output_variants==set(range(variants))
        for boundary in case['boundaries']:
            totals['timing_observations']+=len(boundary['observations'])
        if method['family']=='native':
            assert case['stage_witness'] is not None and first is not None
            prefix=case['stage_witness']['prefix']+'_'
            stage={name[len(prefix):]:value for name,value in arrays.items() if name.startswith(prefix)}
            used.update(name for name in arrays if name.startswith(prefix))
            counts=validate_stage(stage,case['stage_witness'],method,database,queries[0,:batch],allowed,first)
            totals.update(counts)
            meta=case['metadata']
            assert meta['owned_device_bytes']>0 and meta['host_prepared_database_bytes']>=database.shape[1]*128*4
            assert meta['owned_device_bytes']==meta['native']['owned_device_bytes']+meta['mapping_owned_bytes']
            assert meta['padded_database_shape'][0]%128==0
            totals['native_stage_records']+=1
        else:
            assert case['stage_witness'] is None
            meta=case['metadata']
            assert meta['python_owned_array_bytes']>0
            from search_memory import faiss_reservation_bytes
            population=int(allowed.sum()) if percent!=100 else len(database)
            expected_reservation=faiss_reservation_bytes(population,batch) if method['family']=='faiss' else 0
            assert meta['faiss_temporary_reservation_bytes']==expected_reservation
            if method['family']=='faiss':
                assert meta['faiss_memory_info']['0']['TemporaryMemoryBuffer'][1]>=expected_reservation
            assert meta['cuvs_initial_workspace_pool_bytes']==(0 if method['family']=='faiss' else 64<<20)
    assert used==set(arrays),(path,'Unreferenced or missing witness arrays',set(arrays)-used,used-set(arrays))
    totals['groups']+=1
    print('GROUP_REPLAY_PASS',path.name,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',action='append',type=Path)
    parser.add_argument('--dataset',choices=DATASETS)
    parser.add_argument('--summary',type=Path)
    args=parser.parse_args()
    directories=args.directory or [ROOT/'results'/name for name in ('search-smoke-a','search-full-a','search-full-b')]
    totals=Counter()
    source_hashes=sources()
    for directory in directories:
        directory=directory.resolve()
        assert directory.is_dir(),directory
        match=re.fullmatch('search-(smoke|full)-([ab])',directory.name)
        assert match,directory
        smoke,cohort=match[1]=='smoke',match[2]
        names=[args.dataset] if args.dataset else list(DATASETS)
        for name in names:
            truth,truth_arrays,database,queries=verify_truth(directory,name,cohort,smoke,source_hashes,totals)
            batches=(1,4) if smoke else (1,32,128)
            paths=sorted(directory.glob(name+'-b*-f*.json'))
            assert {path.name for path in paths}=={f'{name}-b{b}-f{f}.json' for b in batches for f in (100,10,1)}
            for path in paths:
                verify_group(path,database,queries,truth,truth_arrays,source_hashes,totals)
        totals['cohorts']+=1
    result=dict(scope='Complete declared study-10 collection replay' if not args.directory and not args.dataset else 'Explicit bounded subset only; not whole-study certification',counts=dict(totals),producer_source_sha256=source_hashes,
                verifier_source_sha256={name:sha(ROOT/name) for name in ('verify_search.py','search_replay.py')})
    if args.summary:
        args.summary.write_text(json.dumps(result,indent=2)+'\n')
    print('Independent vector-search replay PASS',json.dumps(result['counts'],sort_keys=True),flush=True)


if __name__=='__main__':
    main()
