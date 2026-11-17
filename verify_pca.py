"""CPU-only source/truth/stage/final-output/timing replay of PCA collections.

Default requires the complete declared full-A/full-B and supplementary smoke-A
collections. Explicit directories/datasets produce a labelled bounded-subset
receipt, never a whole-study certificate. No GPU producer/module is imported.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from pca_cases import COHORTS,DATASETS,inputs,methods
from pca_output_replay import replay_case_outputs,replay_truth
from pca_provenance import BASE_PRODUCERS,GUARDED_ADDITIONS,validate_sources,verify_guarded_observations
from reference import digest

ROOT=Path(__file__).resolve().parent
PRODUCERS=BASE_PRODUCERS
VERIFIERS=('verify_pca.py','pca_output_replay.py','pca_final_replay.py','pca_replay.py','pca_provenance.py','pca_parallel_replay.py')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def leaf(directory,name):
    assert isinstance(name,str) and re.fullmatch(r'[A-Za-z0-9_.-]+',name) and name not in ('.','..'),name
    return directory/name


def check_environment(path,expected,binary):
    assert sha(path)==expected
    environment=json.loads(path.read_text())
    assert environment['gpu_index']==0 and environment['gpu_exclusive'] is False
    assert environment['gpu_name']=='NVIDIA GeForce RTX 5060 Ti' and environment['compute_capability']==[12,0]
    assert type(environment['multiprocessors']) is int and environment['multiprocessors']>0
    assert np.isfinite(environment['bootstrap_wall_ms']) and environment['bootstrap_wall_ms']>0
    for name in ('runtime_api_version','driver_api_version','cublas_lt_version'):assert type(environment[name]) is int and environment[name]>0
    assert environment['cupy_version'] and environment['numpy_version'] and environment['nvcc'] and environment['python']
    assert environment['loaded_libraries']
    for item in environment['loaded_libraries']:
        assert leaf(Path('.'),item['name']).name==item['name']
        assert type(item['bytes']) is int and item['bytes']>0 and re.fullmatch('[0-9a-f]{64}',item['sha256'])
    assert any(item['name']=='libpca.so' and item['sha256']==binary for item in environment['loaded_libraries'])
    for item in environment['packages'].values():
        if item is not None:
            assert item['version']
            assert item['record_sha256'] is None or re.fullmatch('[0-9a-f]{64}',item['record_sha256'])
    return environment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,action='append')
    parser.add_argument('--dataset',choices=DATASETS)
    parser.add_argument('--summary',type=Path)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    assert 1<=args.workers<=16
    if args.workers>1:
        from pca_parallel_replay import run_parallel
        run_parallel(args)
        return
    directories=args.directory or [ROOT/'results/pca-smoke-a',ROOT/'results/pca-full-a',ROOT/'results/pca-full-b']
    producers={name:sha(ROOT/name) for name in PRODUCERS}
    counts=Counter()
    environments=set()
    collections={}
    epochs={}
    binary=None
    for directory in directories:
        match=re.fullmatch(r'pca-(smoke|full)-([ab])',directory.name)
        assert match and directory.is_dir(),directory
        smoke,cohort=match[1]=='smoke',match[2]
        assert cohort in COHORTS
        names=[args.dataset] if args.dataset else ([next(iter(DATASETS))] if smoke else list(DATASETS))
        if not args.dataset:assert {p.name for p in directory.glob('*-collection.json')}=={name+'-collection.json' for name in names},'Incomplete declared PCA dataset set'
        for dataset in names:
            n,d,profile=DATASETS[dataset]
            data,provenance=inputs(n,d,profile,cohort)
            variants=(data,np.ascontiguousarray(-data))
            manifest_path=directory/(dataset+'-collection.json')
            manifest=json.loads(manifest_path.read_text())
            assert manifest['complete'] and manifest['study']==11 and manifest['smoke']==smoke and manifest['cohort']==cohort and manifest['dataset']==dataset
            guarded=validate_sources(manifest['source_sha256'])
            executed_sources=manifest['source_sha256']
            assert re.fullmatch('[0-9a-f]{64}',manifest['binary_sha256'])
            if binary is None:binary=manifest['binary_sha256']
            assert manifest['binary_sha256']==binary
            assert manifest['input_sha256']==[digest(value) for value in variants]
            truth_path=leaf(directory,manifest['truth_file'])
            assert truth_path.name==dataset+'-truth.json' and sha(truth_path)==manifest['truth_record_sha256']
            truth=json.loads(truth_path.read_text())
            assert truth['complete'] and truth['study']==11 and truth['source_sha256']==executed_sources
            assert truth['shape']==[n,d] and truth['cohort']==cohort and truth['dataset']==dataset
            assert truth['provenance']==provenance and truth['input_sha256']==manifest['input_sha256']
            truth_npz=leaf(directory,truth['witness_file'])
            assert sha(truth_npz)==truth['witness_sha256']
            with np.load(truth_npz,allow_pickle=False) as archive:
                assert set(archive.files)=={f'v{v}_{field}' for v in (0,1) for field in ('mean','covariance','eigenvalues','eigenvectors')}
                targets=[{field:archive[f'v{v}_{field}'] for field in ('mean','covariance','eigenvalues','eigenvectors')} for v in (0,1)]
            for variant,target in zip(variants,targets):counts.update(replay_truth(variant,target));counts['truth_input_variants']+=1
            assert truth['common_original_host_bytes']==sum(value.nbytes for value in variants)
            grid=[dict(family='stream',**method) for method in methods()]
            grid.extend(dict(family='batch',precision=precision,centering='global',scaling=False,accumulation='fp64',chunk=n) for precision in ('fp32','fp64'))
            if cohort=='b':grid.reverse()
            assert len(manifest['cases'])==len(grid)
            for index,(item,method) in enumerate(zip(manifest['cases'],grid)):
                path=leaf(directory,item['file'])
                assert path.name==f'{dataset}-m{index:03d}.json' and sha(path)==item['sha256']
                record=json.loads(path.read_text())
                assert record['study']==11 and record['cohort']==cohort and record['dataset']==dataset and record['smoke']==smoke and record['case_index']==index
                assert record['source_sha256']==executed_sources and record['binary_sha256']==binary and record['method']==method
                assert record['truth_file']==truth_path.name and record['truth_record_sha256']==sha(truth_path)
                assert record['input_sha256']==manifest['input_sha256']
                env_path=leaf(directory,record['environment_file'])
                env_identity=(env_path.name,record['environment_sha256'])
                if env_identity not in environments:
                    check_environment(env_path,record['environment_sha256'],binary)
                    environments.add(env_identity)
                witness=leaf(directory,record['witness_file'])
                assert witness.name==path.with_suffix('.npz').name and sha(witness)==record['witness_sha256']
                with np.load(witness,allow_pickle=False) as archive:counts.update(replay_case_outputs(data,record,archive,targets))
                counts['guarded_input_observations']+=verify_guarded_observations(record,guarded)
                counts['guarded_cases']+=int(guarded)
            identity=directory.name+'/'+manifest_path.name
            assert identity not in collections
            collections[identity]=sha(manifest_path)
            epochs[identity]='guarded' if guarded else 'original'
            counts['dataset_collections']+=1
            print('PCA_DATASET_REPLAY_PASS',directory.name,dataset,len(grid),flush=True)
    counts['environment_snapshots']=len(environments)
    counts['collection_directories']=len(directories)
    complete=args.directory is None and args.dataset is None
    if complete:
        assert counts['dataset_collections']==2*len(DATASETS)+1
        assert counts['cases']==(2*len(DATASETS)+1)*(len(methods())+2)
    result=dict(scope='Complete declared study-11 collection replay' if complete else 'Bounded PCA subset replay; not full study completion',
                counts=dict(sorted(counts.items())),producer_source_sha256=producers,
                verifier_source_sha256={name:sha(ROOT/name) for name in VERIFIERS},binary_sha256=binary,collection_sha256=dict(sorted(collections.items())),
                producer_epoch_collections=dict(sorted(epochs.items())),
                additional_producer_source_sha256={name:sha(ROOT/name) for name in GUARDED_ADDITIONS} if counts['guarded_cases'] else {},
                limits='Independent original-data FP64 truth and full retained final covariance/spectrum/leading16 replay; native stage arithmetic is full for uniform8192, all-diagonal plus sampled elsewhere. Unretained full eigenbases, opaque workspace/cache/temporary peak memory and exclusive-system performance are not certified.')
    if args.summary:args.summary.write_text(json.dumps(result,indent=2)+'\n')
    print('Independent PCA replay PASS',json.dumps(result['counts'],sort_keys=True),flush=True)


if __name__=='__main__':main()
