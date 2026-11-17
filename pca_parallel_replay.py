"""Independent dataset subprocesses, one BLAS thread each, exact receipt union.

Every invocation actually replays every requested dataset. Intermediate receipts
are checkpoints, not trusted caches; no cached numerical replay is silently used.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def merge_replays(parts,expected_collections,complete)->dict:
    assert parts
    first=parts[0]
    counts=Counter()
    collections={}
    epochs={}
    additional={}
    for part in parts:
        assert part['scope']=='Bounded PCA subset replay; not full study completion'
        for name in ('producer_source_sha256','verifier_source_sha256','binary_sha256','limits'):
            assert part[name]==first[name],('Incompatible numerical replay dependencies',name)
        assert set(part['producer_epoch_collections'])==set(part['collection_sha256'])
        assert not set(collections).intersection(part['collection_sha256']),'Overlapping PCA replay partitions'
        collections.update(part['collection_sha256'])
        epochs.update(part['producer_epoch_collections'])
        for name,value in part['additional_producer_source_sha256'].items():
            assert name not in additional or additional[name]==value
            additional[name]=value
        counts.update(part['counts'])
    assert collections==expected_collections,'Missing or changed declared PCA replay partition'
    counts['collection_directories']=len({name.split('/')[0] for name in collections})
    return dict(scope='Complete declared study-11 collection replay' if complete else 'Bounded PCA subset replay; not full study completion',
                counts=dict(sorted(counts.items())),producer_source_sha256=first['producer_source_sha256'],
                verifier_source_sha256=first['verifier_source_sha256'],binary_sha256=first['binary_sha256'],
                collection_sha256=dict(sorted(collections.items())),producer_epoch_collections=dict(sorted(epochs.items())),
                additional_producer_source_sha256=additional,limits=first['limits'])


def run_parallel(args):
    from pca_cases import DATASETS,methods
    from pca_provenance import producer_sources,validate_sources
    from verify_pca import ROOT,VERIFIERS,leaf,sha
    directories=args.directory or [ROOT/'results/pca-smoke-a',ROOT/'results/pca-full-a',ROOT/'results/pca-full-b']
    original_sources=producer_sources(False)
    verifiers={name:sha(ROOT/name) for name in VERIFIERS}
    stamp=hashlib.sha256(json.dumps(verifiers,sort_keys=True).encode()).hexdigest()[:16]
    work=ROOT/'build/pca-replay-partitions'/stamp
    work.mkdir(parents=True,exist_ok=True)
    tasks=[]
    expected={}
    environments=set()
    for directory in directories:
        match=re.fullmatch(r'pca-(smoke|full)-([ab])',directory.name)
        assert match and directory.is_dir()
        names=[args.dataset] if args.dataset else ([next(iter(DATASETS))] if match[1]=='smoke' else list(DATASETS))
        if not args.dataset:assert {path.name for path in directory.glob('*-collection.json')}=={name+'-collection.json' for name in names},'Incomplete declared PCA dataset set'
        for name in names:
            path=directory/(name+'-collection.json')
            manifest=json.loads(path.read_text())
            assert manifest['complete']
            validate_sources(manifest['source_sha256'])
            identity=directory.name+'/'+path.name
            assert identity not in expected
            expected[identity]=sha(path)
            # Environment counts are a set union, not the sum of child counts.
            for item in manifest['cases']:
                case_path=leaf(directory,item['file'])
                assert sha(case_path)==item['sha256']
                case=json.loads(case_path.read_text())
                environments.add((case['environment_file'],case['environment_sha256']))
            tasks.append((directory,name,identity))
    environment=os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    def replay(task):
        directory,name,identity=task
        tag=directory.name+'__'+name+'__'+expected[identity][:16]
        receipt=work/(tag+'.json')
        log=work/(tag+'.log')
        command=[sys.executable,str(ROOT/'verify_pca.py'),'--workers','1','--directory',str(directory.resolve()),
                 '--dataset',name,'--summary',str(receipt)]
        with log.open('w') as stream:
            outcome=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=False)
        assert outcome.returncode==0,('PCA replay subprocess failed',identity,outcome.returncode,str(log))
        part=json.loads(receipt.read_text())
        assert part['collection_sha256']=={identity:expected[identity]}
        assert part['producer_source_sha256']==original_sources and part['verifier_source_sha256']==verifiers
        return part
    parts=[]
    failures=[]
    workers=min(args.workers,os.cpu_count() or 1)
    print('PCA_PARALLEL_REPLAY_START',len(tasks),'DATASETS',workers,'PROCESSES','BLAS_THREADS_PER_PROCESS=1',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(replay,task):task[2] for task in tasks}
        for future in as_completed(futures):
            try:
                parts.append(future.result())
                print('PCA_PARALLEL_PARTITION_PASS',len(parts),len(tasks),futures[future],flush=True)
            except Exception as error:
                failures.append((futures[future],str(error)))
                print('PCA_PARALLEL_PARTITION_FAILED',futures[future],str(error),flush=True)
    assert not failures,failures
    assert original_sources==producer_sources(False) and verifiers=={name:sha(ROOT/name) for name in VERIFIERS},'Replay dependencies changed during execution'
    for directory,name,identity in tasks:assert sha(directory/(name+'-collection.json'))==expected[identity]
    complete=args.directory is None and args.dataset is None
    result=merge_replays(parts,expected,complete)
    result['counts']['environment_snapshots']=len(environments)
    if complete:
        assert result['counts']['dataset_collections']==2*len(DATASETS)+1
        assert result['counts']['cases']==(2*len(DATASETS)+1)*(len(methods())+2)
    if args.summary:
        temporary=args.summary.with_suffix(args.summary.suffix+'.part')
        temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(args.summary)
    print('Independent PCA replay PASS',json.dumps(result['counts'],sort_keys=True),flush=True)
    return result
