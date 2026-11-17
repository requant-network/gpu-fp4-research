"""Fresh four-worker CPU replay with exact full-set and launch/source closure.

Every dataset launches a fresh BLAS1 subprocess. Earlier dataset receipts are
not trusted caches. Output unions reject overlap and incompatible source epochs.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from pde_cases import campaign_cases
from pde_cost_audit import audit_collection
from pde_replay import ROOT,file_hash,VERIFIERS

COUNTS=('cases','available','unavailable','operator_observations','operator_stages','poisson_solves','heat_trajectories','heat_steps','outputs')


def merge_replays(parts,expected):
    assert parts,'Empty numerical replay union'
    seen=set();counts=Counter();collections={};qualities={};first=parts[0]
    for part in parts:
        name=part['collection_path']
        assert name not in seen,'Overlapping collection replay'
        for field in ('source_sha256','binary_sha256','verifier_sha256'):assert part[field]==first[field],'Incompatible numerical source/binary/verifier epochs'
        assert part['cases']==part['available']+part['unavailable']
        for key in COUNTS:
            assert type(part[key]) is int and part[key]>=0
            counts[key]+=part[key]
        seen.add(name);collections[name]=part['collection_sha256'];qualities[name]=part['qualities']
    assert seen==set(expected),'Replay does not cover the exact declared collection set'
    return dict(collections=len(seen),counts=dict(counts),collection_sha256=dict(sorted(collections.items())),qualities=dict(sorted(qualities.items())),
                source_sha256=first['source_sha256'],binary_sha256=first['binary_sha256'],verifier_sha256=first['verifier_sha256'])


def declared(smoke_only=False):
    result={'results/pde-smoke-a/e2-q4/collection.json':('smoke','a',2,4)}
    if not smoke_only:
        for case in campaign_cases():
            name='results/pde-full-'+case['cohort']+'/e'+str(case['elements'])+'-q'+str(case['q'])+'/collection.json'
            result[name]=('full',case['cohort'],case['elements'],case['q'])
    return result


def check_launch(expected):
    path=ROOT/'results/pde-full-launch.json';record=json.loads(path.read_text())
    assert record['complete'] and record['launcher_sha256']==file_hash(ROOT/'run_pde_campaign.py')
    assert record['smoke_replay_sha256']==file_hash(ROOT/'results/pde-smoke-replay.json')
    assert len(record['children'])==len(campaign_cases())
    cases=campaign_cases();schedule=[c for c in cases if c['cohort']=='a']+list(reversed([c for c in cases if c['cohort']=='b']))
    assert record['schedule']==schedule and [r['case'] for r in record['children']]==schedule
    names=set()
    for row in record['children']:
        assert row['exit_code']==0 and row['completed_unix_ns']>=row['started_unix_ns']
        assert row['fixture_after_sha256']==record['fixture_sha256']
        binding=row['collection'];name=binding['path'];assert name not in names
        names.add(name);assert binding['sha256']==file_hash(ROOT/name)
        phase,cohort,e,q=expected[name]
        assert phase=='full' and (row['case']['cohort'],row['case']['elements'],row['case']['q'])==(cohort,e,q)
    assert names=={name for name,value in expected.items() if value[0]=='full'}
    for name,sha in record['fixture_sha256'].items():assert file_hash(ROOT/name)==sha
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=file_hash(path))


def run(smoke_only,workers,summary,cost_summary):
    assert type(workers) is int and 1<=workers<=8
    expected=declared(smoke_only)
    actual={p.relative_to(ROOT).as_posix() for p in (ROOT/'results').glob('pde-full-*/*/collection.json')}
    if not smoke_only:assert actual=={n for n,v in expected.items() if v[0]=='full'},'Full on-disk collection set is incomplete or has extras'
    launch=None if smoke_only else check_launch(expected)
    for name,(phase,cohort,e,q) in expected.items():
        manifest=json.loads((ROOT/name).read_text())
        assert (manifest['protocol']['phase'],manifest['cohort'],manifest['elements'],manifest['q'])==(phase,cohort,e,q),'Dataset identity does not match declared geometry/cohort'
    environment=os.environ.copy();environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    output_directory=ROOT/'build'/('pde-replay-'+uuid.uuid4().hex);output_directory.mkdir(parents=True)
    def replay(name):
        target=output_directory/(str(list(expected).index(name))+'.json')
        command=[sys.executable,str(ROOT/'pde_replay.py'),'--collection',name,'--summary',str(target)]
        result=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,text=True)
        assert result.returncode==0,('Actual numerical replay subprocess failed',name,result.stdout,result.stderr)
        part=json.loads(target.read_text())
        assert part['collection_path']==name and part['collection_sha256']==file_hash(ROOT/name)
        assert part['verifier_sha256']=={f:file_hash(ROOT/f) for f in VERIFIERS}
        cost=audit_collection(ROOT/name)
        assert cost['collection_sha256']==part['collection_sha256']
        print('PDE_DATASET_FRESH_NUMERIC_AND_COST_REPLAY_EXIT0',name,flush=True)
        return part,cost
    parts=[];costs=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(replay,name) for name in expected]
        for future in as_completed(futures):
            part,cost=future.result();parts.append(part);costs.append(cost)
    result=merge_replays(parts,expected)
    result.update(scope='Smoke-only numeric/cost replay' if smoke_only else 'Whole exact declared PDE collection numeric replay, with separate named cost audit',
                  full_study_numeric_set=not smoke_only,launch=launch,
                  driver_sha256=file_hash(ROOT/'verify_pde.py'),workers=workers,blas_threads_per_dataset=1,
                  limits='All retained original-data stages/final vectors and original physical residuals; not unretained full CG-intermediate replay, physical peaks, instruction trace or publication completion.')
    Path(summary).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    count=Counter()
    for cost in costs:
        for key,value in cost.items():
            if type(value) is int:count[key]+=value
    cost_result=dict(scope='Exact same collection set named allocation/tuning audit, not physical peaks',counts=dict(count),
                     collection_sha256=result['collection_sha256'],source_sha256=result['source_sha256'],
                     auditor_sha256=file_hash(ROOT/'pde_cost_audit.py'),driver_sha256=file_hash(ROOT/'verify_pde.py'),
                     per_collection={c['collection_path']:c for c in sorted(costs,key=lambda r:r['collection_path'])})
    Path(cost_summary).write_text(json.dumps(cost_result,indent=2,allow_nan=False)+'\n')
    print('PDE_EXACT_SET_FRESH_PARALLEL_REPLAY_PASS',result['collections'],json.dumps(result['counts'],sort_keys=True),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke-only',action='store_true');parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--summary',required=True);parser.add_argument('--cost-summary',required=True);args=parser.parse_args()
    run(args.smoke_only,args.workers,args.summary,args.cost_summary)
