"""Sequential fresh-process full PDE campaign with external fixture closure.

Run after the retained smoke has been independently replayed. One authorized
GPU campaign only. Failed/partial children are never silently called complete.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pde_cases import campaign_cases

ROOT=Path(__file__).resolve().parent


def sha(path):
    result=hashlib.sha256()
    with open(path,'rb') as source:
        while block:=source.read(1024*1024):result.update(block)
    return result.hexdigest()


def schedule():
    cases=campaign_cases()
    return [c for c in cases if c['cohort']=='a']+list(reversed([c for c in cases if c['cohort']=='b']))


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Only authorized physical GPU0'
    receipt_path=ROOT/'results/pde-full-launch.json'
    assert not receipt_path.exists(),'Do not overwrite an executed launch receipt'
    smoke=json.loads((ROOT/'results/pde-smoke-a/e2-q4/collection.json').read_text())
    replay=json.loads((ROOT/'results/pde-smoke-replay.json').read_text())
    assert replay['collection_sha256']==sha(ROOT/replay['collection_path'])
    assert replay['cases']==16 and replay['operator_observations']==128 and replay['heat_steps']==128
    for name,expected in replay['verifier_sha256'].items():assert sha(ROOT/name)==expected
    baseline=smoke['source_sha256']
    fixture_names=('fixtures/pde-gll.json','fixtures/pde-gll.npz')
    fixtures={name:sha(ROOT/name) for name in fixture_names}
    controls=json.loads((ROOT/'results/pde-cuda-preflight.json').read_text())
    assert fixtures==controls['data_sha256'],'Canonical coefficients differ from original GPU control epoch'
    launcher_sha=sha(__file__)
    record=dict(scope='One sequential fresh-process full campaign; numerical/cost/report/publication gates separate',complete=False,
                launcher_sha256=launcher_sha,producer_source_sha256=baseline,fixture_sha256=fixtures,
                smoke_replay_sha256=sha(ROOT/'results/pde-smoke-replay.json'),schedule=schedule(),children=[])
    def save():receipt_path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    save()
    for case in schedule():
        for name,expected in baseline.items():assert sha(ROOT/name)==expected,'Producer changed during full campaign'
        assert sha(__file__)==launcher_sha and {name:sha(ROOT/name) for name in fixture_names}==fixtures
        command=[sys.executable,str(ROOT/'pde_study.py'),'--cohort',case['cohort'],'--elements',str(case['elements']),'--q',str(case['q'])]
        label=case['cohort']+'-e'+str(case['elements'])+'-q'+str(case['q'])
        log=ROOT/'build'/('pde-full-'+label+'.log');log.parent.mkdir(exist_ok=True)
        started=time.time_ns()
        with log.open('w') as output:code=subprocess.call(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        entry=dict(case=case,exit_code=code,started_unix_ns=started,completed_unix_ns=time.time_ns(),
                   fixture_after_sha256={name:sha(ROOT/name) for name in fixture_names})
        record['children'].append(entry);save()
        assert code==0,('Actual dataset subprocess failed; inspect its private build log',label,code)
        assert entry['fixture_after_sha256']==fixtures
        path=ROOT/'results'/('pde-full-'+case['cohort'])/('e'+str(case['elements'])+'-q'+str(case['q']))/'collection.json'
        manifest=json.loads(path.read_text())
        assert manifest['source_sha256']==baseline and manifest['protocol']['phase']=='full' and len(manifest['cases'])==16
        entry['collection']=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path));save()
        print('PDE_FULL_DATASET_PROCESS_EXIT0',label,flush=True)
    assert len(record['children'])==len(schedule()) and sha(__file__)==launcher_sha
    record['complete']=True;save()
    print('PDE_FULL_GPU_COLLECTION_COMPLETE_NOT_YET_WHOLE_REPLAYED',len(record['children']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--describe',action='store_true');args=parser.parse_args()
    if args.describe:print(json.dumps(schedule(),indent=2))
    else:run()
