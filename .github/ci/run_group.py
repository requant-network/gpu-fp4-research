"""Execute one independent, explicitly declared CPU verification group."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def load_groups():
    manifest=json.loads((Path(__file__).with_name('commands.json')).read_text())
    if manifest.get('schema')!=1 or not isinstance(manifest.get('groups'),dict):
        raise ValueError('Invalid CI group manifest')
    groups=manifest['groups']
    if not groups or any(not checks or not isinstance(checks,list) or
                         any(not isinstance(command,str) or not command for command in checks)
                         for checks in groups.values()):
        raise ValueError('Every CI group must contain explicit checks')
    return groups


def run_checks(checks,root=ROOT):
    root=Path(root);(root/'build').mkdir(exist_ok=True)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    for number,command in enumerate(checks,1):
        argv=shlex.split(command)
        if not argv or argv[0] not in ('python','git'):
            raise ValueError('CI commands must invoke Python or Git directly, without a shell')
        if argv[0]=='python':argv[0]=sys.executable
        print(f'::group::Check {number}/{len(checks)}: {command}',flush=True)
        began=time.monotonic()
        result=subprocess.run(argv,cwd=root,env=env,check=False)
        print(f'CI_CHECK_RESULT exit={result.returncode} elapsed_seconds={time.monotonic()-began:.3f}',flush=True)
        print('::endgroup::',flush=True)
        if result.returncode:return result.returncode
    return 0


if __name__=='__main__':
    groups=load_groups();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group',choices=tuple(groups));parser.add_argument('--list',action='store_true')
    args=parser.parse_args()
    if args.list:
        print(json.dumps(groups[args.group],indent=2));raise SystemExit(0)
    raise SystemExit(run_checks(groups[args.group]))
