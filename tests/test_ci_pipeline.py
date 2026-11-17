"""Fail-closed tests for CI orchestration, not replacements for numerical replay."""
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
CI=ROOT/'.github/ci'


def runner():
    spec=importlib.util.spec_from_file_location('cpu_ci_runner',CI/'run_group.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class CiPipelineTests(unittest.TestCase):
    def test_all_published_checks_are_preserved_without_cross_group_duplication(self):
        config=json.loads((CI/'commands.json').read_text())
        baseline=json.loads((CI/'legacy_checks.json').read_text())['commands']
        self.assertEqual(set(config['groups']),{'core','search','pca','pde'})
        actual=[command for group in config['groups'].values() for command in group]
        self.assertFalse(Counter(baseline)-Counter(actual),'Published scientific checks were dropped')
        self.assertTrue(all(count==1 for count in Counter(actual).values()))
        self.assertEqual(set(runner().load_groups()),set(config['groups']))

    def test_group_runner_stops_at_the_first_real_subprocess_failure(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as temporary:
            root=Path(temporary)
            checks=['python -c "from pathlib import Path; Path(\'first\').write_text(\'ok\'); raise SystemExit(17)"',
                    'python -c "from pathlib import Path; Path(\'must-not-run\').write_text(\'bad\')"']
            self.assertEqual(runner().run_checks(checks,root),17)
            self.assertTrue((root/'first').exists())
            self.assertFalse((root/'must-not-run').exists())

    def test_runner_executes_successful_checks_in_the_selected_working_directory(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as temporary:
            root=Path(temporary)
            self.assertEqual(runner().run_checks(['python -c "from pathlib import Path; Path(\'ok\').write_text(\'yes\')"'],root),0)
            self.assertEqual((root/'ok').read_text(),'yes')

    def test_workflow_keeps_parallel_limits_cancellation_and_a_fail_closed_verify(self):
        text=(ROOT/'.github/workflows/cpu.yml').read_text()
        for required in ('max-parallel: 4','fail-fast: false','timeout-minutes: 120',
                         'cancel-in-progress: true','pull_request:','workflow_dispatch:',
                         'branches: [main]','contents: read','needs: research','if: always()',
                         'group: [core, search, pca, pde]','python .github/ci/run_group.py'):
            self.assertIn(required,text)
        self.assertNotIn('pull_request_target',text)
        self.assertNotIn('continue-on-error',text)
        self.assertNotIn('self-hosted',text)
        self.assertNotIn('paths-ignore',text)
        self.assertRegex(text,r'(?m)^  verify:\s*$')
        self.assertIsNotNone(re.search(r'(?m)^      - run: test "\$REPLAY_RESULT" = success$',text))
        shell=shutil.which('bash')
        if os.name=='nt':
            git=shutil.which('git')
            candidates=[parent/relative for parent in Path(git).parents[:3]
                        for relative in ('bin/bash.exe','usr/bin/bash.exe')] if git else []
            shell=next((str(path) for path in candidates if path.is_file()),None)
        if shell:
            for status in ('success','failure','cancelled','skipped',''):
                result=subprocess.run([shell,'-c','test "$REPLAY_RESULT" = success'],
                                      env={**os.environ,'REPLAY_RESULT':status})
                self.assertEqual(result.returncode==0,status=='success')


if __name__=='__main__':unittest.main()
