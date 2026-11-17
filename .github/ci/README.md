# CPU research CI

The workflow in `../workflows/cpu.yml` runs four independent verification groups:

| Group | Scope |
| --- | --- |
| `core` | Unit tests and the previously published checks before the search study |
| `search` | Search environment/input checks, complete replay and both summary checks |
| `pca` | Complete replay, cost/input-failure audits and exact receipt comparisons |
| `pde` | Complete numerical/cost replay, receipt comparisons and report check |

`commands.json` contains the exact commands. `run_group.py` uses the selected
Python interpreter, launches argument vectors without a shell, creates each
job's own `build/` directory and stops at the first unsuccessful check. Log
groups retain each command's output, exit status and elapsed time. BLAS/OpenMP
threads remain fixed at one; explicit replay worker counts are unchanged.

## Preserved certification boundary

`legacy_checks.json` records the 63 verification commands from published commit
`baa2991e2221accfb7f5f26730ab459d46ba46f5`. Tests require every command to remain
present, without cross-group duplication. Dependency installation and creation
of `build/` are common setup, not omitted scientific checks. No experiment,
tolerance, reference, retained result or numerical verifier was changed.

All four groups must succeed. The final job retains the check name **`verify`**
and fails for failed, cancelled or skipped matrix execution. `fail-fast: false`
lets the remaining groups finish even when another group fails. Each group has
a 120-minute job timeout; at most four groups run concurrently. The aggregate
job has a five-minute timeout and does not check out the research data again.

## Triggers and cancellation

- Pushes to `main` run the complete matrix, including documentation-only pushes.
- Pull requests run the complete matrix. A feature-branch push does not launch
  a second identical push workflow alongside its pull-request workflow.
- `workflow_dispatch` allows a complete manual run, including another branch.
- A newer run for the same branch or pull request cancels its superseded run.
- Tokens retain read-only repository permissions. These jobs use hosted Ubuntu
  runners, not the local GPU server and not a privileged pull-request trigger.

Splitting reduces serial waiting; it does not promise a proportional reduction
in total compute minutes. Each group has an isolated checkout and dependency
setup, so startup/checkout overhead is repeated. No artifact uploads or new
large-data caches were added. Existing pip caching keys include both dependency
files. Retained research data remains in Git rather than Actions artifact storage.

## Local checks

```bash
python -m unittest discover -s tests -p test_ci_pipeline.py -v
python .github/ci/run_group.py core --list
python .github/ci/run_group.py pde
```

The tests execute real success/failure subprocesses, verify fail-closed aggregate
statuses, and protect the published command inventory. Validate workflow syntax
with `actionlint` before publication. A local test or syntax pass is not a green
GitHub run: publication still requires successful jobs for the exact pushed SHA.
