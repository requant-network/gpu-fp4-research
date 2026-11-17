"""Reproducible PCA quality/latency/reuse frontiers and bounded findings.

--preliminary derives explicitly provisional tables while numerical replay runs;
only the default final report requires the complete source-bound CPU receipt.
No experiment is replaced by a generated table or a smoke check.
"""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from pca_cases import DATASETS,methods
from pca_frontiers import QUALITY_METRICS,best,effective_ms,feasible,routes
from verify_pca import PRODUCERS,VERIFIERS
from pca_provenance import validate_sources

ROOT=Path(__file__).resolve().parent
BUDGETS=(.001,.01,.1,.5,1.)
REUSES=(1,4,16,64,256,1024,100000)
FIXED_TIMES=(.1,.25,.5,1.,2.,5.,10.,25.,100.)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_text(rows):
    assert rows
    text=io.StringIO()
    writer=csv.DictWriter(text,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue()


def load():
    sources={name:sha(ROOT/name) for name in PRODUCERS}
    groups=defaultdict(list)
    cases=[]
    manifests={}
    environments={}
    epochs={}
    for cohort in ('a','b'):
        directory=ROOT/'results'/f'pca-full-{cohort}'
        assert {p.name for p in directory.glob('*-collection.json')}=={name+'-collection.json' for name in DATASETS}
        for dataset,(n,d,profile) in DATASETS.items():
            manifest_path=directory/(dataset+'-collection.json')
            manifest=json.loads(manifest_path.read_text())
            assert manifest['complete'] and not manifest['smoke'] and manifest['cohort']==cohort and manifest['dataset']==dataset
            guarded=validate_sources(manifest['source_sha256'])
            epoch='guarded' if guarded else 'original'
            epochs[directory.name+'/'+manifest_path.name]=epoch
            assert len(manifest['cases'])==len(methods())+2
            manifests[directory.name+'/'+manifest_path.name]=sha(manifest_path)
            for index,item in enumerate(manifest['cases']):
                path=directory/item['file']
                assert sha(path)==item['sha256']
                case=json.loads(path.read_text())
                assert case['source_sha256']==manifest['source_sha256'] and case['shape']==[n,d] and case['case_index']==index
                assert case['cohort']==cohort and case['dataset']==dataset and not case['smoke']
                env_path=directory/case['environment_file']
                if env_path.name not in environments:
                    assert sha(env_path)==case['environment_sha256']
                    environments[env_path.name]=json.loads(env_path.read_text())
                environment=environments[env_path.name]
                category='tail' if dataset.startswith('tail-') else profile
                identity=directory.name+'/'+path.name
                cases.append((identity,case))
                for boundary,route in routes(case,environment['bootstrap_wall_ms']).items():
                    groups[(cohort,dataset,boundary)].append(dict(case_id=identity,cohort=cohort,dataset=dataset,profile=category,n=n,d=d,producer_epoch=epoch,
                                                               method=case['method'],**route))
    assert len(cases)==2*len(DATASETS)*(len(methods())+2)
    assert len(groups)==2*len(DATASETS)*2
    assert all(len(rows)==len(methods())+2 for rows in groups.values())
    return cases,groups,manifests,environments,sources,epochs


def derive(preliminary):
    cases,groups,manifests,environments,sources,epochs=load()
    audit=json.loads((ROOT/'results/pca-cost-audit.json').read_text())
    assert audit['producer_source_sha256']==sources
    for name,expected in audit['auditor_source_sha256'].items():assert sha(ROOT/name)==expected
    for name,expected in manifests.items():assert audit['collection_sha256'][name]==expected
    for name,epoch in epochs.items():assert audit['producer_epoch_collections'][name]==epoch
    failure=json.loads((ROOT/'results/pca-input-failure-audit.json').read_text())
    for name,expected in failure['verifier_source_sha256'].items():assert sha(ROOT/name)==expected
    assert failure['fixture_manifest_sha256']==sha(ROOT/'fixtures/pca-input-failure/manifest.json')
    change=json.loads((ROOT/'fixtures/pca-input-failure/input-change-forensics.json').read_text())['findings'][0]
    proof=None
    if not preliminary:
        proof=json.loads((ROOT/'results/pca-replay.json').read_text())
        assert proof['scope']=='Complete declared study-11 collection replay' and proof['producer_source_sha256']==sources
        assert proof['counts']['cases']==(2*len(DATASETS)+1)*(len(methods())+2)
        assert proof['verifier_source_sha256']=={name:sha(ROOT/name) for name in VERIFIERS}
        assert proof['collection_sha256']==audit['collection_sha256']
        assert proof['producer_epoch_collections']==audit['producer_epoch_collections']
        assert proof['additional_producer_source_sha256']==audit['additional_producer_source_sha256']
    counts=Counter(attempted=len(cases),groups=2*len(DATASETS),available=sum(case['available'] for _,case in cases),
                   successful=sum(case['successfully_executed'] for _,case in cases),
                   retained_outputs=sum(len(case['outputs']) for _,case in cases),
                   retained_significant_negative_eigenvalue_outputs=sum(output['quality']['negative_eigenvalues']>0 for _,case in cases for output in case['outputs'].values()),
                   timing_observations=sum(len(boundary['observations']) for _,case in cases for boundary in case['boundaries']))
    assert counts['attempted']==counts['available']==counts['successful']
    table=[]
    negatives=[]
    for _,rows in sorted(groups.items()):
        for row in rows:
            method=row['method']
            table.append(dict(cohort=row['cohort'],dataset=row['dataset'],profile=row['profile'],n=row['n'],d=row['d'],case_id=row['case_id'],producer_epoch=row['producer_epoch'],
                              family=method['family'],precision=method['precision'],centering=method['centering'],scaling=method['scaling'],accumulation=method['accumulation'],chunk=method['chunk'],
                              boundary=row['boundary'],median_ms=row['median_ms'],p25_ms=row['p25_ms'],p75_ms=row['p75_ms'],
                              setup_ms=row['setup_ms'],common_bootstrap_ms=row['common_bootstrap_ms'],named_owned_device_bytes=row['named_owned_device_bytes'],
                              finite=row['finite'],joint_quality_error=row['quality_error'],significant_negative_eigenvalues=row['negative_eigenvalues'],
                              minimum_relative_eigengap=row['minimum_relative_eigengap'],timed_output_count=row['timed_output_count'],**row['errors']))
            if not feasible(row,BUDGETS[0]):
                negatives.append(dict(case_id=row['case_id'],boundary=row['boundary'],method=method,joint_quality_error=row['quality_error'],
                                      significant_negative_eigenvalues=row['negative_eigenvalues'],errors=row['errors']))
    frontier=[]
    for (cohort,dataset,boundary),rows in sorted(groups.items()):
        for budget in BUDGETS:
            for reuse in REUSES:
                for cold in (False,True):
                    proposed=best(rows,budget,reuse,cold,fp4=True)
                    reference=best(rows,budget,reuse,cold,fp4=False)
                    frontier.append(dict(cohort=cohort,dataset=dataset,profile=rows[0]['profile'],producer_epoch=rows[0]['producer_epoch'],boundary=boundary,budget=budget,reuse=reuse,include_common_bootstrap=cold,
                                         fp4_feasible=proposed is not None,reference_feasible=reference is not None,
                                         fp4_case=proposed['case_id'] if proposed else None,reference_case=reference['case_id'] if reference else None,
                                         fp4_method=json.dumps(proposed['method'],sort_keys=True) if proposed else None,
                                         reference_method=json.dumps(reference['method'],sort_keys=True) if reference else None,
                                         fp4_ms=proposed['effective_ms'] if proposed else None,reference_ms=reference['effective_ms'] if reference else None,
                                         reference_over_fp4=reference['effective_ms']/proposed['effective_ms'] if proposed and reference else None,
                                         fp4_win=bool(proposed and reference and proposed['effective_ms']<reference['effective_ms']),
                                         fp4_error=proposed['quality_error'] if proposed else None,reference_error=reference['quality_error'] if reference else None,
                                         fp4_named_device_bytes=proposed['named_owned_device_bytes'] if proposed else None,
                                         reference_named_device_bytes=reference['named_owned_device_bytes'] if reference else None))
    fixed=[]
    for (cohort,dataset,boundary),rows in sorted(groups.items()):
        for reuse in (1,64,100000):
            for limit in FIXED_TIMES:
                choices={}
                for fp4 in (False,True):
                    eligible=[row for row in rows if (row['method']['precision']=='fp4')==fp4 and row['finite'] and row['negative_eigenvalues']==0 and effective_ms(row,reuse,False)<=limit]
                    choices[fp4]=min(eligible,key=lambda row:(row['quality_error'],effective_ms(row,reuse,False),row['case_id'])) if eligible else None
                fixed.append(dict(cohort=cohort,dataset=dataset,producer_epoch=rows[0]['producer_epoch'],boundary=boundary,reuse=reuse,time_limit_ms=limit,
                                  fp4_error=choices[True]['quality_error'] if choices[True] else None,
                                  reference_error=choices[False]['quality_error'] if choices[False] else None,
                                  fp4_case=choices[True]['case_id'] if choices[True] else None,
                                  reference_case=choices[False]['case_id'] if choices[False] else None))
    def aggregate(items):
        ratios=[row['reference_over_fp4'] for row in items if row['reference_over_fp4'] is not None]
        return dict(groups=len(items),fp4_feasible=sum(row['fp4_feasible'] for row in items),references_feasible=sum(row['reference_feasible'] for row in items),
                    comparable=len(ratios),fp4_wins=sum(row['fp4_win'] for row in items),median_reference_over_fp4=float(np.median(ratios)) if ratios else None)
    findings={}
    for boundary in ('resident','host-service'):
        for budget in BUDGETS:
            chosen=[row for row in frontier if row['boundary']==boundary and row['budget']==budget and row['reuse']==100000 and not row['include_common_bootstrap']]
            findings[boundary+'-budget-'+str(budget)]=aggregate(chosen)
    summary=dict(status='PRELIMINARY: GPU data-derived; complete independent numerical replay still required' if preliminary else 'Complete declared PCA collection independently replayed; bounded ex-post findings, not a future-input guarantee',
                 counts=dict(counts),budgets=list(BUDGETS),reuse_counts=list(REUSES),findings=findings,
                 producer_source_sha256=sources,additional_producer_source_sha256=audit['additional_producer_source_sha256'],
                 producer_epoch_collections=dict(sorted(epochs.items())),
                 analysis_source_sha256={name:sha(ROOT/name) for name in ('summarize_pca.py','pca_frontiers.py','pca_provenance.py')},
                 input_failure_audit_sha256=sha(ROOT/'results/pca-input-failure-audit.json'),
                 collection_sha256=dict(sorted(manifests.items())),cost_audit_sha256=sha(ROOT/'results/pca-cost-audit.json'))
    if proof:summary['independent_replay_counts']=proof['counts']
    prefix='pca-preliminary' if preliminary else 'pca'
    outputs={ROOT/'results'/f'{prefix}-cases.csv':csv_text(table),ROOT/'results'/f'{prefix}-frontiers.csv':csv_text(frontier),
             ROOT/'results'/f'{prefix}-fixed-time.csv':csv_text(fixed),ROOT/'results'/f'{prefix}-negative-results.json':json.dumps(negatives,indent=2,allow_nan=False)+'\n',
             ROOT/'results'/f'{prefix}-summary.json':json.dumps(summary,indent=2,allow_nan=False)+'\n'}
    lines=['# Streaming covariance and PCA on consumer Blackwell','',
           '**PRELIMINARY — complete independent numerical replay pending. Not study completion.**' if preliminary else '**Bounded study 11: complete GPU collection and independent CPU replay. No novelty, physical-peak-memory or future-input error guarantee.**','',
           '## Scope and protocol','',
           f"All {counts['groups']} cohort/dataset groups contain {counts['attempted']:,} attempted, available and successfully executed configurations. Each group tests 84 streamed configurations plus untiled FP32/FP64 library controls. Full A/B data contain {counts['timing_observations']:,} recorded resident/host observations and {counts['retained_outputs']:,} distinct retained output records. Supplementary smoke is separate and does not create a third independent cohort.",'',
           'Synthetic uniform, shifted-by-32, nearly dependent and drifting inputs use N8192/32768 and D128/256; a nonaligned N8193 tail is included. Genuine SIFT1M descriptor windows use N8192/65536,D128, with disjoint original rows for A (starting at 0) and B (starting at 500000). Seeds and dimensions are in pca_cases.py; seed labels are not acquisition dates. Two alternating sign variants are changing inputs, not independent cohorts. Measurement UTC dates: '+', '.join(sorted({case['finished_utc'].split('T')[0] for _,case in cases}))+'.','',
           'Each streamed operation recomputes original-data centering, optional power-of-two feature scaling, transpose/padding, native encoding, every Gram product, FP32/64 moment accumulation, symmetrization and complete GPU eigh. Raw, globally centered and count-weighted chunk-centered merge are tested, with chunks2048 and merge/scaled/FP64-accumulation controls at256/8192. No prepacked-input or GEMM-only speed claim. Strong references include every non-FP4 streamed precision and the untiled whole-matrix FP32/64 library pipelines, not only the slower chunked variants.','',
           'The pinned CuPy eigensolver dispatches a real symmetric two-dimensional input to its cuSOLVER syevd path; this adapter enables linalg error reporting and independently checks results, rather than treating API return as proof.[1] The native FP32 bridge requests CUBLAS_COMPUTE_32F_PEDANTIC, while half/BF16 candidates are filtered by recorded HMMA numerical implementation flags. These are API/source/flag observations, not newly collected instruction-counter traces.[2] Untiled FP32 and FP64 controls are labelled by their actual library APIs, without an unsupported claim about internal instructions.','',
           '## Independent numerical evidence','']
    if proof:
        replay=proof['counts']
        lines.append(f"The complete source-bound replay covers {replay['cases']:,} records including smoke, {replay['native_stage_records']:,} stage witnesses, {replay['output_covariance_entries']:,} final covariance entries, {replay['output_spectrum_entries']:,} retained eigenvalues, {replay['output_leading_entries']:,} leading-vector entries and {replay['timing_observations']:,} timing observations. Exact scope/hashes/counters are in results/pca-replay.json.")
    else:lines.append('Partial replays are available, but no whole-study numerical certificate is claimed here. Tables use the sealed GPU records and remain provisional until the default verify_pca.py run passes all declared collections.')
    lines+=['',
            'Original FP32 observations are regenerated or read from fingerprinted genuine SIFT fixtures. The independent CPU model uses interleaved FP64 reductions for complete truth covariance, checks its full reference eigenspectrum/eigenbasis, reconstructs preparation/encoding and full-or-sampled native Gram arithmetic, then checks all final covariance entries within propagated product/accumulation envelopes. Full Gram witnesses are retained for uniform8192/D128; elsewhere all diagonal plus fixed sampled Gram entries are checked. These envelopes are not bitwise GPU re-execution.','',
            'Complete means/covariances/eigenvalues and leading16 eigenvectors are retained. All eigenvectors were computed and transferred, but unretained full eigenbases are not independently certified. Actual covariance-to-eigenvalue and leading-vector residual/orthogonality checks are separate from original-data PCA quality; an internally consistent eigendecomposition of a wrong covariance does not establish task correctness. Tests reject such a forged covariance with recomputed spectrum/basis/quality/hashes, and reject stale-input timing bindings.','',
            'Joint quality is the worst, across actually timed output records, of covariance Frobenius relative error, full-spectrum relative error, maximum per-eigenvalue error scaled by max(abs(reference eigenvalue),1e-12*reference spectral radius), mean error normalized by reference total standard deviation and leading16 projector error normalized by sqrt(32). Significant negative eigenvalues below -1e-12*reference spectral radius make a route ineligible. Eigengaps remain diagnostic; a small gap does not silently remove the subspace constraint. FP64 reference and the declared weak-eigenvalue floor limit what can be inferred about near-null directions.','',
            '## Rejected original input and guarded repeat','',
            f"The original strict replay correctly rejected a mean that did not reduce the declared observations. An independently reproduced original mean census covers {failure['original_census_counts']['cases']:,} cases and {failure['original_census_counts']['outputs']:,} retained outputs; {failure['original_census_counts']['retained_output_mean_integrity_failures']} outputs fail the unchanged original-data mean bound, all in full-B shifted32768/D256 variant1. A one-word reconstruction at row{change['row']}, feature{change['column']} changes {change['original']} to {change['observed']}: bits {change['original_bits']} to {change['observed_bits']}, XOR {change['xor']} ({change['bit_count']} bit). It matches the retained operand fingerprint exactly, not approximately.",'',
            f"The public rejected fixture retains the original full collection verbatim. Independent reconstruction checks all {failure['rejected_source_counts']['retained_prepared_sources']:,} archived prepared-source fingerprints using retained center/scale and phase-appropriate inputs, including {failure['rejected_source_counts']['changed_input_phase_sources']:,} in the changed-input phase. It also reproduces all original mean-census failure records. This identifies altered inputs, not the physical origin, host-versus-device location or a hardware/vendor culprit. A retained 100-iteration focused diagnostic reports clean host/device fingerprints and means but does not exclude rare failures. The compute-sanitizer attempt did not finish; it is not counted as a clean sanitizer result.",'',
            f"All 86 configurations of that original dataset collection are excluded from the canonical numerical/performance corpus, not repaired, averaged into good records or counted as low-precision quality failures. Only this dataset was recollected in a fresh guarded process. The full corpus explicitly contains {sum(epoch=='original' for epoch in epochs.values())} original-producer collections and {sum(epoch=='guarded' for epoch in epochs.values())} guarded repeat. The original twelve producer dependencies remain byte-unchanged; the repeat additionally binds pca_study_guarded.py and pca_input_integrity.py. Its host inputs are read-only and every timed observation records exact original-host pre/post and device-original-input post fingerprints. Registry, replay, cost audit and CSV producer_epoch columns distinguish both source epochs.",'',
            'Guard checks and their host/device transfers occur outside the named resident/host-service timers. They change inter-request spacing and may change cache/thermal state, so the replacement is not an unchanged-spacing A/B experiment. Comparisons remain within the same dataset/cohort/source epoch; aggregate group counts do not make source-epoch or thermal equivalence claims. The replacement still passes the original numerical tolerances; no mean/covariance bound was widened to accept damaged data. Rejected evidence is under fixtures/pca-input-failure, independently audited by verify_pca_input_failure.py and results/pca-input-failure-audit.json.','',
            '## Quality-constrained device results','',
            'Each row selects the fastest feasible FP4 configuration and the fastest feasible non-FP4 strong reference for the same cohort/dataset/boundary. This is an ex-post oracle frontier, not a deployable safe selector. The table uses resident CUDA events and plan reuse100000, excluding common bootstrap from this warm-process view but charging constructor and first operation divided by reuse. Ratios are reference time / FP4 time; only comparable feasible groups enter their median.','',
            '| Joint budget | FP4 feasible / groups | Reference feasible | FP4 wins | Comparable | Median reference / FP4 |',
            '|---|---:|---:|---:|---:|---:|']
    for budget in BUDGETS:
        result=findings['resident-budget-'+str(budget)]
        ratio='—' if result['median_reference_over_fp4'] is None else f"{result['median_reference_over_fp4']:.6f}"
        lines.append(f"| {budget:g} | {result['fp4_feasible']}/{result['groups']} | {result['references_feasible']} | {result['fp4_wins']} | {result['comparable']} | {ratio} |")
    lines+=['','### Profile-level diagnosis at joint budget0.1','',
            '| Input category | Groups | FP4 feasible | FP4 wins | Median reference / FP4 |','|---|---:|---:|---:|---:|']
    for category in sorted({row['profile'] for row in frontier}):
        selected=[row for row in frontier if row['profile']==category and row['boundary']=='resident' and row['budget']==.1 and row['reuse']==100000 and not row['include_common_bootstrap']]
        result=aggregate(selected)
        ratio='—' if result['median_reference_over_fp4'] is None else f"{result['median_reference_over_fp4']:.6f}"
        lines.append(f"| {category} | {result['groups']} | {result['fp4_feasible']} | {result['fp4_wins']} | {ratio} |")
    lines+=['','## Preparation, host service and reuse','',
            'Resident events cover complete GPU centering/encoding/products/accumulation/eigh, with the current original FP32 observations already uploaded. Python launch gaps and GPU status checks inside the operation remain inside that boundary; result D2H is outside. Host-service wall time instead includes finite-input validation, adapter host-original copy, pageable H2D, the complete GPU operation/synchronization and full mean/covariance/eigenvalues/eigenvectors D2H. Dataset generation, truth calculation, hashes, quality and serialization are outside both named service boundaries.','',
            'For each route, effective time is median service + (constructor + first operation)/R. The separate common-bootstrap view also charges observed device/library/small-operation initialization/R to every candidate as a sole-mode counterfactual. This begins at main-entry device selection and excludes interpreter startup and earlier Python/CuPy imports. It is shared inside the measured dataset process, not a complete fresh-process, fresh-install or cold-OS startup measurement. CSVs retain R1/4/16/64/256/1024/100000, both bootstrap views and both service boundaries. Fixed-time tables report the best measured joint error under named time limits, without quality-budget preselection.','',
            '## Named memory and calibration audit','',
            f"The independent cost audit passed {audit['counts']['cases']:,} full+smoke records, including {audit['counts']['calibration_observations']:,} eligible heuristic-calibration observations. Selected candidates match the measured eligible median minimum. All live named Python allocations match their original/operand/product/mean/moment/covariance/spectrum/eigenvector geometries. Native encoded/scales/workspace requests match the fixed adapter formula. Named owned requests range from {audit['minimum_named_owned_device_bytes']:,} to {audit['maximum_named_owned_device_bytes']:,} bytes across the declared records.",'',
            'These are live named allocation requests, not physical device peak. Opaque cuSOLVER/cuBLAS internals, allocator cache, temporary expression high-watermarks and shared desktop/other-context residency are excluded. Named memory columns are not whole-device memory savings. Full named buffers and cold/setup costs are retained even when a mode loses. The audit and its exact scope are in results/pca-cost-audit.json.','',
            '## Negative outcomes and interpretation','',
            f"The full GPU collection retains {counts['retained_significant_negative_eigenvalue_outputs']:,} output records with significant negative eigenvalues. {len(negatives):,} route/configuration records fail the strict joint0.001 budget or its finite/PSD gate. Neither group is removed to manufacture a successful low-precision result. Detailed errors and all measured routes are retained in the negative-results and cases artifacts.",'',
            'Raw moment subtraction intentionally exposes a sensitive recipe: quantized second moments minus an original-data mean outer product need not preserve small covariance directions. Global/chunk centering, feature scaling and higher accumulation precision address different sources of error; the measured tables, not a single dominant norm, determine which combinations are feasible. A small covariance norm error can coexist with weak-spectrum or subspace failure. No conclusion about every possible Tensor Core PCA implementation follows from this fixed adapter/grid.','',
            '## Executed environment and limitations','',
            'All measurements target GPU0 RTX5060Ti16GB, SM120, in a shared nonexclusive system. Each dataset ran in a fresh process, with its own common-bootstrap snapshot and normal process exit. GPU1, unrelated services, clocks and power settings were not changed. The original full campaigns reverse method/dataset order for B. The later guarded recovery keeps B method order within its dataset but does not preserve the original global dataset schedule. Neither ordering creates a controlled exclusive thermal/frequency experiment. Installed package metadata, loaded-library fingerprints, observed runtime/driver APIs and nvcc output remain separate in each environment JSON; a version mismatch does not identify its cause.','',
            'Observed runtime API values: '+', '.join(map(str,sorted({env['runtime_api_version'] for env in environments.values()})))+'; driver API values: '+', '.join(map(str,sorted({env['driver_api_version'] for env in environments.values()})))+'. CuPy versions: '+', '.join(sorted({env['cupy_version'] for env in environments.values()}))+'; NumPy versions: '+', '.join(sorted({env['numpy_version'] for env in environments.values()}))+'. The build script pins the CUTLASS source and compiles the SM120 native path. API FP64 is not asserted to be a particular instruction implementation.','',
            'Original observations fit device memory. This is a bounded streamed-chunk covariance/PCA pipeline, not a >VRAM online-arrival experiment. Two cohorts and thirty timing samples per boundary do not establish population confidence intervals, future-input guarantees, architectural ceilings or scientific novelty.','',
            '## Reproduction and artifacts','',
            'Use the pinned GPU environment already described by requirements-search-gpu.txt and its CUDA/CUTLASS build prerequisites. PCA does not import Faiss or cuVS; installing a package is not evidence that it executed. Real descriptor rights/provenance remain in THIRD_PARTY.md and fixtures/search-sift/manifest.json.','',
            '```bash',
            'mkdir -p build',
            'cp native/bridge.cu build/base-128.cuh',
            'bash build_pca.sh /path/to/pinned/cutlass',
            'export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1',
            '# A fresh process per dataset; reverse dataset order as well for B.',
            'for cohort in a b; do',
            '  datasets=$(python -c "from pca_cases import DATASETS; names=list(DATASETS); print(\' \'.join(names[::-1] if \'$cohort\'==\'b\' else names))")',
            '  for dataset in $datasets; do',
            '    producer=pca_study.py',
            '    if [ "$cohort" = b ] && [ "$dataset" = shifted-n32768-d256 ]; then producer=pca_study_guarded.py; fi',
            '    python "$producer" --cohort "$cohort" --dataset "$dataset"',
            '  done',
            'done',
            '# Supplementary, not a full campaign:',
            'CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python pca_study.py --cohort a --smoke',
            'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_pca.py --workers 4 --summary results/pca-replay.json',
            'OPENBLAS_NUM_THREADS=1 python verify_pca_input_failure.py --summary results/pca-input-failure-audit.json',
            'python pca_cost_audit.py --summary results/pca-cost-audit.json',
            'python summarize_pca.py',
            'python summarize_pca.py --check',
            '```','',
            'Executed records are in results/pca-full-a, results/pca-full-b and results/pca-smoke-a. Final derived cases/frontiers/fixed-time/negative/summary artifacts use results/pca-*; preliminary artifacts use pca-preliminary-* and are not completion evidence. CPU-only replay never loads GPU producer modules. Preserve executed sources/binary/witness hashes before changes; resume requires matching bindings. Passing CPU/artifact CI on the exact published head remains a separate completion gate.','',
            'Existing sealed GPU records are resumed, not remeasured. To recollect, first archive and move the existing canonical collection directories out of the way; never overwrite source-bound evidence or interpret RESUME as a new experiment. CPU --workers4 launches independent dataset subprocesses with one BLAS/OpenMP thread each. Every requested partition is actually replayed; intermediate receipts are checkpoints, not silently trusted numerical caches. The reducer rejects overlap, missing collections, changed hashes and mixed verifier dependencies before producing the whole-study receipt.','',
            'The example GPU loop assigns the accepted source epochs for a fresh recollection. It does not recreate the accidental historical input change or its later recovery chronology, nor guarantee equivalent historical cache/thermal conditions. Historical UTC timestamps, source epochs and the rejected original remain in the retained artifacts.','',
            (ROOT/'PCA_SOURCES.md').read_text().strip(),'']
    outputs[ROOT/('build/PCA_PRELIMINARY.md' if preliminary else 'PCA_REPORT.md')]='\n'.join(lines)
    return outputs,summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preliminary',action='store_true')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    outputs,summary=derive(args.preliminary)
    for path,text in outputs.items():
        if args.check:assert path.exists() and path.read_text()==text,('PCA derived artifact mismatch',path)
        else:path.parent.mkdir(exist_ok=True,parents=True);path.write_text(text)
    print('PCA_PRELIMINARY_DERIVED_PASS' if args.preliminary else 'PCA_FINAL_DERIVED_PASS',json.dumps(summary['counts'],sort_keys=True),flush=True)
    print(json.dumps(summary['findings'],sort_keys=True),flush=True)


if __name__=='__main__':main()
