"""Derive ex-post search quality/latency/reuse frontiers and an English report.

Only complete full A/B collections can produce SEARCH_REPORT.md. --smoke writes
separately named diagnostic artifacts, never a study-completion report.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from search_cases import DATASETS
from search_frontiers import routes

ROOT=Path(__file__).resolve().parent
BUDGETS=(.9,.95,.99,1.)
REUSE=(1,4,16,64,256,1024,100000)
BOUNDARIES=('resident-uncaptured','best-resident','host-service')


def method_name(method):
    family=method['family']
    if family=='native':
        names=('CUTLASS-FP4','FP8','FP16','BF16','strict-FP32','cuBLASLt-FP4')
        return names[method['precision']]+f"/C{method['candidates']}/"+('refine' if method['refine'] else 'direct')+f"/norm{method['norm_mode']}"+('/compact' if method['compact'] else '')
    return family+'/'+','.join(f'{k}={v}' for k,v in sorted(method.items()) if k!='family')


def read_records(smoke):
    records=[]
    for cohort in (('a',) if smoke else ('a','b')):
        directory=ROOT/'results'/f'search-{"smoke" if smoke else "full"}-{cohort}'
        paths=sorted(directory.glob('*-b*-f*.json'))
        batches=(1,4) if smoke else (1,32,128)
        expected={f'{name}-b{b}-f{f}.json' for name in DATASETS for b in batches for f in (100,10,1)}
        assert {p.name for p in paths}==expected,(directory,'Incomplete declared collection')
        for path in paths:
            record=json.loads(path.read_text())
            assert record['complete'] and record['smoke']==smoke and record['cohort']==cohort
            record['_file']=path.relative_to(ROOT).as_posix()
            records.append(record)
    assert len({record['binary_sha256'] for record in records})==1
    assert len({json.dumps(record['source_sha256'],sort_keys=True) for record in records})==1
    return records


def extract(records):
    rows=[]
    counts=Counter(groups=len(records))
    negative=[]
    for record in records:
        group=(record['cohort'],record['dataset'],record['batch'],record['allowed_percent'])
        for index,case in enumerate(record['cases']):
            method=case['method']
            counts['attempted']+=1
            if not case['available']:
                counts['unsupported']+=1
                negative.append(dict(group=list(group),case=index,method=method,kind='unavailable',diagnostic=case.get('error')))
                continue
            counts['available']+=1
            if not case['valid']:
                counts['retained_invalid_id_configurations']+=1
            if not case['graph_available']:
                counts['ineligible_graphs']+=1
                negative.append(dict(group=list(group),case=index,method=method,kind='ineligible graph',diagnostic=case.get('graph_error')))
            for output in case['outputs'].values():
                quality=output['quality']
                counts['retained_output_records']+=1
                counts['retained_duplicate_ids']+=quality['duplicate_ids']
                counts['retained_missing_ids']+=quality['missing_ids']
                counts['retained_invalid_or_filtered_ids']+=quality['invalid_or_filtered_ids']
                if quality['duplicate_ids'] or quality['missing_ids'] or quality['invalid_or_filtered_ids'] or quality['nonfinite_reported_valid_ids']:
                    negative.append(dict(group=list(group),case=index,method=method,kind='output quality failure',output_prefix=output['prefix'],variant=output['variant'],quality=quality))
            meta=case.get('metadata',{})
            for name,route in routes(case).items():
                row=dict(cohort=group[0],dataset=group[1],batch=group[2],allowed_percent=group[3],case=index,
                         method=method_name(method),family=method['family'],fp4_proposal=method['family']=='native' and method['precision'] in (0,5),
                         route=name,actual_boundary=route['boundary'],median_ms=route['median_ms'],p25_ms=route['p25_ms'],p75_ms=route['p75_ms'],
                         min_strict_recall=route['recall'],min_tie_aware_recall=route['tie_aware_recall'],finite_complete_ids=route['finite_complete_ids'],
                         setup_ms=route['setup_ms'],native_owned_device_bytes=meta.get('owned_device_bytes'),python_owned_array_bytes=meta.get('python_owned_array_bytes'),
                         faiss_allocation_info=json.dumps(meta.get('faiss_memory_info'),sort_keys=True),
                         cuvs_initial_workspace_pool_bytes=meta.get('cuvs_initial_workspace_pool_bytes'),
                         source_record=record['_file'])
                rows.append(row)
    return rows,counts,negative


def frontier(rows,group,boundary,budget,reuse):
    available=[r for r in rows if (r['cohort'],r['dataset'],r['batch'],r['allowed_percent'])==group and r['route']==boundary and r['finite_complete_ids'] and r['min_tie_aware_recall']>=budget]
    def best(subset):
        if not subset:return None
        result=min(subset,key=lambda r:r['median_ms']+r['setup_ms']/reuse)
        return result,result['median_ms']+result['setup_ms']/reuse
    proposal=best([r for r in available if r['fp4_proposal']])
    reference=best([r for r in available if r['family']!='native'])
    native=best([r for r in available if r['family']=='native'])
    result=dict(cohort=group[0],dataset=group[1],batch=group[2],allowed_percent=group[3],boundary=boundary,tie_aware_recall_budget=budget,reuse_batches=reuse,
                fp4_method=proposal[0]['method'] if proposal else None,fp4_total_ms_per_batch=proposal[1] if proposal else None,
                reference_method=reference[0]['method'] if reference else None,reference_total_ms_per_batch=reference[1] if reference else None,
                any_native_method=native[0]['method'] if native else None,any_native_total_ms_per_batch=native[1] if native else None,
                fp4_speedup=reference[1]/proposal[1] if reference and proposal else None,
                fp4_wins=bool(reference and proposal and proposal[1]<reference[1]),
                any_native_speedup=reference[1]/native[1] if reference and native else None)
    return result


def csv_text(rows):
    output=io.StringIO(newline='')
    if not rows:return ''
    writer=csv.DictWriter(output,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def derive(smoke):
    records=read_records(smoke)
    rows,counts,negative=extract(records)
    groups=sorted({(r['cohort'],r['dataset'],r['batch'],r['allowed_percent']) for r in rows})
    assert len(groups)==len(records)
    # Reuse=None is not inferred: every amortized number uses an explicit batch count.
    fronts=[frontier(rows,group,boundary,budget,reuse) for group in groups for boundary in BOUNDARIES for budget in BUDGETS for reuse in REUSE]
    fixed=[]
    for group in groups:
        feasible=[r for r in rows if (r['cohort'],r['dataset'],r['batch'],r['allowed_percent'])==group and r['route']=='best-resident' and r['family']!='native' and r['finite_complete_ids'] and r['min_tie_aware_recall']>=.99]
        if not feasible:continue
        anchor=min(feasible,key=lambda r:r['median_ms'])
        for multiplier in (.5,1.,2.):
            limit=anchor['median_ms']*multiplier
            eligible=[r for r in rows if (r['cohort'],r['dataset'],r['batch'],r['allowed_percent'])==group and r['route']=='best-resident' and r['finite_complete_ids'] and r['median_ms']<=limit]
            for category in ('fp4','reference','any-native'):
                subset=[r for r in eligible if (r['fp4_proposal'] if category=='fp4' else r['family']!='native' if category=='reference' else r['family']=='native')]
                chosen=max(subset,key=lambda r:(r['min_tie_aware_recall'],-r['median_ms'])) if subset else None
                fixed.append(dict(cohort=group[0],dataset=group[1],batch=group[2],allowed_percent=group[3],category=category,
                                  time_budget_multiplier=multiplier,time_budget_ms=limit,reference_anchor=anchor['method'],
                                  best_tie_aware_recall=chosen['min_tie_aware_recall'] if chosen else None,chosen_method=chosen['method'] if chosen else None))
    summary=dict(scope='Diagnostic smoke only; not study completion' if smoke else 'Declared full A/B vector-search collections; completion also requires independent replay and exact published-head passing CI',
                 counts=dict(counts),group_count=len(groups),binary_sha256=records[0]['binary_sha256'],source_sha256=records[0]['source_sha256'],
                 environment_sha256=records[0]['environment_sha256'],frontiers=fronts,fixed_time_frontiers=fixed)
    stem='search-smoke' if smoke else 'search'
    outputs={ROOT/'results'/f'{stem}-cases.csv':csv_text(rows),ROOT/'results'/f'{stem}-frontiers.csv':csv_text(fronts),
             ROOT/'results'/f'{stem}-fixed-time.csv':csv_text(fixed),ROOT/'results'/f'{stem}-summary.json':json.dumps(summary,indent=2,allow_nan=False)+'\n',
             ROOT/'results'/f'{stem}-negative-results.json':json.dumps(negative,indent=2,allow_nan=False)+'\n'}
    if smoke:
        return outputs,summary
    proof=json.loads((ROOT/'results/search-replay.json').read_text())
    assert proof['scope']=='Complete declared study-10 collection replay'
    assert proof['producer_source_sha256']==records[0]['source_sha256']
    for name,expected in proof['verifier_source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    assert proof['counts']['groups']==len(records)+len(read_records(True))
    replay=proof['counts']
    lines=['# Quantized vector search on consumer Blackwell','',
           '## Scope and evidence','',
           f"The declared two-cohort collection contains {counts['groups']} groups and {counts['attempted']} attempted configurations: {counts['available']} available and {counts['unsupported']} unsupported calls. These are bounded experiments on one RTX 5060 Ti 16 GB, not an architectural ceiling, novelty claim, general recall guarantee, or a result for all vector-search workloads.",'',
           f"The source-bound complete CPU replay passed {replay['groups']} groups ({replay['attempted']} attempted / {replay['available']} available configurations) including the supplementary smoke: {replay['independent_truth_queries']:,} exact truth queries, {replay['independent_truth_distances']:,} direct FP64 database/query distances, {replay['output_ids']:,} retained output IDs, {replay['candidate_ids']:,} candidate IDs and {replay['sampled_dots']:,} sampled native DOTs. The three collection directories are smoke A, full A and full B, not three independent input cohorts. The proof and executed verifier/producer hashes are in results/search-replay.json.",'',
           'The five datasets are uniform-small (16,384 x 128), clustered (65,536 x 256), shifted (262,144 x 128), uniform-million (1,000,000 x 128) and genuine SIFT1M (1,000,000 x 128). Synthetic inputs use independent portable cohorts; SIFT cohorts use disjoint query-source halves. Every full group has B=1/32/128, a static global allow-mask of 100/10/1%, four query variants, five warmups and thirty observations per available boundary. Cohort B reverses dataset, geometry and method order. Timing observations are repeats, not independent inputs. The SIFT attribution, exact original fvecs hashes and lossless shard reconstruction are in fixtures/search-sift/manifest.json and verify_search_sift.py. Numerical descriptors are not relicensed as original images.','',
           '## Implementation and appropriate controls','',
           'The proposal includes native CUTLASS SM120 block-scaled FP4, cuBLASLt FP4, FP8, FP16, BF16 and strict pedantic FP32 products; reconstructed-norm distance formation; native batched cuVS select-k; optional direct-difference original-FP32 refinement; static filtering; and device global-ID mapping. Candidate budgets are 10 without refinement and 32/128/512 with refinement, clipped only to the allowed population. An FP4 original-norm control and explicit compact-filter FP4/FP16 controls retain losses from quantization, cancellation, setup and compaction. Million-row native buffers are padded to 1,000,064 rows; padded rows are never allowed.','',
           'Strong references include resident Faiss GPU FP32/FP16 with both native and cuVS backends, cuVS brute force FP32/FP16, IVF-flat at limited and exhaustive probes, IVF-PQ at two probe counts with and without candidate refinement, and CAGRA with two itopk sizes. Faiss uses device add_c/search_c and initialized-resource stream assignment, not a host-query shortcut. Filtered Faiss statically compacts and remaps global IDs, with compaction in setup. The installed standalone IVF-PQ Python API lacks a filter argument: these calls are recorded as unsupported, never silently replaced by unfiltered search.','',
           '## Correctness and retained failures','',
           'verify_search.py regenerates every input/filter, recomputes every truth query by direct FP64 differences and independently partitioned top-k with explicit ID ties, checks all returned IDs and actual geometric distances, and replays timing/query/output bindings and source/witness hashes. Strict recall is against deterministic ID-tied top-10; tie-aware recall counts unique valid results no farther than the exact tenth distance. Duplicate, missing, out-of-range, filtered and nonfinite valid-ID outputs are separately retained. Quality frontiers require complete finite unique IDs and minimum tie-aware recall across the outputs actually used by that timing boundary; strict recall is also exported. Each output recall is the batch mean, not a per-query lower bound or a certified guarantee on unseen queries.','',
           'Native stage replay checks actual source/encoded bytes and FP4 scales, exact FP32 norm order, sampled DOT accumulation envelopes and score formation, every candidate mapping and score envelope, complete direct refinement and final membership. Large unselected score matrices are not retained: their ranking claim is limited to sampled competitors and all selected candidates; the separate native-kernel preflight independently checks complete small matrices/rank cutoffs and large sampled witnesses. No sampled witness is described as a proof of all large unselected ranks.','',
           f"The collection retains {counts['retained_invalid_id_configurations']} configurations with invalid returned-ID sets, {counts['retained_duplicate_ids']} duplicates and {counts['retained_missing_ids']} missing IDs across deduplicated stored output records. These counts are not independent-query failure rates. Invalid configurations cannot produce a speed win. Low recall, cancellation and nonfinite results are failures, not useful faster search. {counts['ineligible_graphs']} available configurations have ineligible captured graphs; their complete uncaptured route remains recorded. Failed or stale graph outputs are retained but not used as successful captured timings.",'',
           'The original owned-stream CAGRA adapter exposed a shutdown lifetime error: the pinned cuVS implementation caches a filter payload containing the CUDA stream until static cache destruction. The adapter now uses context-lifetime per-thread-default streams for CAGRA only. The failing regression was run before the change; five fresh four-plan subprocesses subsequently passed changed-query, raw distance and clean-exit checks. All old smoke sources/records were archived before the change and fresh smoke was rerun. The pinned source shows the retained stream, destructor and process-lifetime static filter-payload owner.[1] Filtered CAGRA also produced incomplete/duplicate raw results at low selectivity in this protocol; a private raw-ID mapping control reproduced these with both owned and PTDS streams. This is retained observed behavior, not an established vendor root-cause claim.','',
           '## Timing, setup, reuse and memory boundaries','',
           'Resident-uncaptured CUDA events cover query encoding/norms, the product, filtering/select-k, optional original-FP32 refinement and global-ID mapping; CPU launch gaps are inside that interval. Resident-captured is used only after complete changing-query equality controls. Best-resident means the faster eligible captured or complete uncaptured route of each method, so unsupported reference capture does not remove strong references from the primary comparison. Host-service is Python wall time for current FP32-query H2D, all uncaptured work, synchronization and full distance/int64-ID D2H, including allocation/dispatch. It excludes database setup, which is measured separately.','',
           'Setup includes constructor wall time (CPU compaction/padding, uploads, database encoding/norms, allocations, index training/native tuning), first execution, and the actual successful capture or failed-capture attempt. Reuse totals are setup/R plus measured per-batch latency for explicit R=1/4/16/64/256/1024/100000. These are ex-post measured frontiers, not a deployable selector trained on held-out inputs. A median ratio above one is descriptive; shared GPU conditions and no architecture-ceiling inference remain explicit. CSVs include per-route median and interquartile interval.','',
           'Native owned-device bytes include retained original FP32 database/query data, encoded data/scales, scores/DOTs, candidates/refinement, mappings and declared native workspaces; they are not quantized-code-only bytes. Baseline Python arrays are deduplicated by device pointer. Faiss allocation categories and its separate shape-aware temporary reservation (at least 64 MiB, two full FP32 distance tiles plus 16 MiB margin for the declared geometry) are exported. This reserve is charged, not hidden. A fixed-64-MiB B128 shifted search previously aborted after captured scratch overflow while the uncaptured control passed; larger persistent scratch removed the failure in eight separate full-geometry subprocess controls, including a million-row database and both Faiss backends/precisions. Scratch fallback allocation/deallocation behavior is explicit in the pinned resource implementation.[4] Prior records and sources were archived and the entire declared collection was rerun; this is an experimentally verified mitigation, not a claim to have proven every internal vendor race. cuVS has a separately named initial 64 MiB per-resource pool; opaque index/internal allocations are not included in Python-owned bytes. Shared-device free-memory observations are advisory, not physical peaks or reliable cross-method index memory. No end-to-end memory-reduction claim follows from comparing these unequal accounting scopes.','',
           '## Equal-recall, setup-inclusive frontiers','',
           'FP4 means the best tested CUTLASS or cuBLASLt FP4 proposal configuration, not the best arbitrary precision. Reference means the best quality-feasible method among all non-native Faiss/cuVS controls, including indexed search. Both must meet the same recall budget. Missing feasible proposals or references are explicitly not wins. The table uses R=100000 (near steady reuse); exact all-R frontiers are in results/search-frontiers.csv.','',
           '| Boundary | Recall budget | FP4 feasible groups | Reference feasible groups | Comparable groups | FP4 median wins | Median reference/FP4 among comparable |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for boundary in BOUNDARIES:
        for budget in BUDGETS:
            selected=[f for f in fronts if f['boundary']==boundary and f['tie_aware_recall_budget']==budget and f['reuse_batches']==100000]
            ratios=[f['fp4_speedup'] for f in selected if f['fp4_speedup'] is not None]
            median=f'{float(np.median(ratios)):.4f}x' if ratios else 'not comparable'
            lines.append(f"| {boundary} | {budget:g} | {sum(f['fp4_method'] is not None for f in selected)}/{len(groups)} | {sum(f['reference_method'] is not None for f in selected)}/{len(groups)} | {len(ratios)} | {sum(f['fp4_wins'] for f in selected)} | {median} |")
    lines+=['','### Dependence on dataset at recall 0.95, best resident route, R=100000','',
            '| Dataset | FP4 feasible | Reference feasible | FP4 wins | Median reference/FP4 |','|---|---:|---:|---:|---:|']
    for name in DATASETS:
        selected=[f for f in fronts if f['dataset']==name and f['boundary']=='best-resident' and f['tie_aware_recall_budget']==.95 and f['reuse_batches']==100000]
        ratios=[f['fp4_speedup'] for f in selected if f['fp4_speedup'] is not None]
        median=f'{float(np.median(ratios)):.4f}x' if ratios else 'not comparable'
        lines.append(f"| {name} | {sum(f['fp4_method'] is not None for f in selected)}/{len(selected)} | {sum(f['reference_method'] is not None for f in selected)}/{len(selected)} | {sum(f['fp4_wins'] for f in selected)} | {median} |")
    primary=[f for f in fronts if f['boundary']=='best-resident' and f['tie_aware_recall_budget']==.95 and f['reuse_batches']==100000]
    lines+=['','## Bounded findings','',
            f"At minimum tie-aware recall 0.95 and R=100000, FP4 is feasible in {sum(f['fp4_method'] is not None for f in primary)}/{len(primary)} groups and is faster than the best feasible strong reference in {sum(f['fp4_wins'] for f in primary)}/{len(primary)} groups on the best resident route. This result includes losing and infeasible groups rather than conditioning on a favorable subset.",'',
            'Candidate generation and exact refinement are distinct: refinement cannot recover a true neighbor omitted from the candidate set. Increasing C consumes selection/refinement time and keeps the original database resident; low-precision product throughput alone therefore does not establish useful full-search efficiency. Shifted/nearly clustered distances also stress the norm/dot identity even when its product implementation meets its arithmetic envelope.','',
            'results/search-fixed-time.csv reports the best observed recall under 0.5x/1x/2x the fastest reference latency that meets 0.99 recall in the same group; groups without such a reference anchor are omitted explicitly. results/search-cases.csv retains each method/route and its measured quality, latency and named memory fields. results/search-negative-results.json preserves unsupported calls, graph exclusions and invalid/incomplete outputs.','',
            '## Reproduction and completion gates','',
            'Run build_search.sh and build_search_full.sh with the pinned CUTLASS and isolated Faiss/cuVS roots; prepare and independently verify the lossless SIFT fixtures. Run check_search_lifetime.py on GPU 0, then search_study.py --cohort a and --cohort b. The five-dataset fresh smoke is supplementary, not a replacement for these full collections. Run verify_search_preflight.py, verify_search_kernel_preflight.py, verify_search_sift.py, verify_search.py and summarize_search.py --check on CPU. Existing studies must continue passing. Exact source, binary, actual package/library environment and witness SHA bindings are in each record.','',
            'The report is generated only from complete A/B records. Scientific completion additionally requires independent replay of the complete collection and passing CPU/artifact CI on the exact published head; generation of this report alone is not that gate.','',
            '## Sources','',
            '[1] https://raw.githubusercontent.com/NVIDIA/cuvs/v26.10.00/cpp/src/neighbors/detail/cagra/cagra_filter_payload.hpp',
            '[4] https://raw.githubusercontent.com/facebookresearch/faiss/v1.16.0/faiss/gpu/StandardGpuResources.cpp','']
    outputs[ROOT/'SEARCH_REPORT.md']='\n'.join(lines)
    return outputs,summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    outputs,summary=derive(args.smoke)
    for path,text in outputs.items():
        if args.check:
            assert path.read_text(encoding='utf-8')==text,path
        else:
            path.write_text(text,encoding='utf-8',newline='\n')
    print('Search derived artifacts', 'CHECK_PASS' if args.check else 'WRITTEN',json.dumps(summary['counts'],sort_keys=True))


if __name__=='__main__':main()
