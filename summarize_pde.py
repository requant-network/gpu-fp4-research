"""Generate full bounded physical findings from exact numerical/cost receipts.

All frontiers are empirical, include failed observations and strongest eligible
references, and do not establish an unseen-input quality dispatcher.
"""
import argparse
from collections import Counter,defaultdict
import csv
import io
import json
import math
from pathlib import Path
import statistics
import numpy as np
from pde_cases import campaign_cases,configuration_specs
from pde_frontiers import select_frontier
from pde_replay import ROOT,file_hash,load_bound_json,load_payloads,norm,VERIFIERS
from pde_protocol import decode_output
from pde_reference import Mesh,apply_operator


def step_original_residuals(mesh,heat,run,arrays):
    """Derive frontier quality from every actual state/RHS, never scalar claims."""
    result=[]
    for row in run['step_records']:
        rhs=arrays[row['rhs_key']]
        value=decode_output(heat['outputs'][row['output']],arrays,row['input_sha256'])
        if not math.isfinite(norm(value)) or not math.isfinite(norm(rhs)):
            result.append(None);continue
        with np.errstate(over='ignore',invalid='ignore'):
            residual=rhs-apply_operator(mesh,value,heat['reaction'])
            relative=norm(residual)/norm(rhs)
        result.append(relative if math.isfinite(relative) else None)
    return result


def midpoint(values):return statistics.median(values) if values else None


def joint(residual,forward,budget_r,budget_f):
    if residual is None or forward is None:return None
    return max(residual/budget_r,forward/budget_f)


def build_tables():
    replay=json.loads((ROOT/'results/pde-replay.json').read_text());cost=json.loads((ROOT/'results/pde-cost-audit.json').read_text())
    assert replay['full_study_numeric_set'] and replay['collections']==len(campaign_cases())+1,'Whole numerical replay is required'
    assert replay['collection_sha256']==cost['collection_sha256'] and replay['source_sha256']==cost['source_sha256']
    assert replay['driver_sha256']==cost['driver_sha256']==file_hash(ROOT/'verify_pde.py')
    assert cost['auditor_sha256']==file_hash(ROOT/'pde_cost_audit.py')
    assert replay['verifier_sha256']=={n:file_hash(ROOT/n) for n in VERIFIERS}
    for name,expected in replay['source_sha256'].items():assert file_hash(ROOT/name)==expected
    assert replay['launch']['sha256']==file_hash(ROOT/replay['launch']['path'])
    tables={name:[] for name in ('operator','poisson','heat','operator_frontiers','poisson_frontiers','heat_frontiers')}
    operator_groups=defaultdict(list);poisson_groups=defaultdict(list);heat_groups=defaultdict(list)
    negative=defaultdict(list);discretization=[];totals=Counter()
    for collection,expected_sha in replay['collection_sha256'].items():
        assert file_hash(ROOT/collection)==expected_sha
        if '/pde-smoke-' in collection:continue
        manifest=json.loads((ROOT/collection).read_text());cohort,e,q=manifest['cohort'],manifest['elements'],manifest['q'];settings=manifest['protocol']
        dataset=cohort+'-e'+str(e)+'-q'+str(q);qualities=replay['qualities'][collection]
        mesh=Mesh(e,q)
        assert settings['phase']=='full'
        for item in manifest['cases']:
            case=load_bound_json(item);index=case['configuration_index'];config=case['configuration'];available=case['available']
            identity=str(index)+':'+config['family']+':'+config['precision']
            common=dict(dataset=dataset,cohort=cohort,elements=e,q=q,dofs=manifest['dofs'],configuration=index,family=config['family'],precision=config['precision'],available=available,
                        case_path=item['path'],common_benchmark_preparation_ms=manifest['common_preparation_ms'])
            setup=sum(case[k] for k in ('construction_ms','first_data_load_ms','calibration_ms','first_complete_use_ms')) if available else None
            allocation=cost['per_collection'][collection]['geometries'][str(index)] if available else {}
            memory=dict(python_named_bytes=allocation.get('python_named_bytes'),native_named_bytes=allocation.get('native_named_bytes'),
                        libceed_request_qdata_bytes=allocation.get('libceed_request_qdata_bytes'),libceed_request_index_bytes=allocation.get('libceed_request_index_bytes'))
            if not available:negative['unavailable'].append(dict(**common,availability=case['availability']))
            for route in ('resident','host'):
                observed=[r for r in case.get('operator_observations',[]) if r['route']==route]
                errors=[qualities[str(index)]['operator'][r['output']]['relative_original_operator_error'] for r in observed]
                wall=[r['wall_ms'] for r in observed]
                row=dict(**common,**memory,id=identity,route=route,n_observations=len(observed),setup_ms=setup,
                         wall_ms=wall,errors=errors,median_wall_ms=midpoint(wall),median_cuda_span_ms=midpoint([r['cuda_ms'] for r in observed]),
                         worst_original_operator_error=max(errors) if errors else None)
                tables['operator'].append(row)
                operator_groups[(dataset,route)].append(row)
                if available and max(errors)>1e-3:negative['operator_budget_0.001'].append(dict(dataset=dataset,configuration=index,route=route,precision=config['precision'],worst_error=max(errors)))
            for target in settings['targets']:
                for policy in settings['policies']:
                    for route in ('resident','host'):
                        observed=[(r,m) for r,m in zip(case.get('poisson_runs',[]),qualities[str(index)]['poisson'] if available else [],strict=True)
                                  if (r['target'],r['policy'],r['route'])==(target,policy,route)]
                        residuals=[m['original_relative'] for _,m in observed];forward=[m['continuum_mass_relative'] for _,m in observed]
                        wall=[r['wall_ms'] for r,_ in observed]
                        fallbacks=sum(bool(r['controller'].get('fallback',False)) for r,_ in observed)
                        row=dict(**common,id=identity+':'+policy+':'+str(target),route=route,policy=policy,nominal_target=target,setup_ms=setup,
                                 n_observations=len(observed),wall_ms=wall,median_wall_ms=midpoint(wall),original_residuals=residuals,continuum_mass_errors=forward,
                                 all_nominal_targets_met=all(m['target_met'] for _,m in observed) if observed else False,fallback_observations=fallbacks,
                                 precision_label=config['precision']+'-assisted-with-fp64-checks/fallback' if policy=='reliable' else config['precision']+'-fixed')
                        tables['poisson'].append(row);poisson_groups[(dataset,route)].append(row)
                        totals['poisson_observations']+=len(observed);totals['poisson_original_target_misses']+=sum(not m['target_met'] for _,m in observed)
                        if config['precision']=='fp4':totals['fp4_poisson_fallback_observations']+=fallbacks
                        if available and not row['all_nominal_targets_met']:negative['poisson_original_target_misses'].append(dict(dataset=dataset,configuration=index,policy=policy,target=target,route=route,residuals=residuals))
                        if available and config['family']=='cuda' and config['precision']=='fp64' and config['threads']==256 and policy=='fixed' and target==1e-8 and route=='resident':
                            discretization.append(dict(dataset=dataset,elements=e,q=q,cohort=cohort,original_residuals=residuals,continuum_mass_errors=forward))
            if manifest['heat_included']:
                for steps in settings['heat_steps']:
                    heat=next((h for h in case['heat'] if h['steps']==steps),None)
                    heat_arrays=load_payloads(heat['payloads']) if heat else {}
                    for target in settings['targets']:
                        for policy in settings['policies']:
                            for route in ('resident','host'):
                                match=next(((r,m) for r,m in zip(heat['runs'],qualities[str(index)]['heat'][settings['heat_steps'].index(steps)*8:settings['heat_steps'].index(steps)*8+8],strict=True)
                                            if (r['target'],r['policy'],r['route'])==(target,policy,route)),None) if available else None
                                if match:
                                    run,metric=match;step_residuals=step_original_residuals(mesh,heat,run,heat_arrays)
                                    maximum=max(step_residuals) if all(v is not None for v in step_residuals) else None
                                    wall=[run['wall_ms']];forward=[metric['continuum_mass_relative']];residuals=[maximum]
                                    fallbacks=sum(bool(s['controller'].get('fallback',False)) for s in run['step_records'])
                                    complete=run['steps_completed']==steps
                                    totals['heat_trajectories']+=1;totals['heat_steps']+=run['steps_completed']
                                    totals['heat_step_original_target_misses']+=sum(v is None or v>target for v in step_residuals)
                                else:wall=[];forward=[];residuals=[];fallbacks=0;complete=False;metric=None
                                row=dict(**common,id=identity+':s'+str(steps)+':'+policy+':'+str(target),steps=steps,route=route,policy=policy,nominal_target=target,
                                         setup_ms=heat['setup_ms'] if heat else None,n_observations=len(wall),wall_ms=wall,median_wall_ms=midpoint(wall),
                                         continuum_mass_errors=forward,maximum_step_original_residuals=residuals,complete_trajectory=complete,fallback_steps=fallbacks,
                                         backward_euler_continuum_mass_error=metric['backward_euler_continuum_mass_relative'] if metric else None,
                                         inherited_geometry_unavailability=not available)
                                tables['heat'].append(row);heat_groups[(dataset,route)].append(row)
                                if available and (not complete or residuals[0] is None or residuals[0]>target):negative['heat_original_target_misses'].append(dict(dataset=dataset,configuration=index,steps=steps,policy=policy,route=route,target=target,maximum_step_residual=residuals[0]))
    for (dataset,route),rows in sorted(operator_groups.items()):
        for budget in (1e-3,1e-2,1e-1):
            for reuse in (1,100,10000):
                candidates=[dict(id=r['id'],precision=r['precision'],available=r['available'],wall_ms=r['wall_ms'],errors=r['errors'],
                                 setup_ms=r['setup_ms']+r['common_benchmark_preparation_ms'] if r['available'] else None) for r in rows]
                tables['operator_frontiers'].append(dict(dataset=dataset,route=route,error_budget=budget,reuse=reuse,**select_frontier(candidates,budget,reuse)))
    for group_name,groups in (('poisson',poisson_groups),('heat',heat_groups)):
        for (dataset,route),rows in sorted(groups.items()):
            for budget_r in (1e-3,1e-8):
                for budget_f in ((1e-2,1e-3) if group_name=='poisson' else (1e-1,1e-2)):
                    for reuse in (1,100,10000):
                        candidates=[]
                        for r in rows:
                            residuals=r['original_residuals'] if group_name=='poisson' else r['maximum_step_original_residuals']
                            errors=[joint(a,b,budget_r,budget_f) for a,b in zip(residuals,r['continuum_mass_errors'],strict=True)]
                            eligible=r['available'] and (group_name!='heat' or r['complete_trajectory'])
                            candidates.append(dict(id=r['id'],precision=r['precision'],available=eligible,wall_ms=r['wall_ms'],errors=errors,
                                                   setup_ms=r['setup_ms']+r['common_benchmark_preparation_ms'] if eligible else None))
                        tables[group_name+'_frontiers'].append(dict(dataset=dataset,route=route,residual_budget=budget_r,forward_budget=budget_f,reuse=reuse,**select_frontier(candidates,1.,reuse)))
    summary=dict(full_collections=len(campaign_cases()),numeric_counts=replay['counts'],cost_counts=cost['counts'],full_measurement_totals=dict(totals),
                 full_attempted_configurations=len(campaign_cases())*len(configuration_specs()),unavailable_configurations=len(negative['unavailable']),
                 discretization_controls=discretization,negative_results=dict(negative),
                 numeric_receipt_sha256=file_hash(ROOT/'results/pde-replay.json'),cost_receipt_sha256=file_hash(ROOT/'results/pde-cost-audit.json'),
                 analysis_source_sha256={n:file_hash(ROOT/n) for n in ('summarize_pde.py','pde_frontiers.py')})
    return tables,summary


def csv_text(rows):
    fields=list(dict.fromkeys(key for row in rows for key in row));output=io.StringIO(newline='')
    writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for row in rows:writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    return output.getvalue()


def report(tables,summary):
    counts=summary['numeric_counts'];lines=['# Matrix-free Poisson and heat on consumer Blackwell','',
        'This is a bounded measured study, not a hardware ceiling, novelty claim or a quality-safe dispatcher. External context is in [PDE_SOURCES.md](PDE_SOURCES.md); executable details and limits are in [PDE_REPRODUCTION.md](PDE_REPRODUCTION.md).','',
        '## Executed scope and numerical closure','',
        f"The full matrix contains {summary['full_collections']} independent cohort/geometry collections and {summary['full_attempted_configurations']} declared configurations. Including separate smoke, fresh CPU replay covered {counts['cases']} attempted / {counts['available']} available configurations, {counts['operator_observations']} operator observations, {counts['poisson_solves']} complete Poisson solve observations, {counts['heat_trajectories']} heat trajectories and {counts['heat_steps']} retained heat steps. It checked {counts['outputs']} distinct full final-vector records and {counts['operator_stages']} complete retained GEMM witnesses.",'',
        'Each full geometry has two independent sine-mode cohorts; E=1/2/4/8, q=4/8/16. Changing smooth and rough inputs include nonzero operator boundaries. Heat is explicitly bounded to E=1/2, q=4/8/16, final time0.01 and4/8 steps. The remaining larger geometries cover operators and Poisson, not unexecuted large heat trajectories.','',
        'Direct FP32/64 CUDA, pinned libCEED cuda/ref/shared/gen, grouped native FP4/FP8/FP16/BF16/strict-FP32, and ungrouped short-q cuBLAS references were actually attempted. Half/BF16 candidates require recorded eligibility flags; numerical flags/API types/root resources do not constitute an instruction trace. Short-q controls do not inherit the proposal\'s artificial128-wide padding.','',
        '## Complete online operator at fixed error','',
        'Every table choice uses the worst independently checked error among all observations in that timing route, and the fastest eligible non-FP4 implementation. Effective cost includes common benchmark preparation and construction/source load/calibration/first use divided by the stated reuse, plus median complete operation time. Common preparation includes analytic reference-field generation: this is benchmark-inclusive, not minimal deployment initialization.','',
        '| Boundary | Error budget | Reuse | FP4 feasible groups | Reference feasible groups | FP4 wins |','|---|---:|---:|---:|---:|---:|']
    for route in ('resident','host'):
        for budget in (1e-3,1e-2,1e-1):
            selected=[r for r in tables['operator_frontiers'] if r['route']==route and r['error_budget']==budget and r['reuse']==10000]
            lines.append(f"| {route} | {budget:g} | 10000 | {sum(r['fp4'] is not None for r in selected)}/{len(selected)} | {sum(r['reference'] is not None for r in selected)}/{len(selected)} | {sum(r['fp4_wins'] for r in selected)}/{len(selected)} |")
    lines+=['','The complete tables also include reuse1/100. Padding, dynamic source packing, quantization, actual native product, all-axis weighting, deterministic global assembly and context acknowledgment are not replaced by a prepacked microbenchmark. Host service includes H2D and D2H; its CUDA event span ends before host D2H and is not the same boundary as host wall time.','',
        '## Poisson: original residual and continuum forward error','',
        'Recursive convergence of an input-dependent/nonsymmetric low operator is not physical convergence. Fixed runs retain this distinction. Reliable runs pay for bounded low-operator defect corrections, true residual tests, rejected progress and FP64 fallback. FP4-assisted/fallback rows are not pure-FP4 solves.','',
        f"The full data retain {summary['full_measurement_totals']['poisson_original_target_misses']} Poisson observation(s) missing their nominal original residual target and {summary['full_measurement_totals']['fp4_poisson_fallback_observations']} FP4-assisted Poisson observations using fallback. No failed repetition is silently dropped from a configuration's frontier eligibility.",'',
        '| Boundary | Residual budget | Continuum mass-L2 budget | Reuse | FP4-assisted/fixed feasible | Reference feasible | FP4-labelled wins |','|---|---:|---:|---:|---:|---:|---:|']
    for route in ('resident','host'):
        for residual in (1e-3,1e-8):
            selected=[r for r in tables['poisson_frontiers'] if r['route']==route and r['residual_budget']==residual and r['forward_budget']==1e-2 and r['reuse']==10000]
            lines.append(f"| {route} | {residual:g} | 0.01 | 10000 | {sum(r['fp4'] is not None for r in selected)}/{len(selected)} | {sum(r['reference'] is not None for r in selected)}/{len(selected)} | {sum(r['fp4_wins'] for r in selected)}/{len(selected)} |")
    lines+=['','## Discretization and temporal refinement','',
        'The following independently replayed strict fixed-CUDA-FP64 controls show continuum mass-L2 error, not merely an internally small algebraic residual. At high order, iterative stopping and floating-point error can dominate the remaining spatial discretization error; monotonic convergence is not asserted below that floor.','',
        '| Cohort | E | q | Worst original residual | Worst continuum mass-L2 error |','|---|---:|---:|---:|---:|']
    for row in sorted(summary['discretization_controls'],key=lambda r:(r['cohort'],r['q'],r['elements'])):
        lines.append(f"| {row['cohort']} | {row['elements']} | {row['q']} | {max(row['original_residuals']):.7g} | {max(row['continuum_mass_errors']):.7g} |")
    lines+=['','Heat tables retain each step state and RHS and validate RHS = M*previous_state/dt against the preceding retained vector. Continuum decay and sampled continuum backward-Euler targets are separate, exposing time discretization rather than equating a solved implicit system with the exact heat PDE. Frontier step residuals are recomputed from every retained state/RHS, not taken from scalar GPU claims. Frontiers require every step residual and final continuum forward error and may choose between the actually executed 4/8-step trajectories.','',
        '| Dataset | Steps | Maximum step residual | Continuum mass-L2 error | Error against continuum BE target |','|---|---:|---:|---:|---:|']
    heat_controls=[r for r in tables['heat'] if r['configuration']==3 and r['policy']=='fixed' and r['nominal_target']==1e-8 and r['route']=='resident']
    for row in sorted(heat_controls,key=lambda r:(r['cohort'],r['q'],r['elements'],r['steps'])):
        lines.append(f"| {row['dataset']} | {row['steps']} | {row['maximum_step_original_residuals'][0]:.7g} | {row['continuum_mass_errors'][0]:.7g} | {row['backward_euler_continuum_mass_error']:.7g} |")
    lines+=['','The strict FP64 controls distinguish the temporal error reduced by refinement from persistent under-resolved spatial error. No monotonic low-precision or unresolved-mesh convergence claim is inferred from a small recursive residual.','',
        '| Boundary | Residual budget | Continuum mass-L2 budget | Reuse | FP4-labelled feasible | Reference feasible | FP4-labelled wins |','|---|---:|---:|---:|---:|---:|---:|']
    for route in ('resident','host'):
        for forward in (1e-1,1e-2):
            selected=[r for r in tables['heat_frontiers'] if r['route']==route and r['residual_budget']==1e-8 and r['forward_budget']==forward and r['reuse']==10000]
            lines.append(f"| {route} | 1e-08 | {forward:g} | 10000 | {sum(r['fp4'] is not None for r in selected)}/{len(selected)} | {sum(r['reference'] is not None for r in selected)}/{len(selected)} | {sum(r['fp4_wins'] for r in selected)}/{len(selected)} |")
    lines+=['',f"The full heat data retain {summary['full_measurement_totals']['heat_step_original_target_misses']} step observation(s) missing their nominal original residual target. Fixed low-precision misses remain negative evidence, not successful physical acceleration.",'',
        '## Method-local memory categories','',
        'The following E2/q8 sample excludes shared benchmark input/reference buffers. Library request lengths remain separate, not invented opaque-owned-byte totals. Temporary arrays, reduction buffers, allocator/JIT caches and physical peaks are not claimed audited.','',
        '| Implementation | Python named bytes | Native named bytes | CEED qdata request bytes | CEED index request bytes |','|---|---:|---:|---:|---:|']
    for row in tables['operator']:
        if row['dataset']=='a-e2-q8' and row['route']=='resident' and row['configuration'] in (0,3,5,7,9,13):
            values=[row.get(k) for k in ('python_named_bytes','native_named_bytes','libceed_request_qdata_bytes','libceed_request_index_bytes')]
            lines.append('| '+row['id']+' | '+' | '.join(str(v) if v is not None else 'not measured' for v in values)+' |')
    lines+=['','## Retained limits and negatives','',
        f"There are {summary['unavailable_configurations']} full configuration availability failures. The pinned cuda/ref q16 basis exceeds its static shared-allocation limit, while optimized shared/gen operators are retained. The largest grouped geometry encounters the original native bridge N<=8192 guard. These are respectively a specific kernel/compiler allocation failure and an implementation guard, not a blanket GPU/Tensor Core limitation. Unavailable inherited heat combinations are not separately executed heat failures.",'',
        'All negative records are in results/pde-negative-results.json. CUDA, native, libCEED and short-q setup and allocation/request scopes remain visible in master tables. Opaque libCEED memory is not ranked against another family\'s fuller named accounting; no physical peak-memory saving is established.','',
        'This common host-controlled Jacobi-PCG experiment stages each callback and performs FP64-vector reductions; it is not a best-of-all fully GPU-resident solver claim. First use follows explicit calibration, and solver measurements follow operator warmup. Import/device-discovery, warm disk/JIT/allocator history, diagnostics between requests and shared-device activity prevent cold-system/exclusive/saturated-throughput claims.','',
        'Every timed final vector is retained losslessly, including nondeterministic variants. Numeric replay reconstructs original-data encoding/products/post-processing and final original-system residuals. It does not certify every unretained CG intermediate or measure board/system energy. Full source30,4binary,fixture pre/post and child-exit closures are retained; exact numerical/source/cost receipts are prerequisites for these tables.','',
        '## Artifacts','',
        '- [Operator observations and costs](results/pde-operator.csv)','- [Poisson solve observations and quality](results/pde-poisson.csv)','- [Heat trajectories and quality](results/pde-heat.csv)',
        '- [Operator frontiers](results/pde-operator-frontiers.csv)','- [Poisson frontiers](results/pde-poisson-frontiers.csv)','- [Heat frontiers](results/pde-heat-frontiers.csv)',
        '- [Numeric replay](results/pde-replay.json)','- [Named cost audit](results/pde-cost-audit.json)','- [Full launch receipt](results/pde-full-launch.json)','- [Summary and negatives](results/pde-summary.json)','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();tables,summary=build_tables()
    output={ROOT/'results'/('pde-'+name.replace('_','-')+'.csv'):csv_text(rows) for name,rows in tables.items()}
    output[ROOT/'results/pde-summary.json']=json.dumps(summary,indent=2,allow_nan=False)+'\n'
    output[ROOT/'results/pde-negative-results.json']=json.dumps(summary['negative_results'],indent=2,allow_nan=False)+'\n'
    output[ROOT/'PDE_REPORT.md']=report(tables,summary)
    for path,text in output.items():
        if args.check:assert path.read_text()==text,('Generated physical findings differ',path.name)
        else:path.write_text(text)
    print('PDE_COMPLETE_FINDINGS_CHECK_PASS' if args.check else 'PDE_COMPLETE_FINDINGS_GENERATED',json.dumps({k:len(v) for k,v in tables.items()},sort_keys=True))


if __name__=='__main__':main()
