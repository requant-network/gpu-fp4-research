"""Derive held-out error-prediction findings, including unsafe admissions and cost."""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import statistics

from selector_reference import BUDGETS,HELD_SHAPES,PROFILES

ROOT=Path(__file__).resolve().parent


def median(case,key="host_request_ms"):
    return statistics.median(case[key])


def held_out(record,row):
    return record["seed"]!=20261122 or tuple(row["shape"]) in HELD_SHAPES or row["profile"] not in PROFILES[:2] or row["kind"]=="real"


def error(case):
    return max(case["checks"]["relative_frobenius_error"],case["checks"]["exact_sample_relative_error"])


def outputs():
    calibration=json.loads((ROOT/"results/selector-calibration.json").read_text(encoding="utf-8"))
    records=[json.loads((ROOT/f"results/selector-cohort-{suffix}.json").read_text(encoding="utf-8")) for suffix in ("a","b")]
    if not all(r.get("complete") and not r["quick"] for r in [calibration,*records]):raise ValueError("Complete selector campaigns required")
    reference_buffer=io.StringIO(newline="");reference_writer=csv.writer(reference_buffer,lineterminator="\n")
    reference_writer.writerow(["phase","seed","m","n","k","profile","kind","mode","boundary","median_ms","relative_error","exact_sample_error","probe_feature","setup_and_initial_preparation_wall_ms"])
    cases_count=entries=observations=0
    for record in [calibration,*records]:
        for c in record["cases"]:
            cases_count+=1;entries+=c["shape"][0]*c["shape"][1]
            for boundary,values in [*c["timings_ms"].items(),("host-request",c["host_request_ms"])]:
                observations+=len(values)
                reference_writer.writerow([record["phase"],record["seed"],*c["shape"],c["profile"],c["kind"],c["mode"],boundary,f"{statistics.median(values):.9f}",
                    f"{c['checks']['relative_frobenius_error']:.12g}",f"{c['checks']['exact_sample_relative_error']:.12g}",c["feature"],c["setup_and_initial_preparation_wall_ms"]])
    buffer=io.StringIO(newline="");writer=csv.writer(buffer,lineterminator="\n")
    writer.writerow(["seed","model_seed","m","n","k","profile","kind","held_out","maximum_relative_error","selected","predicted_error","full_relative_error","exact_sample_error","budget_met","selector_host_median_ms","selected_direct_host_median_ms","offline_feasible_best","offline_feasible_best_host_median_ms","feasible_selector_regret","native_fp64_over_feasible_selector"])
    decisions=[];failures=[]
    for record in records:
        by_key={(tuple(c["shape"]),c["profile"],c["kind"],c["mode"]):c for c in record["cases"]}
        for row in record["selections"]:
            if len(set(row["selected"]))!=1:raise ValueError("Repeated-input selection changed")
            key=tuple(row["shape"]),row["profile"],row["kind"];selected=row["selected"][0];budget=row["maximum_relative_error"]
            candidates=[c for (*group,_),c in by_key.items() if tuple(group)==key and error(c)<=budget]
            best=min(candidates,key=median) if candidates else None;latency=median(row);feasible=row["budget_met"]
            regret=latency/median(best) if feasible and best else None;fp64_ratio=median(by_key[*key,"fp64"])/latency if feasible else None
            prediction=row["estimates"].get(selected)
            writer.writerow([record["seed"],record["model_seed"],*row["shape"],row["profile"],row["kind"],held_out(record,row),budget,selected,prediction,
                row["checks"]["relative_frobenius_error"],row["checks"]["exact_sample_relative_error"],feasible,latency,median(by_key[*key,selected]),
                None if best is None else best["mode"],None if best is None else median(best),regret,fp64_ratio])
            item={"seed":record["seed"],"shape":row["shape"],"profile":row["profile"],"kind":row["kind"],"held_out":held_out(record,row),"budget":budget,
                  "selected":selected,"prediction":prediction,"full_error":row["checks"]["relative_frobenius_error"],"sample_error":row["checks"]["exact_sample_relative_error"],
                  "feasible":feasible,"regret":regret,"fp64_ratio":fp64_ratio,"latency":latency,"direct_selected":median(by_key[*key,selected])}
            decisions.append(item)
            if not feasible:failures.append(item)
            observations+=len(row["host_request_ms"])
    held=[d for d in decisions if d["held_out"]]
    lines=["# Held-out per-operation format selection","",
           "**Status: the remaining error-prediction gate of study 1 has been experimentally evaluated. The empirical selector has a retained budget violation and is not a certified error-constrained dispatcher. The fifteen-study program remains active.**","",
           f"The campaign retains {cases_count:,} reference configurations, {len(decisions)} complete selection decisions at four budgets, {observations:,} timing observations and {entries:,} reference-output entries checked at three checkpoints. "
           f"Of these decisions, {len(held)} involve unseen shapes, distributions, model data or the independent second cohort; {sum(not d['feasible'] for d in held)} violate their requested budget. "
           "Ten full small-case witnesses and two changing-input witnesses permit independent CPU replay.","",
           "## Hypothesis and fixed protocol","",
           "Can a small native GPU error probe, calibrated on ordinary/outlier products, select a useful format for unseen operations after paying for its own work? "
           "The predictor is calibrated once from seed 20261122, four shapes and the normal/outlier distributions. The maximum training ratio of actual error to probe feature is multiplied by 1.25, separately for each format. "
           "No evaluation error or timing updates that policy. A separate calibration run finishes before either evaluation cohort starts; both evaluation records retain the identical policy and its hash.","",
           "The probe evaluates 64 deterministic sampled output coordinates per format using native CUDA FP64 dot reductions on original FP32 inputs, decoded quantized inputs and absolute decoded products. "
           "Its feature is the sample norm of quantization differences plus 2^-24*sqrt(K) times the absolute-product norm, divided by the original sampled product norm. "
           "This accumulation term is an empirical feature, not a proven error bound. Zero denominators are explicitly handled. CPU dot products independently check every retained probe.","",
           "Eligibility compares the calibrated estimate with the requested relative error. Eligible formats are ranked by prepacked latency at the nearest training shape in squared log-dimension distance. "
           "If none qualifies, execution uses native FP64. The policy does not inspect the current full product or its reference error before dispatch. "
           "Evaluation covers two unseen shapes, cancellation and complementary exponent spreads, four real-activation layer/batch groups, and an independent input/model cohort with reversed execution order.","",
           "Training shapes are 16x512x512, 128x128x128, 256x1024x1024 and 1024x1024x1024. Held-out shapes are 32x768x1280 and 384x768x256. "
           "The portable normal/outlier generator uses a finite integer CLT grid; it does not cover every FP32 mantissa pattern. Cancellation uses rich-mantissa paired inputs with an exact rank-one residual. "
           "Spread inputs use complementary column exponents -12, -6, 0, 6 and 12; zero rows are retained. Real inputs/weights come from the two frozen digits models.","",
           "## Budget failures are findings","",
           "Budget acceptance uses the maximum of the full reference-relative Frobenius error and the retained exact-sample error. The former uses CPU FP64, or the analytic cancellation product; the latter uses arbitrary-precision integer dot products. "
           "A complete low-error guarantee does not follow from 64 samples. The predeclared policy is kept unchanged after seeing its failures.","",
           "| Seed | M x N x K | Profile | Budget | Selected | Prediction | Full error | Exact-sample error |",
           "|---|---|---|---:|---|---:|---:|---:|"]
    for d in failures:lines.append(f"| {d['seed']} | {' x '.join(map(str,d['shape']))} | {d['profile']} | {d['budget']:.5g} | {d['selected']} | {d['prediction']:.6g} | {d['full_error']:.6g} | {d['sample_error']:.6g} |")
    lines += ["","The retained full error can exceed the budget even while the sampled error and calibrated prediction remain below it. "
              "This rejects the hypothesis that the tested calibration safely generalizes to every operation in the held-out protocol. A runtime accuracy guarantee remains unproven; this result does not recommend this predictor for such a guarantee.","",
              "## Complete costs on held-out decisions","",
              "Each request includes H2D A, preparation of A in all five formats, all five native probes, their D2H transfers/synchronization, the host decision, the selected GEMM and full FP64 output D2H. "
              "FP64 fallback additionally converts A to FP64. B representations, plans and heuristic choices are reused. Every format returns FP64 output; native promotion of FP32 candidate output is timed. "
              "Direct references charge their own A conversion, GEMM/promotion and the same input/output transfers. The offline feasible best is known only retrospectively and pays no selection overhead.","",
              "Ratios below include only decisions that actually meet their budget; the unsafe admission above is excluded from equal-error comparisons. "
              "Regret is selector latency divided by the fastest measured direct reference meeting that input's error budget. FP64/selector greater than 1 favors the selector.","",
              "| Budget | Held-out decisions | Budget failures | Feasible median regret | Feasible regret range | Feasible median FP64 / selector |",
              "|---|---:|---:|---:|---:|---:|"]
    for budget in BUDGETS:
        rows=[d for d in held if d["budget"]==budget];valid=[d for d in rows if d["regret"] is not None];regrets=[d["regret"] for d in valid]
        lines.append(f"| {budget:.5g} | {len(rows)} | {sum(not d['feasible'] for d in rows)} | {statistics.median(regrets):.4f} | {min(regrets):.4f}-{max(regrets):.4f} | {statistics.median(d['fp64_ratio'] for d in valid):.4f} |")
    choices=Counter(d["selected"] for d in held)
    lines += ["", "Held-out choices: "+", ".join(f"{mode}: {count}" for mode,count in sorted(choices.items()))+". "
              "The full probe/preparation/decision cost is often substantial even when selected GEMM execution is fast. Cases faster than native FP64 do not establish an advantage over the best feasible low-precision reference.","",
              "[All reference boundaries and setup costs](results/selector-references.csv) and [every decision, error and complete cost](results/selector-decisions.csv) retain the unfavorable cases. "
              "One-time plan allocation and initial preparation are separate from reused-plan requests; algorithm tuning, graph construction and numerical validation are excluded from request/device timers. "
              "Captured direct references use five warmups and thirty observations, eight repetitions for smaller products and one for volume >=2^29. Thirty complete host requests per case repeat A. "
              "Separate witnesses change A four times with fixed B and check all four budgets, fresh encoding/probes and complete outputs; these are correctness transitions, not latency distributions for changing arrivals.","",
              "## Independent checks and interpretation","",
              "Every quantized byte/scale and reference output is checked before timing, after device timing and after direct requests. The final output for each selection/budget is checked in full. "
              "Small witnesses replay full arithmetic; large cases retain 64 actual output values at each check, so exact-integer sample errors can be recomputed without trusting a summary hash. "
              "The native probe also has four invalid/null/count controls. All executed source/data hashes, binary identities, environment, calibration and raw observations remain in the records.","",
              "The implementation uses the cooperative FP4 encoder and pinned CUTLASS/cuBLASLt native paths; Python performs orchestration and the timed host decision. "
              "It does not establish optimal selection overhead, scientific novelty or an architecture ceiling. Other GPU contexts remain present; no clock/power/service settings were changed. "
              "The empirical failure and overhead finish this bounded hypothesis test without establishing a deployable guaranteed-error policy.","",
              "Together with the [grid/encoder work](ENCODER_REPORT.md), [complete application/strong references](APPLICATION_REPORT.md), [native architecture work](HARDWARE_REPORT.md), "
              "[outlier corrections](OUTLIER_REPORT.md) and [held-out pipeline comparisons](PIPELINE_REPORT.md), this supplies the remaining study-1 gate. "
              "Studies 5 and 7-15 still require their own experiments and evidence.","",
              "## Reproduction","",
              "Build with `bash build_selector.sh /path/to/pinned/cutlass` plus the existing extended/precision builds. "
              "Run `selector_study.py --phase calibration --seed 20261122 --model-seed 20261116 --output results/selector-calibration.json` first. "
              "Use its `results/selector-policy.json` with `--phase evaluation --policy results/selector-policy.json`, the same seed/model for cohort A and seed 20261222/model 20261216 with `--reverse` for B. "
              "Keep separate output paths and preserve existing observations. CPU-only checks are `python verify_selector.py` and `python summarize_selector.py --check`.",""]
    return {"SELECTOR_REPORT.md":"\n".join(lines),"results/selector-references.csv":reference_buffer.getvalue(),"results/selector-decisions.csv":buffer.getvalue()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--check",action="store_true");args=parser.parse_args()
    for name,content in outputs().items():
        path=ROOT/name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8")!=content:raise ValueError(f"Stale selector artifact: {name}")
        else:path.write_text(content,encoding="utf-8",newline="\n")
    print("Selector artifacts match" if args.check else "Wrote held-out selector report and complete tables")


if __name__=="__main__":main()
