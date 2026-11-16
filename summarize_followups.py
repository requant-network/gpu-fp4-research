"""Derive encoder/fusion tables and an explicitly held-out configuration rule."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
NAMES=("scalar","lane16-128","pair8-128","pair8-256","pair8-512")


def median(row,boundary):
    return statistics.median(row["timings_ms"][boundary])


def outputs():
    records=[json.loads((ROOT/"results"/name).read_text(encoding="utf-8"))
             for name in ("encoder-cohort-a.json","encoder-cohort-b.json")]
    if not all(r.get("complete") and len(r["cases"])==296 for r in records):
        raise ValueError("Complete encoder cohorts required")
    groups=[]
    for record in records:
        rows={}
        for case in record["cases"]:
            key=(tuple(case["shape"]),case["profile"])
            rows.setdefault(key,{})[(case["tile_m"],case["variant"],bool(case["activation"]),case["fused"])]=case
        groups.append(rows)
    training=[]
    for (shape,profile),cases in groups[0].items():
        if cases[128,0,False,False]["partition"]=="training":
            costs=[median(cases[128,v,False,False],"prepare-a") for v in range(5)]
            training.append((shape[0]*shape[2],costs))
    best=None
    for threshold in [-1]+sorted({size for size,_ in training}):
        choices=[]
        for below in (True,False):
            subset=[costs for size,costs in training if (size<=threshold)==below]
            scores=[sum(math.log(costs[v]/min(costs)) for costs in subset) for v in range(5)]
            choices.append(min(range(5),key=lambda v:scores[v]))
        score=sum(math.log(costs[choices[0] if size<=threshold else choices[1]]/min(costs)) for size,costs in training)
        if best is None or score<best[0]:
            best=(score,threshold,*choices)
    _,threshold,small,large=best
    evaluations=[]
    for record,rows in zip(records,groups):
        for (shape,profile),cases in rows.items():
            if cases[128,0,False,False]["partition"]!="held-out":
                continue
            chosen=small if shape[0]*shape[2]<=threshold else large
            costs=[median(cases[128,v,False,False],"prepare-a") for v in range(5)]
            evaluations.append({"seed":record["seed"],"shape":list(shape),"profile":profile,
                                "selected_variant":chosen,"regret":costs[chosen]/min(costs),
                                "speedup_over_scalar":costs[0]/costs[chosen]})
    selector={"feature":"M*K input elements","training_cohort":records[0]["seed"],
              "training_groups":len(training),"threshold":threshold,
              "at_or_below_variant":small,"above_variant":large,
              "objective":"sum log latency regret for preparation A; five encoder variants; fixed tile128",
              "held_out_evaluations":evaluations,
              "held_out_geomean_regret":math.exp(statistics.mean(math.log(r["regret"]) for r in evaluations)),
              "held_out_geomean_speedup_over_scalar":math.exp(statistics.mean(math.log(r["speedup_over_scalar"]) for r in evaluations))}
    text=io.StringIO(newline="")
    writer=csv.writer(text,lineterminator="\n")
    writer.writerow(["seed","m","n","k","profile","partition","label","boundary","median_ms","p10_ms","p90_ms","relative_frobenius_error"])
    for record in records:
        for case in record["cases"]:
            for boundary,values in case["timings_ms"].items():
                values=sorted(values)
                # Match the empirical nearest-rank percentiles declared for this table.
                writer.writerow([record["seed"],*case["shape"],case["profile"],case["partition"],case["label"],boundary,
                                 f"{statistics.median(values):.9f}",f"{values[2]:.9f}",f"{values[26]:.9f}",
                                 f"{case['checks']['relative_frobenius_error']:.10g}"])
    lines=["# Cooperative FP4 preparation, fusion and tile study","",
           "**Status: this additive encoder/fusion/tile experiment is complete. Program studies 1-4 still have additional requirements.**","",
           "Two independent cohorts cover 592 configurations, eleven shapes, four input profiles (two at the largest shape), "
           "five timing boundaries and 88,800 retained observations. Encoded values and scales are byte-identical to the independent CPU encoder. "
           "Every output is checked before and after timing. Eight small input/output witness files support CPU replay.","",
           "The original scalar encoder, four cooperative implementations and M=128/256 CUTLASS tiles are retained, including regressions. "
           "Bias/ReLU fusion compares the same activation and quantization contract. These are warm, captured, device-resident shared-system observations.","",
           "## Normal-profile comparisons","",
           "Ratios are ranges across cohorts. Above 1 favors the candidate. The fixed pair8-256 encoder uses eight cooperating lanes per block of sixteen values "
           "and 256 threads per CUDA block; it was not selected separately for each test case.","",
           "| M x N x K | Scalar / pair8-256 prepare A | Separate / fused online A | Tile128 / tile256 prepacked GEMM |",
           "|---|---:|---:|---:|"]
    for shape,profile in sorted(groups[0]):
        if profile!="normal":
            continue
        ratios=[[],[],[]]
        for rows in groups:
            cases=rows[shape,profile]
            ratios[0].append(median(cases[128,0,False,False],"prepare-a")/median(cases[128,3,False,False],"prepare-a"))
            ratios[1].append(median(cases[128,3,True,False],"online-a")/median(cases[128,3,True,True],"online-a"))
            ratios[2].append(median(cases[128,3,False,False],"prepacked")/median(cases[256,3,False,False],"prepacked"))
        lines.append(f"| {' x '.join(map(str,shape))} | "+" | ".join(f"{min(v):.2f}-{max(v):.2f}" for v in ratios)+" |")
    lines += ["","## Held-out encoder selection","",
              f"A single-threshold rule was fitted on {len(training)} groups from cohort {records[0]['seed']}, excluding all four held-out shapes. "
              f"Choose `{NAMES[small]}` when M*K <= {threshold}; otherwise choose `{NAMES[large]}`. "
              "The objective uses only preparation-A latency and has no application-quality claim.","",
              f"Across {len(evaluations)} held-out shape/profile/cohort evaluations, geometric-mean regret against the fastest measured encoder is "
              f"{selector['held_out_geomean_regret']:.3f}x; geometric-mean speedup against the scalar encoder is "
              f"{selector['held_out_geomean_speedup_over_scalar']:.3f}x. "
              "Both the fitted rule and every held-out outcome are retained in [selector data](results/encoder-selector.json).","",
              "## Interpretation and remaining scope","",
              "Cooperative preparation is not uniformly faster. Small and large inputs can prefer different work assignment; "
              "extra per-lane scale/conversion work competes with improved memory access. Fusion removes a separate activation pass "
              "but must be evaluated as a complete operation. Tile choice also changes latency without changing the quantization contract.","",
              "This experiment does not finish the complete format selector, real-model quality/latency integration, compiled framework fusion baseline, "
              "or direct loading/pipeline/profiler comparisons required by [the program](PROGRAM.md). "
              "It establishes measured candidates and counterexamples for those follow-ups. "
              "The full [CSV](results/encoder-summary.csv) retains all variants, profiles, boundaries and nearest-rank p10/p90 values.","",
              "Run `python verify_followups.py` for source/input/numerical replay and `python summarize_followups.py --check` for derived artifacts.",""]
    return {"results/encoder-summary.csv":text.getvalue(),"results/encoder-selector.json":json.dumps(selector,indent=2,allow_nan=False)+"\n",
            "ENCODER_REPORT.md":"\n".join(lines)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    for name,contents in outputs().items():
        path=ROOT/name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8")!=contents:
                raise ValueError(f"Stale derived follow-up artifact: {name}")
        else:
            path.write_text(contents,encoding="utf-8",newline="\n")
    print("Follow-up derived artifacts match" if args.check else "Wrote follow-up tables and held-out encoder selector")


if __name__=="__main__":
    main()
