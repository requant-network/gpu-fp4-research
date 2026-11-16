"""Derive real-application comparisons and validation-gated held-out format selection."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
TRAIN=(16,64,256)
TEST=(32,128,192)
BUDGETS=(0,.005,.01)
BOUNDARIES=("captured_device","uncaptured_device","host_request","cold_plan_cached_algorithm")


def label(config):
    first,second,backend,fused=config
    return f"{first}/{second}"+(f" {backend}" if "fp4" in (first,second) else " cuBLASLt")


def med(case,boundary):return statistics.median(case["timings_ms"][boundary])


def predict(training,config,batch,boundary):
    low=max(size for size in TRAIN if size<=batch)
    high=min(size for size in TRAIN if size>=batch)
    a=med(training[low,config],boundary)
    if low==high:return a
    b=med(training[high,config],boundary)
    return a+(b-a)*(batch-low)/(high-low)


def artifacts():
    native=[json.loads((ROOT/"results"/name).read_text(encoding="utf-8")) for name in ("application-cohort-a.json","application-cohort-b.json")]
    torch=[json.loads((ROOT/"results"/name).read_text(encoding="utf-8")) for name in ("pytorch-cohort-a.json","pytorch-cohort-b.json")]
    graphs=[json.loads((ROOT/"results"/name).read_text(encoding="utf-8")) for name in ("pytorch-graphs-cohort-a.json","pytorch-graphs-cohort-b.json")]
    if not all(r.get("complete") and len(r["cases"])==48 for r in native+torch):
        raise ValueError("Four complete classifier cohorts required")
    if not all(r.get("complete") and len(r["cases"])==24 for r in graphs):
        raise ValueError("Both complete internal-graph reference cohorts required")
    cases=[{(r["batch"],tuple(r["configuration"])):r for r in record["cases"]} for record in native]
    qualities=[{tuple(r["configuration"]):r for r in record["quality"]} for record in native]
    configs=sorted(qualities[0])
    selection=[]
    for cohort,(record,quality,observed) in enumerate(zip(native,qualities,cases)):
        baseline=quality["fp32","fp32","cutlass",False]["splits"]["validation"]["accuracy"]
        for budget in BUDGETS:
            eligible=[config for config in configs if quality[config]["splits"]["validation"]["accuracy"]+budget+1e-12>=baseline]
            for boundary in BOUNDARIES:
                for batch in TEST:
                    chosen=min(eligible,key=lambda config:predict(cases[0],config,batch,boundary))
                    best=min(med(observed[batch,config],boundary) for config in eligible)
                    selection.append({"model_seed":record["seed"],"batch":batch,"budget":budget,"boundary":boundary,
                                      "configuration":list(chosen),"predicted_ms":predict(cases[0],chosen,batch,boundary),
                                      "observed_ms":med(observed[batch,chosen],boundary),"regret":med(observed[batch,chosen],boundary)/best,
                                      "validation_accuracy":quality[chosen]["splits"]["validation"]["accuracy"],
                                      "test_accuracy":quality[chosen]["splits"]["test"]["accuracy"]})
    selector={"method":"Linear interpolation in batch size using only native cohort A batches 16,64,256; per-model validation-accuracy eligibility",
              "quality_budgets":list(BUDGETS),"held_out_batches":list(TEST),"evaluations":selection,
              "geomean_regret":math.exp(statistics.mean(math.log(r["regret"]) for r in selection)),
              "scope":"Offline held-out configuration evaluation. Test labels/latencies do not choose configurations; routing overhead is excluded."}
    output=io.StringIO(newline="")
    writer=csv.writer(output,lineterminator="\n")
    writer.writerow(["seed","implementation","configuration","batch","partition","boundary","median_ms","p10_ms","p90_ms"])
    for implementation,records in (("native",native),("pytorch",torch),("pytorch-internal-graphs",graphs)):
        for record in records:
            for row in record["cases"]:
                configuration=label(row["configuration"]) if implementation=="native" else f"{row['mode']} {row['engine']}"
                for boundary,observations in row["timings_ms"].items():
                    values=sorted(observations)
                    writer.writerow([record["seed"],implementation,configuration,row["batch"],row["partition"],boundary,
                                     f"{statistics.median(values):.8f}",f"{values[2]:.8f}",f"{values[26]:.8f}"])
    reuse=io.StringIO(newline="")
    reuse_writer=csv.writer(reuse,lineterminator="\n")
    reuse_writer.writerow(["seed","configuration","batch","reuse_count","total_median_ms","amortized_ms"])
    for record in native:
        for row in record["cases"]:
            for count,values in row["reuse_total_ms"].items():
                total=statistics.median(values)
                reuse_writer.writerow([record["seed"],label(row["configuration"]),row["batch"],count,f"{total:.8f}",f"{total/int(count):.8f}"])
    lines=["# Real-data classifier and compiled-framework comparison","",
           "**Status: this bounded application/reference experiment is complete. The fifteen-study program remains active.**","",
           "Two independently trained digits models compare eight native precision/backend configurations and eager/compiled "
           "PyTorch FP32, FP16, BF16 and FP8, plus a dynamic-shape reduce-overhead reference with internal CUDA Graphs. "
           "Each implementation covers six batch sizes, for 240 timing cases in total. "
           "Native FP4 uses both CUTLASS and cuBLASLt with identical block-scaled operands and FP32 output. "
           "The classifier includes both linear layers, bias/ReLU and final GPU argmax over ten classes.","",
           "The dataset, parameters, exhaustive disjoint splits, numerical witnesses, complete retained predictions, "
           "raw durations and executed-source hashes support CPU-only replay. Native encoded operands and every layer's "
           "arithmetic are checked at runtime; all validation/test predictions also match independent complete CPU simulation.","",
           "## Classification quality","",
           "Values are test accuracy percentages for the two independent models. Each model has 364 held-out test images; "
           "one image is approximately 0.275 percentage points. These small tests do not establish production model quality.","",
           "| Configuration | Model A test | Model B test |","|---|---:|---:|"]
    for config in configs:
        values=[100*q[config]["splits"]["test"]["accuracy"] for q in qualities]
        lines.append(f"| Native {label(config)} | {values[0]:.3f}% | {values[1]:.3f}% |")
    for mode in ("fp32","fp16","bf16","fp8"):
        values=[]
        for record in torch:
            row=next(r for r in record["quality"] if r["mode"]==mode and r["engine"]=="compiled")
            values.append(100*row["splits"]["test"]["accuracy"])
        lines.append(f"| PyTorch compiled {mode} | {values[0]:.3f}% | {values[1]:.3f}% |")
    lines += ["","## Complete operation latency at batch 64","",
              "Ranges span cohort medians, in microseconds. The graph column reports steady-state device work per inference. "
              "The uncaptured column includes host dispatch and stream completion; the host-request column also includes "
              "changing input H2D, output allocation and label D2H. Weights are prepared and reusable in these three columns.","",
              "| Implementation | Captured device us | Uncaptured device us | Host request us |",
              "|---|---:|---:|---:|"]
    def append_row(name,rows):
        cells=[]
        for boundary in BOUNDARIES[:3]:
            if all(boundary in row["timings_ms"] for row in rows):
                values=[1000*med(row,boundary) for row in rows]
                cells.append(f"{min(values):.2f}-{max(values):.2f}")
            else:cells.append("not isolated")
        lines.append(f"| {name} | "+" | ".join(cells)+" |")
    for config in configs:append_row("Native "+label(config),[rows[64,config] for rows in cases])
    for mode in ("fp32","fp16","bf16","fp8"):
        for engine in ("eager","compiled"):
            append_row(f"PyTorch {mode} {engine}",[next(r for r in record["cases"] if r["batch"]==64 and r["mode"]==mode and r["engine"]==engine) for record in torch])
        append_row(f"PyTorch {mode} internal graphs",[next(r for r in record["cases"] if r["batch"]==64 and r["mode"]==mode) for record in graphs])
    lines += ["","## Allocation and weight reuse","",
              "Native records additionally retain fresh-plan requests: arrays/transfers, native plans, weight preparation, "
              "inference, labels and native teardown are timed together with a cached algorithm index. This is a warm process "
              "and a reusable CuPy allocator, not cold driver initialization. Candidate tuning is reported separately.","",
              "The [reuse CSV](results/application-reuse.csv) measures one weight-preparation phase followed by 1, 8 or 64 "
              "complete resident inferences. It excludes weight H2D and allocations, which belong to the fresh-plan/request boundary. "
              "Reuse curves are measured totals rather than extrapolated GEMM throughput.","",
              "## Validation-gated held-out format selection","",
              "An offline selector interpolates latency from cohort-A batch sizes 16,64,256. It is evaluated at 32,128,192 "
              "for both models. Eligibility uses each model's validation accuracy, never its test labels. The complete sweep "
              "retains accuracy-loss budgets of 0,0.5 and 1 percentage points instead of selecting a favorable budget after testing.","",
              f"Across {len(selection)} held-out model/batch/budget/timer evaluations, geometric-mean regret versus the fastest "
              f"eligible measured configuration is {selector['geomean_regret']:.3f}x. All choices and test outcomes are in "
              "[selection data](results/application-selector.json). This evaluates configuration decisions offline; runtime routing "
              "overhead and generalization to other architectures/models are not established.","",
              "## Interpretation and limits","",
              "FP4 kernel speed alone is insufficient to choose an application format. Preparation, host dispatch, transfers, "
              "setup/reuse and task accuracy can change the ranking. Increasing one layer's precision need not monotonically "
              "improve classification on a finite test set. The evidence supports evaluating the complete computation and "
              "reporting accuracy together with each latency boundary.","",
              "PyTorch 2.13.0+cu130 is the installed reference stack; native code uses CUDA 13.3/cuBLASLt as recorded. "
              "`fullgraph=True` succeeds for every compiled case. Maximum-autotune mode was requested, but PyTorch declined "
              "GEMM max-autotuning on this 36-SM device and used its available GEMM backend. The specialization limit was "
              "raised from eight to 64 for the declared static shape/precision grid. Partial first-attempt records are excluded. "
              "TF32 and reduced-precision FP16/BF16 accumulation were disabled; all GEMM outputs are FP32.","",
              "The additional `reduce-overhead` reference enables internal CUDA Graphs, dynamic shapes and static weight addresses. "
              "Every case retains live CUDA Graph roots; the root count grows from one to 24 across the full grid. "
              "Its host timers include graph-tree management and step markers. It has no isolated pure-device column. "
              "Numerical witnesses and complete predictions are independently replayed for this reference too.","",
              "The native API reserves an FP4 backend field even for configurations that contain no FP4; those FP32/FP16/BF16/FP8 "
              "products use cuBLASLt. All executions are exploratory shared-system observations. Small test data, padded model "
              "dimensions, repeated graph inputs, different installed library versions and single-session timing limit conclusions.","",
              "Run `python verify_application.py` for CPU-only numerical/provenance/quality replay and "
              "`python summarize_application.py --check` to verify the [full timing CSV](results/application-summary.csv). "
              "Outlier correction and direct loading/resource characterization remain open under [PROGRAM.md](PROGRAM.md).",""]
    return {"APPLICATION_REPORT.md":"\n".join(lines),"results/application-summary.csv":output.getvalue(),
            "results/application-reuse.csv":reuse.getvalue(),"results/application-selector.json":json.dumps(selector,indent=2,allow_nan=False)+"\n"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    for name,contents in artifacts().items():
        path=ROOT/name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8")!=contents:raise ValueError(f"Stale application artifact: {name}")
        else:path.write_text(contents,encoding="utf-8",newline="\n")
    print("Application artifacts match" if args.check else "Wrote application report, timing/reuse CSV and selector")


if __name__=="__main__":main()
