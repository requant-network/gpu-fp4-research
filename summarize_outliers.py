"""Derive correction accuracy/cost tables and explicitly offline constrained choices."""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
ERROR_BUDGETS=(1e-6,1e-5,1e-4,1e-3,.01,.05,.1,.2)
TIME_MULTIPLIERS=(.5,1,2,4,8)


def median(case,boundary):return statistics.median(case["timings_ms"][boundary])


def outputs():
    records=[json.loads((ROOT/"results"/name).read_text(encoding="utf-8")) for name in ("outlier-cohort-a.json","outlier-cohort-b.json")]
    if not all(r.get("complete") and len(r["cases"])==299 for r in records):raise ValueError("Complete correction cohorts required")
    buffer=io.StringIO(newline="");writer=csv.writer(buffer,lineterminator="\n")
    writer.writerow(["seed","model_seed","m","n","k","profile","kind","partition","label","boundary","median_ms","p10_ms","p90_ms","relative_frobenius_error","setup_wall_ms"])
    constrained={"error_budgets":ERROR_BUDGETS,"time_budget_multipliers_of_fp16":TIME_MULTIPLIERS,
                 "scope":"Offline empirical choices using each case's measured error/latency; not a deployable predictor, held-out selector or quality guarantee for unseen inputs",
                 "fixed_error":[],"fixed_time":[]}
    grouped=[]
    for record in records:
        groups={}
        for case in record["cases"]:
            groups.setdefault((tuple(case["shape"]),case["profile"],case["kind"]),{})[case["label"]]=case
            for boundary,values in case["timings_ms"].items():
                values=sorted(values)
                writer.writerow([record["seed"],record["model_seed"],*case["shape"],case["profile"],case["kind"],case["partition"],case["label"],boundary,
                                 f"{statistics.median(values):.9f}",f"{values[2]:.9f}",f"{values[26]:.9f}",f"{case['checks']['relative_frobenius_error']:.12g}",f"{case['setup_wall_ms']:.9f}"])
        grouped.append(groups)
        for (shape,profile,kind),cases in groups.items():
            for boundary in ("prepacked","online-a","online-both"):
                base={"seed":record["seed"],"shape":list(shape),"profile":profile,"kind":kind,"boundary":boundary}
                for budget in ERROR_BUDGETS:
                    eligible=[c for c in cases.values() if c["checks"]["relative_frobenius_error"]<=budget]
                    chosen=min(eligible,key=lambda c:median(c,boundary)) if eligible else None
                    constrained["fixed_error"].append({**base,"maximum_relative_error":budget,"chosen":None if chosen is None else chosen["label"],
                        "median_ms":None if chosen is None else median(chosen,boundary),"relative_error":None if chosen is None else chosen["checks"]["relative_frobenius_error"]})
                for multiplier in TIME_MULTIPLIERS:
                    budget=multiplier*median(cases["fp16"],boundary)
                    eligible=[c for c in cases.values() if median(c,boundary)<=budget]
                    chosen=min(eligible,key=lambda c:c["checks"]["relative_frobenius_error"]) if eligible else None
                    constrained["fixed_time"].append({**base,"fp16_time_multiplier":multiplier,"maximum_median_ms":budget,
                        "chosen":None if chosen is None else chosen["label"],"median_ms":None if chosen is None else median(chosen,boundary),
                        "relative_error":None if chosen is None else chosen["checks"]["relative_frobenius_error"]})
    total=sum(c["shape"][0]*c["shape"][1] for r in records for c in r["cases"])
    lines=["# FP4 outlier correction: accuracy and complete cost","",
           "**Status: study 2's bounded column/block/residual comparison is complete. The fifteen-study program remains active.**","",
           f"Two independent input/model cohorts contain 598 configurations, 89,700 retained observations and {total:,} output entries checked before and after timing. "
           "The grid includes seven synthetic shapes, four distributions and two real-model layers at three batches. "
           "Twelve complete small numerical witnesses permit CPU replay. All selected columns, block masks and FP16 factors were checked against independent CPU construction.","",
           "## Algorithms and correctness contract","",
           "The common base is block-scaled FP4 GEMM with native packed conversion, the same fixed tensor scale of one and the same 128x128x128 CUTLASS tile as earlier studies. "
           "Comparators are native cuBLASLt FP8, FP16 and strict FP32; each uses up to eight heuristic candidates, tuned outside the reported operation times. "
           "Every correction adds two FP16 Tensor Core products with FP32 accumulation and a final CUDA summation kernel.","",
           "Let Q_A and Q_B denote decoded FP4 operands, and H denote FP16 rounding. Matrices use the repository's C = A B^T convention.","",
           "- **Columns:** GPU reductions compute max(abs(A[:,j])) times max(abs(B[:,j])). A GPU bitonic sort selects 128 or 256 columns (capped at K), breaking ties by ascending index. "
           "The result is Q_A Q_B^T + H(A_S) H(B_S)^T - H(Q_AS) H(Q_BS)^T. Correction GEMM depth is the selected column budget.",
           "- **Blocks:** each operand selects contiguous 16-value blocks whose maximum magnitude exceeds 0.25 or 0.75 of that operand's global maximum. "
           "R_A/R_B contain selected FP32 residuals A-Q_A and B-Q_B, zero elsewhere. The result is Q_A Q_B^T + H(R_A) H(Q_B+R_B)^T + H(Q_A) H(R_B)^T. "
           "Selected B blocks are copied from B before FP16 rounding to avoid an extra residual re-addition.",
           "- **Full residual:** the same residual formula selects every block. It tests whether finer component reconstruction is useful despite two additional full-depth GEMMs.","",
           "Block/residual correction uses dense FP16 factors with zeros in unselected positions. It does not compact arbitrary blocks or claim sparse Tensor Core speedups. "
           "Column correction does reduce GEMM depth. Neither threshold/column score nor budget is optimized separately for a measured test case. "
           "Arithmetic is checked against FP64 products of the exact prepared factors with a conservative FP32 accumulation/summation bound; error is separately measured against the original FP32 inputs in FP64.","",
           "## Synthetic outlier results","",
           "Ranges span cohorts for the element-outlier profile. Error is relative Frobenius product error. "
           "Latency ratios above 1 favor correction over direct FP16 and include online-A detection, selection, gathering and preparation. "
           "Both references start from common FP32 source inputs; B and its required preparation remain resident.","",
           "| M x N x K | Plain FP4 error | Columns128 error | Full residual error | FP16 / columns128 online A | FP16 / residual online A |",
           "|---|---:|---:|---:|---:|---:|"]
    for shape in sorted({tuple(c["shape"]) for c in records[0]["cases"] if c["kind"]=="synthetic"}):
        stats=[[],[],[],[],[]]
        for group in grouped:
            cases=group[shape,"outliers","synthetic"]
            for i,label in enumerate(("fp4","columns-128","residual")):stats[i].append(cases[label]["checks"]["relative_frobenius_error"])
            stats[3].append(median(cases["fp16"],"online-a")/median(cases["columns-128"],"online-a"))
            stats[4].append(median(cases["fp16"],"online-a")/median(cases["residual"],"online-a"))
        lines.append(f"| {' x '.join(map(str,shape))} | "+" | ".join(f"{min(v):.3g}-{max(v):.3g}" for v in stats)+" |")
    lines += ["","Synthetic inputs use the existing integer CLT generator and power-of-two outlier multipliers. Many values are exactly representable in FP16, "
              "so direct FP16 can have unusually small input-rounding error on these distributions. Real learned weights/activations below add general FP32 values. "
              "Synthetic and real-data conclusions must not be conflated.","","## Real activations","",
              "Two independently trained UCI digits classifiers supply validation inputs and hidden ReLU activations, frozen from FP64 layer evaluation then rounded once to FP32. "
              "This isolates layer products on real data; these error measurements are not complete classifier accuracy or evidence that approximate first-layer activations preserve downstream quality. "
              "Batch 64 results retain each model separately.","",
              "| Model seed | Layer | Plain FP4 error | FP8 error | FP16 error | Columns128 error | Blocks0.25 error | Residual error |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for record,group in zip(records,grouped):
        for profile,shape in (("digits-input",(64,256,128)),("digits-hidden",(64,128,256))):
            cases=group[shape,profile,"real"]
            lines.append(f"| {record['model_seed']} | {profile} | "+" | ".join(f"{cases[label]['checks']['relative_frobenius_error']:.4g}" for label in ("fp4","fp8","fp16","columns-128","blocks-0.25","residual"))+" |")
    online=[r for r in constrained["fixed_error"] if r["boundary"]=="online-a" and r["maximum_relative_error"]==.001]
    corrected=sum(r["chosen"] not in (None,"fp4","fp8","fp16","fp32") for r in online)
    lines += ["","## Fixed error and fixed time","",
              "[All constrained outcomes](results/outlier-constrained.json) retain eight error budgets (0.000001 through 0.2) and five time budgets (0.5 through 8 times direct FP16), "
              "for prepacked, online-A and online-both operation boundaries. Infeasible budgets explicitly retain a null choice. "
              "These are offline empirical frontiers using measured error and median latency from the same case; they are not a learned selector or an unseen-input error guarantee.","",
              f"At relative-error budget 0.001, correction is the fastest eligible online-A candidate in {corrected} of {len(online)} cohort/shape/profile groups. "
              "The complete record retains whichever uncorrected format wins as well. Preparation-only improvements are not substituted for complete operation costs.","",
              "## Cost boundaries, reproducibility and limits","",
              "Five CUDA-event boundaries measure prepare A, prepare both, prepacked GEMM including corrections, online A and online both. "
              "Online-A column selection reuses cached B column maxima but must reselect columns and gather both operands; block/residual modes reuse B factors. "
              "Detection, sorting, masks, gathering, correction GEMMs and final addition are included where required. Five warmups and thirty observations use 32 repetitions per graph; "
              "observation order rotates and the second cohort reverses group/configuration order. Nearest-rank p10/p90 and every candidate are in [the complete CSV](results/outlier-summary.csv).","",
              "Fresh native plan allocation/setup is separately recorded once per configuration, including both correction plans and their buffers; it is descriptive, not a sampled cold-process distribution. "
              "Source H2D transfer, Python dispatch, tuning, graph construction, correctness checks and teardown are excluded from device-event times. "
              "Prepacked/online-A/online-both expose preparation reuse; results do not establish changing-input host-request latency or energy. "
              "Correction owns four FP16 factors, two full FP32 output temporaries, reduction/selection/mask buffers and two cuBLASLt workspaces; it is not a memory-saving scheme.","",
              "Other GPU contexts remain present. Measurements are exploratory shared-system observations, not exclusive performance guarantees. "
              "The implementation supports the tested aligned shapes, finite inputs and FP16-representable correction factors, K up to 2048; it does not establish the best possible correction algorithm or scientific novelty. "
              "Sparse block compaction, alternative outlier scores, optimized tensor scales and complete corrected-model deployment remain separate potential improvements. "
              "The broad program's pipeline/resource/configuration gates remain open even though this bounded outlier comparison supplies the required evidence for study 2.","",
              "Reproduce after earlier extended/hardware builds with `bash build_outlier.sh ../cutlass-fp4`. Frozen real activations are prepared by `prepare_outlier_activations.py`; "
              "run `outlier_study.py` with seeds 20261119/model 20261116 and 20261219/model 20261216, separate output paths and `--reverse` for the latter. "
              "Run `python verify_outliers.py` and `python summarize_outliers.py --check` without GPU dependencies to replay the public artifacts.",""]
    return {"OUTLIER_REPORT.md":"\n".join(lines),"results/outlier-summary.csv":buffer.getvalue(),
            "results/outlier-constrained.json":json.dumps(constrained,indent=2,allow_nan=False)+"\n"}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--check",action="store_true");args=parser.parse_args()
    for name,content in outputs().items():
        path=ROOT/name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8")!=content:raise ValueError(f"Stale outlier artifact: {name}")
        else:path.write_text(content,encoding="utf-8",newline="\n")
    print("Outlier derived artifacts match" if args.check else "Wrote outlier report, tables and constrained outcomes")


if __name__=="__main__":main()
