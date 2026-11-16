"""Derive native configuration/cache tables and held-out configuration policies."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
BASELINE="m128-sauto-coop"


def median(case,key):return statistics.median(case["timings_ms"][key])


def outputs():
    records=[json.loads((ROOT/f"results/pipeline-cohort-{suffix}.json").read_text(encoding="utf-8")) for suffix in ("a","b")]
    if not all(r.get("complete") and not r["quick"] for r in records):raise ValueError("Complete pipeline campaigns required")
    groups=[]
    for record in records:
        by_group={}
        for case in record["cases"]:by_group.setdefault((tuple(case["shape"]),case["profile"]),{})[case["label"]]=case
        groups.append(by_group)
    keys=sorted(records[0]["cases"][0]["timings_ms"])
    labels=sorted(next(iter(groups[0].values())))
    policies=[]
    for key in keys:
        training=[]
        for (shape,profile),rows in groups[0].items():
            if rows[BASELINE]["partition"]=="training":training.append((math.prod(shape),{label:median(rows[label],key) for label in labels}))
        best=None
        for threshold in [-1]+sorted({size for size,_ in training}):
            choices=[]
            for below in (True,False):
                costs=[values for size,values in training if (size<=threshold)==below]
                choices.append(min(labels,key=lambda label:sum(math.log(v[label]/min(v.values())) for v in costs)))
            score=sum(math.log(v[choices[0] if size<=threshold else choices[1]]/min(v.values())) for size,v in training)
            if best is None or score<best[0]:best=score,threshold,*choices
        _,threshold,small,large=best
        evaluations=[]
        for record,by_group in zip(records,groups):
            for (shape,profile),rows in by_group.items():
                if rows[BASELINE]["partition"]!="held-out":continue
                chosen=small if math.prod(shape)<=threshold else large
                fastest=min(median(c,key) for c in rows.values())
                evaluations.append({"seed":record["seed"],"shape":shape,"profile":profile,"selected":chosen,
                                    "regret":median(rows[chosen],key)/fastest,"speedup_over_default":median(rows[BASELINE],key)/median(rows[chosen],key)})
        policies.append({"boundary_condition":key,"training_cohort":records[0]["seed"],"training_groups":len(training),"feature":"M*N*K",
                         "threshold":threshold,"at_or_below":small,"above":large,"held_out_evaluations":evaluations,
                         "held_out_geomean_regret":math.exp(statistics.mean(math.log(e["regret"]) for e in evaluations)),
                         "held_out_geomean_speedup_over_default":math.exp(statistics.mean(math.log(e["speedup_over_default"]) for e in evaluations))})
    selector={"scope":"Six independent two-leaf rules fitted on cohort-A training shapes only; cache condition treated as known; both cohorts' four held-out shapes evaluated; no new-distribution or accuracy-budget prediction claim",
              "candidate_labels":labels,"policies":policies}
    buffer=io.StringIO(newline="");writer=csv.writer(buffer,lineterminator="\n")
    writer.writerow(["seed","m","n","k","profile","partition","label","boundary_condition","median_ms","p10_ms","p90_ms","relative_frobenius_error"])
    for record in records:
        for case in record["cases"]:
            for key,values in case["timings_ms"].items():
                values=sorted(values);writer.writerow([record["seed"],*case["shape"],case["profile"],case["partition"],case["label"],key,
                    f"{statistics.median(values):.9f}",f"{values[2]:.9f}",f"{values[26]:.9f}",f"{case['checks']['relative_frobenius_error']:.10g}"])
    total=sum(len(r["cases"]) for r in records);observations=sum(len(v) for r in records for c in r["cases"] for v in c["timings_ms"].values())
    entries=sum(c["shape"][0]*c["shape"][1] for r in records for c in r["cases"])
    families=records[0]["families"]
    calibration=json.loads((ROOT/"results/pipeline-timer-calibration.json").read_text(encoding="utf-8"))
    floor={key:statistics.median(values)*1000 for key,values in calibration["timings_ms"].items()}
    lines=["# Native SM120 pipelines, traversal and cache pressure","",
           "**Status: study 4's bounded configuration investigation is complete. The complete fifteen-study program remains active.**","",
           f"Two cohorts contain {total:,} configurations, {observations:,} retained observations and {entries:,} output entries checked before and after timing. "
           "Eleven shapes and normal/outlier inputs cover twelve usable pipeline builds plus nine traversal configurations. Four additional compiled configurations fail their runtime resource request and remain in the evidence. "
           "Four small input/output witness files support CPU numerical replay. All configurations preserve identical FP4 encoded values/scales; configuration selection changes implementation rather than the quantization contract.","",
           "## Native controls","",
           "The frozen CUTLASS base is transformed only at tile M, mainloop stage count and schedule declarations. Tile M is 128 or 256, N/K tiles remain 128, "
           "and stage policies are automatic, 2, 3 or 4. Explicit cooperative and ping-pong NVFP4 schedules change consumer work assignment and buffering while retaining TMA loads. "
           "The pinned builder uses different MMA atom layouts for cooperative/ping-pong schedules. Automatic stages resolve to four for M=128 and two for M=256 in this environment.","",
           "A separate native library keeps M=128/automatic/cooperative fixed and compares heuristic traversal with along-M/along-N raster order and swizzle bounds 1, 2, 4 and 8. "
           "This is a controlled traversal comparison, not the full cross product of traversal and every pipeline build. Earlier [encoder/loading experiments](HARDWARE_REPORT.md) supply scalar/vector-load/warp-reduction controls. "
           "Only features supported by the pinned SM120 block-scaled implementation are tested; this is not a claim to exercise every feature of every NVIDIA architecture.","",
           "## Pipeline resources and unavailable requests","",
           "Function attributes and CUDA's occupancy calculator describe theoretical resource capacity. They do not measure active warps, dynamic spills, bandwidth or Tensor Core utilization. "
           "Local bytes are reported storage, not proof of executed spill traffic. Nsight Compute counters remain unavailable under the recorded driver permissions.","",
           "| Configuration | Resolved stages | Block threads | Registers/thread | Dynamic shared bytes | Local bytes/thread | Theoretical blocks/SM |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for family in families:
        if family["status"]=="available":
            r=family["resources"];lines.append(f"| {family['label']} | {r['stages']} | {r['block_threads']} | {r['registers_per_thread']} | {r['dynamic_shared_bytes']} | {r['local_bytes']} | {r['theoretical_blocks_per_sm']} |")
        else:lines.append(f"| {family['label']} | "+" | ".join(["unavailable"]*6)+" |")
    lines += ["","The unavailable variants report `invalid argument` during the runtime resource query. Their successful build manifests and actual failure messages are retained in each cohort. "
              "They have no fabricated zero latency. Successful requests expose the substantial shared-memory/local-storage changes introduced by buffering and work assignment; "
              "resource counts alone do not establish the cause of a measured latency change.","",
              "## Fixed pipeline comparisons","",
              "Ratios span cohorts on normal inputs. Above 1 favors the named candidate. All pipeline ratios use primed prepacked GEMM. "
              "The final column divides cache-pressure latency by primed latency for the fixed default; above 1 means the pressure condition is slower.","",
              "| M x N x K | Default / stage2 cooperative | Default / stage4 ping-pong | Default / M256 stage2 cooperative | Default cache-pressure / primed |",
              "|---|---:|---:|---:|---:|"]
    shapes=sorted({tuple(c["shape"]) for c in records[0]["cases"]})
    for shape in shapes:
        ratios=[[],[],[],[]]
        for by_group in groups:
            rows=by_group[shape,"normal"];base=median(rows[BASELINE],"prepacked/warm")
            for index,label in enumerate(("m128-s2-coop","m128-s4-ping","m256-s2-coop")):ratios[index].append(base/median(rows[label],"prepacked/warm"))
            ratios[3].append(median(rows[BASELINE],"prepacked/cache-pressure")/base)
        lines.append(f"| {' x '.join(map(str,shape))} | "+" | ".join(f"{min(v):.3f}-{max(v):.3f}" for v in ratios)+" |")
    lines += ["","## Held-out policies","",
              "For each operation/condition, one volume threshold and two candidate choices minimize training log latency regret. "
              "Only the first cohort's fourteen training shape/profile groups are used. Four rectangular shapes are excluded from fitting in both cohorts. "
              "These are configuration policies within the measured candidate set; cache condition must be supplied by the caller. Both input profiles occur in training, so this is not validation on unseen distributions.","",
              "| Operation/condition | Rule: at or below threshold / above | Held-out geometric-mean regret | Held-out speedup over default |",
              "|---|---|---:|---:|"]
    for policy in policies:
        lines.append(f"| {policy['boundary_condition']} | M*N*K <= {policy['threshold']}: {policy['at_or_below']} / {policy['above']} | {policy['held_out_geomean_regret']:.3f} | {policy['held_out_geomean_speedup_over_default']:.3f} |")
    lines += ["","Every held-out decision and regression is retained in [the policy data](results/pipeline-selector.json). "
              "[The complete CSV](results/pipeline-summary.csv) includes every traversal, pipeline, profile, boundary, condition and nearest-rank p10/p90. "
              "No per-test winner is substituted for the fitted rule.","",
              "## Timing and cache interpretation","",
              "Each graph contains conditioning, an external CUDA-event start node, one measured operation and an external end node. "
              "A primed sample executes an identical unmeasured operation before the start event. The pressure condition reads/writes a separate buffer four times reported L2 capacity before that event. "
              "Both conditioning costs are excluded from operation latency. Three measured boundaries are prepacked GEMM, online A and online both. "
              "Online preparation uses the frozen native float4 FP4 encoder; weights/scales are reused only where the boundary allows. Five warmups and thirty observations rotate all six graphs per configuration; the second cohort reverses groups/configurations.","",
              "The sweep applies cache pressure; without cache counters it does not prove every input line was evicted. Differences may also reflect conditioning/power state, and cannot be attributed exclusively to an L2 miss rate. "
              f"[Empty-interval calibration](results/pipeline-timer-calibration.json) reports median {floor['warm']:.3f} microseconds after warm conditioning and "
              f"{floor['cache-pressure']:.3f} microseconds after cache pressure. This descriptive event-node baseline is not subtracted from the results. "
              "Small differences and near-unity ratios should not be interpreted as a reliable optimization advantage. "
              "One-operation instrumented graph times are not directly interchangeable with earlier 32-repetition uninstrumented graph averages. "
              "Post-timing checks inspect current operands/output before recomputing anything, guarding against corruption during the timed graph.","",
              "Host transfers, allocations, plans, tuning/capture, conditioning and validation are outside the reported interval. "
              "This experiment deliberately studies device-resident reuse; changing-input host requests and setup/reuse costs are separately measured in [the application study](APPLICATION_REPORT.md). "
              "It does not establish streaming bandwidth, whole-request CPU latency, energy or the architecture's best achievable throughput. Other GPU contexts remain present; all measurements remain exploratory shared-system observations.","",
              "Build with `build_pipeline.py` for the declared sixteen combinations and `build_raster.py` for traversal/cache controls. "
              "Run `pipeline_study.py` for seeds 20261120 and 20261220, separate outputs and `--reverse` for the second cohort. "
              "Run `python verify_pipeline.py` and `python summarize_pipeline.py --check` for CPU-only artifact replay. "
              "The tested protocol closes the declared loading/pipeline/warp-role/buffering/traversal/resource/cache/held-out configuration comparison, not a claim of universal kernel optimality.",""]
    fusion=json.loads((ROOT/"results/fusion-resources.json").read_text(encoding="utf-8"))
    fusion_lines=["# Quantization/activation fusion: resource checkpoint","",
                  "**Status: study 3's declared fusion comparison is complete within the retained protocol. The full research program remains active.**","",
                  "The [encoder study](ENCODER_REPORT.md) compares separate and fused bias/ReLU/FP4 preparation across shapes and profiles. "
                  "The [native hardware study](HARDWARE_REPORT.md) adds hardware packed conversion and byte-identical fused/separate controls. "
                  "The [complete-application study](APPLICATION_REPORT.md) evaluates two classifiers, final labels/task accuracy, complete host requests, native cuBLASLt FP4 and eager/compiled/CUDA-Graph PyTorch references. "
                  "Those reports retain both useful gains and slower configurations. Compiled-reference CUDA/toolchain differences are recorded rather than hidden.","",
                  "This checkpoint supplies the remaining resource characterization: thirty function/block-size configurations query actual CUDA attributes and theoretical occupancy. "
                  "The frozen encoders are recompiled in the traversal-control library; source hashes and exact binary identity are retained in [resource data](results/fusion-resources.json). "
                  "These are not counter-derived achieved occupancy or spill measurements.","",
                  "| Encoder variant | Activation fused | Registers/thread | Shared bytes | Local bytes/thread | Theoretical thread occupancy, block128 |",
                  "|---|---|---:|---:|---:|---:|"]
    for row in fusion["rows"]:
        if row["block_threads"]==128:
            fusion_lines.append(f"| {row['variant']} | {row['fused']} | {row['registers_per_thread']} | {row['static_shared_bytes']} | {row['local_bytes']} | {row['theoretical_thread_occupancy']:.3f} |")
    fusion_lines += ["","The native scalar16 implementation uses more registers with activation fused (34 to 40); vector variants retain their measured register counts. "
                     "A theoretical occupancy of one does not mean the grid fills the device or that memory/compute pipelines are saturated. "
                     "The full data retain block sizes 128, 256 and 512, while block size alone does not retune register allocation. "
                     "The [pipeline study](PIPELINE_REPORT.md) independently retains large local/shared-storage footprints, unavailable requests and their latency outcomes. "
                     "Those resource controls do not imply that every slowdown is caused by spilling.","",
                     "Conclusions remain bounded: removing a pass can help while preparation/work assignment/tile choices still regress; complete classifier quality and request latency must be compared separately from isolated preparation speed. "
                     "No scientific novelty, architecture ceiling or achieved hardware-utilization claim is made. Counter access remains denied, and the shared-system timing qualification remains in force.",""]
    return {"PIPELINE_REPORT.md":"\n".join(lines),"FUSION_REPORT.md":"\n".join(fusion_lines),"results/pipeline-summary.csv":buffer.getvalue(),
            "results/pipeline-selector.json":json.dumps(selector,indent=2,allow_nan=False)+"\n"}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--check",action="store_true");args=parser.parse_args()
    for name,content in outputs().items():
        path=ROOT/name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8")!=content:raise ValueError(f"Stale pipeline artifact: {name}")
        else:path.write_text(content,encoding="utf-8",newline="\n")
    print("Pipeline derived artifacts match" if args.check else "Wrote pipeline/fusion reports, tables and held-out rules")


if __name__=="__main__":main()
