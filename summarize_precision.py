"""Derive reconstruction/precision/cost comparisons from verified native runs."""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent


def median(case,key):return statistics.median(case["host_request_ms"] if key=="host-request" else case["timings_ms"][key])


def outputs():
    records=[json.loads((ROOT/f"results/precision-cohort-{suffix}.json").read_text(encoding="utf-8")) for suffix in ("a","b")]
    if not all(r.get("complete") and not r["quick"] for r in records):raise ValueError("Complete precision cohorts required")
    groups=[];buffer=io.StringIO(newline="");writer=csv.writer(buffer,lineterminator="\n")
    writer.writerow(["seed","model_seed","m","n","k","profile","kind","partition","label","selected_components","required_components","retained_emulation_bits","boundary","median_ms","p10_ms","p90_ms","relative_frobenius_error_vs_gold","exact_sample_relative_error","setup_wall_ms","initial_preparation_wall_ms"])
    frontiers={"scope":"Offline empirical choices; both full CPU/analytic-reference error and retained exact-integer sample error must meet the budget. Not a deployable predictor, unseen-input guarantee or certified all-output error selector.","choices":[]}
    for record in records:
        by_group={}
        for case in record["cases"]:
            by_group.setdefault((tuple(case["shape"]),case["profile"],case["kind"]),{})[case["label"]]=case
            checks=case["checks"]
            for key in ("prepacked","online-a","online-both","host-request"):
                values=sorted(case["host_request_ms"] if key=="host-request" else case["timings_ms"][key])
                writer.writerow([record["seed"],record["model_seed"],*case["shape"],case["profile"],case["kind"],case["partition"],case["label"],checks["selected_components"],checks["required_components"],checks["retained_emulation_bits"],key,
                    f"{statistics.median(values):.9f}",f"{values[2]:.9f}",f"{values[26]:.9f}",f"{checks['relative_frobenius_error']:.12g}",f"{checks['exact_sample_relative_error']:.12g}",case["setup_wall_ms"],case["initial_preparation_wall_ms"]])
        for (shape,profile,kind),cases in by_group.items():
            for key in ("prepacked","online-a","online-both","host-request"):
                for budget in (1e-3,1e-6,1e-10,1e-12):
                    eligible=[c for c in cases.values() if max(c["checks"]["relative_frobenius_error"],c["checks"]["exact_sample_relative_error"])<=budget]
                    chosen=min(eligible,key=lambda c:median(c,key)) if eligible else None
                    frontiers["choices"].append({"seed":record["seed"],"shape":shape,"profile":profile,"kind":kind,"boundary":key,"maximum_relative_error":budget,
                        "selected":None if chosen is None else chosen["label"],"median_ms":None if chosen is None else median(chosen,key)})
        groups.append(by_group)
    total=sum(len(r["cases"]) for r in records);entries=sum(c["shape"][0]*c["shape"][1] for r in records for c in r["cases"])
    observations=sum(len(values) for r in records for c in r["cases"] for values in list(c["timings_ms"].values())+[c["host_request_ms"]])
    emulation=[c for r in records for c in r["cases"] if c["mode"]==2]
    active=sum(c["checks"]["retained_emulation_bits"]>0 for c in emulation)
    requested=[c for r in records for c in r["cases"] if c["mode"]==3]
    bf16_capable=sum(c["tuning"][0][c["selected_algorithms"][0]]["emulation_capability"]!=0 for c in requested)
    lines=["# High-precision products from INT8 components","",
           "**Status: study 6's bounded precision/reconstruction comparison is complete. The full fifteen-study program remains active.**","",
           f"Two independent input/model cohorts contain {total:,} configurations, {observations:,} retained device/host timing observations and {entries:,} output entries checked at three checkpoints. "
           "Eight full-product witnesses and two changing-input transition witnesses support CPU replay. Each case additionally retains 64 exact-integer product samples and exact INT8 product samples where components are used, before timing, after timing and after host requests.","",
           "## Problem and methods","",
           "All operands start as finite FP32 and every method returns FP64 output. The question is accurate products of FP32 inputs, not universal emulation of every FP64 input or IEEE-754 special-value behavior. "
           "Native references use pedantic cuBLASLt FP32/FP64; FP32 output is promoted to FP64 as part of its measured operation. The library comparisons request performant/eager dynamic FP64 emulation, fixed 31/55-bit emulation and BF16x9 FP32 emulation. "
           "Each primary/fallback GEMM is tuned among up to eight cuBLASLt heuristics outside reported operation times.","",
           "The custom method normalizes each operand row by a power of two. Consecutive signed base-128 digits are extracted by truncation toward zero, with one INT8 matrix per 7-bit component. "
           "Fixed methods use 1, 2, 3, 4, 6 or 8 components. Every pair of digit matrices is multiplied with INT32 accumulation; selected algorithms must report IMMA implementation flags. "
           "Products on the same component-index diagonal are summed in INT64, then scaled and accumulated into FP64 from less significant diagonals toward more significant ones.","",
           "For row exponents e_A/e_B and digits D, the result is the sum over s of 2^(e_A+e_B-7*(s+2)) times the integer diagonal sum for i+j=s. "
           "For K<=8192 and digits bounded by 127, each INT32 dot is bounded by 8192*127^2; even eight such dots fit signed INT32, while the implementation conservatively uses INT64 diagonal storage. "
           "This integer path avoids FP32 rounding inside partial products. Finite-component truncation and final FP64 reconstruction still require separate error checks.","",
           "The adaptive method examines each row's maximum exponent and least significant nonzero input bit. It selects enough components to represent every input exactly when at most eight suffice; otherwise it uses native FP64. "
           "This is an input-representation decision, not a guarantee that the final sum is correctly rounded. A changed component count/fallback decision re-encodes B when its cached representation is insufficient. "
           "Changed-input witnesses force normal, cancellation, wide-range and normal decisions rather than validating only a static choice.","",
           "## Actual library execution","",
           "The [archived NVIDIA cuBLAS documentation](https://docs.nvidia.com/cuda/archive/13.3.1/cublas/index.html#floating-point-emulation) lists BF16x9 for compute capabilities 10.0/10.3 and FP64 fixed-point emulation for 12.x. "
           "Requested API modes alone do not prove an emulated execution. The library's device mantissa-bit output is retained: positive values indicate emulation, -1 indicates fallback. "
           "Strategy, mantissa control, configured bit cap and special-value mask are queried and recorded. Finite inputs allow special-value support to be disabled explicitly. "
           "Defaults are library/device dependent; a configured zero cap is not interpreted as a fixed universal bit limit.","",
           f"Among {len(emulation)} FP64-emulation cases, {active} report positive retained bits and {len(emulation)-active} report fallback at the initial checkpoint. "
           f"Among {len(requested)} accepted BF16x9 requests, {bf16_capable} selected algorithms report emulation capability; this request must not be mislabeled as BF16x9 Tensor Core acceleration when flags indicate native FP32. "
           "Per-checkpoint retained bits and numerical implementation flags remain in the raw records.","",
           "Each plan supplies 512 MiB of workspace. This is the supplied buffer, not a measured peak-memory bound; the library can require additional internal allocations. "
           "Native conversion, component analysis/extraction, recombination and internal library preparation are included wherever the operation boundary requires them.","",
           "## Normal-profile cost comparison","",
           "Ranges span cohorts. Ratios above 1 favor the named alternative over native FP64. All ratios include preparation of A; B stays prepared. "
           "The adaptive exact-sample column is separately evaluated against correctly rounded integer-gold products.","",
           "| M x N x K | Native FP64 online A, microseconds | FP64 / performant emulation | FP64 / fixed55 emulation | FP64 / adaptive | Adaptive exact-sample relative error |",
           "|---|---:|---:|---:|---:|---:|"]
    shapes=sorted({tuple(c["shape"]) for c in records[0]["cases"] if c["kind"]=="synthetic"})
    for shape in shapes:
        values=[[],[],[],[],[]]
        for by_group in groups:
            cases=by_group[shape,"normal","synthetic"];native=median(cases["fp64"],"online-a")
            values[0].append(native*1000);values[1].append(native/median(cases["emu64-performant"],"online-a"));values[2].append(native/median(cases["emu64-fixed55"],"online-a"))
            values[3].append(native/median(cases["adaptive-8"],"online-a"));values[4].append(cases["adaptive-8"]["checks"]["exact_sample_relative_error"])
        lines.append(f"| {' x '.join(map(str,shape))} | "+" | ".join(f"{min(v):.4g}-{max(v):.4g}" for v in values)+" |")
    lines += ["","## Cancellation, range and subnormal controls","",
              "The normal generator uses portable integer mantissas with rich FP32 low bits. Cancellation pairs have an analytic rank-one residual, so their complete target product is known exactly. "
              "Wide-range inputs use complementary column exponents from -60 to 60: true pairwise products remain moderate while row-wise representations must span a large range. "
              "Subnormal A inputs are scaled by 2^-140; B remains normal. Long dots include K=8192. Zero rows remain in every synthetic case.","",
              "Representative 128-cubed exact-sample errors below include every cohort rather than selecting the more favorable input seed.","",
              "| Seed | Profile | FP32 error | FP64 error | Slices4 error | Slices8 error | Adaptive components (0=fallback) | Adaptive error |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for record,by_group in zip(records,groups):
        for profile in ("normal","cancellation","spread","subnormal"):
            cases=by_group[(128,128,128),profile,"synthetic"]
            errors=[cases[label]["checks"]["exact_sample_relative_error"] for label in ("fp32","fp64","slices-4","slices-8")]
            lines.append(f"| {record['seed']} | {profile} | "+" | ".join(f"{v:.5g}" for v in errors)+f" | {cases['adaptive-8']['checks']['selected_components']} | {cases['adaptive-8']['checks']['exact_sample_relative_error']:.5g} |")
    lines += ["","## Real activations and precision/cost frontiers","",
              "Two independently trained digits classifiers supply frozen input/hidden activations and learned weights. These are real layer-product tests, not a claim of complete model-quality improvement from higher precision. "
              "Batch 64 below compares exact-sample errors and complete host-request costs, which include fresh selection and transfers.","",
              "| Model | Layer | Adaptive components | FP32 exact-sample error | FP64 exact-sample error | Adaptive exact-sample error | Native FP64 / adaptive host-request time |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for record,by_group in zip(records,groups):
        for shape,profile in (((64,256,128),"digits-input"),((64,128,256),"digits-hidden")):
            cases=by_group[shape,profile,"real"]
            errors=[cases[label]["checks"]["exact_sample_relative_error"] for label in ("fp32","fp64","adaptive-8")]
            lines.append(f"| {record['model_seed']} | {profile} | {cases['adaptive-8']['checks']['selected_components']} | "+" | ".join(f"{v:.5g}" for v in errors)+
                         f" | {median(cases['fp64'],'host-request')/median(cases['adaptive-8'],'host-request'):.4f} |")
    lines += ["","Every real case, retained error, component decision, supplied workspace, one-time setup and initial preparation cost is in [the complete CSV](results/precision-summary.csv).","",
              "[Offline precision/cost frontiers](results/precision-frontiers.json) compare error budgets 0.001, 0.000001, 1e-10 and 1e-12 across prepacked, online-A, online-both and host-request boundaries. "
              "Eligibility requires both the full reference-relative error and exact-sample error to meet the budget; infeasible outcomes are retained. "
              "These are empirical outcomes on measured inputs, not the held-out per-operation error predictor still required by study 1.","",
              "## Independent validation and timing limits","",
              "For exact witnesses, each finite FP32 input is represented as an integer times 2^-149. Arbitrary-precision integer dot products are converted once to FP64 and scaled by 2^-298. "
              "Thus cancellation and FP64-reference rounding do not silently invalidate the small-case/error samples. Every prepared digit byte, exponent and component requirement is independently reconstructed. "
              "Full-size arithmetic is checked against CPU FP64 products on the exact prepared values with a conservative accumulation bound. The fixed-library quantization allowance is an experimental screening tolerance, not a proof of the library's opaque internal quantizer. "
              "Normal/real full FP64-reference errors near its own rounding floor are not evidence of superiority over exact arithmetic. Exact retained samples provide the stronger local check.","",
              "Captured timings repeat identical resident inputs. The adaptive host decision is cached inside these graphs; online graphs charge analysis and encoding, but not the decision readback/synchronization. "
              "Host-request timings include H2D A, fresh analysis/selection, any required B re-encoding, computation, full FP64 D2H output and synchronization. Those thirty requests repeat the same A; the separate changed-input witnesses test decision transitions. "
              "B and native plans are reused. Setup/initial preparation are separately retained once per case, not claimed as cold-process timing distributions.","",
              "Five graph warmups and thirty observations use eight repetitions for smaller products and one for large/long products, equally across methods of the same shape. "
              "The second cohort reverses group/configuration order. Transfers, plan creation, tuning, graph construction and validation are excluded from device timers. "
              "Graph averages and host-request latencies answer different questions; they are not interchangeable speedup claims.","",
              "The server retains other GPU contexts and unchanged clock/power/service settings. All results are exploratory shared-system observations. "
              "The protocol does not establish general FP64-input emulation, signed-zero/nonfinite IEEE behavior, scientific novelty, an architecture ceiling, or optimal component fusion. "
              "Separate INT8 GEMMs and diagonal/reconstruction kernels retain launch and memory costs; poor cases remain part of the findings.","",
              "Reproduce with `bash build_precision.sh`, then `precision_study.py` using seeds 20261121/model 20261116 and 20261221/model 20261216, separate outputs and `--reverse` for the second. "
              "Run `python verify_precision.py` and `python summarize_precision.py --check` for CPU-only replay. A preliminary eager-only partial run was preserved privately; the published cohorts execute the final performant/eager protocol and current recorded source hashes.",""]
    return {"PRECISION_REPORT.md":"\n".join(lines),"results/precision-summary.csv":buffer.getvalue(),"results/precision-frontiers.json":json.dumps(frontiers,indent=2,allow_nan=False)+"\n"}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--check",action="store_true");args=parser.parse_args()
    for name,content in outputs().items():
        path=ROOT/name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8")!=content:raise ValueError(f"Stale precision artifact: {name}")
        else:path.write_text(content,encoding="utf-8",newline="\n")
    print("Precision derived artifacts match" if args.check else "Wrote precision report, complete table and offline frontiers")


if __name__=="__main__":main()
