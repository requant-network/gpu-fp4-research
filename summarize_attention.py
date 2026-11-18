"""Derive complete attention cost/error, model-quality and memory comparisons."""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
BUDGETS=(.3,.1,.01,.001,1e-5)


def med(case,boundary):return statistics.median(case["host_request_ms"] if boundary=="host-request" else case["timings_ms"][boundary])


def error(case):return max(case["checks"]["relative_frobenius_error"],case["checks"]["sample_relative_error"])


def outputs():
    core=[json.loads((ROOT/f"results/attention-cohort-{s}.json").read_text(encoding="utf-8")) for s in ("a","b")]
    sage=[json.loads((ROOT/f"results/sage-attention-cohort-{s}.json").read_text(encoding="utf-8")) for s in ("a","b")]
    application=[json.loads((ROOT/f"results/attention-application-{s}.json").read_text(encoding="utf-8")) for s in ("a","b")]
    if not all(r.get("complete") and not r["quick"] for r in [*core,*sage,*application]):raise ValueError("Complete attention and model cohorts required")
    core_buffer=io.StringIO(newline="");writer=csv.writer(core_buffer,lineterminator="\n")
    writer.writerow(["seed","b","h","length","head_dim","mask","profile","label","boundary","median_ms","p10_ms","p90_ms","full_relative_error","sample_relative_error","owned_native_device_buffer_bytes","extra_peak_torch_allocated_bytes","setup_wall_ms"])
    groups=[];frontiers={"scope":"Offline empirical choices at fixed observed error, not an online policy or certified accuracy guarantee. Both full Frobenius and retained sampled errors must meet the budget.","choices":[]}
    observations=0
    for first,second in zip(core,sage):
        by_group={}
        for c in first["cases"]+second["cases"]:
            by_group.setdefault((tuple(c["shape"]),c["mask"],c["profile"]),{})[c["label"]]=c
            for boundary,values in [*c["timings_ms"].items(),("host-request",c["host_request_ms"])]:
                ordered=sorted(values);observations+=len(values)
                writer.writerow([first["seed"],*c["shape"],c["mask"],c["profile"],c["label"],boundary,f"{statistics.median(values):.9f}",f"{ordered[2]:.9f}",f"{ordered[26]:.9f}",
                    c["checks"]["relative_frobenius_error"],c["checks"]["sample_relative_error"],c.get("owned_native_device_buffer_bytes"),c.get("extra_peak_torch_allocated_bytes"),c.get("setup_and_tuning_wall_ms",c.get("setup_and_first_call_wall_ms"))])
        for (shape,kind,profile),cases in by_group.items():
            for boundary in ("prepared","online","host-request"):
                for budget in BUDGETS:
                    eligible=[c for c in cases.values() if error(c)<=budget];native=[c for c in eligible if c["engine"]=="native"];fused=[c for c in eligible if c["engine"] not in ("native","math")]
                    best=min(eligible,key=lambda c:med(c,boundary)) if eligible else None
                    frontiers["choices"].append({"seed":first["seed"],"shape":shape,"mask":kind,"profile":profile,"boundary":boundary,"maximum_relative_error":budget,
                        "selected":None if best is None else best["label"],"median_ms":None if best is None else med(best,boundary),
                        "best_native_ms":min((med(c,boundary) for c in native),default=None),"best_fused_ms":min((med(c,boundary) for c in fused),default=None)})
        groups.append(by_group)
    model_buffer=io.StringIO(newline="");writer=csv.writer(model_buffer,lineterminator="\n")
    writer.writerow(["model_seed","batch","label","boundary","median_ms","p10_ms","p90_ms","validation_correct","validation_count","test_correct","test_count","test_logit_relative_error","weight_bytes","owned_native_device_buffer_bytes","extra_peak_torch_allocated_bytes","setup_and_tuning_wall_ms"])
    for record in application:
        quality={q["label"]:q for q in record["quality"]}
        for c in record["cases"]:
            q=quality[c["label"]]["splits"]
            for boundary,values in c["timings_ms"].items():
                ordered=sorted(values);observations+=len(values)
                writer.writerow([record["model_seed"],c["batch"],c["label"],boundary,f"{statistics.median(values):.9f}",f"{ordered[2]:.9f}",f"{ordered[26]:.9f}",q["validation"]["correct"],q["validation"]["count"],q["test"]["correct"],q["test"]["count"],q["test"]["logit_relative_error_vs_original"],c["weight_bytes"],c.get("owned_native_device_buffer_bytes"),c["extra_peak_torch_allocated_bytes"],c["setup_and_tuning_wall_ms"]])
    cases=sum(len(r["cases"]) for r in core+sage);model_cases=sum(len(r["cases"]) for r in application);missing=sum(len(r["unavailable"]) for r in core+sage+application)
    lines=["# Mixed-precision attention on RTX 5060 Ti","",
           "**Status: study 5 is complete within this declared inference protocol. See the [completed programme ledger](PROGRAM.md).**","",
           f"Two independent input/model cohorts retain {cases:,} attention configurations, {model_cases} complete transformer configurations, {observations:,} timing observations and {missing} rejected/unsupported attempts. "
           "CPU replay separately checks native arithmetic, fused reference outputs, frozen model inference, complete quality arrays and changing-image requests. No training-speed or LLM-perplexity claim is made.","",
           "## Native implementation and strong controls","",
           "The native CUDA C++ plan materializes QK, masked FP32 softmax and PV. QK and PV independently use FP4, FP8 or FP16 in a 3x3 grid, with BF16/BF16 and pedantic FP32/FP32 controls. "
           "Heads are batched in CUTLASS SM120 NVF4 or cuBLASLt strided-batch GEMMs. A separate cooperative encoder writes each head's interleaved FP4 scale layout. "
           "Q/K rows and transposed V are padded to multiples of 128; every pad, transpose, encoding, dynamic probability encoding and output crop is charged in the online boundary. Softmax divides by the original head dimension's square root.","",
           "The [cuBLAS layout documentation](https://docs.nvidia.com/cuda/archive/13.3.1/cublas/index.html#cublasltmatrixlayoutattribute-t) describes the strided-batch attributes used by this plan. "
           "Each native GEMM stage is separately tuned among up to eight available library heuristics outside operation timings. [Emitted-code evidence](results/attention-native-inspection.json) from the executed binary contains E2M1 Tensor Core MMA and TMA instructions. "
           "Those static instructions/resources do not establish achieved utilization or an architecture ceiling.","",
           "Strong controls explicitly force PyTorch math, FlashAttention, efficient attention and cuDNN SDPA for FP32/FP16/BF16 where supported; unavailable cases are retained. "
           "The initial 144-case probe retains actual dispatch-operation names. FlashAttention in this environment rejects arbitrary boolean padding masks, so efficient/cuDNN controls cover that mask. The wrappers do not silently label a fallback as a requested backend.","",
           "The [official SageAttention3 implementation](https://github.com/thu-ml/SageAttention/tree/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention3_blackwell) supplies an additional specialized fused FP4 control, built unmodified against pinned CUTLASS in a project-local target. "
           "Its upstream smoothing/quantization, FP16/BF16 input casts and FP32 output promotion are included. The wrapper rejects arbitrary masks and head dimensions outside 64/128, rather than allowing ignored masks or an undocumented fallback. "
           "The upstream preprocessor centers K in place: prepared-input timings clone K on every call, preventing benchmark repetitions from silently changing inputs. "
           "Its own compiler flags include fast math; the materialized native build does not enable it. Build/source/wheel/module hashes are retained in [the build manifest](results/sage-build.json).", "",
           "## Grid and timing boundaries","",
           "Shapes (B,H,L,D) are (1,1,128,64), (2,4,512,64), (1,4,1024,128), (1,2,256,32), (1,2,193,48) and (1,1,768,96). "
           "The final two shapes are held out from the earlier regular grid; no selector is fitted in this study. Inputs are portable normal-like integer CLT values or peaky Q/K scaled by four. "
           "Masks are none, causal, or a boolean mask excluding the last key quarter and completely masking query row zero. Padded rows/keys and completely masked outputs must be exactly zero.","",
           "Prepared timings reuse encoded native Q/K/V or framework input casts; dynamic softmax and probability preparation remain inside every operation. "
           "Online timings start from the same resident FP32 Q/K/V and include all conversion/preparation. Complete host requests include all three inputs' H2D transfers, computation and full FP32 output D2H. "
           "Plans and masks are reused. Five warmups and thirty observations use four captured repetitions for smaller products and one when B*H*L>=4096; every method of a shape uses the same count. "
           "The second cohort reverses groups/configurations. One-time setup/tuning costs are separately retained. Host and repeated-graph numbers answer different questions.","",
           "## Representative normal-input attention costs","",
           "Ranges span both cohorts for noncausal L=1024, H=4, D=128. Lower online latency favors a method; errors are shown beside it because these are not equal-accuracy speed ratios.","",
           "| Method | Online microseconds | Full relative error |",
           "|---|---:|---:|"]
    for label in ("native-fp4-fp4","native-fp4-fp16","native-fp8-fp8","native-fp8-fp16","native-fp16-fp16","native-fp32-fp32","flash-fp16","cudnn-fp16","efficient-fp32","sage3-fp16","sage3-bf16"):
        rows=[g[((1,4,1024,128),"none","normal")][label] for g in groups];times=[med(c,"online")*1000 for c in rows];errors=[c["checks"]["relative_frobenius_error"] for c in rows]
        lines.append(f"| {label} | {min(times):.4g}-{max(times):.4g} | {min(errors):.4g}-{max(errors):.4g} |")
    lines += ["","The materialized FP8 path uses tensor scales of one. Softmax probabilities near zero can therefore incur large FP8 underflow/rounding errors. "
              "These outcomes characterize this encoder and scaling choice, not an inherent accuracy limit of FP8 or an optimized fused FP8 attention implementation. FP4 uses dynamic 16-value block scales, with the previously specified scale floor; SageAttention3 uses its own smoothing/scaling. "
              "Probability rescaling and a fused materialized-path replacement are untested optimization directions. The fused reference is necessary to avoid presenting a fast GEMM as a complete attention win.","",
              "## Fixed observed error","",
              "Eligibility requires both full Frobenius error and sampled error against original FP64 attention to meet the budget. The following online comparison covers 72 input/mask/shape/cohort groups. "
              "[All offline frontiers](results/attention-frontiers.json) also retain prepared and complete host boundaries, missing eligibility and exact selected methods; these are not online guarantees.","",
              "| Error budget | Groups with native and fused eligible | Native faster | Fused faster | No method eligible |",
              "|---|---:|---:|---:|---:|"]
    for budget in BUDGETS:
        rows=[r for r in frontiers["choices"] if r["boundary"]=="online" and r["maximum_relative_error"]==budget];both=[r for r in rows if r["best_native_ms"] is not None and r["best_fused_ms"] is not None]
        lines.append(f"| {budget:g} | {len(both)} | {sum(r['best_native_ms']<r['best_fused_ms'] for r in both)} | {sum(r['best_native_ms']>r['best_fused_ms'] for r in both)} | {sum(r['selected'] is None for r in rows)} |")
    lines += ["","## Complete real-model quality and costs","",
              "Two independently initialized/trained pixel-token transformers use the pinned real digits dataset, stratified train/validation/test splits, one attention block, FP32 projections/norms/MLP and a ten-class head. "
              "Each model has 64 tokens, two heads and head dimension 64. Training runs sixty epochs; the highest validation accuracy, then lowest validation cross-entropy, selects a frozen checkpoint. Test labels do not choose it. "
              "Independent NumPy FP64 full inference reproduces all 1,446 frozen validation/test reference predictions. This is a bounded image-classification workload, not evidence for a language/video model.","",
              "Only the attention implementation changes during complete inference. All available methods run complete validation and test sets. Quality is evaluated at batch 16; latency/memory is separately measured at batches 4/16/32. "
              "Changing-image host requests cycle four distinct image batches and return labels. Native padding, Q/K/V copies, encodings and dynamic attention remain inside the complete model. "
              "The raw numerical witnesses independently replay dense model arithmetic around the actual checked attention output, avoiding false CPU/GPU encoder disagreements from recomputing slightly different Q/K/V inputs.","",
              "| Model | Method | Test correct / count | Batch-16 captured microseconds | Batch-16 changing-request microseconds |",
              "|---|---|---:|---:|---:|"]
    for record in application:
        for label in ("native-fp4-fp4","native-fp4-fp16","native-fp8-fp16","native-fp16-fp16","math-fp32","flash-fp16","cudnn-fp16","sage3-fp16","sage3-bf16"):
            q=next(q for q in record["quality"] if q["label"]==label)["splits"]["test"];c=next(c for c in record["cases"] if c["label"]==label and c["batch"]==16)
            lines.append(f"| {record['model_seed']} | {label} | {q['correct']} / {q['count']} | {statistics.median(c['timings_ms']['captured_device'])*1000:.4f} | {statistics.median(c['timings_ms']['host_request'])*1000:.4f} |")
    lines += ["","A few changed predictions can raise or lower these small test counts; no statistically established quality improvement is claimed. "
              "A block-level attention speed advantage does not imply faster complete transformer inference. Captured execution, uncaptured resident latency and changing-request latency remain separate in [the complete model table](results/attention-model-summary.csv). "
              "All methods' validation/test logits, predictions, changed-label counts and arithmetic witnesses are retained, including unfavorable outcomes.","",
              "## Memory, verification and limits","",
              "Native `owned_native_device_buffer_bytes` counts explicit padding, materialized scores/probabilities, encoded operands/scales and supplied workspaces. It excludes external inputs/output tensors and opaque library internal allocations. "
              "Framework `extra_peak_torch_allocated_bytes` is the incremental Torch allocator peak during the measured function, not whole-device/process peak. Model weight bytes are separate. "
              "These metrics have different accounting boundaries and cannot be directly treated as equivalent peak-VRAM measurements. [The full attention table](results/attention-summary.csv) retains both explicitly named metrics.","",
              "Native validation checks every padded/transposed input, encoded operand/scale byte, QK element, FP32 softmax probability, PV element and output crop against independent CPU arithmetic. "
              "After timing and host requests, complete buffer/encoding snapshots must match the independently validated initial result; unchanged bytes reuse that reference. "
              "The last actual captured output is retained and compared, rather than validating only a fresh post-timing call. Small witnesses replay full arrays; large cases retain 64 actual output samples. "
              "The specialized upstream FP4 control retains observed full/sample errors and stable repeated outputs; this does not prove every internal upstream rounding operation.","",
              "An initial partial campaign exposed a shared-memory softmax race: different warps could overwrite the maximum before every warp read it. "
              "The failed run, sources, binary and witnesses were archived privately. An added barrier fixes the race; both published full cohorts execute the final source and rerun the complete grid. "
              "Six invalid/null configuration controls and the independent full model checks remain in the protocol.","",
              "Measurements run on the shared RTX 5060 Ti 16 GB without changing clocks, power settings, services or other GPU contexts. "
              "This study does not establish optimal fused kernels, all sequence/head/mask combinations, training convergence under low precision, general model-quality preservation, scientific novelty or achieved hardware utilization. "
              "Studies 7-15 still require their own experiments and evidence.","",
              "## Reproduction","",
              "Build the native bridge with `bash build_attention.sh /path/to/pinned/cutlass`. Build the pinned fused control with `build_sage_reference.py --sage /path/to/pinned/SageAttention --cutlass /path/to/pinned/cutlass --python /path/to/torch/python`. "
              "The tested environment is recorded Torch 2.13/CUDA-13.0 with Toolkit 13.3 for native extensions; [dependency versions](results/attention-dependencies.json) additionally retain Triton 3.7.1 and cuDNN runtime 9.20. CPU replay needs only Python and pinned NumPy. Train both `prepare_attention_models.py` seeds 20261123/20261223, preserving their datasets, splits, histories and weights.","",
              "Run `attention_study.py` and `sage_attention_study.py` with seeds 20261123 and 20261223, separate outputs and `--reverse` for B. Run `attention_application.py` with the corresponding model seeds and the same order control. "
              "CPU checks are `verify_attention_models.py`, `verify_attention.py`, `verify_sage_attention.py`, `verify_attention_application.py` and `summarize_attention.py --check`. Preserve published records before any new run.",""]
    return {"ATTENTION_REPORT.md":"\n".join(lines),"results/attention-summary.csv":core_buffer.getvalue(),"results/attention-model-summary.csv":model_buffer.getvalue(),"results/attention-frontiers.json":json.dumps(frontiers,indent=2,allow_nan=False)+"\n"}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--check",action="store_true");args=parser.parse_args()
    for name,content in outputs().items():
        path=ROOT/name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8")!=content:raise ValueError(f"Stale attention artifact: {name}")
        else:path.write_text(content,encoding="utf-8",newline="\n")
    print("Attention artifacts match" if args.check else "Wrote attention report, all cost/quality/memory tables and offline frontiers")


if __name__=="__main__":main()
