# Held-out per-operation format selection

**Status: the remaining error-prediction gate of study 1 has been experimentally evaluated. The empirical selector has a retained budget violation and is not a certified error-constrained dispatcher. The fifteen-study program remains active.**

The campaign retains 384 reference configurations, 224 complete selection decisions at four budgets, 52,800 timing observations and 96,632,832 reference-output entries checked at three checkpoints. Of these decisions, 192 involve unseen shapes, distributions, model data or the independent second cohort; 1 violate their requested budget. Ten full small-case witnesses and two changing-input witnesses permit independent CPU replay.

## Hypothesis and fixed protocol

Can a small native GPU error probe, calibrated on ordinary/outlier products, select a useful format for unseen operations after paying for its own work? The predictor is calibrated once from seed 20261122, four shapes and the normal/outlier distributions. The maximum training ratio of actual error to probe feature is multiplied by 1.25, separately for each format. No evaluation error or timing updates that policy. A separate calibration run finishes before either evaluation cohort starts; both evaluation records retain the identical policy and its hash.

The probe evaluates 64 deterministic sampled output coordinates per format using native CUDA FP64 dot reductions on original FP32 inputs, decoded quantized inputs and absolute decoded products. Its feature is the sample norm of quantization differences plus 2^-24*sqrt(K) times the absolute-product norm, divided by the original sampled product norm. This accumulation term is an empirical feature, not a proven error bound. Zero denominators are explicitly handled. CPU dot products independently check every retained probe.

Eligibility compares the calibrated estimate with the requested relative error. Eligible formats are ranked by prepacked latency at the nearest training shape in squared log-dimension distance. If none qualifies, execution uses native FP64. The policy does not inspect the current full product or its reference error before dispatch. Evaluation covers two unseen shapes, cancellation and complementary exponent spreads, four real-activation layer/batch groups, and an independent input/model cohort with reversed execution order.

Training shapes are 16x512x512, 128x128x128, 256x1024x1024 and 1024x1024x1024. Held-out shapes are 32x768x1280 and 384x768x256. The portable normal/outlier generator uses a finite integer CLT grid; it does not cover every FP32 mantissa pattern. Cancellation uses rich-mantissa paired inputs with an exact rank-one residual. Spread inputs use complementary column exponents -12, -6, 0, 6 and 12; zero rows are retained. Real inputs/weights come from the two frozen digits models.

## Budget failures are findings

Budget acceptance uses the maximum of the full reference-relative Frobenius error and the retained exact-sample error. The former uses CPU FP64, or the analytic cancellation product; the latter uses arbitrary-precision integer dot products. A complete low-error guarantee does not follow from 64 samples. The predeclared policy is kept unchanged after seeing its failures.

| Seed | M x N x K | Profile | Budget | Selected | Prediction | Full error | Exact-sample error |
|---|---|---|---:|---|---:|---:|---:|
| 20261222 | 1024 x 1024 x 1024 | outliers | 0.1 | fp4 | 0.0908831 | 0.112613 | 0.0701206 |

The retained full error can exceed the budget even while the sampled error and calibrated prediction remain below it. This rejects the hypothesis that the tested calibration safely generalizes to every operation in the held-out protocol. A runtime accuracy guarantee remains unproven; this result does not recommend this predictor for such a guarantee.

## Complete costs on held-out decisions

Each request includes H2D A, preparation of A in all five formats, all five native probes, their D2H transfers/synchronization, the host decision, the selected GEMM and full FP64 output D2H. FP64 fallback additionally converts A to FP64. B representations, plans and heuristic choices are reused. Every format returns FP64 output; native promotion of FP32 candidate output is timed. Direct references charge their own A conversion, GEMM/promotion and the same input/output transfers. The offline feasible best is known only retrospectively and pays no selection overhead.

Ratios below include only decisions that actually meet their budget; the unsafe admission above is excluded from equal-error comparisons. Regret is selector latency divided by the fastest measured direct reference meeting that input's error budget. FP64/selector greater than 1 favors the selector.

| Budget | Held-out decisions | Budget failures | Feasible median regret | Feasible regret range | Feasible median FP64 / selector |
|---|---:|---:|---:|---:|---:|
| 0.1 | 48 | 1 | 3.0485 | 1.0348-6.1031 | 0.8220 |
| 0.01 | 48 | 0 | 2.8717 | 1.0348-6.0276 | 0.8273 |
| 0.001 | 48 | 0 | 2.8349 | 1.0345-6.0272 | 0.8239 |
| 1e-05 | 48 | 0 | 2.7637 | 1.0347-6.0198 | 0.8075 |

Held-out choices: bf16: 16, fp16: 94, fp32: 10, fp4: 1, fp64: 48, fp8: 23. The full probe/preparation/decision cost is often substantial even when selected GEMM execution is fast. Cases faster than native FP64 do not establish an advantage over the best feasible low-precision reference.

[All reference boundaries and setup costs](results/selector-references.csv) and [every decision, error and complete cost](results/selector-decisions.csv) retain the unfavorable cases. One-time plan allocation and initial preparation are separate from reused-plan requests; algorithm tuning, graph construction and numerical validation are excluded from request/device timers. Captured direct references use five warmups and thirty observations, eight repetitions for smaller products and one for volume >=2^29. Thirty complete host requests per case repeat A. Separate witnesses change A four times with fixed B and check all four budgets, fresh encoding/probes and complete outputs; these are correctness transitions, not latency distributions for changing arrivals.

## Independent checks and interpretation

Every quantized byte/scale and reference output is checked before timing, after device timing and after direct requests. The final output for each selection/budget is checked in full. Small witnesses replay full arithmetic; large cases retain 64 actual output values at each check, so exact-integer sample errors can be recomputed without trusting a summary hash. The native probe also has four invalid/null/count controls. All executed source/data hashes, binary identities, environment, calibration and raw observations remain in the records.

The implementation uses the cooperative FP4 encoder and pinned CUTLASS/cuBLASLt native paths; Python performs orchestration and the timed host decision. It does not establish optimal selection overhead, scientific novelty or an architecture ceiling. Other GPU contexts remain present; no clock/power/service settings were changed. The empirical failure and overhead finish this bounded hypothesis test without establishing a deployable guaranteed-error policy.

Together with the [grid/encoder work](ENCODER_REPORT.md), [complete application/strong references](APPLICATION_REPORT.md), [native architecture work](HARDWARE_REPORT.md), [outlier corrections](OUTLIER_REPORT.md) and [held-out pipeline comparisons](PIPELINE_REPORT.md), this supplies the remaining study-1 gate. Studies 5 and 7-15 still require their own experiments and evidence.

## Reproduction

Build with `bash build_selector.sh /path/to/pinned/cutlass` plus the existing extended/precision builds. Run `selector_study.py --phase calibration --seed 20261122 --model-seed 20261116 --output results/selector-calibration.json` first. Use its `results/selector-policy.json` with `--phase evaluation --policy results/selector-policy.json`, the same seed/model for cohort A and seed 20261222/model 20261216 with `--reverse` for B. Keep separate output paths and preserve existing observations. CPU-only checks are `python verify_selector.py` and `python summarize_selector.py --check`.
