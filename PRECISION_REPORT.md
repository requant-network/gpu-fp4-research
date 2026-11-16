# High-precision products from INT8 components

**Status: study 6's bounded precision/reconstruction comparison is complete. The full fifteen-study program remains active.**

Two independent input/model cohorts contain 1,008 configurations, 120,960 retained device/host timing observations and 645,005,312 output entries checked at three checkpoints. Eight full-product witnesses and two changing-input transition witnesses support CPU replay. Each case additionally retains 64 exact-integer product samples and exact INT8 product samples where components are used, before timing, after timing and after host requests.

## Problem and methods

All operands start as finite FP32 and every method returns FP64 output. The question is accurate products of FP32 inputs, not universal emulation of every FP64 input or IEEE-754 special-value behavior. Native references use pedantic cuBLASLt FP32/FP64; FP32 output is promoted to FP64 as part of its measured operation. The library comparisons request performant/eager dynamic FP64 emulation, fixed 31/55-bit emulation and BF16x9 FP32 emulation. Each primary/fallback GEMM is tuned among up to eight cuBLASLt heuristics outside reported operation times.

The custom method normalizes each operand row by a power of two. Consecutive signed base-128 digits are extracted by truncation toward zero, with one INT8 matrix per 7-bit component. Fixed methods use 1, 2, 3, 4, 6 or 8 components. Every pair of digit matrices is multiplied with INT32 accumulation; selected algorithms must report IMMA implementation flags. Products on the same component-index diagonal are summed in INT64, then scaled and accumulated into FP64 from less significant diagonals toward more significant ones.

For row exponents e_A/e_B and digits D, the result is the sum over s of 2^(e_A+e_B-7*(s+2)) times the integer diagonal sum for i+j=s. For K<=8192 and digits bounded by 127, each INT32 dot is bounded by 8192*127^2; even eight such dots fit signed INT32, while the implementation conservatively uses INT64 diagonal storage. This integer path avoids FP32 rounding inside partial products. Finite-component truncation and final FP64 reconstruction still require separate error checks.

The adaptive method examines each row's maximum exponent and least significant nonzero input bit. It selects enough components to represent every input exactly when at most eight suffice; otherwise it uses native FP64. This is an input-representation decision, not a guarantee that the final sum is correctly rounded. A changed component count/fallback decision re-encodes B when its cached representation is insufficient. Changed-input witnesses force normal, cancellation, wide-range and normal decisions rather than validating only a static choice.

## Actual library execution

The [archived NVIDIA cuBLAS documentation](https://docs.nvidia.com/cuda/archive/13.3.1/cublas/index.html#floating-point-emulation) lists BF16x9 for compute capabilities 10.0/10.3 and FP64 fixed-point emulation for 12.x. Requested API modes alone do not prove an emulated execution. The library's device mantissa-bit output is retained: positive values indicate emulation, -1 indicates fallback. Strategy, mantissa control, configured bit cap and special-value mask are queried and recorded. Finite inputs allow special-value support to be disabled explicitly. Defaults are library/device dependent; a configured zero cap is not interpreted as a fixed universal bit limit.

Among 288 FP64-emulation cases, 232 report positive retained bits and 56 report fallback at the initial checkpoint. Among 72 accepted BF16x9 requests, 0 selected algorithms report emulation capability; this request must not be mislabeled as BF16x9 Tensor Core acceleration when flags indicate native FP32. Per-checkpoint retained bits and numerical implementation flags remain in the raw records.

Each plan supplies 512 MiB of workspace. This is the supplied buffer, not a measured peak-memory bound; the library can require additional internal allocations. Native conversion, component analysis/extraction, recombination and internal library preparation are included wherever the operation boundary requires them.

## Normal-profile cost comparison

Ranges span cohorts. Ratios above 1 favor the named alternative over native FP64. All ratios include preparation of A; B stays prepared. The adaptive exact-sample column is separately evaluated against correctly rounded integer-gold products.

| M x N x K | Native FP64 online A, microseconds | FP64 / performant emulation | FP64 / fixed55 emulation | FP64 / adaptive | Adaptive exact-sample relative error |
|---|---:|---:|---:|---:|---:|
| 16 x 512 x 512 | 33.58-33.96 | 0.9952-1.007 | 0.1465-0.1477 | 0.625-0.6331 | 0-0 |
| 32 x 768 x 1280 | 174.9-175.6 | 0.9998-1 | 0.2182-0.2184 | 1.384-1.386 | 0-0 |
| 128 x 128 x 128 | 28.94-29.13 | 1-1 | 0.239-0.2395 | 0.5027-0.5036 | 0-0 |
| 128 x 128 x 8192 | 775.1-778.3 | 0.9999-1.001 | 0.3035-0.3036 | 2.704-2.735 | 0-0 |
| 256 x 512 x 512 | 428.1-430 | 0.9999-1 | 1.781-1.782 | 2.967-2.972 | 0-0 |
| 384 x 768 x 256 | 429.3-431 | 0.9999-0.9999 | 1.446-1.448 | 1.957-1.972 | 0-0 |
| 1024 x 1024 x 1024 | 6178-6178 | 6.342-6.353 | 8.243-8.247 | 6.982-6.99 | 0-0 |
| 2048 x 2048 x 2048 | 5.088e+04-5.11e+04 | 8.079-8.108 | 10.47-10.47 | 6.725-6.747 | 0-0 |

## Cancellation, range and subnormal controls

The normal generator uses portable integer mantissas with rich FP32 low bits. Cancellation pairs have an analytic rank-one residual, so their complete target product is known exactly. Wide-range inputs use complementary column exponents from -60 to 60: true pairwise products remain moderate while row-wise representations must span a large range. Subnormal A inputs are scaled by 2^-140; B remains normal. Long dots include K=8192. Zero rows remain in every synthetic case.

Representative 128-cubed exact-sample errors below include every cohort rather than selecting the more favorable input seed.

| Seed | Profile | FP32 error | FP64 error | Slices4 error | Slices8 error | Adaptive components (0=fallback) | Adaptive error |
|---|---|---:|---:|---:|---:|---:|---:|
| 20261121 | normal | 1.6832e-07 | 0 | 0 | 0 | 4 | 0 |
| 20261121 | cancellation | 260.98 | 0 | 1 | 0 | 8 | 0 |
| 20261121 | spread | 1.6072e-07 | 0 | 1 | 1 | 0 | 0 |
| 20261121 | subnormal | 0.0017519 | 0 | 0 | 0 | 4 | 0 |
| 20261221 | normal | 1.3231e-07 | 0 | 0 | 0 | 4 | 0 |
| 20261221 | cancellation | 171.01 | 0 | 1 | 0 | 8 | 0 |
| 20261221 | spread | 1.4696e-07 | 0 | 1 | 1 | 0 | 0 |
| 20261221 | subnormal | 0.0013077 | 0 | 0 | 0 | 4 | 0 |

## Real activations and precision/cost frontiers

Two independently trained digits classifiers supply frozen input/hidden activations and learned weights. These are real layer-product tests, not a claim of complete model-quality improvement from higher precision. Batch 64 below compares exact-sample errors and complete host-request costs, which include fresh selection and transfers.

| Model | Layer | Adaptive components | FP32 exact-sample error | FP64 exact-sample error | Adaptive exact-sample error | Native FP64 / adaptive host-request time |
|---|---|---:|---:|---:|---:|---:|
| 20261116 | digits-input | 6 | 9.1458e-08 | 0 | 0 | 0.2374 |
| 20261116 | digits-hidden | 6 | 3.4203e-08 | 1.6723e-16 | 0 | 0.3066 |
| 20261216 | digits-input | 6 | 8.5118e-08 | 0 | 0 | 0.2333 |
| 20261216 | digits-hidden | 6 | 6.989e-08 | 2.196e-16 | 0 | 0.2992 |

Every real case, retained error, component decision, supplied workspace, one-time setup and initial preparation cost is in [the complete CSV](results/precision-summary.csv).

[Offline precision/cost frontiers](results/precision-frontiers.json) compare error budgets 0.001, 0.000001, 1e-10 and 1e-12 across prepacked, online-A, online-both and host-request boundaries. Eligibility requires both the full reference-relative error and exact-sample error to meet the budget; infeasible outcomes are retained. These are empirical outcomes on measured inputs. The separate [held-out per-operation predictor](SELECTOR_REPORT.md) retains its own accuracy failures and selection costs.

## Independent validation and timing limits

For exact witnesses, each finite FP32 input is represented as an integer times 2^-149. Arbitrary-precision integer dot products are converted once to FP64 and scaled by 2^-298. Thus cancellation and FP64-reference rounding do not silently invalidate the small-case/error samples. Every prepared digit byte, exponent and component requirement is independently reconstructed. Full-size arithmetic is checked against CPU FP64 products on the exact prepared values with a conservative accumulation bound. The fixed-library quantization allowance is an experimental screening tolerance, not a proof of the library's opaque internal quantizer. Normal/real full FP64-reference errors near its own rounding floor are not evidence of superiority over exact arithmetic. Exact retained samples provide the stronger local check.

Captured timings repeat identical resident inputs. The adaptive host decision is cached inside these graphs; online graphs charge analysis and encoding, but not the decision readback/synchronization. Host-request timings include H2D A, fresh analysis/selection, any required B re-encoding, computation, full FP64 D2H output and synchronization. Those thirty requests repeat the same A; the separate changed-input witnesses test decision transitions. B and native plans are reused. Setup/initial preparation are separately retained once per case, not claimed as cold-process timing distributions.

Five graph warmups and thirty observations use eight repetitions for smaller products and one for large/long products, equally across methods of the same shape. The second cohort reverses group/configuration order. Transfers, plan creation, tuning, graph construction and validation are excluded from device timers. Graph averages and host-request latencies answer different questions; they are not interchangeable speedup claims.

The server retains other GPU contexts and unchanged clock/power/service settings. All results are exploratory shared-system observations. The protocol does not establish general FP64-input emulation, signed-zero/nonfinite IEEE behavior, scientific novelty, an architecture ceiling, or optimal component fusion. Separate INT8 GEMMs and diagonal/reconstruction kernels retain launch and memory costs; poor cases remain part of the findings.

Reproduce with `bash build_precision.sh`, then `precision_study.py` using seeds 20261121/model 20261116 and 20261221/model 20261216, separate outputs and `--reverse` for the second. Run `python verify_precision.py` and `python summarize_precision.py --check` for CPU-only replay. A preliminary eager-only partial run was preserved privately; the published cohorts execute the final performant/eager protocol and current recorded source hashes.
