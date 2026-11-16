# Cooperative FP4 preparation, fusion and tile study

**Status: this additive encoder/fusion/tile experiment is complete. Program studies 1-4 still have additional requirements.**

Two independent cohorts cover 592 configurations, eleven shapes, four input profiles (two at the largest shape), five timing boundaries and 88,800 retained observations. Encoded values and scales are byte-identical to the independent CPU encoder. Every output is checked before and after timing. Eight small input/output witness files support CPU replay.

The original scalar encoder, four cooperative implementations and M=128/256 CUTLASS tiles are retained, including regressions. Bias/ReLU fusion compares the same activation and quantization contract. These are warm, captured, device-resident shared-system observations.

## Normal-profile comparisons

Ratios are ranges across cohorts. Above 1 favors the candidate. The fixed pair8-256 encoder uses eight cooperating lanes per block of sixteen values and 256 threads per CUDA block; it was not selected separately for each test case.

| M x N x K | Scalar / pair8-256 prepare A | Separate / fused online A | Tile128 / tile256 prepacked GEMM |
|---|---:|---:|---:|
| 16 x 512 x 512 | 3.43-3.46 | 1.09-1.11 | 0.60-0.60 |
| 32 x 768 x 1280 | 3.34-3.49 | 1.08-1.08 | 0.57-0.57 |
| 64 x 1024 x 1024 | 2.39-2.41 | 1.11-1.12 | 0.58-0.58 |
| 128 x 128 x 128 | 4.22-4.58 | 1.12-1.12 | 0.64-0.64 |
| 192 x 768 x 768 | 2.03-2.08 | 1.15-1.15 | 0.59-0.59 |
| 256 x 1024 x 1024 | 1.38-1.41 | 1.15-1.15 | 0.58-0.59 |
| 384 x 1536 x 512 | 1.57-1.58 | 1.15-1.16 | 0.66-0.67 |
| 512 x 512 x 512 | 1.37-1.39 | 1.17-1.17 | 0.61-0.62 |
| 768 x 1536 x 1024 | 1.13-1.13 | 1.15-1.16 | 0.98-0.98 |
| 1024 x 1024 x 1024 | 1.10-1.11 | 1.17-1.17 | 0.98-0.98 |
| 2048 x 2048 x 2048 | 0.89-0.90 | 1.41-1.42 | 1.03-1.04 |

## Held-out encoder selection

A single-threshold rule was fitted on 26 groups from cohort 20261117, excluding all four held-out shapes. Choose `lane16-128` when M*K <= 16384; otherwise choose `pair8-128`. The objective uses only preparation-A latency and has no application-quality claim.

Across 32 held-out shape/profile/cohort evaluations, geometric-mean regret against the fastest measured encoder is 1.015x; geometric-mean speedup against the scalar encoder is 1.767x. Both the fitted rule and every held-out outcome are retained in [selector data](results/encoder-selector.json).

## Interpretation and remaining scope

Cooperative preparation is not uniformly faster. Small and large inputs can prefer different work assignment; extra per-lane scale/conversion work competes with improved memory access. Fusion removes a separate activation pass but must be evaluated as a complete operation. Tile choice also changes latency without changing the quantization contract.

This experiment does not finish the complete format selector, real-model quality/latency integration, compiled framework fusion baseline, or direct loading/pipeline/profiler comparisons required by [the program](PROGRAM.md). It establishes measured candidates and counterexamples for those follow-ups. The full [CSV](results/encoder-summary.csv) retains all variants, profiles, boundaries and nearest-rank p10/p90 values.

Run `python verify_followups.py` for source/input/numerical replay and `python summarize_followups.py --check` for derived artifacts.
