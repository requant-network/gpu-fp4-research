# FP4 outlier correction: accuracy and complete cost

**Status: study 2's bounded column/block/residual comparison is complete. The fifteen-study program remains active.**

Two independent input/model cohorts contain 598 configurations, 89,700 retained observations and 421,564,416 output entries checked before and after timing. The grid includes seven synthetic shapes, four distributions and two real-model layers at three batches. Twelve complete small numerical witnesses permit CPU replay. All selected columns, block masks and FP16 factors were checked against independent CPU construction.

## Algorithms and correctness contract

The common base is block-scaled FP4 GEMM with native packed conversion, the same fixed tensor scale of one and the same 128x128x128 CUTLASS tile as earlier studies. Comparators are native cuBLASLt FP8, FP16 and strict FP32; each uses up to eight heuristic candidates, tuned outside the reported operation times. Every correction adds two FP16 Tensor Core products with FP32 accumulation and a final CUDA summation kernel.

Let Q_A and Q_B denote decoded FP4 operands, and H denote FP16 rounding. Matrices use the repository's C = A B^T convention.

- **Columns:** GPU reductions compute max(abs(A[:,j])) times max(abs(B[:,j])). A GPU bitonic sort selects 128 or 256 columns (capped at K), breaking ties by ascending index. The result is Q_A Q_B^T + H(A_S) H(B_S)^T - H(Q_AS) H(Q_BS)^T. Correction GEMM depth is the selected column budget.
- **Blocks:** each operand selects contiguous 16-value blocks whose maximum magnitude exceeds 0.25 or 0.75 of that operand's global maximum. R_A/R_B contain selected FP32 residuals A-Q_A and B-Q_B, zero elsewhere. The result is Q_A Q_B^T + H(R_A) H(Q_B+R_B)^T + H(Q_A) H(R_B)^T. Selected B blocks are copied from B before FP16 rounding to avoid an extra residual re-addition.
- **Full residual:** the same residual formula selects every block. It tests whether finer component reconstruction is useful despite two additional full-depth GEMMs.

Block/residual correction uses dense FP16 factors with zeros in unselected positions. It does not compact arbitrary blocks or claim sparse Tensor Core speedups. Column correction does reduce GEMM depth. Neither threshold/column score nor budget is optimized separately for a measured test case. Arithmetic is checked against FP64 products of the exact prepared factors with a conservative FP32 accumulation/summation bound; error is separately measured against the original FP32 inputs in FP64.

## Synthetic outlier results

Ranges span cohorts for the element-outlier profile. Error is relative Frobenius product error. Latency ratios above 1 favor correction over direct FP16 and include online-A detection, selection, gathering and preparation. Both references start from common FP32 source inputs; B and its required preparation remain resident.

| M x N x K | Plain FP4 error | Columns128 error | Full residual error | FP16 / columns128 online A | FP16 / residual online A |
|---|---:|---:|---:|---:|---:|
| 16 x 512 x 512 | 0.107-0.119 | 0.0608-0.0746 | 4.02e-08-5.44e-08 | 0.101-0.101 | 0.188-0.191 |
| 32 x 768 x 1280 | 0.112-0.113 | 0.0847-0.0849 | 6.43e-08-8.23e-08 | 0.068-0.0687 | 0.251-0.254 |
| 128 x 128 x 128 | 0.1-0.106 | 7.3e-08-1.66e-07 | 3.68e-08-4.51e-08 | 0.123-0.123 | 0.167-0.168 |
| 192 x 768 x 768 | 0.112-0.113 | 0.0928-0.0929 | 6.19e-08-6.27e-08 | 0.152-0.153 | 0.253-0.254 |
| 256 x 1024 x 1024 | 0.112-0.113 | 0.098-0.099 | 6.76e-08-7.57e-08 | 0.257-0.257 | 0.282-0.283 |
| 768 x 1536 x 1024 | 0.113-0.113 | 0.102-0.102 | 7.24e-08-7.45e-08 | 0.522-0.523 | 0.303-0.304 |
| 2048 x 2048 x 2048 | 0.112-0.113 | 0.108-0.108 | 1.22e-07-1.27e-07 | 0.751-0.753 | 0.322-0.322 |

Synthetic inputs use the existing integer CLT generator and power-of-two outlier multipliers. Many values are exactly representable in FP16, so direct FP16 can have unusually small input-rounding error on these distributions. Real learned weights/activations below add general FP32 values. Synthetic and real-data conclusions must not be conflated.

## Real activations

Two independently trained UCI digits classifiers supply validation inputs and hidden ReLU activations, frozen from FP64 layer evaluation then rounded once to FP32. This isolates layer products on real data; these error measurements are not complete classifier accuracy or evidence that approximate first-layer activations preserve downstream quality. Batch 64 results retain each model separately.

| Model seed | Layer | Plain FP4 error | FP8 error | FP16 error | Columns128 error | Blocks0.25 error | Residual error |
|---|---|---:|---:|---:|---:|---:|---:|
| 20261116 | digits-input | 0.114 | 0.0237 | 0.0001822 | 0.0001822 | 0.007326 | 2.412e-05 |
| 20261116 | digits-hidden | 0.04489 | 0.01134 | 8.938e-05 | 0.02399 | 0.007084 | 1.15e-05 |
| 20261216 | digits-input | 0.1139 | 0.02299 | 0.0001779 | 0.0001779 | 0.008453 | 2.456e-05 |
| 20261216 | digits-hidden | 0.05078 | 0.01298 | 0.0001075 | 0.02199 | 0.01188 | 1.346e-05 |

## Fixed error and fixed time

[All constrained outcomes](results/outlier-constrained.json) retain eight error budgets (0.000001 through 0.2) and five time budgets (0.5 through 8 times direct FP16), for prepacked, online-A and online-both operation boundaries. Infeasible budgets explicitly retain a null choice. These are offline empirical frontiers using measured error and median latency from the same case; they are not a learned selector or an unseen-input error guarantee.

At relative-error budget 0.001, correction is the fastest eligible online-A candidate in 0 of 68 cohort/shape/profile groups. The complete record retains whichever uncorrected format wins as well. Preparation-only improvements are not substituted for complete operation costs.

All online-A error budgets are shown below, counting the fastest eligible measured implementation per group. Correction wins at some stricter budgets even though it does not win at 0.001.

| Maximum relative error | FP4 wins | FP8 wins | FP16 wins | FP32 wins | Correction wins | Infeasible |
|---|---:|---:|---:|---:|---:|---:|
| 1e-06 | 0 | 0 | 38 | 24 | 6 | 0 |
| 1e-05 | 0 | 0 | 56 | 12 | 0 | 0 |
| 0.0001 | 0 | 0 | 59 | 9 | 0 | 0 |
| 0.001 | 0 | 0 | 68 | 0 | 0 | 0 |
| 0.01 | 0 | 0 | 68 | 0 | 0 | 0 |
| 0.05 | 4 | 64 | 0 | 0 | 0 | 0 |
| 0.1 | 4 | 64 | 0 | 0 | 0 | 0 |
| 0.2 | 16 | 52 | 0 | 0 | 0 | 0 |

## Cost boundaries, reproducibility and limits

Five CUDA-event boundaries measure prepare A, prepare both, prepacked GEMM including corrections, online A and online both. Online-A column selection reuses cached B column maxima but must reselect columns and gather both operands; block/residual modes reuse B factors. Detection, sorting, masks, gathering, correction GEMMs and final addition are included where required. Five warmups and thirty observations use 32 repetitions per graph; observation order rotates and the second cohort reverses group/configuration order. Nearest-rank p10/p90 and every candidate are in [the complete CSV](results/outlier-summary.csv).

Fresh native plan allocation/setup is separately recorded once per configuration, including both correction plans and their buffers; it is descriptive, not a sampled cold-process distribution. Source H2D transfer, Python dispatch, tuning, graph construction, correctness checks and teardown are excluded from device-event times. Prepacked/online-A/online-both expose preparation reuse; results do not establish changing-input host-request latency or energy. Correction owns four FP16 factors, two full FP32 output temporaries, reduction/selection/mask buffers and two cuBLASLt workspaces; it is not a memory-saving scheme.

Other GPU contexts remain present. Measurements are exploratory shared-system observations, not exclusive performance guarantees. The implementation supports the tested aligned shapes, finite inputs and FP16-representable correction factors, K up to 2048; it does not establish the best possible correction algorithm or scientific novelty. Sparse block compaction, alternative outlier scores, optimized tensor scales and complete corrected-model deployment remain separate potential improvements. This bounded comparison supplies study 2's evidence; subsequent pipeline/resource/configuration results are retained in [the pipeline report](PIPELINE_REPORT.md).

Reproduce after earlier extended/hardware builds with `bash build_outlier.sh ../cutlass-fp4`. Frozen real activations are prepared by `prepare_outlier_activations.py`; run `outlier_study.py` with seeds 20261119/model 20261116 and 20261219/model 20261216, separate output paths and `--reverse` for the latter. Run `python verify_outliers.py` and `python summarize_outliers.py --check` without GPU dependencies to replay the public artifacts.
