# Real-data classifier and compiled-framework comparison

**Status: this bounded application/reference experiment is complete. The fifteen-study program remains active.**

Two independently trained digits models compare eight native precision/backend configurations and eager/compiled PyTorch FP32, FP16, BF16 and FP8, plus a dynamic-shape reduce-overhead reference with internal CUDA Graphs. Each implementation covers six batch sizes, for 240 timing cases in total. Native FP4 uses both CUTLASS and cuBLASLt with identical block-scaled operands and FP32 output. The classifier includes both linear layers, bias/ReLU and final GPU argmax over ten classes.

The dataset, parameters, exhaustive disjoint splits, numerical witnesses, complete retained predictions, raw durations and executed-source hashes support CPU-only replay. Native encoded operands and every layer's arithmetic are checked at runtime; all validation/test predictions also match independent complete CPU simulation.

## Classification quality

Values are test accuracy percentages for the two independent models. Each model has 364 held-out test images; one image is approximately 0.275 percentage points. These small tests do not establish production model quality.

| Configuration | Model A test | Model B test |
|---|---:|---:|
| Native bf16/bf16 cuBLASLt | 98.901% | 96.703% |
| Native fp16/fp16 cuBLASLt | 98.901% | 96.978% |
| Native fp32/fp32 cuBLASLt | 98.901% | 96.978% |
| Native fp4/fp16 cublaslt | 98.352% | 95.879% |
| Native fp4/fp16 cutlass | 98.352% | 95.879% |
| Native fp4/fp4 cublaslt | 98.626% | 95.879% |
| Native fp4/fp4 cutlass | 98.626% | 95.879% |
| Native fp8/fp8 cuBLASLt | 98.901% | 96.429% |
| PyTorch compiled fp32 | 98.901% | 96.978% |
| PyTorch compiled fp16 | 98.901% | 96.978% |
| PyTorch compiled bf16 | 98.901% | 96.703% |
| PyTorch compiled fp8 | 98.901% | 96.429% |

## Complete operation latency at batch 64

Ranges span cohort medians, in microseconds. The graph column reports steady-state device work per inference. The uncaptured column includes host dispatch and stream completion; the host-request column also includes changing input H2D, output allocation and label D2H. Weights are prepared and reusable in these three columns.

| Implementation | Captured device us | Uncaptured device us | Host request us |
|---|---:|---:|---:|
| Native bf16/bf16 cuBLASLt | 6.46-6.54 | 33.81-34.65 | 58.75-59.41 |
| Native fp16/fp16 cuBLASLt | 6.52-6.53 | 33.49-34.09 | 58.35-59.15 |
| Native fp32/fp32 cuBLASLt | 13.32-13.47 | 36.49-36.71 | 59.30-59.59 |
| Native fp4/fp16 cublaslt | 7.90-7.92 | 35.56-35.80 | 61.63-63.20 |
| Native fp4/fp16 cutlass | 9.33-9.42 | 32.72-33.68 | 59.52-64.37 |
| Native fp4/fp4 cublaslt | 8.90-9.02 | 29.55-30.38 | 54.49-56.41 |
| Native fp4/fp4 cutlass | 11.64-11.70 | 25.80-25.87 | 45.81-46.36 |
| Native fp8/fp8 cuBLASLt | 6.15-6.18 | 34.29-34.43 | 57.90-59.02 |
| PyTorch fp32 eager | 14.20-14.26 | 95.34-96.02 | 149.82-150.18 |
| PyTorch fp32 compiled | 11.23-11.28 | 130.14-130.70 | 195.67-196.39 |
| PyTorch fp32 internal graphs | not isolated | 121.87-122.85 | 181.27-182.48 |
| PyTorch fp16 eager | 9.03-9.08 | 111.48-112.08 | 165.50-166.96 |
| PyTorch fp16 compiled | 5.52-5.53 | 134.03-134.75 | 202.02-204.38 |
| PyTorch fp16 internal graphs | not isolated | 109.48-112.67 | 187.97-189.69 |
| PyTorch bf16 eager | 9.22-9.27 | 110.54-115.07 | 164.12-166.48 |
| PyTorch bf16 compiled | 5.70-5.70 | 133.26-134.78 | 202.21-209.05 |
| PyTorch bf16 internal graphs | not isolated | 111.78-112.19 | 187.60-196.67 |
| PyTorch fp8 eager | 11.11-11.12 | 135.23-139.46 | 189.14-191.50 |
| PyTorch fp8 compiled | 7.51-7.56 | 158.65-161.52 | 226.61-229.26 |
| PyTorch fp8 internal graphs | not isolated | 111.72-112.76 | 188.33-191.63 |

## Allocation and weight reuse

Native records additionally retain fresh-plan requests: arrays/transfers, native plans, weight preparation, inference, labels and native teardown are timed together with a cached algorithm index. This is a warm process and a reusable CuPy allocator, not cold driver initialization. Candidate tuning is reported separately.

The [reuse CSV](results/application-reuse.csv) measures one weight-preparation phase followed by 1, 8 or 64 complete resident inferences. It excludes weight H2D and allocations, which belong to the fresh-plan/request boundary. Reuse curves are measured totals rather than extrapolated GEMM throughput.

## Validation-gated held-out format selection

An offline selector interpolates latency from cohort-A batch sizes 16,64,256. It is evaluated at 32,128,192 for both models. Eligibility uses each model's validation accuracy, never its test labels. The complete sweep retains accuracy-loss budgets of 0,0.5 and 1 percentage points instead of selecting a favorable budget after testing.

Across 72 held-out model/batch/budget/timer evaluations, geometric-mean regret versus the fastest eligible measured configuration is 1.006x. All choices and test outcomes are in [selection data](results/application-selector.json). This evaluates configuration decisions offline; runtime routing overhead and generalization to other architectures/models are not established.

## Interpretation and limits

FP4 kernel speed alone is insufficient to choose an application format. Preparation, host dispatch, transfers, setup/reuse and task accuracy can change the ranking. Increasing one layer's precision need not monotonically improve classification on a finite test set. The evidence supports evaluating the complete computation and reporting accuracy together with each latency boundary.

PyTorch 2.13.0+cu130 is the installed reference stack; native code uses CUDA 13.3/cuBLASLt as recorded. `fullgraph=True` succeeds for every compiled case. Maximum-autotune mode was requested, but PyTorch declined GEMM max-autotuning on this 36-SM device and used its available GEMM backend. The specialization limit was raised from eight to 64 for the declared static shape/precision grid. Partial first-attempt records are excluded. TF32 and reduced-precision FP16/BF16 accumulation were disabled; all GEMM outputs are FP32.

The additional `reduce-overhead` reference enables internal CUDA Graphs, dynamic shapes and static weight addresses. Every case retains live CUDA Graph roots; the root count grows from one to 24 across the full grid. Its host timers include graph-tree management and step markers. It has no isolated pure-device column. Numerical witnesses and complete predictions are independently replayed for this reference too.

The native API reserves an FP4 backend field even for configurations that contain no FP4; those FP32/FP16/BF16/FP8 products use cuBLASLt. All executions are exploratory shared-system observations. Small test data, padded model dimensions, repeated graph inputs, different installed library versions and single-session timing limit conclusions.

Run `python verify_application.py` for CPU-only numerical/provenance/quality replay and `python summarize_application.py --check` to verify the [full timing CSV](results/application-summary.csv). Outlier correction and direct loading/resource characterization remain open under [PROGRAM.md](PROGRAM.md).
