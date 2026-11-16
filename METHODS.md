# Methods

## Arithmetic and storage

The mathematical product is `C = A @ B.T`, where A is MxK and B is NxK.
Source operands are finite FP32, outputs are FP32, and all matrix accumulations
use FP32. The strict FP32 comparator requests `CUBLAS_COMPUTE_32F_PEDANTIC`.
FP8/FP16/BF16 use cuBLASLt with `CUBLAS_COMPUTE_32F` and explicit input types.

FP4 uses E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 and 6. Each consecutive
16-value row block gets an unsigned E4M3 scale: round `max(abs(block))/6` to
the nearest representable scale after clamping to [2^-9,448]. A zero block
uses scale 1. Values are divided by that scale and rounded to E2M1 with
nearest-even midpoint handling and finite saturation. Two values share a byte.
The top-level tensor multiplier is fixed to one. The encoder is a bounded
baseline, not a complete full-dynamic-range NVFP4 model quantization recipe.

The native quantizer writes both the CUTLASS interleaved scale layout and a
logical scale array for independent checks. This validation-oriented extra write
remains inside online preparation timers. The initial scalar-per-block encoder
is deliberately straightforward and has not been optimized.

FP8 uses unscaled E4M3 finite conversion, FP16 uses IEEE half nearest-even,
and BF16 uses nearest-even truncation of FP32. These are specified baselines,
not claims about the best quantization recipe for every format.

## Inputs and correctness

`reference.inputs` uses NumPy SeedSequence from cohort, M, N, K and profile.
The `normal` profile is a bounded, approximately normal integer-CLT construction:
sum twelve independent uniform 8-bit values, subtract 1530 and divide by 256.
Two PCG64 raw words supply six bytes each. Integer sums and power-of-two scaling
make generated FP32 inputs bit-identical across the tested Linux/Windows builds.
The outlier profile multiplies every
97th A element and every 89th B element by 32. The first row of each operand
is zero. These are synthetic stress profiles, not measured model activations.
An initial internal standard-normal generator failed exact cross-platform replay
on larger inputs and was replaced before the retained campaign. The final inputs
are discrete CLT samples, not exact Gaussian samples.

The independent CPU implementation enumerates representable FP4/FP8 values.
Every GPU-packed byte and logical block-scale byte must agree exactly with it.
Every GPU output is checked against an FP64 product of the reconstructed operands.
The elementwise absolute bound is `8*gamma_K*(abs(Aq)@abs(Bq).T)+1e-6`, where
`gamma_K = K*2^-24/(1-K*2^-24)`. It is a conservative arithmetic check, not a
target application error tolerance. Quantization error relative to the original
operands is reported separately as relative Frobenius and maximum absolute error.

The small 128x128x128 cases retain full inputs and output in NPZ witnesses.
CPU CI replays their arithmetic independently. Larger cases are fully checked
at execution time; CI regenerates input/encoding hashes but does not reconstruct
unretained GPU output or replay GPU performance.

## Timing boundaries

| Boundary | Included device work |
|---|---|
| prepacked | GEMM on already converted/quantized operands |
| online-a | Prepare A from FP32, then GEMM; B is prepacked |
| online-both | Prepare A and B from FP32, then GEMM |

Even FP32 performs a copy in online preparation, so this is a common FP32-source
interface, not a framework's optimal native-FP32 or native-BF16 pipeline.
Allocation, library/context initialization, candidate selection, graph capture,
host transfers, validation and environment queries are outside measured graphs.
All allocations and output buffers are reused. Graphs repeat the same operands:
these are warm steady-state reuse measurements and do not simulate streaming
fresh data or one-shot request latency.

Each observation times one graph of 32 repeated operations using CUDA events,
then divides by 32. Candidate selection uses two warmups and three observations
of 16 GEMMs for each available heuristic candidate. Up to eight candidates are
retained; this is a bounded heuristic search, not exhaustive autotuning.

Thirty rounds rotate the first measured format/boundary combination. Each graph
gets five warmup launches. The second cohort reverses case and initial mode order.
Medians and all original observations are retained. Input cohorts are independent
replications; thirty repeated observations of one input are not thirty independent
datasets. Profile separately from timing: profiler replay overhead changes timing.

## Environment and scope

Target: RTX 5060 Ti 16 GB, SM120, CUDA Toolkit 13.3, pinned CUTLASS and the recorded
cuBLASLt version. The native bridge links CUDA 13.3 libraries while the existing
CuPy environment uses CUDA 12 runtime components; versions are retained explicitly.
The same driver/context interoperates through device pointers and an explicit stream.

The shared server retains other GPU contexts. Boundary snapshots record utilization,
temperature and clocks, but cannot exclude activity during an observation. Results
are exploratory observations rather than exclusive-hardware performance guarantees.
No clocks, power limits, services or unrelated workloads are modified.

## Additive encoder, activation and tile experiment

The original four executed pilot sources remain byte unchanged. `optimized_bridge.cu`
adds one-value-per-lane (16-lane groups) and two-values-per-lane (8-lane groups)
cooperation with fixed 128/256/512-thread launches. Shuffle reductions compute the
same block maximum and the original conversion contract; unused layout padding
retains its initialized scale. Two generated base headers differ only in the
declared M tile (128 or 256). Source transformations and binary digests are retained.

Fusion incorporates bias addition and ReLU into FP4 input preparation. Its separate
control performs the same activation in one kernel followed by the same cooperative
encoder. The activation and quantization contract is checked against independent
CPU computations. Both routes include bias/ReLU inside online/preparation timers.

Six historical shapes, four held-out shapes and one 2048-cubed shape are tested.
Normal and outlier profiles are supplemented with small-magnitude values exercising
minimum scales and explicit rounding midpoints. The largest shape uses normal and
outlier profiles. Five boundaries distinguish preparation A, preparation A+B,
prepacked GEMM, online A+GEMM and online A+B+GEMM. Shared-system qualifications and
5/30/32 warmup/observation/repetition counts match the declared protocol.

All encoded operands and scales must match the CPU encoder exactly. New arithmetic
checks use `8*gamma_K*(abs(Aq)@abs(Bq).T)+float32.tiny`, which also resolves small
signals without the historical pilot's absolute 1e-6 floor. Eight NPZ files retain
all small-case outputs. The two new cohorts contain 336,068,608 entries checked
before and after timing; CPU replay covers 917,504 retained output entries.

The encoder selector uses only cohort-A training shapes and preparation-A times.
Its input feature is M*K, with one threshold and one fixed encoder on each side.
No held-out shape or cohort-B duration is used to fit it. Held-out outcomes include
every numerical profile in both cohorts. This selects an equivalent encoder,
not a format at equal application quality. The new CSV uses nearest-rank empirical
p10/p90; the historical pilot CSV retains its original interpolated percentiles.

## Real-data model preparation

The attributed [digits subset](fixtures/DATA.md) supplies 1,797 small images and
ten labels. Two independent stratified 60/20/20 splits train a 128-256-128 padded
ReLU classifier, using only ten output classes. Each model uses a fixed 80-epoch
Adam schedule. Test data is not used to choose a checkpoint. Padding preserves the
real 64-feature task while meeting the bounded matrix interface.

Frozen parameter arrays, split indices, training source and dataset hashes,
validation history and train/validation/test accuracy are retained. CPU verification
replays the frozen models, checks disjoint exhaustive splits and rejects changed
data or parameters. Re-running training can differ between BLAS builds; distributed
parameters are reference artifacts rather than a promise of bit-identical training.
This preparation does not yet establish GPU model quality or latency.

## Complete classifier and framework references

The subsequent application study executes both trained models on GPU. The first
matrix product prepares the request; the second combines bias/ReLU preparation
with its matrix product, followed by GPU bias/argmax over ten classes. FP32 output
is used for every GEMM. FP4 compares CUTLASS with an independent cuBLASLt backend
using the same encoded operands and `VEC16_UE4M3` block scales in the required TN
layout. The implementation follows the [cuBLASLt block-scaling interface](https://docs.nvidia.com/cuda/cublas/index.html).

The first-layer output is independently checked, then copied outside timing to
reconstruct the exact second-layer encoded input. Every encoded operand and each
layer's output are validated. Public small witnesses support both layer replays;
all retained validation/test predictions additionally match independent complete
CPU simulation. Accuracy and score error are distinct from arithmetic correctness.

Host requests alternate among four actual input batches and check their returned
labels. Uncaptured wall timers include dispatch and stream completion. Fresh-plan
timers additionally include arrays/transfers, native plan creation, prepared weights,
inference, returned labels and native teardown, using a cached algorithm index and
a warm CuPy allocator/process. The allocation/setup observation and tuning search
are separate fields. Weight-reuse graphs prepare both weights once, then execute
1,8 or 64 complete inferences; their totals exclude allocation and weight H2D.

PyTorch references use the same inputs, frozen parameters and task, with explicit
FP32 GEMM outputs. FP32 is strict; FP16/BF16 reduced-precision accumulation is disabled.
FP8 uses `_scaled_mm` with unit scalar scales. Eager and fully compiled no-internal-graph
paths have isolated captured-device and uncaptured/host timers. A separate dynamic
`reduce-overhead` reference enables internal CUDA Graphs and static weight addresses;
live graph-tree roots are recorded for every shape/precision case. Its timers
include graph management/step markers and do not claim an isolated kernel time.

Maximum-autotune mode was requested in the static compiled reference; the installed
framework declines GEMM max-autotuning on this 36-SM GPU. The 24-specialization grid
requires a limit above the default eight; the actual run uses 64 and fullgraph=True.
The failed partial first attempt is excluded. Native CUDA 13.3 and the available
PyTorch 2.13.0+cu130 reference stack have different library versions, explicitly retained.

The offline application selector uses cohort-A timings at batches16,64,256 with
linear interpolation. It evaluates batches32,128,192 in both models. Eligibility
uses each model's validation accuracy with a complete 0/0.5/1-percentage-point
budget sweep. Test labels and held-out timings do not choose the format. Actual
chosen-configuration test accuracy/latency and regret are retained; runtime routing
overhead and generalization to other models are unmeasured.

This application checkpoint extends rather than rewrites the historical pilot.
All observations retain the shared-system limitations. Direct native instruction,
loading, pipeline and resource investigations remain open.
