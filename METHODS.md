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
