# Native FFT and spectral filtering on consumer Blackwell

This bounded study tests complete window/FFT/filter/inverse processing on one
RTX 5060 Ti 16 GB, SM120. The proposed radix-8/16 matrix FFT uses actual
FP16/BF16 HMMA GEMMs, but its packing, small DFT products and intermediate global
buffers do not beat the strong native references in the principal comparison.
The practical positive result is fusing the complete FP32 pipeline with cuFFTDx.
Low transform precision can destroy a weak component despite a small error relative
to the unfiltered input. These results do not establish an architectural limit or
a general impossibility of a faster Tensor Core FFT.

## Executed scope

The two independent input cohorts attempt **2,184 configurations** across
**182 length/batch/profile groups**: **1,876 available**, **308 unavailable**,
and **168,840 timing observations**. Lengths are 128, 512, 1000, 1024, 4096
and 16384; batch counts are 1 and 16, plus 1024-by-64. Seven signal profiles cover
impulse with identity/lowpass, dyadic noise with Hann/lowpass, off-bin tones with
Hann, strong/weak on-bin components separated by nominal 72.25 and 120.41 dB,
and a high-range 96.33 dB case. The complex notch removes the strong bin and its
two neighbours. Complex lowpass coefficients include a three-sample phase delay.
There are four changing inputs per group; cohort B reverses group and method order.
Carrier phase, impulse position and noise change with the input seed; coefficient
tables are shared immutable parameters. Inputs and output interfaces are FP32.

The nominal range describes the synthesized components. FP32 input addition already
rounds the weakest component; `input_weak_amplitude_ratio` records its actual Fourier
amplitude relative to the nominal amplitude. The FP64 reference uses the stored FP32
input and coefficient values, rather than claiming access to an unrounded physical signal.

The [raw A record](results/fft-a.json), [raw B record](results/fft-b.json),
[available-case CSV](results/fft-summary.csv) and [quality/cost frontiers](results/fft-frontiers.json)
retain failures and the measurement boundaries. The [quick control](results/fft-quick.json)
adds 75 available configurations; its timings are excluded from these comparative tables.

## Native implementations

cuFFT 12.3.0, delivered with CUDA 13.3, executes forward and inverse complex transforms
with explicitly owned workspace. Separate native kernels apply the window, complex
frequency response and normalization. FP32, FP64, FP16 and BF16 transforms are attempted.
FP64 computes the window and filter in FP64 and returns FP32 output; other paths
round their typed intermediates. BF16 cuFFT uses mixed single/BF16 operations, as
documented by NVIDIA; it is not labelled pure BF16 arithmetic.

The proposed matrix FFT is mixed-radix decimation in time. Each small complex DFT is
an augmented real matrix multiply `[Re(W),-Im(W);Im(W),Re(W)]`, with a column-major
packed input. A digit permutation, coalesced shared-memory transpose, FP32 twiddles,
typed FP16/BF16 encoding and per-stage normalization are included in the timed path.
Radix-2/4 tails pad the real GEMM dimension to 16; radix 16 uses dimension 32.
Forward and inverse stages each divide by their radix, with a final scale restoring
the usual inverse convention. cuBLASLt chooses among up to eight heuristics during
setup; every low-precision stage must report the HMMA implementation flag. A strict
FP32 pedantic matrix-GEMM control uses the same factorization and packing.
Each plan reserves 64 MiB for Lt workspace; this is charged in memory and setup.

The stronger fused reference uses external NVIDIA MathDx 26.06.1, whose actual
cuFFTDx header version is **1.7.3**, with `SM<1200>`. Its single native kernel performs
windowing, register/shared-memory forward FFT, response multiplication, inverse FFT
and output normalization. One/four FFTs per block are attempted. Diagnostic kernels
add spectrum writes outside performance timing. Shared-memory, register and theoretical
resident-block limits are recorded; they do not measure achieved occupancy.
The SDK archive and 1022 used header files are [pinned by SHA256](results/fft-dx-dependency.json).

For length 1000, actual cuFFT FP16/BF16 planning returns status 16. The power-of-two
matrix factorization rejects it, and the selected cuFFTDx descriptors are unsupported.
At length 16384, the selected four-FFT cuFFTDx descriptor is unsupported; the one-FFT
descriptor runs. These are outcomes of the named implementations, not claims that
all non-power-of-two or four-FFT implementations are impossible. Ordinary FP32/64
cuFFT supplies a working reference at every shape.

## Timing, preparation and memory

Every available method has five warmups and thirty observations for each of two
resident boundaries: complete captured pipeline and complete uncaptured pipeline.
Small graph launches repeat the operation eight times; the largest uses one.
Input encoding, packing, windowing, both transforms, filtering and normalization
are included. Resident repetitions reuse input zero and fixed parameters. Events
surround the graph launch externally, so dispatch gaps may contribute; values are
per-operation averages, not isolated kernel durations or architecture throughput.
The [same-boundary empty-graph control](results/fft-timer.json) reports 1 repetition(s): 0.448 us/op, 8 repetition(s): 0.056 us/op.
No calibration is subtracted. Small timing differences near this floor are not
evidence of a hardware mechanism.

Thirty additional changing-input host requests include pageable H2D, the complete
uncaptured pipeline, blocking full FP32 D2H and host output allocation. Actual outputs
are read before any diagnostic rerun. Native CPU planning/allocation/upload/Lt tuning,
graph setup and canonical coefficient-file loading are recorded separately. Common
input synthesis, CPU oracles, untimed spectrum writes and witness persistence are
excluded from performance timing. Coefficients are provided stored parameters;
request-time generation of a new window or arbitrary filter is outside this protocol.
The reuse tables add measured coefficient loading, plan setup and graph construction
once per R operations. They are hot-process amortization models, not measured cold
process requests or a changing-filter/planning service.

Requested owned device buffers range from 5,632
to 90,622,528 bytes. This includes diagnostic buffers
even though their writes are untimed. It excludes opaque library/context allocations,
allocator peaks, shared-memory capacity and other processes' VRAM. No total physical
peak memory or whole-system energy claim is made.

## Complete latency and quality

For 16384-by-16 Hann-windowed noise/lowpass, the table combines cohort medians and
uses the worst output relative L2 error over both cohorts and all four inputs:

| Method | Captured complete us/op | Worst output relative L2 |
|---|---:|---:|
| cufft-fp32 | 30.496 | 3.92e-07 |
| cufft-fp64 | 255.776 | 2.54e-08 |
| cufft-fp16 | 24.568 | 0.00232 |
| cufft-bf16 | 26.672 | 0.00405 |
| matrix-r16-fp32 | 259.960 | 2.61e-07 |
| matrix-r16-fp16 | 251.448 | 0.000944 |
| matrix-r16-bf16 | 251.384 | 0.00572 |
| cufftdx-fp32-pack1 | 22.312 | 3.92e-07 |

Across the 154 power-of-two groups with a fused reference, the best
attempted cuFFTDx packing is faster than ordinary FP32 cuFFT in
145/154 groups, with median ratio
2.129x (cuFFT time / fused time). Requiring both methods to satisfy
the 0.001 output-error budget and weak-component limits leaves 111
groups; median ratio is 2.130x. These are empirical comparisons
on the recorded shared system, without confidence claims across hardware or load states.

Quality-constrained comparisons require every actual input variant to be finite,
output relative L2 <= the stated budget, and, for weak-component profiles, weak-bin
error, quiet-bin artifact and output error each <= 0.1 of the nominal weak signal.
The reference family includes all available cuFFT precisions and fused cuFFTDx.
`HMMA` includes only the FP16/BF16 matrix FFT; `matrix` also includes its pedantic
FP32 control. A missing feasible candidate is retained as a failure, not discarded.

| Output budget | Boundary | Feasible HMMA groups | HMMA faster than best reference | Any matrix FFT faster |
|---|---|---:|---:|---:|
| 0.01 | captured_complete | 88/182 | 0/182 | 0/182 |
| 0.01 | uncaptured_complete | 88/182 | 0/182 | 0/182 |
| 0.01 | host | 88/182 | 0/182 | 0/182 |
| 0.001 | captured_complete | 80/182 | 0/182 | 0/182 |
| 0.001 | uncaptured_complete | 80/182 | 0/182 | 0/182 |
| 0.001 | host | 80/182 | 0/182 | 0/182 |
| 1e-05 | captured_complete | 0/182 | 0/182 | 0/182 |
| 1e-05 | uncaptured_complete | 0/182 | 0/182 | 0/182 |
| 1e-05 | host | 0/182 | 0/182 | 0/182 |

At budget 0.001, modeled captured setup/reuse HMMA wins are:
R=1: 0/182; R=4: 0/182; R=16: 0/182; R=64: 0/182; R=256: 0/182; R=1024: 0/182. The complete JSON also retains the other budgets and
method identities. This is an offline measured frontier, not a deployable selector,
an independent test of prediction, or a guaranteed-error rule.

## Weak components and failures

For the nominal 120.41 dB profile, the following summarizes the per-group worst
four-input output error across all available lengths, batches and both cohorts:

| Method | Median output relative L2 | Maximum output relative L2 | Maximum weak-bin relative error |
|---|---:|---:|---:|
| cufft-fp64 | 2.35e-08 | 2.56e-08 | 1.08e-12 |
| cufft-fp32 | 0.0204 | 0.179 | 0.00229 |
| cufftdx-fp32-pack1 | 0.0358 | 0.0546 | 1.95e-07 |
| cufft-fp16 | 233 | 339 | 73.5 |
| cufft-bf16 | 1.1e+03 | 1.9e+03 | 650 |
| matrix-r16-fp16 | 162 | 263 | 73.5 |
| matrix-r16-bf16 | 1.31e+03 | 2.29e+03 | 651 |

The quiet-bin metric excludes the deliberately strong neighbourhood and target weak
bin, then measures the largest *error against the FP64 spectrum*, divided by nominal
weak spectral amplitude. The output weak metric divides complete output error norm
by `weak_amplitude*sqrt(batch*N)`. Neither uses the strong input norm as an accuracy
denominator. Spectra are from an untimed diagnostic pass whose final output must
match the previously read actual timed output, including nonfinite outcomes.

cuFFT FP16 produces nonfinite values on the 16384-point high-range cases. The output,
spectrum and filtered witnesses retain these values; JSON uses null for undefined
errors and records nonfinite counts. They remain timed attempted cases but are never
eligible for a quality-constrained win. Per-stage scaling keeps the matrix FFT finite
in that case, but finiteness alone does not recover the weak component or establish
a latency advantage. A small error normalized by the original strong input is only
an implementation screening check; it is not the scientific quality criterion.

## Independent validation and scope limits

The runtime checks 283,319,680 actual output entries across input variants over
the two full campaigns. CPU replay checks **23,309,184 retained complete complex entries**
across output/spectrum/filtered arrays, **1,273,656 sampled entries**, and
**1,142,784 selected GEMM-product entries**. Quick and capability controls add
separate witnesses. Full arrays are retained for all small batch-one cases and
selected large/reference/low-precision cases; the remaining large full arrays are
checked at runtime and are not recoverable from CPU CI. Sample replay does not
re-establish unretained global norms. Every witness, source, canonical coefficient,
input, uploaded parameter and actual timed output is hash-bound to its record.

An FP64 mixed-radix implementation is tested against NumPy FFT before using the
factorization on GPU. For every noise-case matrix FFT, all actual packed bytes,
typed basis coefficients, full real GEMM output and exact unpacking/normalization
are checked against independent CPU encoding and FP64 accumulation envelopes.
Selected columns retain source operands, typed packed values, products and unpacked
outputs for CPU CI; fully retained stage chains are replayed end to end. The capability
probe separately retains 44 supported raw cuFFT forward/inverse pairs and four
unsupported planning outcomes. Unit tests cover mixed radices, weak denominators,
overflow policy, signal replay, nonfinite decoding and artifact corruption.

The setup does not provide exclusive GPU access or hardware performance counters;
existing contexts are recorded, and clocks/power settings are not changed. HMMA
flags establish the selected arithmetic family, not achieved Tensor Core utilization.
The tested matrix FFT uses multiple global-memory stages; a more deeply fused custom
FFT, other factorizations, FP8/FP4 formats, callbacks, larger transforms, real-input
transforms, multidimensional FFT and arbitrary streaming filters need new experiments.
No claim of research novelty is made without a separate literature assessment.

## Reproduction

CPU checks require Python 3.12 and the pinned NumPy package:

```sh
python -m unittest discover -s tests -v
python verify_fft_controls.py
python verify_fft.py
python summarize_fft.py --check
```

Large NPZ files are published as content-addressed 16 MiB parts listed in
[the artifact manifest](results/fft-artifacts.json). The loader assembles and hashes
the exact original NPZ bytes in memory; locally regenerated original files take
precedence. No GPU or external SDK is required for CPU replay.

For a new GPU run, use Linux/CUDA 13.3 on SM120 and a new output basename:

```sh
python prepare_fft_dx.py
bash build_fft.sh
python fft_study.py --seed 20261126 --output fft-reproduced-a
python fft_study.py --seed 20261226 --output fft-reproduced-b --reverse
python verify_fft.py results/fft-reproduced-a.json results/fft-reproduced-b.json
```

The supplied canonical coefficient fixtures are already present. Regenerating new
fixtures is a new provenance chain (`fft_coefficients.py --output <new-directory>`),
not an instruction to overwrite the executed fixtures. GPU timings are not expected
to be byte-identical to the public records. `summarize_fft.py` derives the published
A/B tables; after deliberately replacing/reproducing those campaigns, regenerate
derived artifacts before using its `--check` mode. External SDK licenses remain
separate; see [third-party attribution](THIRD_PARTY.md).

Primary API references: [cuFFT precision and planning](https://docs.nvidia.com/cuda/archive/13.3.1/cufft/index.html),
[cuFFTDx operators and SM120](https://docs.nvidia.com/cuda/cufftdx/api/operators.html),
[cuFFTDx execution contracts](https://docs.nvidia.com/cuda/cufftdx/api/methods.html).
