# Low-Precision GPU Research

A research project by [Requant Network](https://requant.network).

**Status: initial pilot and additive encoder/fusion/tile experiment complete; the full research program is active.**

Reproducible experiments on **NVIDIA RTX 5060 Ti 16 GB / SM120** asking when
FP4 is useful after quantization cost, operand reuse and numerical error are
included. The first experiment compares FP4, FP8 E4M3, FP16, BF16 and strict FP32.

This studies approximate real-valued products. The earlier
[exact modular Tensor Core study](https://github.com/requant-network/gpu-exact-tensorcore)
addresses a different arithmetic contract.

## What has actually run

| Evidence | Retained coverage |
|---|---|
| GPU campaign | Two independent input cohorts, six shapes, two synthetic profiles, five formats |
| Timing boundaries | Prepacked; prepare A online; prepare A and B online |
| Observations | 120 format/shape/profile cases; 10,800 graph timing observations |
| Encoding validation | Every packed operand and FP4 logical scale byte checked against independent NumPy encoding |
| Arithmetic validation | 33,259,520 output entries checked before and again after timing |
| CPU replay | Twenty full small-case input/output witnesses; source, build and encoding-hash checks |
| Native negative controls | Invalid mode, invalid aligned shape and negative dimension, using valid device pointers |

The [pilot tables](RESULTS.md) show that preparation changes which format has
the lowest observed latency. The baseline FP4 quantizer also introduces substantial
error. Reported FP4/FP16 speed ratios are **not comparisons at equal accuracy**.

The server retains other GPU contexts. These are exploratory shared-system
observations, not exclusive-hardware performance guarantees. Graphs repeatedly
reuse the same operands; they do not establish streaming or one-shot request latency.

## Read the study

- [All fifteen investigations and completion requirements](PROGRAM.md)
- [Cooperative encoder, fusion, tile and held-out rule results](ENCODER_REPORT.md)
- [Research questions and follow-up gates](RESEARCH.md)
- [Arithmetic, independent validation and timing methods](METHODS.md)
- [Derived results](RESULTS.md) and [complete CSV](results/summary.csv)
- [Pilot interpretation and next experiment](PILOT_REPORT.md)
- [Raw records and numerical witnesses](results)
- [Licenses and upstream attribution](THIRD_PARTY.md)

## CPU verification

Python 3.12 and the pinned NumPy dependency are sufficient. GPU libraries are not
needed to replay the public witnesses or check the artifacts.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python verify_results.py
python summarize.py --check
python verify_followups.py
python summarize_followups.py --check
```

CPU CI performs these checks. It cannot rerun GPU performance or recover the
unretained large-case output arrays. Every large-case output was checked at runtime;
the public small-case witnesses support independent numerical replay.

## GPU reproduction

The tested environment uses Linux, CUDA Toolkit 13.3, a working CuPy 13.6.0
CUDA-12 runtime environment and the recorded NVIDIA driver. The native bridge
links CUDA 13.3/cuBLASLt; versions are explicit in each raw record. Existing working
CuPy/CUDA compatibility is required in addition to the Python packages.

```sh
python -m pip install -r requirements-gpu.txt
git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git ../cutlass-fp4
git -C ../cutlass-fp4 checkout --detach b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd
bash build.sh ../cutlass-fp4

# Correctness preflight; not a full timing campaign.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python gpu_study.py --quick --output build/preflight.json

# Use a separate directory to preserve the published observations.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python gpu_study.py \
  --seed 20261116 --output build/reproduction/pilot-20261116.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python gpu_study.py \
  --seed 20261216 --reverse --output build/reproduction/pilot-20261216.json
```

Use `--device` to select the SM120 GPU. The bounded native interface accepts
dimensions up to 8192, with N and K multiples of 128. The retained campaign
covers only the six shapes listed in `reference.py`; accepted API dimensions
are not an experimentally validated performance range.

The native build has no fast-math flag. Original observations, source hashes and
the clean CUTLASS pin are retained. New observations must keep their actual source,
binary, toolchain, generator and timing boundaries; changed code does not reproduce
the published experiment merely by using the same seed.

## Current limits and next step

The additive study adds four byte-identical cooperative encoders, fused bias/ReLU
preparation, two CUTLASS tile shapes, four numerical profiles and held-out sizes.
Its two cohorts retain 592 configurations and 88,800 observations. Both improvements
and regressions remain in [the report](ENCODER_REPORT.md). Two independently trained
digits classifiers and their dataset provenance are also retained for the next
complete-application experiments; GPU classifier results are not yet claimed.

```sh
bash build_extended.sh ../cutlass-fp4 128
bash build_extended.sh ../cutlass-fp4 256
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python encoder_study.py \
  --seed 20261117 --output build/reproduction/encoder-cohort-a.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python encoder_study.py \
  --seed 20261217 --reverse --output build/reproduction/encoder-cohort-b.json
```

The historical pilot sources and their records remain unchanged. New experiments
have separate source/binary manifests and numerical witnesses.

The initial FP4 implementation uses one CUTLASS tile configuration, a simple
per-block encoder and a fixed top-level tensor scale of one. cuBLASLt baselines
select among up to eight heuristic candidates. This is a specified baseline
comparison, not the best achievable implementation of each format.

Remaining work adds full format selection at fixed error, outlier correction,
compiled framework comparisons, loading/pipeline characterization, complete request
costs and real application quality. The [program ledger](PROGRAM.md) also preserves
the other planned investigations. No general superiority or novelty claim is made.

## License

Original Python code and documentation are [MIT licensed](LICENSE).
The NVIDIA-derived native bridge carries its retained BSD-3-Clause notice.
See [third-party attribution](THIRD_PARTY.md).
