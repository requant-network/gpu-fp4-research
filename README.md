# Low-Precision GPU Research

A research project by [Requant Network](https://requant.network).

**Status: pilot, encoder/fusion/tile, classifier/framework, native-conversion and outlier-correction experiments complete; the full fifteen-study program is active.**

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

Subsequent experiments add 592 encoder/fusion/tile configurations and 240 complete
classifier/reference cases. The latter separately measure captured computation,
uncaptured dispatch, changing-input host requests, fresh plans and weight reuse.
See [the complete-application report](APPLICATION_REPORT.md) for task quality and
scope; its host-request results are distinct from the original repeated-graph pilot.

## Read the study

- [All fifteen investigations and completion requirements](PROGRAM.md)
- [Cooperative encoder, fusion, tile and held-out rule results](ENCODER_REPORT.md)
- [Real classifier, cuBLASLt FP4 and PyTorch/CUDA-Graph references](APPLICATION_REPORT.md)
- [Native FP4 conversion, vector loads, emitted instructions and resources](HARDWARE_REPORT.md)
- [FP4 column/block/residual correction at fixed error and fixed time](OUTLIER_REPORT.md)
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
python verify_application.py
python summarize_application.py --check
python verify_hardware.py
python summarize_hardware.py --check
python verify_outliers.py
python summarize_outliers.py --check
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
digits classifiers and their dataset provenance support completed GPU application
experiments, including CUTLASS/cuBLASLt FP4 and eager/compiled/internal-graph PyTorch
references. The native implementations are CUDA C++ with an explicit C interface;
Python drives experiments and is included in the separately reported host timers.

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

Build `libapplication.so` with `bash build_application.sh ../cutlass-fp4` after the
extended builds. Run `application_study.py` with model seeds 20261116/20261216 and
separate output paths for the two cohorts (add `--reverse` for the latter). The
`pytorch_study.py` and `pytorch_graph_study.py` references use the recorded external
PyTorch 2.13.0+cu130/Triton environment. CPU artifact replay needs neither package.

The initial FP4 implementation uses one CUTLASS tile configuration, a simple
per-block encoder and a fixed top-level tensor scale of one. cuBLASLt baselines
select among up to eight heuristic candidates. This is a specified baseline
comparison, not the best achievable implementation of each format.

Offline held-out format decisions now include validation-accuracy budgets and
complete request costs. Outlier experiments add fixed-error/time comparisons.
Remaining work includes held-out per-operation error prediction and direct
loading/pipeline/resource characterization. Native
Tensor Core paths are measured implementations, not an established architectural
ceiling. The [program ledger](PROGRAM.md) preserves all other investigations.

The additive native-conversion experiment contains 680 configurations and 102,000
observations with byte-identical FP4 encoding. Emitted SASS confirms Tensor Core,
TMA and packed FP4 conversion instructions. Runtime resource queries retain
theoretical occupancy. Nsight Compute counter collection was attempted and denied
by the driver's current permissions; achieved utilization remains unmeasured.
See [the hardware report](HARDWARE_REPORT.md) for comparisons and reproduction.

The [outlier study](OUTLIER_REPORT.md) adds 598 configurations comparing native
column, block and full-residual correction against FP4/FP8/FP16/strict-FP32,
including real-model activations. Detection and correction preparation are timed.
At error budget 0.001, correction wins none of the 68 online-A groups; both useful
accuracy improvements and unfavorable complete costs remain in the record.

## License

Original Python code and documentation are [MIT licensed](LICENSE).
The NVIDIA-derived native bridge carries its retained BSD-3-Clause notice.
See [third-party attribution](THIRD_PARTY.md).
