# Low-Precision GPU Research

A research project by [Requant Network](https://requant.network).

**Status: studies 1-10 complete within their declared protocols; the full fifteen-study program is active. Studies 11-15 remain pending. CPU/artifact CI checks each published head.**

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
- [Native pipeline, traversal, buffering, resource and cache comparisons](PIPELINE_REPORT.md)
- [Fusion resource characterization and complete evidence](FUSION_REPORT.md)
- [FP4 column/block/residual correction at fixed error and fixed time](OUTLIER_REPORT.md)
- [INT8 component reconstruction versus native and emulated FP64](PRECISION_REPORT.md)
- [Held-out error prediction, its budget violation and complete selection costs](SELECTOR_REPORT.md)
- [Mixed-precision attention, fused FP4/SDPA references and complete model quality](ATTENTION_REPORT.md)
- [Adaptive SPD solvers, native Cholesky/IR references and forward-error failures](SOLVER_REPORT.md)
- [Hybrid sparse/dense SpMM versus native CSR/BSR/ELL/dense references](SPMM_REPORT.md)
- [Complete spectral filtering: Tensor Core FFT, cuFFT and fused cuFFTDx](FFT_REPORT.md)
- [Quantized vector search versus resident Faiss/cuVS and indexed references](SEARCH_REPORT.md)
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
python verify_pipeline.py
python summarize_pipeline.py --check
python verify_precision.py
python summarize_precision.py --check
python verify_selector.py
python summarize_selector.py --check
python verify_attention_models.py
python verify_attention.py
python verify_sage_attention.py
python verify_attention_application.py
python summarize_attention.py --check
python verify_solver.py
python summarize_solver.py --check
python verify_spmm.py
python summarize_spmm.py --check
python verify_fft_controls.py
python verify_fft.py
python summarize_fft.py --check
python verify_search_environment.py
python verify_search_sift.py
python verify_search_preflight.py
python verify_search_kernel_preflight.py
python verify_search.py
python summarize_search.py --check
python summarize_search.py --smoke --check
python verify_pca.py
python pca_cost_audit.py
python summarize_pca.py --check
```

CPU CI performs these checks. It cannot rerun GPU performance or recover the
unretained large-case output arrays. Every large-case output was checked at runtime;
the public small-case witnesses support independent numerical replay.
The solver study additionally retains every actual solver output in full,
including failed solutions, and links each measured observation to its witness.
The SpMM study retains selected complete outputs and samples for every check;
its large complete outputs are checked at runtime. Its gzip cohort archives
preserve the original JSON bytes and support CPU-only replay.
The FFT study retains complete small/selected large output and spectral arrays,
plus samples and independent stage witnesses. Its largest NPZ files are published
in byte-preserving parts; the CPU loader verifies and assembles them automatically.
The search study retains every returned top-10 ID/distance array and independent
candidate/refinement stage witnesses. CPU replay recomputes all exact FP64 truth
queries; large unselected score rankings have explicitly sampled scope. Original
SIFT fvecs hashes are reconstructed from losslessly stored numerical descriptors.

The PCA collection retains all final means, covariances and eigenvalues, with
leading16 eigenvectors, independent original-data/stage witnesses and per-timing
input/output bindings. It does not certify unretained full eigenbases or opaque
library/cache/temporary peak-memory allocations. Named buffer requests and
recorded heuristic selection have a separate geometry-based audit.

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
The [held-out per-operation predictor](SELECTOR_REPORT.md) retains one budget
violation and substantial selection costs; it establishes no runtime accuracy guarantee. Native
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

The [high-precision product study](PRECISION_REPORT.md) adds 1,008 configurations
comparing signed INT8 component products with native FP32/FP64 and cuBLASLt
emulation. It includes cancellation, wide exponent ranges, subnormal inputs,
long dots, real activations and changing-input adaptive decisions. Exact integer
witnesses check finite FP32-input products. Preparation and complete host costs
are retained; custom reconstruction does not uniformly beat library emulation.
Build with `bash build_precision.sh` and run `precision_study.py` with independent
seed/model pairs and separate output paths, as documented in the report.

The [attention study](ATTENTION_REPORT.md) compares 1,512 native/fused attention
configurations and 138 complete transformer configurations across two cohorts.
It includes independent QK/PV formats, masks, irregular sizes, preparation/transfer
costs, explicit memory boundaries, FlashAttention/efficient/cuDNN and pinned fused
SageAttention3 FP4 controls. Both real-model cohorts retain complete quality arrays.
Native FP8 probability errors depend on the specified unit scaling; the report
does not present them as a format-wide accuracy limit or an optimal kernel result.

## Vector-search reproduction

Search uses a separate Linux/Python 3.12 environment, pinned in
`requirements-search-gpu.txt`. The actual snapshot is
`results/search-environment.json`: the executed runtime API reports CUDA 13.2
despite the CUDA-13.3 toolkit/package pins. Do not infer the runtime from package
versions or attribute this observation to one library without a loader trace.
No CPU verification requires these GPU packages.

Use a separate reproduction checkout and preserve the published record directories
and environment snapshot before fresh GPU work. The search drivers resume only
matching sealed source/binary records; resuming existing records is not a new GPU
experiment. Archive `results/search-smoke-a`, `results/search-full-a`,
`results/search-full-b` and the environment/probe records together when replacing
them. Rebuild and rerun the probes as well if replacing the snapshot.

```sh
python3.12 -m venv ../search-env
../search-env/bin/python -m pip install -r requirements-search-gpu.txt
../search-env/bin/python -m pip check
mkdir -p build
cp native/bridge.cu build/base-128.cuh
bash build_search.sh ../cutlass-fp4 ../search-env
bash build_search_full.sh ../cutlass-fp4 ../search-env
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
../search-env/bin/python verify_search_sift.py
../search-env/bin/python inspect_search_environment.py
../search-env/bin/python search_preflight.py
../search-env/bin/python search_kernel_preflight.py
../search-env/bin/python check_search_lifetime.py
../search-env/bin/python check_search_capture_memory.py
for dataset in uniform-small clustered shifted uniform-million sift-million; do
  ../search-env/bin/python search_study.py --cohort a --smoke --dataset "$dataset"
done
../search-env/bin/python search_study.py --cohort a
../search-env/bin/python search_study.py --cohort b
../search-env/bin/python verify_search.py
../search-env/bin/python summarize_search.py
../search-env/bin/python summarize_search.py --check
```

The lossless SIFT fixtures are included. `prepare_search_sift.py --cache <directory>`
can refetch their pinned original files; `verify_search_sift.py` reconstructs both
whole-source hashes offline. CAGRA uses context-lifetime PTDS streams, and Faiss's
shape-aware scratch reservation is explicitly charged. Unsupported/stale graph
routes and incomplete/duplicate-ID outcomes are retained, not speed wins.

## Streaming covariance/PCA reproduction

PCA uses the separate pinned Python3.12/CuPy14.2 GPU environment in
`requirements-search-gpu.txt`; it does not import Faiss or cuVS. The legacy pilot
environment above is not the PCA snapshot. Each retained PCA dataset uses a fresh
process with its own environment/bootstrap record; source/binary/witness bindings
must match when resuming. Use a separate checkout or archive complete record sets
before a fresh experiment. Resuming sealed results is not a new GPU run.

```sh
mkdir -p build
cp native/bridge.cu build/base-128.cuh
bash build_pca.sh ../cutlass-fp4
# Activate the pinned GPU environment first.
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for cohort in a b; do
  datasets=$(python -c "from pca_cases import DATASETS; names=list(DATASETS); print(' '.join(names[::-1] if '$cohort'=='b' else names))")
  for dataset in $datasets; do
    producer=pca_study.py
    if [ "$cohort" = b ] && [ "$dataset" = shifted-n32768-d256 ]; then producer=pca_study_guarded.py; fi
    python "$producer" --cohort "$cohort" --dataset "$dataset"
  done
done
# Supplementary smoke, not study completion:
python pca_study.py --cohort a --smoke
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_pca.py --workers 4 --summary results/pca-replay.json
OPENBLAS_NUM_THREADS=1 python verify_pca_input_failure.py --summary results/pca-input-failure-audit.json
python pca_cost_audit.py --summary results/pca-cost-audit.json
python summarize_pca.py
python summarize_pca.py --check
```

The 2,580 full configurations compare original-data centering/scaling, FP32/64
moment accumulation, native FP4/FP8/FP16/BF16/pedantic-FP32 products, FP64 APIs and
untiled whole-matrix library controls on shifted, nearly dependent, drifting and
genuine SIFT inputs. Quality/latency tables are ex-post frontiers, not safe
future-input dispatch rules. Preserve numerical failures and all cost boundaries.

The [PCA report](PCA_REPORT.md) documents a rejected original collection: retained
fingerprints reconstruct a one-bit input-word change, not harmless mean roundoff.
Its original bytes remain in `fixtures/pca-input-failure`; they are excluded from
performance wins and independently audited. Only that dataset was recollected
with read-only host inputs and timed-observation host/device fingerprint checks.
The original twelve producer dependencies remain unchanged; its guarded epoch
adds two explicitly bound modules. CSVs and receipts identify each epoch. Guard
checks are outside service timers and change inter-request spacing; no unchanged
cache/thermal-state or hardware/vendor fault attribution is claimed. The sanitizer
attempt did not finish. CPU replay uses independent dataset processes, one BLAS
thread each, and rejects missing/overlapping or differently source-bound partitions.

## License

Original Python code and documentation are [MIT licensed](LICENSE).
The NVIDIA-derived native bridge carries its retained BSD-3-Clause notice.
See [third-party attribution](THIRD_PARTY.md).
