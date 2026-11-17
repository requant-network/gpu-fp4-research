# Streaming covariance and PCA on consumer Blackwell

**Bounded study 11: complete GPU collection and independent CPU replay. No novelty, physical-peak-memory or future-input error guarantee.**

## Scope and protocol

All 30 cohort/dataset groups contain 2,580 attempted, available and successfully executed configurations. Each group tests 84 streamed configurations plus untiled FP32/FP64 library controls. Full A/B data contain 154,800 recorded resident/host observations and 5,160 distinct retained output records. Supplementary smoke is separate and does not create a third independent cohort.

Synthetic uniform, shifted-by-32, nearly dependent and drifting inputs use N8192/32768 and D128/256; a nonaligned N8193 tail is included. Genuine SIFT1M descriptor windows use N8192/65536,D128, with disjoint original rows for A (starting at 0) and B (starting at 500000). Seeds and dimensions are in pca_cases.py; seed labels are not acquisition dates. Two alternating sign variants are changing inputs, not independent cohorts. Measurement UTC dates: 2026-11-17.

Each streamed operation recomputes original-data centering, optional power-of-two feature scaling, transpose/padding, native encoding, every Gram product, FP32/64 moment accumulation, symmetrization and complete GPU eigh. Raw, globally centered and count-weighted chunk-centered merge are tested, with chunks2048 and merge/scaled/FP64-accumulation controls at256/8192. No prepacked-input or GEMM-only speed claim. Strong references include every non-FP4 streamed precision and the untiled whole-matrix FP32/64 library pipelines, not only the slower chunked variants.

The pinned CuPy eigensolver dispatches a real symmetric two-dimensional input to its cuSOLVER syevd path; this adapter enables linalg error reporting and independently checks results, rather than treating API return as proof.[1] The native FP32 bridge requests CUBLAS_COMPUTE_32F_PEDANTIC, while half/BF16 candidates are filtered by recorded HMMA numerical implementation flags. These are API/source/flag observations, not newly collected instruction-counter traces.[2] Untiled FP32 and FP64 controls are labelled by their actual library APIs, without an unsupported claim about internal instructions.

## Independent numerical evidence

The complete source-bound replay covers 2,666 records including smoke, 90,856 stage witnesses, 154,992,640 final covariance entries, 858,624 retained eigenvalues, 13,737,984 leading-vector entries and 155,660 timing observations. Exact scope/hashes/counters are in results/pca-replay.json.

Original FP32 observations are regenerated or read from fingerprinted genuine SIFT fixtures. The independent CPU model uses interleaved FP64 reductions for complete truth covariance, checks its full reference eigenspectrum/eigenbasis, reconstructs preparation/encoding and full-or-sampled native Gram arithmetic, then checks all final covariance entries within propagated product/accumulation envelopes. Full Gram witnesses are retained for uniform8192/D128; elsewhere all diagonal plus fixed sampled Gram entries are checked. These envelopes are not bitwise GPU re-execution.

Complete means/covariances/eigenvalues and leading16 eigenvectors are retained. All eigenvectors were computed and transferred, but unretained full eigenbases are not independently certified. Actual covariance-to-eigenvalue and leading-vector residual/orthogonality checks are separate from original-data PCA quality; an internally consistent eigendecomposition of a wrong covariance does not establish task correctness. Tests reject such a forged covariance with recomputed spectrum/basis/quality/hashes, and reject stale-input timing bindings.

Joint quality is the worst, across actually timed output records, of covariance Frobenius relative error, full-spectrum relative error, maximum per-eigenvalue error scaled by max(abs(reference eigenvalue),1e-12*reference spectral radius), mean error normalized by reference total standard deviation and leading16 projector error normalized by sqrt(32). Significant negative eigenvalues below -1e-12*reference spectral radius make a route ineligible. Eigengaps remain diagnostic; a small gap does not silently remove the subspace constraint. FP64 reference and the declared weak-eigenvalue floor limit what can be inferred about near-null directions.

## Rejected original input and guarded repeat

The original strict replay correctly rejected a mean that did not reduce the declared observations. An independently reproduced original mean census covers 2,666 cases and 5,333 retained outputs; 51 outputs fail the unchanged original-data mean bound, all in full-B shifted32768/D256 variant1. A one-word reconstruction at row6376, feature215 changes -31.997573852539062 to -27.997573852539062: bits 0xc1fffb08 to 0xc1dffb08, XOR 0x200000 (1 bit). It matches the retained operand fingerprint exactly, not approximately.

The public rejected fixture retains the original full collection verbatim. Independent reconstruction checks all 3,892 archived prepared-source fingerprints using retained center/scale and phase-appropriate inputs, including 800 in the changed-input phase. It also reproduces all original mean-census failure records. This identifies altered inputs, not the physical origin, host-versus-device location or a hardware/vendor culprit. A retained 100-iteration focused diagnostic reports clean host/device fingerprints and means but does not exclude rare failures. The compute-sanitizer attempt did not finish; it is not counted as a clean sanitizer result.

All 86 configurations of that original dataset collection are excluded from the canonical numerical/performance corpus, not repaired, averaged into good records or counted as low-precision quality failures. Only this dataset was recollected in a fresh guarded process. The full corpus explicitly contains 29 original-producer collections and 1 guarded repeat. The original twelve producer dependencies remain byte-unchanged; the repeat additionally binds pca_study_guarded.py and pca_input_integrity.py. Its host inputs are read-only and every timed observation records exact original-host pre/post and device-original-input post fingerprints. Registry, replay, cost audit and CSV producer_epoch columns distinguish both source epochs.

Guard checks and their host/device transfers occur outside the named resident/host-service timers. They change inter-request spacing and may change cache/thermal state, so the replacement is not an unchanged-spacing A/B experiment. Comparisons remain within the same dataset/cohort/source epoch; aggregate group counts do not make source-epoch or thermal equivalence claims. The replacement still passes the original numerical tolerances; no mean/covariance bound was widened to accept damaged data. Rejected evidence is under fixtures/pca-input-failure, independently audited by verify_pca_input_failure.py and results/pca-input-failure-audit.json.

## Quality-constrained device results

Each row selects the fastest feasible FP4 configuration and the fastest feasible non-FP4 strong reference for the same cohort/dataset/boundary. This is an ex-post oracle frontier, not a deployable safe selector. The table uses resident CUDA events and plan reuse100000, excluding common bootstrap from this warm-process view but charging constructor and first operation divided by reuse. Ratios are reference time / FP4 time; only comparable feasible groups enter their median.

| Joint budget | FP4 feasible / groups | Reference feasible | FP4 wins | Comparable | Median reference / FP4 |
|---|---:|---:|---:|---:|---:|
| 0.001 | 0/30 | 30 | 0 | 0 | — |
| 0.01 | 0/30 | 30 | 0 | 0 | — |
| 0.1 | 0/30 | 30 | 0 | 0 | — |
| 0.5 | 24/30 | 30 | 10 | 24 | 0.985525 |
| 1 | 24/30 | 30 | 14 | 24 | 1.015777 |

### Profile-level diagnosis at joint budget0.1

| Input category | Groups | FP4 feasible | FP4 wins | Median reference / FP4 |
|---|---:|---:|---:|---:|
| dependent | 6 | 0 | 0 | — |
| drift | 6 | 0 | 0 | — |
| shifted | 6 | 0 | 0 | — |
| sift | 4 | 0 | 0 | — |
| tail | 2 | 0 | 0 | — |
| uniform | 6 | 0 | 0 | — |

## Preparation, host service and reuse

Resident events cover complete GPU centering/encoding/products/accumulation/eigh, with the current original FP32 observations already uploaded. Python launch gaps and GPU status checks inside the operation remain inside that boundary; result D2H is outside. Host-service wall time instead includes finite-input validation, adapter host-original copy, pageable H2D, the complete GPU operation/synchronization and full mean/covariance/eigenvalues/eigenvectors D2H. Dataset generation, truth calculation, hashes, quality and serialization are outside both named service boundaries.

For each route, effective time is median service + (constructor + first operation)/R. The separate common-bootstrap view also charges observed device/library/small-operation initialization/R to every candidate as a sole-mode counterfactual. This begins at main-entry device selection and excludes interpreter startup and earlier Python/CuPy imports. It is shared inside the measured dataset process, not a complete fresh-process, fresh-install or cold-OS startup measurement. CSVs retain R1/4/16/64/256/1024/100000, both bootstrap views and both service boundaries. Fixed-time tables report the best measured joint error under named time limits, without quality-budget preselection.

## Named memory and calibration audit

The independent cost audit passed 2,666 full+smoke records, including 42,966 eligible heuristic-calibration observations. Selected candidates match the measured eligible median minimum. All live named Python allocations match their original/operand/product/mean/moment/covariance/spectrum/eigenvector geometries. Native encoded/scales/workspace requests match the fixed adapter formula. Named owned requests range from 4,827,136 to 102,764,544 bytes across the declared records.

These are live named allocation requests, not physical device peak. Opaque cuSOLVER/cuBLAS internals, allocator cache, temporary expression high-watermarks and shared desktop/other-context residency are excluded. Named memory columns are not whole-device memory savings. Full named buffers and cold/setup costs are retained even when a mode loses. The audit and its exact scope are in results/pca-cost-audit.json.

## Negative outcomes and interpretation

The full GPU collection retains 822 output records with significant negative eigenvalues. 3,342 route/configuration records fail the strict joint0.001 budget or its finite/PSD gate. Neither group is removed to manufacture a successful low-precision result. Detailed errors and all measured routes are retained in the negative-results and cases artifacts.

Raw moment subtraction intentionally exposes a sensitive recipe: quantized second moments minus an original-data mean outer product need not preserve small covariance directions. Global/chunk centering, feature scaling and higher accumulation precision address different sources of error; the measured tables, not a single dominant norm, determine which combinations are feasible. A small covariance norm error can coexist with weak-spectrum or subspace failure. No conclusion about every possible Tensor Core PCA implementation follows from this fixed adapter/grid.

## Executed environment and limitations

All measurements target GPU0 RTX5060Ti16GB, SM120, in a shared nonexclusive system. Each dataset ran in a fresh process, with its own common-bootstrap snapshot and normal process exit. GPU1, unrelated services, clocks and power settings were not changed. The original full campaigns reverse method/dataset order for B. The later guarded recovery keeps B method order within its dataset but does not preserve the original global dataset schedule. Neither ordering creates a controlled exclusive thermal/frequency experiment. Installed package metadata, loaded-library fingerprints, observed runtime/driver APIs and nvcc output remain separate in each environment JSON; a version mismatch does not identify its cause.

Observed runtime API values: 13020; driver API values: 13030. CuPy versions: 14.2.0; NumPy versions: 2.2.6. The build script pins the CUTLASS source and compiles the SM120 native path. API FP64 is not asserted to be a particular instruction implementation.

Original observations fit device memory. This is a bounded streamed-chunk covariance/PCA pipeline, not a >VRAM online-arrival experiment. Two cohorts and thirty timing samples per boundary do not establish population confidence intervals, future-input guarantees, architectural ceilings or scientific novelty.

## Reproduction and artifacts

Use the pinned GPU environment already described by requirements-search-gpu.txt and its CUDA/CUTLASS build prerequisites. PCA does not import Faiss or cuVS; installing a package is not evidence that it executed. Real descriptor rights/provenance remain in THIRD_PARTY.md and fixtures/search-sift/manifest.json.

```bash
mkdir -p build
cp native/bridge.cu build/base-128.cuh
bash build_pca.sh /path/to/pinned/cutlass
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# A fresh process per dataset; reverse dataset order as well for B.
for cohort in a b; do
  datasets=$(python -c "from pca_cases import DATASETS; names=list(DATASETS); print(' '.join(names[::-1] if '$cohort'=='b' else names))")
  for dataset in $datasets; do
    producer=pca_study.py
    if [ "$cohort" = b ] && [ "$dataset" = shifted-n32768-d256 ]; then producer=pca_study_guarded.py; fi
    python "$producer" --cohort "$cohort" --dataset "$dataset"
  done
done
# Supplementary, not a full campaign:
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python pca_study.py --cohort a --smoke
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_pca.py --workers 4 --summary results/pca-replay.json
OPENBLAS_NUM_THREADS=1 python verify_pca_input_failure.py --summary results/pca-input-failure-audit.json
python pca_cost_audit.py --summary results/pca-cost-audit.json
python summarize_pca.py
python summarize_pca.py --check
```

Executed records are in results/pca-full-a, results/pca-full-b and results/pca-smoke-a. Final derived cases/frontiers/fixed-time/negative/summary artifacts use results/pca-*; preliminary artifacts use pca-preliminary-* and are not completion evidence. CPU-only replay never loads GPU producer modules. Preserve executed sources/binary/witness hashes before changes; resume requires matching bindings. Passing CPU/artifact CI on the exact published head remains a separate completion gate.

Existing sealed GPU records are resumed, not remeasured. To recollect, first archive and move the existing canonical collection directories out of the way; never overwrite source-bound evidence or interpret RESUME as a new experiment. CPU --workers4 launches independent dataset subprocesses with one BLAS/OpenMP thread each. Every requested partition is actually replayed; intermediate receipts are checkpoints, not silently trusted numerical caches. The reducer rejects overlap, missing collections, changed hashes and mixed verifier dependencies before producing the whole-study receipt.

The example GPU loop assigns the accepted source epochs for a fresh recollection. It does not recreate the accidental historical input change or its later recovery chronology, nor guarantee equivalent historical cache/thermal conditions. Historical UTC timestamps, source epochs and the rejected original remain in the retained artifacts.

## Sources

[1] https://github.com/cupy/cupy/blob/v14.2.0/cupy/linalg/_eigenvalue.py — CuPy v14.2.0 pinned eigensolver implementation
[2] https://docs.nvidia.com/cuda/cublas/index.html — NVIDIA cuBLAS API and numerical implementation flags
