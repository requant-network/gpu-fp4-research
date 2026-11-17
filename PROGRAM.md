# Consumer Blackwell research program

A research program by [Requant Network](https://requant.network), targeting one
RTX 5060 Ti 16 GB. All code, protocols, public data and reports are in English.

The objective is to finish all fifteen planned investigations. The first GEMM
pilot is evidence for study 1, not completion of this program. Related studies
may share a repository and pinned implementations; repository count is not an
experimental completion criterion.

## Completion contract

Every study requires executable CPU/GPU work, an independently checked correctness
contract, useful reference implementations, two input cohorts where applicable,
retained negative outcomes, explicit setup/transfer/reuse boundaries, environment
and executed-source provenance, a substantive English findings report and passing
CPU/artifact CI on its published head. Sampling repeated timing observations does
not replace independent inputs. Shared-system limitations remain explicit.

A negative finding can finish a hypothesis test when the proposed approach and
appropriate controls were actually run. An unavailable experiment, missing baseline,
untested quality metric or documentation-only scaffold cannot finish a study.
Claims of scientific novelty require a separate literature assessment.

## Required studies and evidence

| ID | Investigation | Required comparisons and outcome evidence | Current state |
|---|---|---|---|
| 1 | FP4/FP8/FP16 efficiency map | Extend rectangular/small/large shapes and distributions; prepacked, online, transfers, allocation and reuse costs; model-derived inputs; held-out error-constrained selector; complete real application quality/latency. Preserve the existing six follow-up gates. | Complete within the combined bounded protocols: pilot, grid, application/strong references, outlier/cost/native configuration work and [held-out per-operation prediction](SELECTOR_REPORT.md). One held-out budget violation and selection overhead reject safe/useful generalization of the tested empirical rule; no guaranteed-error dispatcher is established |
| 2 | FP4 outlier treatment | Compare plain FP4 with corrected columns/blocks/residuals using distribution-derived and real activations; charge detection and correction costs; evaluate time at fixed error and error at fixed time. | Complete within the declared bounded protocol: [report](OUTLIER_REPORT.md), 598 configurations, two input/model cohorts, independent factors/output checks and fixed-error/fixed-time frontiers; negative cost results retained |
| 3 | Fused quantization and activation | Compare separate, fused and compiled library pipelines across shapes; include activation, quantization, GEMM and final output; verify arithmetic and application quality; retain resource-pressure regressions. | Complete within the declared protocol: [fusion checkpoint](FUSION_REPORT.md), native separate/fused controls, application quality/latency and compiled/graph references, thirty resource queries; counter access limitation retained |
| 4 | Data movement and configuration | Compare supported loading/pipelining strategies, tile sizes, buffering, warp roles and traversal; characterize register/shared-memory limits and cache reuse; validate configuration selection on held-out shapes. | Complete within the declared protocol: [pipeline report](PIPELINE_REPORT.md), sixteen attempted builds, twelve usable pipelines, nine traversal controls, cache-pressure/primed comparisons, resource/negative outcomes and held-out rules; no architecture-ceiling claim |
| 5 | Mixed-precision attention | Compare QK/PV precision combinations, lengths, head sizes and masks against strong attention baselines; separate attention arithmetic error from complete model quality, memory and latency. | Complete within the declared inference protocol: [attention report](ATTENTION_REPORT.md), 1,512 attention and 138 complete transformer configurations, two independent cohorts, native/fused SDPA and pinned SageAttention3 FP4 controls, full CPU witnesses/model quality, preparation/transfer costs and named memory boundaries; unit-FP8-scaling and generalization limits retained |
| 6 | High precision from low-precision products | Compare component counts/selection with native FP32/FP64 and available library emulation; include cancellation, exponent spread and long dots; retain rigorous reconstruction/error checks and complete preparation costs. | Complete within the finite-FP32-input product protocol: [precision report](PRECISION_REPORT.md), 1,008 configurations, two cohorts, exact integer witnesses/samples, native/library references, changing-input decisions and complete preparation/host costs; fallback and losing cases retained |
| 7 | Adaptive-precision linear solver | Compare fixed low/high and adaptive factorization/preconditioning; controlled conditioning and real matrices; measure time to target residual and forward error, iterations and failures. | Complete within the dense SPD protocol: [solver report](SOLVER_REPORT.md), 1,664 attempted / 1,612 available configurations, two cohorts, native FP64 Cholesky/LU and mixed-precision IR/GMRES/fallback controls, real structural matrices, complete output replay, original-system residual/forward-error failures, transfers/scaling/factor-reuse costs; residual-only adaptation does not certify forward error |
| 8 | Hybrid sparse/dense SpMM | Compare dense Tensor Core blocks plus sparse remainder with cuSPARSE and suitable strong alternatives; vary block size, fill, row ordering and reuse; charge format conversion and memory; include real sparse matrices. | Complete within the operator protocol: [SpMM report](SPMM_REPORT.md), 2,496 configurations and two cohorts, native CSR/BSR/Blocked-ELL/dense references, coalesced Tensor Core block packing plus FP32 remainder, real matrices, independently decoded formats/encoding hashes and full/sample witnesses, complete conversion/row-order/memory/reuse costs; active hybrid wins 0/52 online-B groups at error 0.001 |
| 9 | FFT and spectral filtering | Compare complete window/FFT/filter/inverse pipelines with cuFFT; vary length, batch and precision; test tones, impulse, noise and weak/strong components; measure spectral artifacts and complete time. | Complete within the complex one-dimensional filtering protocol: [FFT report](FFT_REPORT.md), 2,184 attempted / 1,876 available configurations across two cohorts, native cuFFT FP32/64/16/BF16 and fused SM120 cuFFTDx references, radix-8/16 HMMA proposal and strict FP32 control, independent spectral/stage witnesses, weak-component/overflow failures, preparation/transfer/memory/reuse costs; HMMA wins 0/182 captured groups at error 0.001, fused FP32 reference provides the positive latency result |
| 10 | Quantized vector search | Compare quantized candidate generation/refinement with exact and suitable indexed Faiss/cuVS baselines; vary query batch, candidate count and filtering; measure recall, latency and memory. | Complete within the declared static-filter top-10 protocol: [search report](SEARCH_REPORT.md), 3,750 full attempted / 3,510 available configurations across two independent cohorts, synthetic and genuine SIFT1M inputs, native FP4/FP8/FP16/BF16/strict-FP32 candidates plus original-FP32 refinement, resident Faiss/cuVS exact/indexed references, complete preparation/transfers/reuse and named memory scopes; all exact truth queries and retained outputs independently replayed. FP4 is infeasible on shifted inputs at the tested recall budgets; duplicate/missing reference outputs, unsupported filtered IVF-PQ, capture failures and opaque-index memory limits are retained |
| 11 | Streaming covariance and PCA | Compare centering/scaling/accumulation choices with stable FP64 references; include shifted, nearly dependent and real data; measure covariance, eigenvalues, subspace error, memory and total time. | Pending |
| 12 | Matrix-free physical operator | Implement Poisson/heat operator on regular high-order elements; compare CUDA and Tensor Core operators at several precisions; validate discretization error, complete solver convergence, memory and latency. | Pending |
| 13 | RT geometry queries | Implement an OptiX LiDAR/visibility or point-location workload and a CUDA comparator; vary geometry, queries and scene updates; include acceleration-structure build/update costs and boundary correctness. | Pending |
| 14 | Compressed CPU-to-GPU movement | Compare plain transfer with compress/transfer/GPU-decompress/compute, including nvCOMP reference; vary chunking, entropy, overlap and data exceeding VRAM; charge all preparation and buffer costs and check exact restoration. | Pending |
| 15 | Energy/latency mode selection | Sample actual GPU energy/power for a completed workload across precision, batching and kernel choices; include steady and irregular arrival patterns, idle periods and tail latency; evaluate mode selection. Board measurements do not establish whole-system energy. | Pending |

## Evidence status

Completed pilot evidence is in [RESULTS.md](RESULTS.md). Study completion will be
updated only when its actual required evidence and final report exist. Future
reports must distinguish bounded experimental conclusions from general claims.
