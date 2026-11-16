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
| 1 | FP4/FP8/FP16 efficiency map | Extend rectangular/small/large shapes and distributions; prepacked, online, transfers, allocation and reuse costs; model-derived inputs; held-out error-constrained selector; complete real application quality/latency. Preserve the existing six follow-up gates. | Pilot, encoder grid, application/cost/reference and held-out policies complete; outlier/error and native-resource gates open |
| 2 | FP4 outlier treatment | Compare plain FP4 with corrected columns/blocks/residuals using distribution-derived and real activations; charge detection and correction costs; evaluate time at fixed error and error at fixed time. | Pending; real-model activations and task baselines retained |
| 3 | Fused quantization and activation | Compare separate, fused and compiled library pipelines across shapes; include activation, quantization, GEMM and final output; verify arithmetic and application quality; retain resource-pressure regressions. | Native fusion, application quality and compiled/graph references complete; resource characterization open |
| 4 | Data movement and configuration | Compare supported loading/pipelining strategies, tile sizes, buffering, warp roles and traversal; characterize register/shared-memory limits and cache reuse; validate configuration selection on held-out shapes. | Encoder/tile comparisons and held-out encoder rule complete; loading/pipeline/profiling gates open |
| 5 | Mixed-precision attention | Compare QK/PV precision combinations, lengths, head sizes and masks against strong attention baselines; separate attention arithmetic error from complete model quality, memory and latency. | Pending |
| 6 | High precision from low-precision products | Compare component counts/selection with native FP32/FP64 and available library emulation; include cancellation, exponent spread and long dots; retain rigorous reconstruction/error checks and complete preparation costs. | Pending |
| 7 | Adaptive-precision linear solver | Compare fixed low/high and adaptive factorization/preconditioning; controlled conditioning and real matrices; measure time to target residual and forward error, iterations and failures. | Pending |
| 8 | Hybrid sparse/dense SpMM | Compare dense Tensor Core blocks plus sparse remainder with cuSPARSE and suitable strong alternatives; vary block size, fill, row ordering and reuse; charge format conversion and memory; include real sparse matrices. | Pending |
| 9 | FFT and spectral filtering | Compare complete window/FFT/filter/inverse pipelines with cuFFT; vary length, batch and precision; test tones, impulse, noise and weak/strong components; measure spectral artifacts and complete time. | Pending |
| 10 | Quantized vector search | Compare quantized candidate generation/refinement with exact and suitable indexed Faiss/cuVS baselines; vary query batch, candidate count and filtering; measure recall, latency and memory. | Pending |
| 11 | Streaming covariance and PCA | Compare centering/scaling/accumulation choices with stable FP64 references; include shifted, nearly dependent and real data; measure covariance, eigenvalues, subspace error, memory and total time. | Pending |
| 12 | Matrix-free physical operator | Implement Poisson/heat operator on regular high-order elements; compare CUDA and Tensor Core operators at several precisions; validate discretization error, complete solver convergence, memory and latency. | Pending |
| 13 | RT geometry queries | Implement an OptiX LiDAR/visibility or point-location workload and a CUDA comparator; vary geometry, queries and scene updates; include acceleration-structure build/update costs and boundary correctness. | Pending |
| 14 | Compressed CPU-to-GPU movement | Compare plain transfer with compress/transfer/GPU-decompress/compute, including nvCOMP reference; vary chunking, entropy, overlap and data exceeding VRAM; charge all preparation and buffer costs and check exact restoration. | Pending |
| 15 | Energy/latency mode selection | Sample actual GPU energy/power for a completed workload across precision, batching and kernel choices; include steady and irregular arrival patterns, idle periods and tail latency; evaluate mode selection. Board measurements do not establish whole-system energy. | Pending |

## Evidence status

Completed pilot evidence is in [RESULTS.md](RESULTS.md). Study completion will be
updated only when its actual required evidence and final report exist. Future
reports must distinguish bounded experimental conclusions from general claims.
