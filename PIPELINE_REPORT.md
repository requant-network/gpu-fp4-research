# Native SM120 pipelines, traversal and cache pressure

**Status: study 4's bounded configuration investigation is complete. See the [completed programme ledger](PROGRAM.md).**

Two cohorts contain 924 configurations, 166,320 retained observations and 655,097,856 output entries checked before and after timing. Eleven shapes and normal/outlier inputs cover twelve usable pipeline builds plus nine traversal configurations. Four additional compiled configurations fail their runtime resource request and remain in the evidence. Four small input/output witness files support CPU numerical replay. All configurations preserve identical FP4 encoded values/scales; configuration selection changes implementation rather than the quantization contract.

## Native controls

The frozen CUTLASS base is transformed only at tile M, mainloop stage count and schedule declarations. Tile M is 128 or 256, N/K tiles remain 128, and stage policies are automatic, 2, 3 or 4. Explicit cooperative and ping-pong NVFP4 schedules change consumer work assignment and buffering while retaining TMA loads. The pinned builder uses different MMA atom layouts for cooperative/ping-pong schedules. Automatic stages resolve to four for M=128 and two for M=256 in this environment.

A separate native library keeps M=128/automatic/cooperative fixed and compares heuristic traversal with along-M/along-N raster order and swizzle bounds 1, 2, 4 and 8. This is a controlled traversal comparison, not the full cross product of traversal and every pipeline build. Earlier [encoder/loading experiments](HARDWARE_REPORT.md) supply scalar/vector-load/warp-reduction controls. Only features supported by the pinned SM120 block-scaled implementation are tested; this is not a claim to exercise every feature of every NVIDIA architecture.

## Pipeline resources and unavailable requests

Function attributes and CUDA's occupancy calculator describe theoretical resource capacity. They do not measure active warps, dynamic spills, bandwidth or Tensor Core utilization. Local bytes are reported storage, not proof of executed spill traffic. Nsight Compute counters remain unavailable under the recorded driver permissions.

| Configuration | Resolved stages | Block threads | Registers/thread | Dynamic shared bytes | Local bytes/thread | Theoretical blocks/SM |
|---|---:|---:|---:|---:|---:|---:|
| m128-sauto-coop | 4 | 384 | 168 | 100352 | 0 | 1 |
| m128-sauto-ping | 4 | 384 | 168 | 100352 | 24 | 1 |
| m128-s2-coop | 2 | 384 | 168 | 63488 | 0 | 1 |
| m128-s2-ping | 2 | 384 | 168 | 63488 | 24 | 1 |
| m128-s3-coop | 3 | 384 | 168 | 81920 | 0 | 1 |
| m128-s3-ping | 3 | 384 | 168 | 81920 | 24 | 1 |
| m128-s4-coop | 4 | 384 | 168 | 100352 | 0 | 1 |
| m128-s4-ping | 4 | 384 | 168 | 100352 | 24 | 1 |
| m256-sauto-coop | 2 | 384 | 168 | 81920 | 16 | 1 |
| m256-sauto-ping | 2 | 384 | 168 | 81920 | 1160 | 1 |
| m256-s2-coop | 2 | 384 | 168 | 81920 | 16 | 1 |
| m256-s2-ping | 2 | 384 | 168 | 81920 | 1160 | 1 |
| m256-s3-coop | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| m256-s3-ping | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| m256-s4-coop | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| m256-s4-ping | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

The unavailable variants report `invalid argument` during the runtime resource query. Their successful build manifests and actual failure messages are retained in each cohort. They have no fabricated zero latency. Successful requests expose the substantial shared-memory/local-storage changes introduced by buffering and work assignment; resource counts alone do not establish the cause of a measured latency change.

## Fixed pipeline comparisons

Ratios span cohorts on normal inputs. Above 1 favors the named candidate. All pipeline ratios use primed prepacked GEMM. The final column divides cache-pressure latency by primed latency for the fixed default; above 1 means the pressure condition is slower.

| M x N x K | Default / stage2 cooperative | Default / stage4 ping-pong | Default / M256 stage2 cooperative | Default cache-pressure / primed |
|---|---:|---:|---:|---:|
| 16 x 512 x 512 | 1.000-1.000 | 0.750-0.750 | 0.600-0.600 | 1.333-1.333 |
| 32 x 768 x 1280 | 0.800-0.800 | 0.800-0.800 | 0.571-0.571 | 1.250-1.250 |
| 64 x 1024 x 1024 | 1.000-1.000 | 1.000-1.000 | 0.571-0.667 | 1.246-1.250 |
| 128 x 128 x 128 | 1.000-1.000 | 1.000-1.000 | 0.750-0.750 | 1.000-1.000 |
| 192 x 768 x 768 | 1.000-1.000 | 1.000-1.000 | 0.667-0.667 | 0.996-0.996 |
| 256 x 1024 x 1024 | 1.000-1.000 | 0.800-0.800 | 0.571-0.573 | 1.248-1.250 |
| 384 x 1536 x 512 | 1.000-1.000 | 1.000-1.000 | 0.800-0.800 | 0.996-0.996 |
| 512 x 512 x 512 | 1.000-1.000 | 0.750-0.750 | 0.600-0.600 | 1.328-1.331 |
| 768 x 1536 x 1024 | 1.000-1.000 | 1.000-1.000 | 1.000-1.000 | 1.143-1.285 |
| 1024 x 1024 x 1024 | 1.000-1.002 | 1.000-1.000 | 1.000-1.000 | 1.143-1.143 |
| 2048 x 2048 x 2048 | 0.944-0.944 | 1.063-1.063 | 1.000-1.000 | 1.029-1.029 |

## Held-out policies

For each operation/condition, one volume threshold and two candidate choices minimize training log latency regret. Only the first cohort's fourteen training shape/profile groups are used. Four rectangular shapes are excluded from fitting in both cohorts. These are configuration policies within the measured candidate set; cache condition must be supplied by the caller. Both input profiles occur in training, so this is not validation on unseen distributions.

| Operation/condition | Rule: at or below threshold / above | Held-out geometric-mean regret | Held-out speedup over default |
|---|---|---:|---:|
| online-a/cache-pressure | M*N*K <= 1073741824: raster-1-1 / m128-s4-ping | 1.001 | 1.002 |
| online-a/warm | M*N*K <= 268435456: raster-1-1 / m128-sauto-ping | 1.001 | 1.000 |
| online-both/cache-pressure | M*N*K <= 1073741824: raster-2-2 / m128-s4-ping | 1.023 | 0.995 |
| online-both/warm | M*N*K <= 1073741824: raster-2-8 / m128-s4-ping | 1.030 | 0.991 |
| prepacked/cache-pressure | M*N*K <= 134217728: m128-s4-coop / m128-s4-ping | 1.058 | 0.960 |
| prepacked/warm | M*N*K <= 268435456: m128-s2-coop / m128-sauto-ping | 1.058 | 0.946 |

Every held-out decision and regression is retained in [the policy data](results/pipeline-selector.json). [The complete CSV](results/pipeline-summary.csv) includes every traversal, pipeline, profile, boundary, condition and nearest-rank p10/p90. No per-test winner is substituted for the fitted rule.

## Timing and cache interpretation

Each graph contains conditioning, an external CUDA-event start node, one measured operation and an external end node. A primed sample executes an identical unmeasured operation before the start event. The pressure condition reads/writes a separate buffer four times reported L2 capacity before that event. Both conditioning costs are excluded from operation latency. Three measured boundaries are prepacked GEMM, online A and online both. Online preparation uses the frozen native float4 FP4 encoder; weights/scales are reused only where the boundary allows. Five warmups and thirty observations rotate all six graphs per configuration; the second cohort reverses groups/configurations.

The sweep applies cache pressure; without cache counters it does not prove every input line was evicted. Differences may also reflect conditioning/power state, and cannot be attributed exclusively to an L2 miss rate. [Empty-interval calibration](results/pipeline-timer-calibration.json) reports median 0.448 microseconds after warm conditioning and 0.448 microseconds after cache pressure. This descriptive event-node baseline is not subtracted from the results. Small differences and near-unity ratios should not be interpreted as a reliable optimization advantage. One-operation instrumented graph times are not directly interchangeable with earlier 32-repetition uninstrumented graph averages. Post-timing checks inspect current operands/output before recomputing anything, guarding against corruption during the timed graph.

Host transfers, allocations, plans, tuning/capture, conditioning and validation are outside the reported interval. This experiment deliberately studies device-resident reuse; changing-input host requests and setup/reuse costs are separately measured in [the application study](APPLICATION_REPORT.md). It does not establish streaming bandwidth, whole-request CPU latency, energy or the architecture's best achievable throughput. Other GPU contexts remain present; all measurements remain exploratory shared-system observations.

Build with `build_pipeline.py` for the declared sixteen combinations and `build_raster.py` for traversal/cache controls. Run `pipeline_study.py` for seeds 20261120 and 20261220, separate outputs and `--reverse` for the second cohort. Run `python verify_pipeline.py` and `python summarize_pipeline.py --check` for CPU-only artifact replay. The tested protocol closes the declared loading/pipeline/warp-role/buffering/traversal/resource/cache/held-out configuration comparison, not a claim of universal kernel optimality.
