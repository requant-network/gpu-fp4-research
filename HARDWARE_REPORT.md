# Native FP4 conversion and memory access

**Status: this bounded native-instruction experiment is complete. The full architecture/configuration investigation remains open.**

Two independent cohorts contain 680 configurations, 102,000 retained timing observations and 398,458,880 output entries checked before and after timing. All packed values and logical scale bytes match the independent CPU encoder. Eight small witnesses allow CPU-only numerical replay.

## What is native

GPU work is CUDA C++ compiled for SM120a. CUTLASS performs block-scaled FP4 Tensor Core GEMM; cuBLASLt supplies the other formats and an additional FP4 application baseline. Python/CuPy performs orchestration and memory management. Captured CUDA-event times exclude Python dispatch, while the separate application report retains changing-input host-request costs. A pure C++ request driver has not been measured.

The new encoders use `__nv_cvt_float2_to_fp4x2` with round-to-nearest, vector `float2`/`float4` loads and either subgroup shuffles or integer warp reduction of nonnegative absolute-value bit patterns. All retain the original scale calculation, range and byte layout. Comparisons therefore change implementation without changing numerical quality. Their advantage over original encoders is distinct from FP4-versus-FP16 comparisons, which do not guarantee equal accuracy.

## Fixed candidate comparisons

Ratios span the two cohorts on the normal profile. Above 1 favors the named candidate. All encoders use 128 threads per CUDA block and the same M=128 CUTLASS tile. The float2 and float4 candidates are fixed across shapes; this table does not select a winner using each measured test case.

| M x N x K | Original pair8 / native float2: prepare A | Original pair8 / native float4: prepare A | Original pair8 / native float4: online A | Native float4 separate / fused: online A |
|---|---:|---:|---:|---:|
| 16 x 512 x 512 | 1.27-1.32 | 1.09-1.12 | 1.02-1.03 | 1.10-1.11 |
| 32 x 768 x 1280 | 1.45-1.47 | 1.27-1.33 | 1.05-1.05 | 1.08-1.09 |
| 64 x 1024 x 1024 | 1.54-1.65 | 1.48-1.50 | 1.09-1.09 | 1.11-1.12 |
| 128 x 128 x 128 | 1.34-1.37 | 1.16-1.21 | 1.04-1.05 | 1.13-1.13 |
| 192 x 768 x 768 | 1.79-1.80 | 2.15-2.17 | 1.25-1.26 | 1.19-1.20 |
| 256 x 1024 x 1024 | 1.89-1.91 | 2.45-2.48 | 1.35-1.35 | 1.21-1.21 |
| 384 x 1536 x 512 | 1.82-1.83 | 2.23-2.27 | 1.32-1.33 | 1.22-1.23 |
| 512 x 512 x 512 | 1.92-1.94 | 2.47-2.49 | 1.43-1.44 | 1.25-1.25 |
| 768 x 1536 x 1024 | 2.11-2.11 | 3.13-3.17 | 1.55-1.56 | 1.23-1.23 |
| 1024 x 1024 x 1024 | 2.14-2.15 | 3.28-3.29 | 1.70-1.71 | 1.27-1.27 |
| 2048 x 2048 x 2048 | 2.19-2.20 | 3.53-3.54 | 1.24-1.25 | 1.52-1.54 |

The native integer `REDUX.MAX` alternative is slower than the shuffle-based float4 encoder at 2048 cubed: shuffle/redux latency ratio 0.761-0.761. Both produce identical bytes. A dedicated instruction does not guarantee a faster complete kernel.

The complete [CSV](results/hardware-summary.csv) retains scalar/native/reduction alternatives, tiny and midpoint-tie inputs, outliers, separate/fused activation and all five boundaries, including regressions. Preparation-only acceleration need not produce the same complete online-GEMM acceleration. Warm graphs reuse operands, so this experiment does not establish streaming DRAM throughput.

## Emitted instructions and resources

`cuobjdump` of the measured binary confirms FP4 `OMMA.SF...E2M1` Tensor Core operations and `UTMALDG`/`UTMASTG` TMA traffic in GEMM. New encoders contain `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`; variant 13 also contains `REDUX.MAX`. Original encoders retain their software FP4 conversion path. Static instruction counts include branches and fallback code; they are not executed instruction counts.

| Encoder | Registers/thread | Static shared bytes | Local bytes | Theoretical thread occupancy at 128 threads |
|---|---:|---:|---:|---:|
| original scalar16 | 38 | 0 | 0 | 1.000 |
| original pair8 | 22 | 0 | 0 | 1.000 |
| native scalar16 | 34 | 0 | 0 | 1.000 |
| native float2/shuffle8 | 21 | 0 | 0 | 1.000 |
| native float4/shuffle4 | 23 | 0 | 0 | 1.000 |
| native float4/redux4 | 25 | 0 | 0 | 1.000 |

Resources come from `cudaFuncGetAttributes` and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for the unactivated A encoder. This is a register/shared-memory capacity calculation, not measured active warps or achieved throughput. Fused activation can change resources. The [hardware inspection](results/native-inspection-hardware.json) retains every kernel's static instructions/resources and the exact measured binary hash. [Tile128](results/native-inspection-128.json) and [tile256](results/native-inspection-256.json) retain earlier emitted-code evidence.

## Profiler availability

Nsight Compute 2026.2.1 was actually invoked on a separate 2048-cubed encoder probe with LaunchStats, Occupancy and SpeedOfLight sections. The installed driver returned `ERR_NVGPUCTRPERM`: the current account cannot access performance counters. The exact failure, tool version, source and binary hashes are retained in [profiler evidence](results/native-profiler.json). No achieved occupancy, bandwidth, Tensor Core utilization or counter-derived bottleneck is reported. Profiling permissions were not changed. This limits profiling evidence but does not invalidate the separately timed, numerically checked campaigns.

## Limits and next architectural questions

These are shared-system measurements with other GPU contexts present; clocks, power settings and services were not changed. No claim is made that every available architectural feature is used or that the best achievable throughput has been reached. Explicit loading/pipeline controls, buffering/warp roles, traversal, cold-cache/streaming behavior and held-out configuration selection beyond the earlier encoder rule remain open. Relevant mechanisms must improve a particular workload under its correctness contract; enabling more mechanisms alone is not evidence of a useful speedup.

Run `python verify_hardware.py` and `python summarize_hardware.py --check`. GPU reproduction: after the extended builds, run `bash build_hardware.sh ../cutlass-fp4`; run `hardware_study.py --seed 20261118` and a separate reversed cohort with seed 20261218/output path. Run `inspect_native.py` separately for each library with the recorded CUDA toolkit.
