# Matrix-free Poisson and heat on consumer Blackwell

This is a bounded measured study, not a hardware ceiling, novelty claim or a quality-safe dispatcher. External context is in [PDE_SOURCES.md](PDE_SOURCES.md); executable details and limits are in [PDE_REPRODUCTION.md](PDE_REPRODUCTION.md).

## Executed scope and numerical closure

The full matrix contains 24 independent cohort/geometry collections and 384 declared configurations. Including separate smoke, fresh CPU replay covered 400 attempted / 382 available configurations, 11840 operator observations, 5920 complete Poisson solve observations, 3072 heat trajectories and 18176 retained heat steps. It checked 11858 distinct full final-vector records and 430 complete retained GEMM witnesses.

Each full geometry has two independent sine-mode cohorts; E=1/2/4/8, q=4/8/16. Changing smooth and rough inputs include nonzero operator boundaries. Heat is explicitly bounded to E=1/2, q=4/8/16, final time0.01 and4/8 steps. The remaining larger geometries cover operators and Poisson, not unexecuted large heat trajectories.

Direct FP32/64 CUDA, pinned libCEED cuda/ref/shared/gen, grouped native FP4/FP8/FP16/BF16/strict-FP32, and ungrouped short-q cuBLAS references were actually attempted. Half/BF16 candidates require recorded eligibility flags; numerical flags/API types/root resources do not constitute an instruction trace. Short-q controls do not inherit the proposal's artificial128-wide padding.

## Complete online operator at fixed error

Every table choice uses the worst independently checked error among all observations in that timing route, and the fastest eligible non-FP4 implementation. Effective cost includes common benchmark preparation and construction/source load/calibration/first use divided by the stated reuse, plus median complete operation time. Common preparation includes analytic reference-field generation: this is benchmark-inclusive, not minimal deployment initialization.

| Boundary | Error budget | Reuse | FP4 feasible groups | Reference feasible groups | FP4 wins |
|---|---:|---:|---:|---:|---:|
| resident | 0.001 | 10000 | 0/24 | 24/24 | 0/24 |
| resident | 0.01 | 10000 | 0/24 | 24/24 | 0/24 |
| resident | 0.1 | 10000 | 2/24 | 24/24 | 0/24 |
| host | 0.001 | 10000 | 0/24 | 24/24 | 0/24 |
| host | 0.01 | 10000 | 0/24 | 24/24 | 0/24 |
| host | 0.1 | 10000 | 2/24 | 24/24 | 0/24 |

The complete tables also include reuse1/100. Padding, dynamic source packing, quantization, actual native product, all-axis weighting, deterministic global assembly and context acknowledgment are not replaced by a prepacked microbenchmark. Host service includes H2D and D2H; its CUDA event span ends before host D2H and is not the same boundary as host wall time.

## Poisson: original residual and continuum forward error

Recursive convergence of an input-dependent/nonsymmetric low operator is not physical convergence. Fixed runs retain this distinction. Reliable runs pay for bounded low-operator defect corrections, true residual tests, rejected progress and FP64 fallback. FP4-assisted/fallback rows are not pure-FP4 solves.

The full data retain 1628 Poisson observation(s) missing their nominal original residual target and 172 FP4-assisted Poisson observations using fallback. No failed repetition is silently dropped from a configuration's frontier eligibility.

| Boundary | Residual budget | Continuum mass-L2 budget | Reuse | FP4-assisted/fixed feasible | Reference feasible | FP4-labelled wins |
|---|---:|---:|---:|---:|---:|---:|
| resident | 0.001 | 0.01 | 10000 | 19/24 | 21/24 | 0/24 |
| resident | 1e-08 | 0.01 | 10000 | 19/24 | 21/24 | 0/24 |
| host | 0.001 | 0.01 | 10000 | 19/24 | 21/24 | 0/24 |
| host | 1e-08 | 0.01 | 10000 | 19/24 | 21/24 | 0/24 |

## Discretization and temporal refinement

The following independently replayed strict fixed-CUDA-FP64 controls show continuum mass-L2 error, not merely an internally small algebraic residual. At high order, iterative stopping and floating-point error can dominate the remaining spatial discretization error; monotonic convergence is not asserted below that floor.

| Cohort | E | q | Worst original residual | Worst continuum mass-L2 error |
|---|---:|---:|---:|---:|
| a | 1 | 4 | 2.141778e-16 | 0.01303956 |
| a | 2 | 4 | 1.18588e-15 | 0.0006494484 |
| a | 4 | 4 | 6.08952e-15 | 2.271411e-05 |
| a | 8 | 4 | 2.208934e-14 | 7.278705e-07 |
| a | 1 | 8 | 7.896608e-15 | 1.894045e-07 |
| a | 2 | 8 | 5.348962e-09 | 1.636015e-09 |
| a | 4 | 8 | 7.245568e-09 | 5.336853e-11 |
| a | 8 | 8 | 8.504597e-09 | 1.813636e-11 |
| a | 1 | 16 | 7.838373e-09 | 1.236693e-10 |
| a | 2 | 16 | 8.772635e-09 | 2.07999e-10 |
| a | 4 | 16 | 7.787941e-09 | 3.218917e-11 |
| a | 8 | 16 | 7.996211e-09 | 7.88523e-12 |
| b | 1 | 4 | 1.065891e-16 | 1.680864 |
| b | 2 | 4 | 1.308054e-15 | 0.0605276 |
| b | 4 | 4 | 7.121209e-09 | 0.003099077 |
| b | 8 | 4 | 8.102881e-09 | 0.0001017411 |
| b | 1 | 8 | 2.688511e-09 | 0.003657159 |
| b | 2 | 8 | 4.090404e-09 | 1.656554e-05 |
| b | 4 | 8 | 7.762741e-09 | 3.627192e-08 |
| b | 8 | 8 | 7.971073e-09 | 1.553481e-10 |
| b | 1 | 16 | 6.287105e-09 | 1.913364e-09 |
| b | 2 | 16 | 9.286973e-09 | 9.690416e-10 |
| b | 4 | 16 | 9.247177e-09 | 5.373842e-10 |
| b | 8 | 16 | 8.694172e-09 | 8.274099e-11 |

Heat tables retain each step state and RHS and validate RHS = M*previous_state/dt against the preceding retained vector. Continuum decay and sampled continuum backward-Euler targets are separate, exposing time discretization rather than equating a solved implicit system with the exact heat PDE. Frontier step residuals are recomputed from every retained state/RHS, not taken from scalar GPU claims. Frontiers require every step residual and final continuum forward error and may choose between the actually executed 4/8-step trajectories.

| Dataset | Steps | Maximum step residual | Continuum mass-L2 error | Error against continuum BE target |
|---|---:|---:|---:|---:|
| a-e1-q4 | 4 | 6.064554e-17 | 0.006828696 | 0.003633984 |
| a-e1-q4 | 8 | 1.291584e-16 | 0.00157759 | 0.003764261 |
| a-e2-q4 | 4 | 5.631665e-10 | 0.01052485 | 0.000465411 |
| a-e2-q4 | 8 | 4.302338e-09 | 0.005398345 | 0.0004888059 |
| a-e1-q8 | 4 | 1.65826e-09 | 0.01050084 | 1.663075e-07 |
| a-e1-q8 | 8 | 6.267415e-09 | 0.005362035 | 1.831392e-07 |
| a-e2-q8 | 4 | 4.172605e-09 | 0.01050084 | 3.065198e-09 |
| a-e2-q8 | 8 | 5.261417e-09 | 0.005362036 | 7.359424e-09 |
| a-e1-q16 | 4 | 7.980944e-09 | 0.01050084 | 4.653204e-09 |
| a-e1-q16 | 8 | 8.586193e-09 | 0.005362036 | 3.392483e-09 |
| a-e2-q16 | 4 | 9.443818e-09 | 0.01050084 | 3.415621e-09 |
| a-e2-q16 | 8 | 9.94945e-09 | 0.005362036 | 7.620412e-09 |
| b-e1-q4 | 4 | 8.923691e-17 | 1.320913 | 0.9483076 |
| b-e1-q4 | 8 | 2.056438e-16 | 1.289349 | 1.080724 |
| b-e2-q4 | 4 | 5.103955e-09 | 0.2383739 | 0.06010021 |
| b-e2-q4 | 8 | 1.322375e-09 | 0.1481426 | 0.06681127 |
| b-e1-q8 | 4 | 7.659991e-09 | 0.2058686 | 0.004081265 |
| b-e1-q8 | 8 | 3.331709e-09 | 0.1066888 | 0.00450431 |
| b-e2-q8 | 4 | 8.280975e-09 | 0.2089056 | 1.885377e-05 |
| b-e2-q8 | 8 | 7.565249e-09 | 0.1098098 | 1.939737e-05 |
| b-e1-q16 | 4 | 9.121471e-09 | 0.2089056 | 1.092981e-08 |
| b-e1-q16 | 8 | 8.922668e-09 | 0.1098098 | 2.329457e-08 |
| b-e2-q16 | 4 | 9.961207e-09 | 0.2089056 | 5.562938e-09 |
| b-e2-q16 | 8 | 9.159662e-09 | 0.1098098 | 8.345791e-09 |

The strict FP64 controls distinguish the temporal error reduced by refinement from persistent under-resolved spatial error. No monotonic low-precision or unresolved-mesh convergence claim is inferred from a small recursive residual.

| Boundary | Residual budget | Continuum mass-L2 budget | Reuse | FP4-labelled feasible | Reference feasible | FP4-labelled wins |
|---|---:|---:|---:|---:|---:|---:|
| resident | 1e-08 | 0.1 | 10000 | 6/12 | 6/12 | 0/12 |
| resident | 1e-08 | 0.01 | 10000 | 6/12 | 6/12 | 0/12 |
| host | 1e-08 | 0.1 | 10000 | 6/12 | 6/12 | 0/12 |
| host | 1e-08 | 0.01 | 10000 | 6/12 | 6/12 | 0/12 |

The full heat data retain 4510 step observation(s) missing their nominal original residual target. Fixed low-precision misses remain negative evidence, not successful physical acceleration.

## Method-local memory categories

The following E2/q8 sample excludes shared benchmark input/reference buffers. Library request lengths remain separate, not invented opaque-owned-byte totals. Temporary arrays, reduction buffers, allocator/JIT caches and physical peaks are not claimed audited.

| Implementation | Python named bytes | Native named bytes | CEED qdata request bytes | CEED index request bytes |
|---|---:|---:|---:|---:|
| 0:cuda:fp32 | 117719 | 0 | not measured | not measured |
| 3:cuda:fp64 | 134103 | 0 | not measured | not measured |
| 5:ceed:fp64 | 84375 | not measured | 131072 | 16384 |
| 7:tensor:fp4 | 330711 | 20480 | not measured | not measured |
| 9:tensor:fp16 | 330711 | 33619968 | not measured | not measured |
| 13:small:fp16 | 232663 | 33579136 | not measured | not measured |

## Retained limits and negatives

There are 18 full configuration availability failures. The pinned cuda/ref q16 basis exceeds its static shared-allocation limit, while optimized shared/gen operators are retained. The largest grouped geometry encounters the original native bridge N<=8192 guard. These are respectively a specific kernel/compiler allocation failure and an implementation guard, not a blanket GPU/Tensor Core limitation. Unavailable inherited heat combinations are not separately executed heat failures.

All negative records are in results/pde-negative-results.json. CUDA, native, libCEED and short-q setup and allocation/request scopes remain visible in master tables. Opaque libCEED memory is not ranked against another family's fuller named accounting; no physical peak-memory saving is established.

This common host-controlled Jacobi-PCG experiment stages each callback and performs FP64-vector reductions; it is not a best-of-all fully GPU-resident solver claim. First use follows explicit calibration, and solver measurements follow operator warmup. Import/device-discovery, warm disk/JIT/allocator history, diagnostics between requests and shared-device activity prevent cold-system/exclusive/saturated-throughput claims.

Every timed final vector is retained losslessly, including nondeterministic variants. Numeric replay reconstructs original-data encoding/products/post-processing and final original-system residuals. It does not certify every unretained CG intermediate or measure board/system energy. Full source30,4binary,fixture pre/post and child-exit closures are retained; exact numerical/source/cost receipts are prerequisites for these tables.

## Artifacts

- [Operator observations and costs](results/pde-operator.csv)
- [Poisson solve observations and quality](results/pde-poisson.csv)
- [Heat trajectories and quality](results/pde-heat.csv)
- [Operator frontiers](results/pde-operator-frontiers.csv)
- [Poisson frontiers](results/pde-poisson-frontiers.csv)
- [Heat frontiers](results/pde-heat-frontiers.csv)
- [Numeric replay](results/pde-replay.json)
- [Named cost audit](results/pde-cost-audit.json)
- [Full launch receipt](results/pde-full-launch.json)
- [Summary and negatives](results/pde-summary.json)
