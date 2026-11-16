# Adaptive precision for dense SPD linear systems

**Status: study 7 is complete within the protocol below. Studies 8-15 remain part of the active fifteen-study program.**

Two cohorts attempted 1,664 configurations; 1,612 are available and 52 explicitly reject multi-RHS GMRES. The campaign retains 96,720 solve timing observations, 3,861 complete output witnesses (9,953,817 FP64 entries) and 142,992 exact-integer residual samples. Every measured output is copied and linked to its complete retained witness.

## Contract and inputs

All methods start from the same stored FP64 A/B and return FP64 X. A solve passes only if every RHS has relative L2 residual <=1e-12 AND relative L2 forward error <=1e-8. Eligibility in the tables requires passing every retained observation for that boundary, including all four changing RHS variants for host requests. Forward error is measured against a mathematically exact manufactured solution of the stored system; it is not measured against another rounded GPU solver.

Controlled matrices use a signed and row-permuted Walsh basis, dimensions 128/512/1024/2048, and exact spectral condition numbers 1, 2^12, 2^24, 2^36 and 2^40. Integer Walsh transforms of dyadic eigenvalues construct every A entry exactly in FP64. RHS vectors combine three basis directions, including smallest/largest eigenvalues; B and truth are exactly representable. The four RHS variants keep A unchanged. Cases have one or four RHS columns. The second cohort independently changes the matrix/RHS seed and reverses matrix/method order.

Real inputs are [BCSSTK09](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc1/bcsstk09.html), a clamped plate stiffness matrix (1083), and [BCSSTK10](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc1/bcsstk10.html), a hot washer stiffness matrix (1086), supplied by John Lewis to the Harwell-Boeing collection and distributed by NIST Matrix Market. Each is tested at exact global scales 2^-20, 1 and 2^20 with one/four RHS columns. Truth consists of selected unit vectors and B of the corresponding original columns, so no RHS-rounding ambiguity is hidden. The two cohorts change real-matrix RHS directions, not the underlying two physical matrices. These are dense-storage precision tests with manufactured loads, not original physical-load accuracy tests or sparse-solver comparisons. Original compressed files, URLs, hashes and attribution are in [the data manifest](fixtures/solver-matrices.json).

## Native methods and costs

The native bridge directly calls CUDA 13.3 cuSOLVER on GPU 0 / SM120. Strong FP64 references include SPD Cholesky (Dpotrf/Dpotrs) and general LU. Other references request FP32/FP16/BF16/TF32 factorization with no refinement or classical iterative refinement, FP16 GMRES for one RHS, and the library's FP16 classical-refinement fallback. Equilibrated controls compute D=diag(A)^-1/2, solve DAD y=Db, and return x=Dy; their external residual is always evaluated against original A/B. Requested lowest-precision enums are configuration evidence, not proof that every opaque library operation uses Tensor Cores. No FP4 factorization capability is asserted.

The custom adaptive policy attempts FP16 classical refinement with limit 4, FP32 with limit 12, then FP64 Cholesky. Each attempt starts from original inputs, charges restore/scaling and gates on native status/device info plus full original-system relative L2 residual <=1e-12. An equilibrated version applies Jacobi scaling to the two lower-precision attempts. The online policy never sees the manufactured truth, condition label or forward error. Raw negative iteration/fallback/nonconvergence codes are retained and are not interpreted as negative computational work.

Resident timers include fresh factorization, solve, full FP64 original-system residual, scalar readback and host decisions. CUDA events span library host orchestration gaps; separate host wall times are retained. Complete host-request timers additionally include pageable FP64 A+B H2D and full FP64 X D2H, cycling through four RHS variants. Plan creation, allocation, CPU input generation, numerical replay and history extraction are outside solve timers; plan setup wall time and explicit owned device bytes remain in the CSV. Opaque cuSOLVER/cuBLAS handle/internal allocation costs are additional, so owned bytes are not total peak memory. Thirty observations follow five resident warmups, without changing clocks, services or power settings.

## Accuracy-constrained outcomes

A ratio above 1 favors the named method. Ratios use only cases where BOTH the candidate and comparator pass the full contract. Best-library comparisons include the fastest feasible direct/IR/fallback reference in the same group and boundary. Failed solves retain their timings in the CSV but cannot earn an accuracy-constrained speedup.

| Method | Resident passing groups | Host passing groups | Median resident speedup vs fresh FP64 Cholesky | Median resident speedup vs best feasible library |
|---|---:|---:|---:|---:|
| cholesky64 | 72/104 | 72/104 | 1.0000 | 0.5999 |
| lu64 | 72/104 | 72/104 | 0.6730 | 0.4206 |
| fixed_fp32 | 16/104 | 16/104 | 2.2725 | 1.0000 |
| fixed_fp16 | 16/104 | 16/104 | 2.0424 | 0.9163 |
| fixed_bf16 | 16/104 | 16/104 | 2.0537 | 0.9278 |
| fixed_tf32 | 16/104 | 16/104 | 2.1017 | 0.9474 |
| ir_fp32 | 66/104 | 65/104 | 1.6883 | 1.0000 |
| ir_fp16 | 48/104 | 45/104 | 1.3377 | 0.7951 |
| ir_bf16 | 60/104 | 60/104 | 1.1958 | 0.7367 |
| ir_tf32 | 60/104 | 60/104 | 1.3760 | 0.8315 |
| gmres_fp16 | 25/52 | 23/52 | 1.0223 | 0.5965 |
| library_fallback_fp16 | 72/104 | 71/104 | 1.0288 | 0.6474 |
| equilibrated_library_fp16 | 72/104 | 72/104 | 1.1839 | 0.7475 |
| adaptive | 66/104 | 65/104 | 0.7644 | 0.4570 |
| equilibrated_adaptive | 68/104 | 64/104 | 1.1780 | 0.7854 |
| cholesky64_reuse | 72/104 | 72/104 | unavailable | unavailable |

Complete host requests include changing B and both transfers. The ratios below retain the same strict feasibility condition; they are separate from resident timings.

| Method | Median host speedup vs fresh FP64 Cholesky | Median host speedup vs best feasible library |
|---|---:|---:|
| ir_fp32 | 1.3984 | 1.0000 |
| ir_fp16 | 1.2420 | 0.8774 |
| library_fallback_fp16 | 1.0038 | 0.7214 |
| equilibrated_library_fp16 | 1.1759 | 0.8260 |
| adaptive | 0.8207 | 0.6112 |
| equilibrated_adaptive | 1.1638 | 0.8459 |

## Controlled conditioning and forward-error failures

High conditioning can defeat even native FP64 at the fixed 1e-8 forward-error target. This is retained as an unmet accuracy outcome, not silently relaxed. The following resident success counts span dimensions, RHS counts and both seeds.

| Condition number | Cholesky64 | FP32 IR | FP16 IR | Adaptive | Equilibrated adaptive |
|---|---:|---:|---:|---:|---:|
| 2^0 | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 |
| 2^12 | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 |
| 2^24 | 16/16 | 10/16 | 4/16 | 10/16 | 12/16 |
| 2^36 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| 2^40 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |

There are 2,251 changing-RHS adaptive observations whose online residual gate passes while the offline combined accuracy contract fails. Repeated observations of a RHS are not independent failure trials. The [complete failure list](results/solver-frontiers.json) keeps seed/specification, method, RHS variant and exact witness links. The tested residual-only gate therefore does not certify forward error. A trusted condition estimate or stronger certified stopping rule is future work, not an established capability.

| Seed | Controlled n / condition | Method | RHS variant | GPU relative residual | Actual forward error |
|---|---|---|---:|---:|---:|
| 20261124 | 128 / 2^24 | adaptive | 0 | 8.0363e-15 | 7.7838e-08 |
| 20261124 | 128 / 2^24 | equilibrated_adaptive | 1 | 9.6322e-13 | 2.6061e-06 |
| 20261224 | 2048 / 2^40 | equilibrated_adaptive | 0 | 4.771e-16 | 4.8236e-05 |
| 20261224 | 2048 / 2^40 | adaptive | 0 | 4.771e-16 | 4.8236e-05 |

## Real-matrix results at original scale

Reported times are resident milliseconds. A failed candidate is labeled failed rather than included in an equal-accuracy comparison. Global-scale variants and complete host requests are in the CSV.

| Seed | Matrix / RHS | Cholesky64 | FP32 IR | Equilibrated library FP16 | Adaptive | Equilibrated adaptive |
|---|---|---:|---:|---:|---:|---:|
| 20261124 | bcsstk09 / 1 | 3.1972 | 1.8944 | 2.5233 | 4.2871 | 2.3938 |
| 20261124 | bcsstk09 / 4 | 4.9163 | 3.5118 | 4.2259 | 7.7415 | 4.2230 |
| 20261124 | bcsstk10 / 1 | 3.2032 | 1.7758 | 2.1713 | 4.1891 | 2.1750 |
| 20261124 | bcsstk10 / 4 | 4.9980 | 3.0937 | 4.2286 | 7.3327 | 4.2286 |
| 20261224 | bcsstk10 / 4 | 5.0009 | 2.6924 | 4.2523 | 6.9525 | 4.2528 |
| 20261224 | bcsstk10 / 1 | 3.2032 | 1.7995 | 2.1950 | 4.2446 | 2.1927 |
| 20261224 | bcsstk09 / 4 | 4.9665 | 3.1390 | 5.1519 | 7.4171 | 7.4567 |
| 20261224 | bcsstk09 / 1 | 3.2126 | 1.9354 | 2.4434 | 4.3841 | 2.4393 |

## Factor reuse and verification

`cholesky64_reuse` factors A once and solves changing RHS with the retained factor. The factor setup is separately measured thirty times; host setup includes A+B upload. [Amortization records](results/solver-frontiers.json) report factor_setup/R + solve for R=1/4/16/64. The setup restores a B work buffer even though factorization does not use it, so its small copy cost is conservatively included. Reuse solves still charge the original-system residual and host gate; host requests upload only B. This separate fixed-A workload cannot be mixed into the fresh-factorization frontier.

| Reused RHS request count | Median resident amortized speedup vs fresh Cholesky | Median host amortized speedup vs fresh Cholesky |
|---|---:|---:|
| 1 | 0.9997 | 0.9993 |
| 4 | 2.4413 | 2.4382 |
| 16 | 3.9158 | 3.9870 |
| 64 | 4.6464 | 4.8105 |

Every unique actual output is retained in full; every timed observation names its exact solution hash/witness. CPU replay reconstructs all inputs and checks every output's residual and analytic forward error. Sixteen rows per system/RHS are also evaluated by arbitrary-precision integer dot products with exact binary64 input representations and one final rounding, exposing ordinary FP64 residual cancellation. Cross-platform CPU residual comparisons use a gamma_n dot-rounding bound, while exact integer samples must match bit-for-bit and accuracy pass/fail labels must remain unchanged. Runtime preparation checks compare every scaled matrix/RHS/diagonal entry against independent NumPy operations; CI checks their provenance/counts and independently replays complete solution witnesses.

Native controls cover invalid dimensions/RHS/mode, null plan, unsupported multi-RHS GMRES, singular Cholesky and repeated IRS calls. A first quick attempt reused IRSInfos and produced ALLOC_FAILED on subsequent calls; its source/binary/raw outputs were archived privately. Final campaigns destroy/create this per-solve object inside the timed operation and include a four-call regression control. History is requested only for refinement and retained only for successful positive iteration counts with a defined prefix; unrefined/zero/negative-code calls do not claim initialized histories. The separate sixty-case capability preflight is replayed but is not counted as a performance study or assumed to meet this study's stricter targets.

The implementation follows the [NVIDIA cuSOLVER API](https://docs.nvidia.com/cuda/archive/13.3.1/cusolver/index.html), with explicit refinement, precision, fallback and iteration limits. Library history entries are its own residual norms/iteration counts, not the external relative L2 metric. Library default stopping tolerances are left unchanged; the external original-system gate is a separate check. GPU contexts from other processes remain present, so measurements are exploratory shared-system observations. No scientific-novelty, architecture-ceiling, universal solver or certified forward-error claim is made.

## Reproduction

```sh
bash build_solver.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python solver_study.py --seed 20261124 --quick --output solver-quick.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python solver_study.py --seed 20261124 --output solver-a.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python solver_study.py --seed 20261224 --reverse --output solver-b.json
python verify_solver.py
python summarize_solver.py --check
```

Use the recorded CUDA 13.3 toolchain and NumPy environment. The [complete table](results/solver-summary.csv), [first cohort](results/solver-a.json) and [second cohort](results/solver-b.json) retain setup, failures, raw iteration/status codes, histories, timing series, source/data/binary hashes and output witnesses.
