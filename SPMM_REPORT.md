# Hybrid sparse/dense SpMM on consumer Blackwell

**Status: study 8 is complete within the operator protocol below. See the [completed programme ledger](PROGRAM.md).**

Two independent input cohorts attempted 2,496 configurations; 2,496 ran. There are 299,520 retained timing observations, 30,882 distinct input/output check records covering 8,794,996,992 full output entries at runtime, 1,976,448 retained numerical samples and 1,181 complete numerical witnesses (25,598,976 FP32 entries) for independent CPU replay.

At error 0.001 the active hybrid wins 0/52 online-B groups against the strongest feasible native reference. The tested decomposition therefore does not establish a general device-speed benefit. Native blocked/dense libraries are essential comparisons; setup and gathered/intermediate storage remain real costs. The few host-request ratios above one include dispatch/transfers and do not establish a faster device kernel.

## Question and arithmetic contract

Can dense Tensor Core blocks plus a sparse remainder outperform strong native sparse/block/dense libraries after encoding, gather/scatter, row ordering, conversion and reuse costs? A/B start as the same finite FP32 data and every path returns the original unpermuted FP32 matrix product. Fixed-error choices require BOTH relative Frobenius error and maximum per-column relative L2 error <=1e-5, 1e-3 or 1e-2 against an independent FP64 product of the original FP32 values. One exactly zero B column must produce exactly zero output. Different precision paths are compared only where they meet the same error budget.

CSR references use algorithms 1/2/3 with the documented column-major preference for algorithm 1 and row-major for algorithms 2/3, in FP32/FP16/BF16. Block references include cuSPARSE BSR and Blocked-ELL at block sizes 16/32/64 and those three storage precisions. Dense cuBLASLt FP32/FP16/BF16 is also included; FP32 requests pedantic compute. Up to eight cuBLASLt candidates are attempted and three calibration timings per usable candidate select the fastest outside measured operations. The custom dense-block and full dense low-precision cuBLASLt calls must report the HMMA numerical implementation flag. Library flags and actual SM120 execution establish the selected native implementation, not achieved Tensor Core utilization or an architecture ceiling.

The custom hybrid selects every occupied block whose stored nonzero count reaches 25%, 50% or 75% of its full padded 16/32/64 square. It keeps the exact remaining entries in FP32 CSR, encodes selected blocks/B slices as FP16 or BF16, computes batched cuBLASLt products and adds them to the native FP32 CSR remainder. A 32-by-32 shared-memory transpose gathers B with coalesced input/output access; the final row-block reduction is deterministic and avoids atomics. Zero dense-block cases are explicitly labeled and cannot count as Tensor Core hybrid wins.

## Data, shapes and ordering

Synthetic matrices have planted 32-square blocks at 128, mixed block density at 1024/4096, and unstructured 1% sparsity at 2048. Block rows alternate approximately 7/8 and 3/8 fill, with a small unstructured remainder, so changing thresholds moves actual work. Inputs use portable PCG64 integer mantissas; synthetic row zero is empty. Dense widths are 64/256. Natural and stable descending-degree row ordering are crossed with every matrix/width; two additional scrambled-order controls test 1024 mixed blocks and BCSSTK09 at width 64. The second cohort uses an independent matrix/B seed and reverses group/method order. Four independent dense B matrices cycle through host requests.

Real matrices are the structural stiffness matrices [BCSSTK09](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc1/bcsstk09.html) (1083) and [BCSSTK10](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc1/bcsstk10.html) (1086), attributed to John Lewis / Harwell-Boeing / NIST Matrix Market. The original files are retained with hashes. They are cast to common FP32 and scaled globally by an exact power of two so the largest magnitude is below one; this preparation is common input generation outside every method's timers. Both cohorts use these same two real matrices with independent B inputs; they are not four independent physical matrices or a complete graph-learning application. Irregular 193-by-257 padding and non-square/zero controls are additionally retained in the quick campaign.

## Complete timing and setup boundaries

Prepared timings reuse encoded A/B and measure compute plus needed reduction/crop/unpermutation. Online-B additionally encodes/gathers FP32 B; online-both also re-encodes FP32 A format values with an unchanged sparsity pattern. FP32 row-major paths alias their source buffers. Natural-order native references write directly to the requested output where layout permits; dense GEMM uses row-major layouts. Device timings are CUDA Graph averages with five warmups and thirty observations: eight operations per graph, or one when m*k*n>=2^30. These repeat B0 and do not establish changing-B latency.

Complete uncaptured host requests include pageable FP32 B H2D, online-B computation and full FP32 C D2H, with four B inputs cycling through thirty requests. CPU CSR-based reordering/format construction, native allocation/A upload/preprocessing/heuristic calibration and graph construction are measured separately. Natural CSR consumes the already provided canonical source arrays without sorting/copying them. Hybrid format construction uses CSR block keys/counts rather than materializing a full dense A; BSR/ELL retain the padding their vendor format requires. CPU conversion times include validation and sorting where needed; they are implementation costs, not optimal conversion lower bounds.

The [reuse estimates](results/spmm-frontiers.json) add CPU conversion plus native plan setup divided by R=1/4/16/64/256/1024 to each measured steady operation. Device graph estimates additionally charge construction of the three measured graphs; uncaptured host estimates do not. These are component-based fixed-A models, not measured cold-process latency distributions or changing-pattern GPU timings. Input generation, validation/readbacks outside the declared host boundary and CPU numerical witnesses are excluded equally from timed operations.

Explicit owned device allocations request 102,065 to 182,468,608 buffer bytes. They include source/encoded buffers, padded/gathered B, dense block outputs, reduction/output buffers and supplied workspaces. Opaque library/graph allocations and other GPU contexts are additional; this is not a total peak-memory measurement. The CSV separately retains CPU format payload bytes, which may reference the original CSR arrays and are not a CPU allocator peak.

## Fixed-error comparison with the strongest feasible reference

Each group compares the fastest eligible hybrid with actual dense blocks against the fastest eligible CSR/BSR/ELL/dense library path. A ratio above one favors the hybrid. Groups with no feasible active hybrid remain infeasible rather than disappearing from the denominator. Results aggregate dimensions/order/seed; repeated timings are not independent input trials.

| Boundary | Maximum error | Active hybrid feasible groups | Faster active hybrid groups | Median native / active-hybrid time where feasible |
|---|---:|---:|---:|---:|
| prepared | 1e-05 | 8/52 | 0/52 | 0.4604 |
| prepared | 0.001 | 42/52 | 0/52 | 0.3051 |
| prepared | 0.01 | 42/52 | 0/52 | 0.3054 |
| online-b | 1e-05 | 8/52 | 0/52 | 0.4328 |
| online-b | 0.001 | 42/52 | 0/52 | 0.3634 |
| online-b | 0.01 | 42/52 | 0/52 | 0.3634 |
| online-both | 1e-05 | 8/52 | 0/52 | 0.4107 |
| online-both | 0.001 | 42/52 | 0/52 | 0.3644 |
| online-both | 0.01 | 42/52 | 0/52 | 0.3644 |
| host-request | 1e-05 | 8/52 | 2/52 | 0.9557 |
| host-request | 0.001 | 42/52 | 1/52 | 0.8980 |
| host-request | 0.01 | 42/52 | 1/52 | 0.8927 |

## Natural-order examples at error 0.001

Ranges span the two cohorts. Online-B numbers include gather/encoding/reduction. A single best native label is not assumed across cohorts; all chosen methods remain in the raw frontier.

| Matrix / width | Best native online-B, microseconds | Native / active-hybrid online-B | Native / active-hybrid host request |
|---|---:|---:|---:|
| bcsstk09, 1083x1083 / 64 | 4.06-4.062 | 0.2841-0.2942 | 0.9031-0.9136 |
| bcsstk09, 1083x1083 / 256 | 7.684-7.694 | 0.3193-0.3199 | 0.8939-0.9754 |
| bcsstk10, 1086x1086 / 64 | 3.822-3.85 | 0.2672-0.2696 | 0.9177-0.9229 |
| bcsstk10, 1086x1086 / 256 | 7.682-7.688 | 0.3347-0.3349 | 0.8696-0.9284 |
| mixed, 1024x1024 / 64 | 6.892-6.894 | 0.3644-0.3699 | 0.8743-0.8815 |
| mixed, 1024x1024 / 256 | 13.81-15.08 | 0.3236-0.3485 | 0.8916-0.9197 |
| mixed, 4096x4096 / 64 | 20.21-20.5 | 0.5198-0.5289 | 0.9075-0.9636 |
| mixed, 4096x4096 / 256 | 65.04-65.3 | 0.4918-0.4977 | 0.9207-0.9509 |
| planted, 128x128 / 64 | 2.268-2.306 | 0.1922-0.2148 | 0.7939-0.8 |
| planted, 128x128 / 256 | 2.784-2.796 | 0.2148-0.2344 | 0.8669-0.8736 |
| unstructured, 2048x2048 / 64 | 7.132-7.164 | infeasible | infeasible |
| unstructured, 2048x2048 / 256 | 24.88-25.01 | infeasible | infeasible |

## Reuse and negative controls

The table below summarizes modeled online-B costs at error 0.001. Native and active-hybrid candidates are reselected at each reuse count using complete setup plus steady operation cost. A short warmed kernel alone does not establish a benefit for a newly converted matrix.

| Reuse R | Active hybrid feasible groups | Active hybrid faster after setup | Median native / active-hybrid amortized time |
|---|---:|---:|---:|
| 1 | 42/52 | 0/52 | 0.4070 |
| 4 | 42/52 | 0/52 | 0.4079 |
| 16 | 42/52 | 0/52 | 0.4159 |
| 64 | 42/52 | 0/52 | 0.4424 |
| 256 | 42/52 | 0/52 | 0.4241 |
| 1024 | 42/52 | 0/52 | 0.4064 |

Among hybrid configurations, 352 have actual dense blocks and 584 have none. A zero-block choice executes a CSR path and cannot substantiate Tensor Core acceleration. Degree/scrambled ordering can destroy block locality, increase ELL padding or leave no eligible dense blocks; those regressions and conversion costs are retained.

There are 156 FP32-storage ELL configurations. On this library/device they are approximate and are not labeled strict FP32 arithmetic. The report does not certify their opaque internal quantizer. Their complete original-input error is still measured against FP64 and used for eligibility. Strict FP32 CSR/dense and independently decoded FP16/BF16 block references remain available. API storage dtype alone must not be mistaken for full arithmetic precision.

Zero-matrix controls cover native CSR and both hybrid precisions, including all graph boundaries and no dense blocks. Wrong B shapes and null native plans are rejected. The initial 216-configuration CSR capability probe covers algorithm/layout/preprocessing combinations, including changed B after preprocessing; it is independently replayed and is not counted as study completion by itself. A preliminary flat/strided B-gather version was archived privately (one complete and one partial cohort) before improving coalesced packing, direct output and row-major dense references. Both published full cohorts execute the final source hashes; preliminary timings are not mixed into these results.

## Independent verification and limits

A separate dense NumPy oracle checks lossless CSR/dense-block decomposition; source CSR format preparation is independently decoded for every shape/order/format. GPU encoded A/B and source buffers are checked byte-for-byte before timing, after device timings and after host requests (72,466,385,280 checked bytes). CPU CI reconstructs their complete expected bytes and checks all retained hashes, including padding and row permutations.

Full actual outputs are checked against original and independently rounded-input FP64 products before timing, after the last actual graph output of each boundary, and for each distinct changing-B output hash. Every host output is read back and hash-linked to a check; distinct results from library reduction-order differences are checked separately. Intermediate graph outputs are not retained. Sixty-four actual output coordinates per check support independent CPU row-dot replay with FP32 accumulation envelopes. Complete witnesses cover every method on natural 128-square/width-64 inputs and selected CSR/block/dense/hybrid references on both real matrices at width 64. Other large complete output arrays are not retained; full large-case quality relies on source-bound runtime CPU checks plus retained samples, not a claim of full large-output CI replay. Rounded-product verification uses relative error 2e-5 as an experimental screening contract, not a universally correctly-rounded Tensor Core proof. The opaque FP32 ELL exception is explicitly flagged.

Native runs use the recorded CUDA 13.3 toolchain on RTX 5060 Ti / SM120. The [NVIDIA cuSPARSE API](https://docs.nvidia.com/cuda/archive/13.2.0/cusparse/index.html#cusparsespmm) documents the algorithm/layout/preprocessing interfaces; actual runtime versions, flags, supported outcomes and outputs are retained. Other GPU contexts remain present and no clocks, power settings or services were changed. Results are exploratory shared-system observations and do not establish an architecture ceiling, optimal converter/kernel, scientific novelty or general graph-workload performance.

## Reproduction

```sh
bash build_spmm.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python spmm_study.py --seed 20261125 --quick --output spmm-quick.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python spmm_study.py --seed 20261125 --output spmm-a.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python spmm_study.py --seed 20261225 --reverse --output spmm-b.json
python verify_spmm.py
python summarize_spmm.py --check
```

See the [complete CSV](results/spmm-summary.csv), [fixed-error/reuse frontiers](results/spmm-frontiers.json), [first cohort archive](results/spmm-a.json.gz) and [second cohort archive](results/spmm-b.json.gz) for all configurations, failures, timings, quality, memory, setup and source/data/binary bindings. The gzip files preserve the executed JSON bytes exactly. CPU tools accept these archives or newly reproduced uncompressed JSON; the latter takes precedence when present.
