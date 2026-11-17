# Quantized vector search on consumer Blackwell

## Scope and evidence

The declared two-cohort collection contains 90 groups and 3750 attempted configurations: 3510 available and 240 unsupported calls. These are bounded experiments on one RTX 5060 Ti 16 GB, not an architectural ceiling, novelty claim, general recall guarantee, or a result for all vector-search workloads.

The source-bound complete CPU replay passed 120 groups (5000 attempted / 4680 available configurations) including the supplementary smoke: 15,480 exact truth queries, 2,685,174,216 direct FP64 database/query distances, 15,823,760 retained output IDs, 21,062,724 candidate IDs and 480,985 sampled native DOTs. The three collection directories are smoke A, full A and full B, not three independent input cohorts. The proof and executed verifier/producer hashes are in results/search-replay.json.

The five datasets are uniform-small (16,384 x 128), clustered (65,536 x 256), shifted (262,144 x 128), uniform-million (1,000,000 x 128) and genuine SIFT1M (1,000,000 x 128). Synthetic inputs use independent portable cohorts; SIFT cohorts use disjoint query-source halves. Every full group has B=1/32/128, a static global allow-mask of 100/10/1%, four query variants, five warmups and thirty observations per available boundary. Cohort B reverses dataset, geometry and method order. Timing observations are repeats, not independent inputs. The SIFT attribution, exact original fvecs hashes and lossless shard reconstruction are in fixtures/search-sift/manifest.json and verify_search_sift.py. Numerical descriptors are not relicensed as original images.

## Implementation and appropriate controls

The proposal includes native CUTLASS SM120 block-scaled FP4, cuBLASLt FP4, FP8, FP16, BF16 and strict pedantic FP32 products; reconstructed-norm distance formation; native batched cuVS select-k; optional direct-difference original-FP32 refinement; static filtering; and device global-ID mapping. Candidate budgets are 10 without refinement and 32/128/512 with refinement, clipped only to the allowed population. An FP4 original-norm control and explicit compact-filter FP4/FP16 controls retain losses from quantization, cancellation, setup and compaction. Million-row native buffers are padded to 1,000,064 rows; padded rows are never allowed.

Strong references include resident Faiss GPU FP32/FP16 with both native and cuVS backends, cuVS brute force FP32/FP16, IVF-flat at limited and exhaustive probes, IVF-PQ at two probe counts with and without candidate refinement, and CAGRA with two itopk sizes. Faiss uses device add_c/search_c and initialized-resource stream assignment, not a host-query shortcut. Filtered Faiss statically compacts and remaps global IDs, with compaction in setup. The installed standalone IVF-PQ Python API lacks a filter argument: these calls are recorded as unsupported, never silently replaced by unfiltered search.

## Correctness and retained failures

verify_search.py regenerates every input/filter, recomputes every truth query by direct FP64 differences and independently partitioned top-k with explicit ID ties, checks all returned IDs and actual geometric distances, and replays timing/query/output bindings and source/witness hashes. Strict recall is against deterministic ID-tied top-10; tie-aware recall counts unique valid results no farther than the exact tenth distance. Duplicate, missing, out-of-range, filtered and nonfinite valid-ID outputs are separately retained. Quality frontiers require complete finite unique IDs and minimum tie-aware recall across the outputs actually used by that timing boundary; strict recall is also exported. Each output recall is the batch mean, not a per-query lower bound or a certified guarantee on unseen queries.

Native stage replay checks actual source/encoded bytes and FP4 scales, exact FP32 norm order, sampled DOT accumulation envelopes and score formation, every candidate mapping and score envelope, complete direct refinement and final membership. Large unselected score matrices are not retained: their ranking claim is limited to sampled competitors and all selected candidates; the separate native-kernel preflight independently checks complete small matrices/rank cutoffs and large sampled witnesses. No sampled witness is described as a proof of all large unselected ranks.

The collection retains 29 configurations with invalid returned-ID sets, 424 duplicates and 78701 missing IDs across deduplicated stored output records. These counts are not independent-query failure rates. Invalid configurations cannot produce a speed win. Low recall, cancellation and nonfinite results are failures, not useful faster search. 807 available configurations have ineligible captured graphs; their complete uncaptured route remains recorded. Failed or stale graph outputs are retained but not used as successful captured timings.

The original owned-stream CAGRA adapter exposed a shutdown lifetime error: the pinned cuVS implementation caches a filter payload containing the CUDA stream until static cache destruction. The adapter now uses context-lifetime per-thread-default streams for CAGRA only. The failing regression was run before the change; five fresh four-plan subprocesses subsequently passed changed-query, raw distance and clean-exit checks. All old smoke sources/records were archived before the change and fresh smoke was rerun. The pinned source shows the retained stream, destructor and process-lifetime static filter-payload owner.[1] Filtered CAGRA also produced incomplete/duplicate raw results at low selectivity in this protocol; a private raw-ID mapping control reproduced these with both owned and PTDS streams. This is retained observed behavior, not an established vendor root-cause claim.

## Timing, setup, reuse and memory boundaries

Resident-uncaptured CUDA events cover query encoding/norms, the product, filtering/select-k, optional original-FP32 refinement and global-ID mapping; CPU launch gaps are inside that interval. Resident-captured is used only after complete changing-query equality controls. Best-resident means the faster eligible captured or complete uncaptured route of each method, so unsupported reference capture does not remove strong references from the primary comparison. Host-service is Python wall time for current FP32-query H2D, all uncaptured work, synchronization and full distance/int64-ID D2H, including allocation/dispatch. It excludes database setup, which is measured separately.

Setup includes constructor wall time (CPU compaction/padding, uploads, database encoding/norms, allocations, index training/native tuning), first execution, and the actual successful capture or failed-capture attempt. Reuse totals are setup/R plus measured per-batch latency for explicit R=1/4/16/64/256/1024/100000. These are ex-post measured frontiers, not a deployable selector trained on held-out inputs. A median ratio above one is descriptive; shared GPU conditions and no architecture-ceiling inference remain explicit. CSVs include per-route median and interquartile interval.

Native owned-device bytes include retained original FP32 database/query data, encoded data/scales, scores/DOTs, candidates/refinement, mappings and declared native workspaces; they are not quantized-code-only bytes. Baseline Python arrays are deduplicated by device pointer. Faiss allocation categories and its separate shape-aware temporary reservation (at least 64 MiB, two full FP32 distance tiles plus 16 MiB margin for the declared geometry) are exported. This reserve is charged, not hidden. A fixed-64-MiB B128 shifted search previously aborted after captured scratch overflow while the uncaptured control passed; larger persistent scratch removed the failure in eight separate full-geometry subprocess controls, including a million-row database and both Faiss backends/precisions. Scratch fallback allocation/deallocation behavior is explicit in the pinned resource implementation.[4] Prior records and sources were archived and the entire declared collection was rerun; this is an experimentally verified mitigation, not a claim to have proven every internal vendor race. cuVS has a separately named initial 64 MiB per-resource pool; opaque index/internal allocations are not included in Python-owned bytes. Shared-device free-memory observations are advisory, not physical peaks or reliable cross-method index memory. No end-to-end memory-reduction claim follows from comparing these unequal accounting scopes.

## Equal-recall, setup-inclusive frontiers

FP4 means the best tested CUTLASS or cuBLASLt FP4 proposal configuration, not the best arbitrary precision. Reference means the best quality-feasible method among all non-native Faiss/cuVS controls, including indexed search. Both must meet the same recall budget. Missing feasible proposals or references are explicitly not wins. The table uses R=100000 (near steady reuse); exact all-R frontiers are in results/search-frontiers.csv.

| Boundary | Recall budget | FP4 feasible groups | Reference feasible groups | Comparable groups | FP4 median wins | Median reference/FP4 among comparable |
|---|---:|---:|---:|---:|---:|---:|
| resident-uncaptured | 0.9 | 66/90 | 90/90 | 66 | 54 | 1.8673x |
| resident-uncaptured | 0.95 | 66/90 | 90/90 | 66 | 54 | 1.8796x |
| resident-uncaptured | 0.99 | 66/90 | 90/90 | 66 | 54 | 1.8796x |
| resident-uncaptured | 1 | 66/90 | 88/90 | 64 | 51 | 1.8796x |
| best-resident | 0.9 | 66/90 | 90/90 | 66 | 45 | 1.1512x |
| best-resident | 0.95 | 66/90 | 90/90 | 66 | 48 | 1.1649x |
| best-resident | 0.99 | 66/90 | 90/90 | 66 | 42 | 1.1563x |
| best-resident | 1 | 66/90 | 88/90 | 64 | 36 | 1.1000x |
| host-service | 0.9 | 66/90 | 90/90 | 66 | 56 | 1.4899x |
| host-service | 0.95 | 66/90 | 90/90 | 66 | 56 | 1.4964x |
| host-service | 0.99 | 66/90 | 90/90 | 66 | 55 | 1.5069x |
| host-service | 1 | 66/90 | 88/90 | 64 | 51 | 1.5069x |

### Dependence on dataset at recall 0.95, best resident route, R=100000

| Dataset | FP4 feasible | Reference feasible | FP4 wins | Median reference/FP4 |
|---|---:|---:|---:|---:|
| uniform-small | 18/18 | 18/18 | 11 | 1.0271x |
| clustered | 12/18 | 18/18 | 11 | 1.7517x |
| shifted | 0/18 | 18/18 | 0 | not comparable |
| uniform-million | 18/18 | 18/18 | 16 | 1.2841x |
| sift-million | 18/18 | 18/18 | 10 | 1.0463x |

## Bounded findings

At minimum tie-aware recall 0.95 and R=100000, FP4 is feasible in 66/90 groups and is faster than the best feasible strong reference in 48/90 groups on the best resident route. This result includes losing and infeasible groups rather than conditioning on a favorable subset.

Candidate generation and exact refinement are distinct: refinement cannot recover a true neighbor omitted from the candidate set. Increasing C consumes selection/refinement time and keeps the original database resident; low-precision product throughput alone therefore does not establish useful full-search efficiency. Shifted/nearly clustered distances also stress the norm/dot identity even when its product implementation meets its arithmetic envelope.

results/search-fixed-time.csv reports the best observed recall under 0.5x/1x/2x the fastest reference latency that meets 0.99 recall in the same group; groups without such a reference anchor are omitted explicitly. results/search-cases.csv retains each method/route and its measured quality, latency and named memory fields. results/search-negative-results.json preserves unsupported calls, graph exclusions and invalid/incomplete outputs.

## Reproduction and completion gates

Run build_search.sh and build_search_full.sh with the pinned CUTLASS and isolated Faiss/cuVS roots; prepare and independently verify the lossless SIFT fixtures. Run check_search_lifetime.py on GPU 0, then search_study.py --cohort a and --cohort b. The five-dataset fresh smoke is supplementary, not a replacement for these full collections. Run verify_search_preflight.py, verify_search_kernel_preflight.py, verify_search_sift.py, verify_search.py and summarize_search.py --check on CPU. Existing studies must continue passing. Exact source, binary, actual package/library environment and witness SHA bindings are in each record.

The report is generated only from complete A/B records. Scientific completion additionally requires independent replay of the complete collection and passing CPU/artifact CI on the exact published head; generation of this report alone is not that gate.

## Sources

[1] https://raw.githubusercontent.com/NVIDIA/cuvs/v26.10.00/cpp/src/neighbors/detail/cagra/cagra_filter_payload.hpp
[4] https://raw.githubusercontent.com/facebookresearch/faiss/v1.16.0/faiss/gpu/StandardGpuResources.cpp
