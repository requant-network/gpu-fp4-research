# Study 13: Vulkan Ray Query and software CUDA for segment visibility

The historical acquisition records identify one RTX 5060 Ti (16 GB). On the retained outputs, both the Vulkan Ray Query and CUDA software BVH routes exactly matched an independent rational-arithmetic CPU reference on two synthetic scenes. The public replay checks those outputs, not the device identification. The separate boundary fixture retained disagreements. For one fixed-topology edit, rebuilding took longer than refitting or updating in every measured after-edit pair within each backend. The whole-child differences had mixed signs. These results describe the retained runs; they do not establish a general backend ranking or isolate hardware traversal time.

## Implementations and numerical contract

The Vulkan route uses a compute shader with [Khronos `VK_KHR_ray_query`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_ray_query.html), opaque nonindexed triangles, a bottom-level acceleration structure (BLAS), and an identity instance in a top-level acceleration structure (TLAS). The CUDA control builds its BVH on the CPU and traverses it in a CUDA kernel. Its refit also runs on the CPU. No OptiX SDK was used. This comparison tests two complete implementations, not which units or instructions the Vulkan driver used.

Both routes receive the same serialized binary32 triangles and ray records. The exact CPU oracle constructs the open ray segment between the stored parameters using rational arithmetic. Its Boolean test includes triangle boundaries and qualifying coplanar overlap, and excludes zero-area triangles. Conservative bounding boxes prune candidates without replacing the exact intersection test. An active output is `0` (clear) or `1` (blocked); inactive lanes must retain `0xfeedface`. The two regular scenes use distinct geometry: 4,096 spread triangles and 8,192 layered triangles. Both run two changing 4,097-ray batches with 63 padding lanes each. Every active answer flips between the batches, so substituting the first batch's answers for the second fails. The two regular scenes have identical expected answer vectors, so numerical replay alone cannot distinguish a stale copy that already contains the correct batch's answers. The scenes differ in both size and candidate density; this is not a one-variable density study.

| Scene | Triangles | Active queries | Checked words per backend | CUDA mismatches | Vulkan mismatches |
|---|---:|---:|---:|---:|---:|
| spread | 4,096 | 8,194 | 8,320 | 0 | 0 |
| clustered | 8,192 | 8,194 | 8,320 | 0 | 0 |
| boundary | 3 | 9 | 64 | 1 | 2 |

In the boundary fixture, both routes disagree with the oracle at lane 4 (coplanar overlap); Vulkan also disagrees at lane 5 (vertex contact). Padding and output encoding remained valid. These are differences from the declared open-segment reference, not demonstrated API conformance failures. The [Khronos ray-traversal specification](https://docs.vulkan.org/spec/latest/chapters/raytraversal.html#ray-traversal-watertightness) states a watertightness expectation for intersections with shared edges or with shared vertices of closed fans. The boundary fixture is excluded from the cost matrix. See the [matched dispatch replay](RT_VISIBILITY_DISPATCH_REPLAY.md).

## Build once and dispatch twice

Eight paired runs used a fresh child per backend. Order is balanced within each scene. Each child builds once for the first batch and reuses the structure for the second. All 16 retained readbacks passed the exact replay, covering 133,120 words including padding. The enclosing monotonic host interval starts just before process creation and stops after wait/exit; it includes initialization, setup, scene build, both batches, output writes, and child teardown. It excludes parent lock wait, readback snapshotting and replay. A fresh child does not imply cold driver state, cold caches, or controlled temperature.

| Pair | Scene | Order | CUDA whole child (ms) | Vulkan whole child (ms) | Vulkan − CUDA (ms) |
|---|---|---|---:|---:|---:|
| p01 | spread | cuda-first | 265.342 | 1207.429 | 942.088 |
| p02 | clustered | vulkan-first | 199.784 | 480.051 | 280.268 |
| p03 | spread | vulkan-first | 198.825 | 474.883 | 276.058 |
| p04 | clustered | cuda-first | 206.879 | 496.314 | 289.435 |
| p05 | clustered | cuda-first | 199.609 | 483.797 | 284.188 |
| p06 | spread | vulkan-first | 195.967 | 480.049 | 284.082 |
| p07 | clustered | vulkan-first | 199.465 | 476.950 | 277.485 |
| p08 | spread | cuda-first | 197.026 | 478.915 | 281.889 |

Vulkan child wall was longer in these eight recorded pairs. This is a descriptive result for the complete children, not a traversal speed comparison. CUDA separately measures CPU BVH build and scene upload; batch service encloses packing, transfer, dispatch/wait and full return. CUDA subtimers are not counted again in the child cost. Vulkan uses mapped writes/reads and separately records host stages; its first batch stage sum includes BLAS, TLAS and dispatch within one submission. Neither stage sum isolates a structure build or a query. The second batch reuses the existing structures, but the cost still includes host work. The unaccounted wall remainder is not measured teardown.

| Scene | Selected ledger item | ms, median [min, max] |
|---|---|---:|
| spread | CUDA CPU BVH build | 1.757 [1.747, 2.295] |
| spread | CUDA scene upload | 0.105 [0.104, 0.106] |
| spread | CUDA first batch service | 0.270 [0.253, 0.315] |
| spread | CUDA reused batch service | 0.101 [0.098, 0.108] |
| spread | Vulkan initialization | 388.306 [385.066, 1012.123] |
| spread | Vulkan first batch stage sum | 1.313 [1.189, 4.078] |
| spread | Vulkan reused batch stage sum | 0.161 [0.136, 0.165] |
| spread | CUDA unattributed wall | 61.136 [60.492, 61.697] |
| spread | Vulkan unattributed wall | 13.427 [12.316, 23.069] |
| clustered | CUDA CPU BVH build | 3.560 [3.548, 3.654] |
| clustered | CUDA scene upload | 0.166 [0.165, 0.183] |
| clustered | CUDA first batch service | 0.270 [0.264, 0.283] |
| clustered | CUDA reused batch service | 0.116 [0.115, 0.120] |
| clustered | Vulkan initialization | 390.101 [385.060, 405.990] |
| clustered | Vulkan first batch stage sum | 1.278 [1.260, 1.289] |
| clustered | Vulkan reused batch stage sum | 0.146 [0.134, 0.151] |
| clustered | CUDA unattributed wall | 61.146 [60.953, 63.513] |
| clustered | Vulkan unattributed wall | 13.272 [12.288, 14.384] |

| Scene | CUDA owned device bytes | CUDA scene H2D once | CUDA rays + poison H2D/batch | CUDA D2H/batch | Vulkan requested bytes | Vulkan allocated bytes | Vulkan mapped scene once | Vulkan mapped rays + poison/batch | Vulkan mapped readback/batch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spread | 412,384 | 262,624 | 149,760 | 16,640 | 577,617 | 578,048 | 147,520 | 149,760 | 16,640 |
| clustered | 674,016 | 524,256 | 149,760 | 16,640 | 994,897 | 995,328 | 294,976 | 149,760 | 16,640 |

Named allocations and logical I/O use different accounting scopes. Mapped accesses are not measured PCIe traffic; named buffers are not a physical VRAM peak or a complete inventory of runtime workspaces.

## Geometry changes

The same-topology control translates triangle slot 0 by +16 along X without changing primitive count/order or ray bytes. CUDA refits and reuses its device scene buffers. Vulkan updates the BLAS in place after an update-enabled baseline build. The different-topology control appends a primitive; both routes perform a rebuild instead. The Vulkan TLAS is built again in each epoch in both cases. Equal primitive count alone does not establish update legality; see [Khronos acceleration-structure build and update requirements](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBuildAccelerationStructuresKHR.html).

| Control | Triangles before → after | CUDA after | Vulkan BLAS after | Changed active lanes/backend | Exact words | CUDA child (ms) | Vulkan child (ms) |
|---|---|---|---|---:|---:|---:|---:|
| same-topology | 2 → 2 | refit | UPDATE | 4 | 512 | 225.014 | 517.799 |
| different-topology | 2 → 3 | rebuild | BUILD | 2 | 512 | 208.276 | 472.742 |

These two cases change different geometry. Their child walls are diagnostic and cannot isolate refit/update from rebuild. The four-arm same-edit experiment holds both fixture snapshots and query batches fixed. **A** is CUDA host refit with existing device scene buffers; **B** is CUDA host rebuild with old device scene release and new allocations; **C** is Vulkan in-place BLAS UPDATE; **D** builds a fresh Vulkan BLAS with separate build scratch. Both Vulkan arms use the same update-enabled baseline build and repeat the same TLAS-build and dispatch schedule. The orders `ABDC`, `BCAD`, `CDBA`, and `DACB` balance positions, yielding 16 distinct child records and 4,096 exact words, including 3,776 canaries.

The **after-edit** interval includes the transition through the second synchronized full readback. CUDA charges route-specific release and allocation to that interval. Vulkan reserves relevant resources up front, outside its after-edit interval; its composite interval includes staging, command recording, synchronization, and the submission combining BLAS/TLAS work with the first dispatch. The **whole-child** interval includes initialization, baseline, allocations, writes and final teardown through observed wait. Neither scope is isolated GPU acceleration-structure time. Only the paired differences within a quartet are aggregated here.

| Contrast | After edit (µs), median [min, max] | Positive pairs | Whole child (ms), median [min, max] | Positive pairs |
|---|---:|---:|---:|---:|
| B-A | 25.317 [4.889, 33.804] | 4/4 | -0.034 [-49.826, 0.404] | 2/4 |
| D-C | 46.202 [41.626, 57.761] | 4/4 | 0.403 [-0.022, 50.474] | 3/4 |
| C-A | 139.301 [134.448, 155.611] | 4/4 | 300.518 [200.487, 300.996] | 4/4 |
| D-B | 168.393 [163.153, 171.185] | 4/4 | 300.867 [300.570, 300.962] | 4/4 |

Both within-backend after-edit contrasts favor the refit/update path for this edit in all four blocks; both whole-child contrasts change sign. The cross-backend deltas are shown for audit only: their narrower scopes do not charge the same work. One small repeated edit cannot establish a size crossover, a general update policy, steady-state throughput, or a traversal ranking.

## Evidence, reproduction and limits

- [Exact dispatch replay](RT_VISIBILITY_DISPATCH_REPLAY.md) covers the regular and boundary cases with their original fixture/readback bytes.
- [Matched-cost projection](results/study13-cost-projection/README.md) reproduces the checked result for all pairs and validates the recorded ledgers. The replay receipt SHA-256 is `693aba784a51af0f7c5705045c3627d948020245b172f835f440269de1c587c7`.
- [Geometry and same-edit projection](results/study13-geometry-projection/README.md) independently checks every epoch word and the paired arithmetic. The replay receipt SHA-256 is `889950bd9508b6d5a6d46cbb702fd1499534aa84505e6e27e09145ddfe8949e4`.
- Check report regeneration with `python -B render_study13_report.py --check`. This compares the Markdown against a render from pinned receipts and selected ledgers; it does **not** read or replay raw witnesses. Run the separate CPU replay commands in the three links above to check the evidence. Those replays and the document check run in the existing CPU test discovery without Vulkan, CUDA, or a private archive.

These public projections bind exact numerical payloads and minimal approved records. They omit private launch material, physical identity values, and binaries. Stored original SHA-256 values identify private records; the public replay does not reauthenticate omitted bytes, rerun GPU work, reproduce timing, or attest physical execution. Eligibility and ranking flags remain false. Native build and source reproducibility are separate from replay of recorded data. The experimental scope is one shared-system adapter, two performance scenes with two batches per child, one diagnostic boundary fixture, and the declared geometry edits. No claim is made about energy, other geometries, virtualized or cold conditions, or scientific novelty.
