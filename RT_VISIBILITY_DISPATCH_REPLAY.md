# Study 13: matched visibility dispatch replay

From the repository root, run:

```sh
python -B verify_study13_dispatch.py results/study13-matched-dispatch
```

This is an offline check of the archived output from the CUDA software BVH and
Vulkan ray-query routes. It does not launch a GPU or establish timing eligibility.
The public subset contains exact fixture bytes, six raw and fixture-bound output
pairs, the archived exit/stderr records, original replay receipts, and the exact
fixture parser and numerical replay code. The independently pinned root oracle
uses rational arithmetic on the serialized FP32 coordinates. The wrapper checks
the full file inventory and manifest SHA-256 before replaying every padded word;
it also checks that each bound output contains the corresponding raw readback.

The two regular workloads check **33,280 words** across both backends, including
**504 padding canaries**, with no mismatches. The separate boundary workload
checks 128 words. CUDA differs from the open-segment oracle at active lane 4;
Vulkan differs at active lanes 4 and 5. No boundary differences are treated as
regular-workload successes or used for a performance comparison. Lane 4 is a
coplanar overlap under the exact reference; lane 5 is a vertex contact. These
are differences from the reference's open-segment convention, not demonstrated
Vulkan conformance failures. The Khronos specification treats watertightness
as an expectation for shared edges or shared vertices of closed fans; this fixture does
not establish a failure of that condition.

The compute route uses the Vulkan [`VK_KHR_ray_query` extension](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_ray_query.html).
For the API's intersection and watertightness scope, see the Khronos
[Vulkan specification, Ray Traversal](https://docs.vulkan.org/spec/latest/chapters/raytraversal.html).

`SHA256SUMS` binds this subset, not a freshly attested GPU run. Its SHA-256 is
`6c56d357b6fcd0c033f7f552d1d0f819733521ca9caa8079629748abca1e8e55`.
The original acquisition manifest, including omitted host-specific identity and
stdout records, is retained privately with SHA-256
`f1a813bf68b518c8129101bdb7c02c02fcee298af2f38b4999da862f571fcf40`.
Neither that archive nor its exact digest substitutes for public evidence of
cost, device identity, or a native execution replay.
