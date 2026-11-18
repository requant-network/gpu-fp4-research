# Study 13 rebuilt Linux dispatch

The two matched-dispatch programs from the clean Linux link ran on the
RTX 5060 Ti with the selected NVIDIA Vulkan ICD. Each child exited zero
and emitted no stderr. Independent rational-arithmetic replay checked
all returned words against the original serialized scenes.

| Scene | Words per backend | CUDA mismatches | Vulkan mismatches |
| --- | ---: | ---: | ---: |
| Spread | 8,320 | 0 | 0 |
| Clustered | 8,320 | 0 | 0 |
| Boundary | 64 | 1 | 2 |

The three retained boundary differences match the previous acquisition.
Both backends missed the declared coplanar overlap and Vulkan also
differed at the vertex contact. Every padding word was valid. These
tests establish dispatch correctness for the rebuilt matched programs
on the two regular scenes. Dedicated build, reuse and edit timings
remain in the original report.

| Artifact | SHA-256 |
| --- | --- |
| CUDA executable | 155f56c103394d0e1c6175af5ebae465da246b3ecc4b2b4e56602e1aac5dc79c |
| Vulkan executable | 06d898d70acc632f4a9fb750d1ac67fe0e29b60938f4d508b73d530cbbf87b84 |
| Private run archive | 66019d8ef07e7dde8e24c97a87a65fac902b87c0f116858608d427825079d4b3 |

The procedures and release scope are in the [findings report](RT_VISIBILITY_REPORT.md)
and [MIT replay package](https://github.com/requant-network/gpu-fp4-research/releases/tag/study13-mit-replay-v1).
The archived native source and runtime evidence are held privately. The
published replay package passed its [exact-revision CI](https://github.com/requant-network/gpu-fp4-research/actions/runs/44375094474).
