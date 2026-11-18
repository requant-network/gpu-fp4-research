# Study 13 — build and reproducibility notes

The eight native targets were linked in a new x86-64 Linux directory using selected, checksum-pinned inputs. Four targets implement the CUDA software comparator and four implement the Vulkan route. Both the retained and regenerated Vulkan 1.2 SPIR-V passed validation, and their bytes were identical. A separate static ELF audit agreed with all eight recorded outputs. The linked programs were not executed during this build. The historical device runs and the independent replay of their outputs are described in the [Study 13 report](RT_VISIBILITY_REPORT.md).

| Evidence | Scope |
| --- | --- |
| Clean Linux build | Owning build wait 0, eight linked programs and eight agreeing static ELF records; no runtime driver, ICD, or device attestation. |
| Historical executable comparison | Four of four Vulkan hashes matched the corresponding historical binaries; none of the four CUDA hashes matched. No cause has been established. This is not a bit-for-bit reproducible-build claim. The local readback binds the static inspection records, not fresh copies of the eight ELF files. |
| Windows loader-free CPU controls | On a relocated build of the exact source candidate, `plan`, `batches`, and `raw` each exited 0 with empty stderr. The clustered fixture used by that recorded run is byte-identical to the [retained fixture](results/study13-matched-dispatch/clustered.s13). The control executable had no direct Vulkan or CUDA imports. This is the retained control result, not an independent second execution. |

Selected inputs covered 14 tool roles, 22 support rows, 2,259 CUDA include files, 42 Vulkan and video include files, and the Vulkan loader link input. This does not cover every implicit or transitive compiler input. Neither a static link nor the CPU controls certify RT-core execution, the provenance of a separately archived measurement, or GPU timings.

| Retained artifact (SHA-256 commitment) | Digest |
| --- | --- |
| Native source ZIP used for the build | `b2d365bfe79d9ce1c913244b7d79c2212cbcc35d99da33d8bc0fa549f856e1ed` |
| Linux gate source | `98880bd3b513accde952af82adaac4ba101e82681e01b71b30a26d472929b831` |
| Decoded and regenerated SPIR-V | `896174d5c320010130ab8f661ab69342dfea4072dcdf3f538c82886e41a7b5f5` |
| Linux build record | `fcb59654de901d198d438ec2db65ea4cfa9d8a1771989e30d40b17397611de6d` |
| Separate static ELF record | `6ec3b2123b83ca5a3914c1921c879dfc5ed19ffaef37159c567bb4630b4a2b9c` |
| Windows source-candidate control receipt | `697493161df9c98a1a812434d04c2413c17f12d54617020b8d941d8b62ee0521` |

The private records, build environment inventory and executable files are not distributed with the replay. The commitments identify retained records; they do not provide public verification of omitted bytes. The independent public replay operates on the disclosed synthetic fixture and output data without the native binaries or a GPU.
