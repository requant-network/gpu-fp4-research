# Study 13: geometry-change and same-edit quartet replay

This bundle contains sanitized projections of two retained Study 13 collections. It checks the exact visibility output for every before/after readback on CPU. It does not rerun or attest the original launches, identify a physical device, establish comparison eligibility, or rank the routes.

| Projection | Retained coverage | Active words | Padding canaries | Total words |
| --- | --- | ---: | ---: | ---: |
| `evidence/geometry.json` | Two geometry cases, two backends per case, both epochs | 80 | 944 | 1,024 |
| `evidence/quartet.json` | Four declared orders, 16 distinct specimens, both epochs | 320 | 3,776 | 4,096 |
| **Total** | **40 readbacks** | **400** | **4,720** | **5,120** |

## Replay

Use a trusted copy of `replay.py` and the manifest SHA-256 supplied separately by the publisher (do not compute a trust pin from the bundle being verified). For this edition the pin is `0cfed9098f220a0b9f18c2dcde813ce9bfa968304ba53366e5fe45aa2c9660a5`. From the bundle directory:

```sh
python -I -B replay.py --manifest-sha256 0cfed9098f220a0b9f18c2dcde813ce9bfa968304ba53366e5fe45aa2c9660a5 --check
python -B -m unittest -v
```

`--check` compares the deterministic result to `receipt.json` without rewriting anything. Omit it to print the receipt. Use Python 3.12 or later; only the standard library is needed. Preserve LF in the pinned files when distributing this bundle.

The verifier enumerates the two evidence files, checks the public manifest and record bindings, decodes the unchanged fixture/readback bytes from hex, and interprets stored binary32 coordinates as exact rationals. It intersects the open ray segment with closed, two-sided triangles using barycentric clipping. It compares all 128 little-endian words in each epoch, including every `0xfeedface` padding lane. It does not load the archived oracle, native code, or the shader; the shader bytes are checked only for identity.

## Geometry and timing controls

- **Same topology:** translate triangle slot 0 by +16 along x; keep both triangles and the ray bytes fixed. Four active lanes change. CUDA records `build → refit`; Vulkan records `BUILD → UPDATE` (in place) with TLAS `BUILD` in both epochs.
- **Different topology:** append one triangle (2 → 3); keep the prior triangles and the ray bytes fixed. Two active lanes change. CUDA records `build → rebuild`; Vulkan records `BUILD → BUILD`, not an attempted `UPDATE`, with TLAS `BUILD` in both epochs.
- **Same-edit quartets:** orders `ABDC`, `BCAD`, `CDBA`, and `DACB` replay the same fixed edit. A = CUDA refit, B = CUDA rebuild, C = Vulkan in-place BLAS UPDATE, D = Vulkan BLAS BUILD. Vulkan uses the same baseline build flags and rebuilds the TLAS for every epoch. The receipt reports the signed paired differences `B-A`, `D-C`, `C-A`, and `D-B` in each order and aggregates those differences, not the differences between unpaired medians.
- **Timing scope:** the recorded quartet after-edit intervals enclose the transition through the second full readback. They are not isolated acceleration-structure times. Whole-route observations retain separate fresh-child spawn-through-wait measurements; geometry measurements also enclose child launch through wait. CUDA's disjoint ledger is reconciled against the corresponding enclosing time; unattributed remainder is not a separately measured stage. Arithmetic checks do not independently establish where native timer hooks executed.

Original SHA-256 fields refer to archived records; each transformed record also carries the SHA-256 of its canonical public `data`. Unchanged payloads have one SHA-256 for the decoded bytes. Producer hashes are identifiers only. The manifest binds the public projections and replay source. The bundle omits physical identifiers, private paths, command lines, native binaries, and raw logs. The archive, original manifests, and original receipts are not distributed here; public replay neither verifies their bytes nor reproduces their receipts.
