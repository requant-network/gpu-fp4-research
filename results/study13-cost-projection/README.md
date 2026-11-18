# Study 13: matched-cost evidence projection v1

This tree is a public-safe projection of eight **recorded** native CUDA/Vulkan pairs. The evidence directory contains all 16 readbacks, two shared fixture snapshots, a shader snapshot, the frozen pair order, producer IDs and hashes, recorded exits, and route-specific cost and byte ledgers. Base64 is only a text transport: replay hashes the decoded byte snapshots. No producer executables or private launch commands are needed.

## CPU replay

From this directory, using a trusted copy of `replay_cost.py` and `exact_geometry.py`:

```sh
python -I -B replay_cost.py evidence b65e7185a4ffcdf955db3a0178e8886028100d79ac02f9ac9f8d9fa0f48f0c60
```

Use the manifest digest from a trusted channel independently of the evidence directory, not a digest computed from the evidence under test. The command emits canonical JSON; successful output can be compared directly with `replay-receipt.json`. It has no dependency on CUDA, Vulkan, the private archive, or non-stdlib packages. It checks the exact data-file census before reading the records and refuses source files supplied as evidence.

The oracle interprets each stored FP32 value as an exact rational. A conservative bounding-box filter limits triangle candidates; the exact triangle test determines visibility on each open ray segment. Replay checks binary active words and poison canaries on all padded readbacks. There are two changing-input batches per route, with 4,097 active lanes and 4,160 total lanes per batch. This replay passed on **133,120 words**, including **2,016 canaries**, across all 16 routes with no mismatches. Boundary, geometry update and same-edit controls are separate evidence and are not admitted to this matched-cost matrix.

## Evidence and cost boundaries

`manifest.json` maps each allowlisted public file to its original and public SHA-256. JSON records and nested launch, stdout and sidecar projections also carry the original record hash and the hash of the canonical projected record. A collection row binds each acquisition projection to its original acquisition and normalized recorded exit/stdout/stderr status. The original archive/ledger identities are retained as provenance; the public replay cannot reauthenticate omitted private bytes. In particular, a recorded wrapper exit is not an authenticated external wait. The producer hashes identify the original ELFs; they neither include those binaries nor establish a reproducible native build.

The fresh-child host interval starts before the subprocess call and ends after wait/exit. CUDA's named accounting adds setup/release/write stages and one enclosing service interval per batch, not the nested batch substages again. Vulkan's accounting adds its own host stages and measured batch-stage sums; its first batch includes build and dispatch together. The receipt checks that each route's accounted stages fit within the recorded wall and reports the nonnegative remainder as unattributed wall time, not measured teardown. CUDA transfer bytes and Vulkan mapped I/O or owned-buffer requests/allocations have different accounting scopes.

These are descriptive historical measurements and exact numerical replay of sanitized readbacks, not GPU execution attestation, timing reproduction, timing or memory comparison eligibility, or a ranking.
