# Study 14 lossless CPU-to-GPU streaming

Across both tested SIFT reference epochs, neither ndzip nor XBP1 beat the
stronger plain pinned-transfer control in any of the 20 matched cells
at reuse counts 1, 10 or 100. On a distinct 20 GiB smooth synthetic
stream, stock ndzip compressed the transfer to 14.1% of the original
bytes and XBP1 to 77.2%. Plain transfer had the shortest measured host
service and total route wall in this bounded streaming run.

## Real data and controls

The SIFT results cover 400 recorded route cases and 163,680 timed frame
fingerprint observations. Both independently replayed epochs compared
plain pinned transfers at queue depths one and two against online and
precompressed XBP1 and ndzip. One epoch used the ordered-copy fix; the
other used the unmodified serial CPU encoder. Two deterministic input
cohorts, FP32 and FP64 representations, and five frame budgets vary the
workload within the same descriptor dataset. The comparison prices the
named setup, validation and reuse costs. See the [SIFT accounting and
compact replay](COMPRESSION_SIFT_ACCOUNTING.md).

## Bounded stream beyond VRAM

One fresh RTX 5060 Ti process ran the three online routes. Each consumed
320 successive 64 MiB frames from the same independently committed
smooth FP32 stream. The logical input was 21,474,836,480 bytes per route.
The selected board reported 17,103,323,136 physical framebuffer bytes
and CUDA exposed 16,657,285,120 bytes. Only two source buffers were
pinned at a time. The engine regenerated and checked each pair of source
frames, submitted both nonblocking slots, drained both and checked the
returned digest tables before reuse. The SHA-256 check covered every
4 KiB fragment of the decoded word sequence.

| Route | Logical H2D wire bytes | Fraction of input | Preparation (s) | Host service (s) | Route wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain pinned | 21,474,836,480 | 100.0% | 118.049 | 3.425 | 122.169 |
| ndzip stock serial | 3,027,842,176 | 14.1% | 122.388 | 4.680 | 127.551 |
| XBP1 | 16,571,378,464 | 77.2% | 124.265 | 4.320 | 129.193 |

Service includes online encoding and validation where applicable,
transfer, GPU restore and hashing, synchronization, and host digest
check. Preparation is charged separately. Route wall also includes setup,
first use, output records and teardown. Setup took 0.145–0.169 s per
route and first use 0.051–0.204 s. The owning process wall was
380.657 s. Separate source-buffer requests
were 128 MiB per route. Slot requests are retained in the receipt as
named allocations. Those figures do not establish physical memory
peaks, opaque workspace size, or realized CPU/GPU overlap.

Independent verification matched all 960 fresh digest tables to the
CPU-replayed reference bound to the 320 original frame identities. The
owning wait returned zero, the monotonic clock bracket contained the
observed interval, and the replay reconciled every route's host cost
with the enclosing process wall. The result measures one shared-host
run of one smooth stream and one queue depth. The size-dependent real
descriptor comparison is reported separately above.

## Retained failures and reproduction

The earlier whole-cache run was terminated under memory pressure before
any route completed. Its worker wait retained the termination signal.
The earlier expanded synthetic input mismatch remains a rejected
collection. Neither attempt contributes to the timing comparison.
The loader qualification initially exposed a lazy CUDA Runtime load;
after pinning the actual runtime and NVRTC mappings, the six-observation
native plain smoke passed under the canonical device lock.

The [streamed receipt](results/study14-streamed-replay.json) records the
complete host, wire and named-memory census. The [collector](reproduction/study14/streamed_collect.py),
[owning wait](reproduction/study14/observe.py), and
[independent verifier](reproduction/study14/streamed_verify.py) are the
executed source paths in this release. Fresh reproduction uses
the staged native libraries, the retained source-pinned reference
manifest and its CPU-replayed witness tables. See the [XBP1
format](XBP_FORMAT.md) for its bit-preserving representation and raw-frame
fallback. The shared-host figures describe the tested native XBP1 and
external pinned ndzip variants.

The private source-bound full collection SHA-256 is
152765c08a095bb7d17cec5b9d25b68220e4b1d7ac02346c0036bee215e9778d.
The independently verified compact readback SHA-256 is
8c22f584915e828d291043ccf8a0d41c48bfa666ab077eb883ff922030a26317.
