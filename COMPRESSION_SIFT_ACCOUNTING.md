# Study 14: full SIFT output and cost replay

Both full SIFT campaigns completed on the RTX 5060 Ti and passed independent output and named cost replay. One campaign used the pinned ndzip source plus the ordered-copy fix; the other used the unmodified pin in serial mode. Each also contains plain pinned-transfer and XBP1 controls. These are separate fresh processes on the same hardware, not independent descriptor datasets.

| Reference epoch | Route cases | Timed output fingerprints | Whole child (s) | Named stages (s) | Unattributed child time (s) |
|---|---:|---:|---:|---:|---:|
| Ordered copy | 200 | 81,840 | 545.442 | 261.572 | 283.870 |
| Stock serial | 200 | 81,840 | 529.947 | 263.826 | 266.121 |

The verifier checks that per-case preparation subtimers sum to preparation time without counting them twice in process accounting. It independently reconciles the named device and pinned slot requests, source cache intervals, wire bytes, digest bytes and stage ordering. Source caches and slots are named allocation requests; neither the device nor the host numbers describe physical peaks or include an inventory of opaque library workspaces. The unattributed remainder is the difference between measured whole-child wall and the audited named intervals. It is not a measured teardown stage. The whole-child totals include multiple routes, preparation, evidence I/O and other work, so the difference between these two totals is not a codec speedup.

The exact output and cost receipts are retained in `results/sift-full-accounting-receipts/`. The cost verifier binds the receipts to the exact collection and launch manifests. To independently replay it alongside the original collection files:

```sh
python verify_compression_genuine_cost.py results/compression-genuine-ordered-copy-full \
    --summary new-ordered-copy-cost-replay.json
python verify_compression_genuine_cost.py results/compression-genuine-stock-serial-full \
    --summary new-stock-serial-cost-replay.json
```

## Matched latency and wire-byte comparison

The comparison reopens the two hash-pinned full manifests and both accepted receipts per epoch. Each of the 20 cells per epoch fixes cohort, representation and frame budget. For a route at reuse count `R`, the estimate is the mean of its two measured host-service times plus the other named case stages divided by `R`. The baseline is the faster of the two plain pinned-transfer queue depths at that same reuse count, not a fixed or weaker plain route. The estimate includes named validation costs, excludes unattributed process time, and is not cold-process latency or a prospective selector.

| Reference epoch | Cells with any strictly faster compressed route at R=1 | R=10 | R=100 |
|---|---:|---:|---:|
| Ordered copy | 0/20 | 0/20 | 0/20 |
| Stock serial | 0/20 | 0/20 | 0/20 |

For each epoch, XBP1 reduced wire bytes in all 80 of its routes, by 0.77–1.56%; ndzip increased wire bytes in all 80 of its routes, by 1.57–3.10%. The wire-byte result is not a latency win. This is descriptive evidence from two timed repeats per route on these SIFT windows, not a statement about all transfer workloads. The detailed [comparison receipt](results/sift-full-accounting-receipts/sift-frontier.json) retains the baseline and the best measured compressed route per cell and reuse count. Regenerate it with `compression_sift_frontier.py COLLECTIONS --receipts results/sift-full-accounting-receipts --summary NEW.json`.

## Compact receipt check

```sh
python -B verify_study14_sift_receipts.py
```

This check requires only the five retained receipts and the pinned verification
sources. It checks their exact hashes and links, the two complete 20-cell
censuses, whole-child wall accounting, and the strict inequalities for the
recorded plain and compressed choices. The measured result is zero strict wins
in each epoch at each declared reuse count. The compact check does not reselect
the fastest route from all candidates, reconstruct every frame fingerprint,
or replay encoded packets; use the full collection and the original commands
above for those stages of the evidence chain.

This checkpoint does not certify the guarded larger-than-VRAM synthetic
experiment. The previously rejected synthetic collection remains excluded from
performance conclusions.
