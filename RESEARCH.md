# Research protocol

A research project by [Requant Network](https://requant.network).

## Question

When does FP4 reduce the time to a useful matrix product on a consumer NVIDIA
RTX 5060 Ti 16 GB once input preparation and numerical error are made explicit?

This begins the low-precision research program. It is separate from the earlier
exact modular Tensor Core study: these are approximate real-valued products,
with genuine data-dependent 16-value FP4 block scales.

## Hypotheses

1. The fastest prepacked format need not be the fastest online format.
2. Shape and input distribution affect the speed/error trade-off.
3. A small preparation kernel can dominate small matrix products.
4. A format selector must consider allowed error and operand reuse, rather than
   relying only on nominal Tensor Core throughput.

## First bounded experiment

- Six shapes, two synthetic distributions and five formats: FP4, FP8 E4M3,
  FP16, BF16 and strict FP32.
- Two independent input cohorts. Replication reverses case and initial mode order.
- Three device-resident boundaries: prepacked; online A with prepacked B;
  online A and B. Every format starts from the same FP32 source operands.
- Five graph warmups and thirty retained observations. Each observation is a
  graph containing thirty-two repetitions, reported per operation.
- Candidate selection from up to eight cuBLASLt heuristic candidates.
  The FP4 comparator initially uses one pinned CUTLASS 128x128x128 configuration.
- Verify every encoded operand against an independent CPU encoder, then every
  output against FP64 products of reconstructed operands. Separately measure
  error against the original FP32 operands evaluated in FP64.
- Retain complete small-case numerical witnesses, raw observations, source and
  binary hashes, environment snapshots and negative correctness controls.

These are exploratory shared-system observations. Existing GPU contexts remain
present; no exclusive access, dedicated benchmark, energy efficiency or model
quality claim is made.

## Follow-up gates

The first campaign starts the project; it does not complete the research program.

1. Extend the shape/distribution grid and include model-derived shapes/activations.
2. Tune FP4 tile/schedule and quantization kernels; compare stronger FP4 backends.
3. Test outlier correction and per-operation error constraints.
4. Measure host transfers, allocation, uncaptured request latency and reuse amortization.
5. Fit a simple selector and evaluate on held-out shapes and distributions.
6. Integrate one real application and measure its quality and complete latency.

A publication claim requires a literature comparison and a defensible contribution
beyond a product-specific benchmark. Negative outcomes remain part of the record.

## Evidence checkpoint

The [encoder report](ENCODER_REPORT.md) completes a bounded grid extension,
cooperative/fused preparation comparison, two-tile comparison and held-out encoder
rule. The [application report](APPLICATION_REPORT.md) adds stronger native cuBLASLt
FP4, complete real-model quality, host/allocation/reuse costs, eager/compiled/graph
framework references and validation-gated held-out format decisions.

The [native hardware report](HARDWARE_REPORT.md) adds byte-identical hardware FP4
packing, vector-load and warp-reduction comparisons, emitted-code evidence and
theoretical resource queries. Actual Nsight Compute counter collection was denied
by the current driver permissions; achieved utilization is not established.

The [outlier report](OUTLIER_REPORT.md) adds native column/block/residual correction,
real activations, complete preparation costs and fixed-error/fixed-time frontiers.
These are empirical frontiers, not held-out error prediction for unseen inputs.

Held-out per-operation error prediction and the loading/pipeline/resource
investigation remain open. The program has not established the best possible SM120
implementation or scientific novelty; the fifteen-study goal remains active.
