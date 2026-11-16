# First pilot: interpretation and next experiment

## What the pilot answers

The complete retained grid distinguishes three different workloads: a prepared
matrix product, changing A with reusable B, and preparation of both operands.
It also separates errors introduced by quantization from errors in the GPU
matrix arithmetic. See [the results](RESULTS.md) for cohort ranges and all formats.

In the tested baseline, preparation reduces or removes FP4's observed latency
advantage in several shapes. Small matrix products also do not benefit simply
because FP4 has high nominal Tensor Core throughput. Larger tested products can
still favor FP4 with preparation included, but with materially higher error.

These observations support investigating preparation and shape-dependent selection.
They do not establish that FP4 is useful for a particular model or scientific
application. Relative Frobenius error is not an application quality metric.

## What remains unresolved

- Equal-error comparisons and task-level quality.
- Optimized and alternative FP4 encoders, tile shapes and schedules.
- More shapes, real activations, fresh-data streaming and uncaptured request latency.
- Host transfer, allocation and operand-reuse amortization costs.
- Exclusive-device replication and statistical stability across separate sessions.
- Generalization of a format/configuration selector to held-out workloads.

Existing GPU contexts remained present. The two cohorts use independent inputs
and reverse ordering, but are not independent machines or separate-day trials.
Frequency, temperature and cache effects are not eliminated by this protocol.

## Next bounded experiment

Optimize the online FP4 encoder while preserving its exact quantization contract.
Keep the same retained inputs, scale rule, packed-byte oracle and GEMM configuration.
Compare the original encoder with a vectorized or warp-cooperative implementation.
Measure preparation alone, online-A plus GEMM and online-A+B plus GEMM.

Require exact agreement of every encoded byte and scale. A speed improvement that
changes the quantization recipe belongs in a separate error/quality experiment.
Include small shapes where extra synchronization can make cooperation slower.
Retain unsuccessful configurations rather than reporting only the fastest one.

## Research status

The bounded first pilot is complete. The research program remains in progress
under the gates in [RESEARCH.md](RESEARCH.md). This is a starting evidence record,
not a final publication or an exhausted search for the best low-precision method.
