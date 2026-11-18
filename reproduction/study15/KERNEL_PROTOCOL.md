# Study 15 balanced kernel and arithmetic comparison

Repeat the four offered-load pairs from v5 with three implementations.
The unsigned tree uses the original integer kernel and shared-memory
tree reduction. The unsigned warp kernel preserves integer arithmetic
and replaces the block reduction with two shuffle reductions. The FP64
warp kernel changes the arithmetic to fused double multiply-add while
keeping the shuffle reduction. The inputs are the same uint16 bits, and
the maximum possible exact integer result is below 2^44. Every mode must
deliver the same independently computed 64-bit integer outputs.

The launch freezes one block per pattern and input cohort, interleaves
the three kernels within each block, uses an opposite arm order for the
second cohort, and retains every terminal status. Both batching arms
within a pair use the same executable. The original counter,
arrival, cutoff, phase, provenance and exact output limits apply.
The SHA-256 pins cover the staged kernel source and executable images.

Report the directed effect of batching within each accepted pair.
Compare kernel and arithmetic route energies only as descriptive
observations within a complete block, because order and background
board activity can vary. For each arrival pattern and each offered
p99 objective (2 ms and 12 ms), among accepted cohort-1 modes that
meet the objective choose the lowest observed board energy per delivery.
Evaluate that exact kernel and batching arm on cohort 2. Availability
requires every declared variant in the training block to be accepted.
The held-out selected pair must independently pass or yields no
held-out result. No energy per instruction, sensor update period or
whole-host energy is inferred.
