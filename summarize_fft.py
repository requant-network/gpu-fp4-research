"""Derive bounded FFT conclusions and offline quality-constrained cost frontiers."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BUDGETS = (.01, .001, .00001)
REUSE = (1, 4, 16, 64, 256, 1024)


def median(values):
    return float(np.median(values))


def feasible(row, budget):
    if not row["finite"] or row["output_relative_l2"] > budget:
        return False
    return row["weak_bin_relative_error"] is None or all(row[key] <= .1 for key in ("weak_bin_relative_error", "quiet_bin_artifact_over_weak", "output_error_over_weak"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    records = [json.loads((ROOT/"results"/f"fft-{cohort}.json").read_text()) for cohort in ("a", "b")]
    rows, grouped = [], {}
    attempts = unsupported = checked_outputs = full_replay = sampled_replay = stage_replay = 0
    for cohort, record in zip(("a", "b"), records):
        assert record["complete"]
        for group in record["groups"]:
            key = (cohort, group["n"], group["batch"], group["profile"])
            grouped[key] = []
            for case in group["cases"]:
                attempts += 1
                if not case["available"]:
                    unsupported += 1
                    continue
                meta = case["initial_metadata"]
                row = {"cohort": cohort, "n": group["n"], "batch": group["batch"], "profile": group["profile"],
                       "method": case["method"]["name"], "backend": case["method"]["backend"], "precision": case["method"]["precision"],
                       "finite": all(all(item["quality"][key] for key in ("finite_output", "finite_spectrum", "finite_filtered")) for item in case["outputs"]),
                       "coefficient_load_ms": group["coefficient_load_ms"], "plan_setup_ms": case["plan_setup_ms"], "graph_setup_ms": case["graph_setup_ms"],
                       "owned_device_bytes": meta["owned_device_bytes"], "workspace_bytes": meta["workspace_bytes"],
                       "dx_registers": meta["dx_registers"], "dx_shared_bytes": meta["dx_shared_bytes"]}
                for metric in ("output_relative_l2", "spectrum_relative_l2", "filtered_relative_l2", "weak_bin_relative_error", "quiet_bin_artifact_over_weak", "output_error_over_weak", "input_weak_amplitude_ratio"):
                    values = [item["quality"][metric] for item in case["outputs"]]
                    row[metric] = max(values) if all(value is not None for value in values) else None
                for name, boundary in case["boundaries"].items():
                    row[name+"_ms"] = median([o["cuda_ms"] for o in boundary["observations"]])
                    row[name+"_wall_ms"] = median([o["wall_ms"] for o in boundary["observations"]])
                row["host_ms"] = median([o["wall_ms"] for o in case["host_requests"]])
                checked_outputs += sum(item["checked_output_entries"] for item in case["outputs"])
                sample_indices = np.unique(np.concatenate([np.linspace(0, group["n"]*group["batch"]-1, 64, dtype=np.int64),
                                                          np.arange(group["batch"])*group["n"]+group["parameters"]["weak_bin"] if group["parameters"]["weak_bin"] is not None else np.empty(0, dtype=np.int64)]))
                for item in case["outputs"]:
                    if item["full"]:
                        full_replay += 3*group["n"]*group["batch"]
                    else:
                        sampled_replay += 3*len(sample_indices)
                stage_replay += sum(item["geometry"]["tile"]*min(64, item["geometry"]["columns"]) for item in case.get("stage_checks", []))
                rows.append(row)
                grouped[key].append(row)
    frontiers = []
    for key, candidates in grouped.items():
        for budget in BUDGETS:
            valid = [row for row in candidates if feasible(row, budget)]
            references = [row for row in valid if row["backend"] != 1]
            matrix = [row for row in valid if row["backend"] == 1]
            hmma = [row for row in matrix if row["precision"] != 0]
            for boundary in ("captured_complete", "uncaptured_complete", "host"):
                for reuse in ((None,)+REUSE if boundary == "captured_complete" else (None,)):
                    def cost(row):
                        base = row[boundary+"_ms"]
                        return base if reuse is None else base+(row["coefficient_load_ms"]+row["plan_setup_ms"]+row["graph_setup_ms"])/reuse
                    entry = {"cohort": key[0], "n": key[1], "batch": key[2], "profile": key[3], "error_budget": budget,
                             "weak_budget": .1, "boundary": boundary, "reuse": reuse}
                    for name, family in (("reference", references), ("matrix", matrix), ("hmma", hmma)):
                        best = min(family, key=cost) if family else None
                        entry[name] = None if best is None else {"method": best["method"], "ms": cost(best)}
                    frontiers.append(entry)
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    summary = {"scope": "Offline empirical frontier, not a fitted dispatcher or accuracy guarantee. Each decision requires every retained input variant to satisfy the stated quality limits.",
               "attempts": attempts, "available": len(rows), "unsupported": unsupported, "groups": len(grouped),
               "timing_observations": len(rows)*90, "runtime_input_output_entries_checked": checked_outputs,
               "full_complex_entries_replayed": full_replay, "sampled_complex_entries_replayed": sampled_replay,
               "selected_gemm_entries_replayed": stage_replay, "frontiers": frontiers}
    counts = []
    for budget in BUDGETS:
        for boundary in ("captured_complete", "uncaptured_complete", "host"):
            selected = [e for e in frontiers if e["error_budget"] == budget and e["boundary"] == boundary and e["reuse"] is None]
            possible = sum(e["hmma"] is not None for e in selected)
            wins = sum(e["hmma"] is not None and e["reference"] is not None and e["hmma"]["ms"] < e["reference"]["ms"] for e in selected)
            matrix_wins = sum(e["matrix"] is not None and e["reference"] is not None and e["matrix"]["ms"] < e["reference"]["ms"] for e in selected)
            counts.append(f"| {budget:g} | {boundary} | {possible}/{len(selected)} | {wins}/{len(selected)} | {matrix_wins}/{len(selected)} |")
    example = []
    for method in ("cufft-fp32", "cufft-fp64", "cufft-fp16", "cufft-bf16", "matrix-r16-fp32", "matrix-r16-fp16", "matrix-r16-bf16", "cufftdx-fp32-pack1"):
        sample = [r for r in rows if r["n"] == 16384 and r["batch"] == 16 and r["profile"] == "noise" and r["method"] == method]
        example.append(f"| {method} | {median([r['captured_complete_ms'] for r in sample])*1000:.3f} | {max(r['output_relative_l2'] for r in sample):.3g} |")
    weak = []
    for method in ("cufft-fp64", "cufft-fp32", "cufftdx-fp32-pack1", "cufft-fp16", "cufft-bf16", "matrix-r16-fp16", "matrix-r16-bf16"):
        sample = [r for r in rows if r["profile"] == "weak120" and r["method"] == method]
        weak.append(f"| {method} | {median([r['output_relative_l2'] for r in sample]):.3g} | {max(r['output_relative_l2'] for r in sample):.3g} | {max(r['weak_bin_relative_error'] for r in sample):.3g} |")
    dx_ratios, dx_quality_ratios = [], []
    for candidates in grouped.values():
        ordinary = next(r for r in candidates if r["method"] == "cufft-fp32")
        dx = [r for r in candidates if r["backend"] == 2]
        if dx:
            best = min(dx, key=lambda r: r["captured_complete_ms"])
            ratio = ordinary["captured_complete_ms"]/best["captured_complete_ms"]
            dx_ratios.append(ratio)
            if feasible(ordinary, .001) and feasible(best, .001):
                dx_quality_ratios.append(ratio)
    reuse_counts = []
    for reuse in REUSE:
        selected = [e for e in frontiers if e["error_budget"] == .001 and e["boundary"] == "captured_complete" and e["reuse"] == reuse]
        wins = sum(e["hmma"] is not None and e["reference"] is not None and e["hmma"]["ms"] < e["reference"]["ms"] for e in selected)
        reuse_counts.append(f"R={reuse}: {wins}/{len(selected)}")
    timer = json.loads((ROOT/"results/fft-timer.json").read_text())
    timer_floor = ", ".join(f"{c['repetitions']} repetition(s): {median([o['cuda_ms'] for o in c['observations']])*1000:.3f} us/op" for c in timer["cases"])
    report = f'''# Native FFT and spectral filtering on consumer Blackwell

This bounded study tests complete window/FFT/filter/inverse processing on one
RTX 5060 Ti 16 GB, SM120. The proposed radix-8/16 matrix FFT uses actual
FP16/BF16 HMMA GEMMs, but its packing, small DFT products and intermediate global
buffers do not beat the strong native references in the principal comparison.
The practical positive result is fusing the complete FP32 pipeline with cuFFTDx.
Low transform precision can destroy a weak component despite a small error relative
to the unfiltered input. These results do not establish an architectural limit or
a general impossibility of a faster Tensor Core FFT.

## Executed scope

The two independent input cohorts attempt **{attempts:,} configurations** across
**{len(grouped)} length/batch/profile groups**: **{len(rows):,} available**, **{unsupported} unavailable**,
and **{len(rows)*90:,} timing observations**. Lengths are 128, 512, 1000, 1024, 4096
and 16384; batch counts are 1 and 16, plus 1024-by-64. Seven signal profiles cover
impulse with identity/lowpass, dyadic noise with Hann/lowpass, off-bin tones with
Hann, strong/weak on-bin components separated by nominal 72.25 and 120.41 dB,
and a high-range 96.33 dB case. The complex notch removes the strong bin and its
two neighbours. Complex lowpass coefficients include a three-sample phase delay.
There are four changing inputs per group; cohort B reverses group and method order.
Carrier phase, impulse position and noise change with the input seed; coefficient
tables are shared immutable parameters. Inputs and output interfaces are FP32.

The nominal range describes the synthesized components. FP32 input addition already
rounds the weakest component; `input_weak_amplitude_ratio` records its actual Fourier
amplitude relative to the nominal amplitude. The FP64 reference uses the stored FP32
input and coefficient values, rather than claiming access to an unrounded physical signal.

The [raw A record](results/fft-a.json), [raw B record](results/fft-b.json),
[available-case CSV](results/fft-summary.csv) and [quality/cost frontiers](results/fft-frontiers.json)
retain failures and the measurement boundaries. The [quick control](results/fft-quick.json)
adds 75 available configurations; its timings are excluded from these comparative tables.

## Native implementations

cuFFT 12.3.0, delivered with CUDA 13.3, executes forward and inverse complex transforms
with explicitly owned workspace. Separate native kernels apply the window, complex
frequency response and normalization. FP32, FP64, FP16 and BF16 transforms are attempted.
FP64 computes the window and filter in FP64 and returns FP32 output; other paths
round their typed intermediates. BF16 cuFFT uses mixed single/BF16 operations, as
documented by NVIDIA; it is not labelled pure BF16 arithmetic.

The proposed matrix FFT is mixed-radix decimation in time. Each small complex DFT is
an augmented real matrix multiply `[Re(W),-Im(W);Im(W),Re(W)]`, with a column-major
packed input. A digit permutation, coalesced shared-memory transpose, FP32 twiddles,
typed FP16/BF16 encoding and per-stage normalization are included in the timed path.
Radix-2/4 tails pad the real GEMM dimension to 16; radix 16 uses dimension 32.
Forward and inverse stages each divide by their radix, with a final scale restoring
the usual inverse convention. cuBLASLt chooses among up to eight heuristics during
setup; every low-precision stage must report the HMMA implementation flag. A strict
FP32 pedantic matrix-GEMM control uses the same factorization and packing.
Each plan reserves 64 MiB for Lt workspace; this is charged in memory and setup.

The stronger fused reference uses external NVIDIA MathDx 26.06.1, whose actual
cuFFTDx header version is **1.7.3**, with `SM<1200>`. Its single native kernel performs
windowing, register/shared-memory forward FFT, response multiplication, inverse FFT
and output normalization. One/four FFTs per block are attempted. Diagnostic kernels
add spectrum writes outside performance timing. Shared-memory, register and theoretical
resident-block limits are recorded; they do not measure achieved occupancy.
The SDK archive and 1022 used header files are [pinned by SHA256](results/fft-dx-dependency.json).

For length 1000, actual cuFFT FP16/BF16 planning returns status 16. The power-of-two
matrix factorization rejects it, and the selected cuFFTDx descriptors are unsupported.
At length 16384, the selected four-FFT cuFFTDx descriptor is unsupported; the one-FFT
descriptor runs. These are outcomes of the named implementations, not claims that
all non-power-of-two or four-FFT implementations are impossible. Ordinary FP32/64
cuFFT supplies a working reference at every shape.

## Timing, preparation and memory

Every available method has five warmups and thirty observations for each of two
resident boundaries: complete captured pipeline and complete uncaptured pipeline.
Small graph launches repeat the operation eight times; the largest uses one.
Input encoding, packing, windowing, both transforms, filtering and normalization
are included. Resident repetitions reuse input zero and fixed parameters. Events
surround the graph launch externally, so dispatch gaps may contribute; values are
per-operation averages, not isolated kernel durations or architecture throughput.
The [same-boundary empty-graph control](results/fft-timer.json) reports {timer_floor}.
No calibration is subtracted. Small timing differences near this floor are not
evidence of a hardware mechanism.

Thirty additional changing-input host requests include pageable H2D, the complete
uncaptured pipeline, blocking full FP32 D2H and host output allocation. Actual outputs
are read before any diagnostic rerun. Native CPU planning/allocation/upload/Lt tuning,
graph setup and canonical coefficient-file loading are recorded separately. Common
input synthesis, CPU oracles, untimed spectrum writes and witness persistence are
excluded from performance timing. Coefficients are provided stored parameters;
request-time generation of a new window or arbitrary filter is outside this protocol.
The reuse tables add measured coefficient loading, plan setup and graph construction
once per R operations. They are hot-process amortization models, not measured cold
process requests or a changing-filter/planning service.

Requested owned device buffers range from {min(r['owned_device_bytes'] for r in rows):,}
to {max(r['owned_device_bytes'] for r in rows):,} bytes. This includes diagnostic buffers
even though their writes are untimed. It excludes opaque library/context allocations,
allocator peaks, shared-memory capacity and other processes' VRAM. No total physical
peak memory or whole-system energy claim is made.

## Complete latency and quality

For 16384-by-16 Hann-windowed noise/lowpass, the table combines cohort medians and
uses the worst output relative L2 error over both cohorts and all four inputs:

| Method | Captured complete us/op | Worst output relative L2 |
|---|---:|---:|
{chr(10).join(example)}

Across the {len(dx_ratios)} power-of-two groups with a fused reference, the best
attempted cuFFTDx packing is faster than ordinary FP32 cuFFT in
{sum(r > 1 for r in dx_ratios)}/{len(dx_ratios)} groups, with median ratio
{median(dx_ratios):.3f}x (cuFFT time / fused time). Requiring both methods to satisfy
the 0.001 output-error budget and weak-component limits leaves {len(dx_quality_ratios)}
groups; median ratio is {median(dx_quality_ratios):.3f}x. These are empirical comparisons
on the recorded shared system, without confidence claims across hardware or load states.

Quality-constrained comparisons require every actual input variant to be finite,
output relative L2 <= the stated budget, and, for weak-component profiles, weak-bin
error, quiet-bin artifact and output error each <= 0.1 of the nominal weak signal.
The reference family includes all available cuFFT precisions and fused cuFFTDx.
`HMMA` includes only the FP16/BF16 matrix FFT; `matrix` also includes its pedantic
FP32 control. A missing feasible candidate is retained as a failure, not discarded.

| Output budget | Boundary | Feasible HMMA groups | HMMA faster than best reference | Any matrix FFT faster |
|---|---|---:|---:|---:|
{chr(10).join(counts)}

At budget 0.001, modeled captured setup/reuse HMMA wins are:
{'; '.join(reuse_counts)}. The complete JSON also retains the other budgets and
method identities. This is an offline measured frontier, not a deployable selector,
an independent test of prediction, or a guaranteed-error rule.

## Weak components and failures

For the nominal 120.41 dB profile, the following summarizes the per-group worst
four-input output error across all available lengths, batches and both cohorts:

| Method | Median output relative L2 | Maximum output relative L2 | Maximum weak-bin relative error |
|---|---:|---:|---:|
{chr(10).join(weak)}

The quiet-bin metric excludes the deliberately strong neighbourhood and target weak
bin, then measures the largest *error against the FP64 spectrum*, divided by nominal
weak spectral amplitude. The output weak metric divides complete output error norm
by `weak_amplitude*sqrt(batch*N)`. Neither uses the strong input norm as an accuracy
denominator. Spectra are from an untimed diagnostic pass whose final output must
match the previously read actual timed output, including nonfinite outcomes.

cuFFT FP16 produces nonfinite values on the 16384-point high-range cases. The output,
spectrum and filtered witnesses retain these values; JSON uses null for undefined
errors and records nonfinite counts. They remain timed attempted cases but are never
eligible for a quality-constrained win. Per-stage scaling keeps the matrix FFT finite
in that case, but finiteness alone does not recover the weak component or establish
a latency advantage. A small error normalized by the original strong input is only
an implementation screening check; it is not the scientific quality criterion.

## Independent validation and scope limits

The runtime checks {checked_outputs:,} actual output entries across input variants over
the two full campaigns. CPU replay checks **{full_replay:,} retained complete complex entries**
across output/spectrum/filtered arrays, **{sampled_replay:,} sampled entries**, and
**{stage_replay:,} selected GEMM-product entries**. Quick and capability controls add
separate witnesses. Full arrays are retained for all small batch-one cases and
selected large/reference/low-precision cases; the remaining large full arrays are
checked at runtime and are not recoverable from CPU CI. Sample replay does not
re-establish unretained global norms. Every witness, source, canonical coefficient,
input, uploaded parameter and actual timed output is hash-bound to its record.

An FP64 mixed-radix implementation is tested against NumPy FFT before using the
factorization on GPU. For every noise-case matrix FFT, all actual packed bytes,
typed basis coefficients, full real GEMM output and exact unpacking/normalization
are checked against independent CPU encoding and FP64 accumulation envelopes.
Selected columns retain source operands, typed packed values, products and unpacked
outputs for CPU CI; fully retained stage chains are replayed end to end. The capability
probe separately retains 44 supported raw cuFFT forward/inverse pairs and four
unsupported planning outcomes. Unit tests cover mixed radices, weak denominators,
overflow policy, signal replay, nonfinite decoding and artifact corruption.

The setup does not provide exclusive GPU access or hardware performance counters;
existing contexts are recorded, and clocks/power settings are not changed. HMMA
flags establish the selected arithmetic family, not achieved Tensor Core utilization.
The tested matrix FFT uses multiple global-memory stages; a more deeply fused custom
FFT, other factorizations, FP8/FP4 formats, callbacks, larger transforms, real-input
transforms, multidimensional FFT and arbitrary streaming filters need new experiments.
No claim of research novelty is made without a separate literature assessment.

## Reproduction

CPU checks require Python 3.12 and the pinned NumPy package:

```sh
python -m unittest discover -s tests -v
python verify_fft_controls.py
python verify_fft.py
python summarize_fft.py --check
```

Large NPZ files are published as content-addressed 16 MiB parts listed in
[the artifact manifest](results/fft-artifacts.json). The loader assembles and hashes
the exact original NPZ bytes in memory; locally regenerated original files take
precedence. No GPU or external SDK is required for CPU replay.

For a new GPU run, use Linux/CUDA 13.3 on SM120 and a new output basename:

```sh
python prepare_fft_dx.py
bash build_fft.sh
python fft_study.py --seed 20261126 --output fft-reproduced-a
python fft_study.py --seed 20261226 --output fft-reproduced-b --reverse
python verify_fft.py results/fft-reproduced-a.json results/fft-reproduced-b.json
```

The supplied canonical coefficient fixtures are already present. Regenerating new
fixtures is a new provenance chain (`fft_coefficients.py --output <new-directory>`),
not an instruction to overwrite the executed fixtures. GPU timings are not expected
to be byte-identical to the public records. `summarize_fft.py` derives the published
A/B tables; after deliberately replacing/reproducing those campaigns, regenerate
derived artifacts before using its `--check` mode. External SDK licenses remain
separate; see [third-party attribution](THIRD_PARTY.md).

Primary API references: [cuFFT precision and planning](https://docs.nvidia.com/cuda/archive/13.3.1/cufft/index.html),
[cuFFTDx operators and SM120](https://docs.nvidia.com/cuda/cufftdx/api/operators.html),
[cuFFTDx execution contracts](https://docs.nvidia.com/cuda/cufftdx/api/methods.html).
'''
    artifacts = {"results/fft-summary.csv": table.getvalue(), "results/fft-frontiers.json": json.dumps(summary, indent=2, allow_nan=False)+"\n", "FFT_REPORT.md": report}
    for name, content in artifacts.items():
        path = ROOT/name
        if args.check:
            assert path.read_text(encoding="utf-8") == content, f"Stale derived artifact: {name}"
        else:
            path.write_bytes(content.encode("utf-8"))
    print(f"FFT derived artifacts {'verified' if args.check else 'written'}: {len(rows)} available configurations")


if __name__ == "__main__":
    main()
