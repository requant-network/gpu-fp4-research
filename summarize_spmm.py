"""Derive fixed-error SpMM comparisons, conversion costs and reuse models."""
import argparse
import csv
import io
import json
import statistics

import numpy as np

from spmm_study import ROOT
from spmm_records import load_record

BOUNDARIES = ("prepared", "online-b", "online-both", "host-request")


def observations(case, boundary):
    return [o["wall_ms"] for o in case["host_observations"]] if boundary == "host-request" else case["timings_ms"][boundary]


def latency(case, boundary):
    return statistics.median(observations(case, boundary))


def worst_error(case, boundary):
    keys = [o["check"] for o in case["host_observations"]] if boundary == "host-request" else [case["before"], case["after_device"][boundary]]
    checks = [case["checks"][key]["metrics"] for key in keys]
    return max(max(c["relative_frobenius_error"], c["max_relative_column_error"]) for c in checks)


def active_blocks(group, case):
    return case["method"]["kind"] == 1 and group["formats"][case["format"]]["block_count"] > 0


def setup(group, case, boundary):
    value = group["formats"][case["format"]]["conversion_wall_ms"] + case["plan_setup_wall_ms"]
    return value if boundary == "host-request" else value + case["graph_setup_wall_ms"]


def outputs():
    records = [load_record(ROOT / f"results/spmm-{name}.json") for name in ("a", "b")]
    if not all(r["complete"] and not r["configuration"]["quick"] for r in records):
        raise ValueError("Both complete full SpMM cohorts are required")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["seed", "kind", "matrix", "m", "k", "width", "row_order", "nnz", "method", "family", "block", "minimum_fill",
                     "dense_blocks", "sparse_remainder_nnz", "boundary", "median_ms", "p10_ms", "p90_ms", "maximum_frobenius_or_column_error",
                     "cpu_format_wall_ms", "native_plan_wall_ms", "graph_setup_wall_ms", "owned_device_bytes", "host_format_payload_bytes",
                     "numerical_impl_flags", "direct_output", "arithmetic_model_checked"])
    frontier = {"scope": "Offline measured fixed-error choices. Eligibility requires BOTH original-input relative Frobenius and maximum per-column relative L2 error <= budget. Prepared/device choices use original B0 checkpoints; host choices require all four changing inputs. No deployable predictor or unseen-input error guarantee. Reuse estimates add separately measured one-time CPU format conversion/native plan setup and graph creation where applicable, not measured cold-process request distributions.",
                "choices": [], "reuse": [], "counts": {}}
    counts = {"attempted": 0, "available": 0, "timing_observations": 0, "checks": 0, "runtime_unique_entries": 0,
              "retained_sample_entries": 0, "full_witness_entries": 0, "full_witnesses": 0,
              "hybrid_with_dense_blocks": 0, "hybrid_without_dense_blocks": 0, "opaque_fp32_ell": 0,
              "preparation_bytes_checked": 0}
    by_example, summary = {}, {}
    owned = []
    for record in records:
        seed = record["configuration"]["seed"]
        for group in record["groups"]:
            spec = group["specification"]
            available = [c for c in group["cases"] if c["available"]]
            counts["attempted"] += len(group["cases"])
            counts["available"] += len(available)
            for case in available:
                method = case["method"]
                f = group["formats"][case["format"]]
                owned.append(case["native"]["owned_device_bytes"])
                if method["kind"] == 1:
                    counts["hybrid_with_dense_blocks" if active_blocks(group, case) else "hybrid_without_dense_blocks"] += 1
                counts["opaque_fp32_ell"] += int(method["kind"] == 3 and method["precision"] == 0)
                counts["preparation_bytes_checked"] += sum(case[key]["bytes_checked"] for key in ("prepared_before", "prepared_after_device", "prepared_after_host"))
                for check in case["checks"].values():
                    counts["checks"] += 1
                    counts["runtime_unique_entries"] += check["full_output_entries_checked"]
                    counts["retained_sample_entries"] += len(check["samples"])
                    if "witness_key" in check:
                        counts["full_witnesses"] += 1
                        counts["full_witness_entries"] += check["full_output_entries_checked"]
                for boundary in BOUNDARIES:
                    times = observations(case, boundary)
                    lo, hi = np.percentile(times, [10, 90])
                    writer.writerow([seed, spec["kind"], spec.get("name", spec.get("profile")), *group["shape"], spec["width"], spec["order"], group["nnz"],
                                     method["name"], method["kind"], f["block"], method.get("fill", 0), f["block_count"],
                                     case["native"]["buffer_bytes"][2] // 4 if method["kind"] == 1 else "", boundary,
                                     f"{latency(case,boundary):.9f}", f"{lo:.9f}", f"{hi:.9f}", f"{worst_error(case,boundary):.12g}",
                                     f["conversion_wall_ms"], case["plan_setup_wall_ms"], case["graph_setup_wall_ms"], case["native"]["owned_device_bytes"], f["host_format_bytes"],
                                     case["native"]["lt_numerical_flags"], case["native"]["direct_output"], case["checks"][case["before"]]["metrics"]["arithmetic_model_checked"]])
                    counts["timing_observations"] += len(times)
            for boundary in BOUNDARIES:
                for budget in record["configuration"]["error_budgets"]:
                    eligible = [c for c in available if worst_error(c, boundary) <= budget]
                    native = [c for c in eligible if c["method"]["kind"] != 1]
                    hybrid = [c for c in eligible if c["method"]["kind"] == 1]
                    active = [c for c in hybrid if active_blocks(group, c)]
                    best = min(eligible, key=lambda c: latency(c, boundary)) if eligible else None
                    base = min(native, key=lambda c: latency(c, boundary)) if native else None
                    candidate = min(hybrid, key=lambda c: latency(c, boundary)) if hybrid else None
                    dense_candidate = min(active, key=lambda c: latency(c, boundary)) if active else None
                    ratio = None if base is None or dense_candidate is None else latency(base, boundary) / latency(dense_candidate, boundary)
                    choice = {"seed": seed, "specification": spec, "shape": group["shape"], "boundary": boundary, "maximum_error": budget,
                              "best": None if best is None else best["method"]["name"], "best_ms": None if best is None else latency(best, boundary),
                              "best_native": None if base is None else base["method"]["name"], "native_ms": None if base is None else latency(base, boundary),
                              "best_hybrid": None if candidate is None else candidate["method"]["name"],
                              "hybrid_has_dense_blocks": None if candidate is None else active_blocks(group, candidate),
                              "best_active_hybrid": None if dense_candidate is None else dense_candidate["method"]["name"],
                              "active_hybrid_ms": None if dense_candidate is None else latency(dense_candidate, boundary), "active_hybrid_speedup": ratio}
                    frontier["choices"].append(choice)
                    stats = summary.setdefault((boundary, budget), {"groups": 0, "active_feasible": 0, "active_wins": 0, "ratios": []})
                    stats["groups"] += 1
                    stats["active_feasible"] += int(ratio is not None)
                    stats["active_wins"] += int(ratio is not None and ratio > 1)
                    if ratio is not None:
                        stats["ratios"].append(ratio)
                    if spec["order"] == "natural" and boundary in ("online-b", "host-request") and budget == .001:
                        by_example.setdefault((spec.get("name", spec.get("profile")), tuple(group["shape"]), spec["width"], boundary), []).append(choice)
                    for reuse in (1, 4, 16, 64, 256, 1024):
                        amortized = lambda c: setup(group, c, boundary) / reuse + latency(c, boundary)
                        reference = min(native, key=amortized) if native else None
                        alternative = min(active, key=amortized) if active else None
                        frontier["reuse"].append({"seed": seed, "specification": spec, "shape": group["shape"], "boundary": boundary, "maximum_error": budget,
                                                   "requests": reuse, "native": None if reference is None else reference["method"]["name"],
                                                   "native_amortized_ms": None if reference is None else amortized(reference),
                                                   "active_hybrid": None if alternative is None else alternative["method"]["name"],
                                                   "active_hybrid_amortized_ms": None if alternative is None else amortized(alternative),
                                                   "active_hybrid_speedup": None if reference is None or alternative is None else amortized(reference) / amortized(alternative)})
    frontier["counts"] = counts
    lines = ["# Hybrid sparse/dense SpMM on consumer Blackwell", "",
             "**Status: study 8 is complete within the operator protocol below. Studies 9-15 remain in the active fifteen-study program.**", "",
             f"Two independent input cohorts attempted {counts['attempted']:,} configurations; {counts['available']:,} ran. "
             f"There are {counts['timing_observations']:,} retained timing observations, {counts['checks']:,} distinct input/output check records covering "
             f"{counts['runtime_unique_entries']:,} full output entries at runtime, {counts['retained_sample_entries']:,} retained numerical samples and "
             f"{counts['full_witnesses']:,} complete numerical witnesses ({counts['full_witness_entries']:,} FP32 entries) for independent CPU replay.", "",
             f"At error 0.001 the active hybrid wins {summary['online-b', .001]['active_wins']}/{summary['online-b', .001]['groups']} online-B groups against the strongest feasible native reference. "
             "The tested decomposition therefore does not establish a general device-speed benefit. Native blocked/dense libraries are essential comparisons; setup and gathered/intermediate storage remain real costs. "
             "The few host-request ratios above one include dispatch/transfers and do not establish a faster device kernel.", "",
             "## Question and arithmetic contract", "",
             "Can dense Tensor Core blocks plus a sparse remainder outperform strong native sparse/block/dense libraries after encoding, gather/scatter, row ordering, conversion and reuse costs? "
             "A/B start as the same finite FP32 data and every path returns the original unpermuted FP32 matrix product. Fixed-error choices require BOTH relative Frobenius error and maximum per-column relative L2 error <=1e-5, 1e-3 or 1e-2 against an independent FP64 product of the original FP32 values. "
             "One exactly zero B column must produce exactly zero output. Different precision paths are compared only where they meet the same error budget.", "",
             "CSR references use algorithms 1/2/3 with the documented column-major preference for algorithm 1 and row-major for algorithms 2/3, in FP32/FP16/BF16. "
             "Block references include cuSPARSE BSR and Blocked-ELL at block sizes 16/32/64 and those three storage precisions. Dense cuBLASLt FP32/FP16/BF16 is also included; FP32 requests pedantic compute. "
             "Up to eight cuBLASLt candidates are attempted and three calibration timings per usable candidate select the fastest outside measured operations. The custom dense-block and full dense low-precision cuBLASLt calls must report the HMMA numerical implementation flag. "
             "Library flags and actual SM120 execution establish the selected native implementation, not achieved Tensor Core utilization or an architecture ceiling.", "",
             "The custom hybrid selects every occupied block whose stored nonzero count reaches 25%, 50% or 75% of its full padded 16/32/64 square. "
             "It keeps the exact remaining entries in FP32 CSR, encodes selected blocks/B slices as FP16 or BF16, computes batched cuBLASLt products and adds them to the native FP32 CSR remainder. "
             "A 32-by-32 shared-memory transpose gathers B with coalesced input/output access; the final row-block reduction is deterministic and avoids atomics. Zero dense-block cases are explicitly labeled and cannot count as Tensor Core hybrid wins.", "",
             "## Data, shapes and ordering", "",
             "Synthetic matrices have planted 32-square blocks at 128, mixed block density at 1024/4096, and unstructured 1% sparsity at 2048. "
             "Block rows alternate approximately 7/8 and 3/8 fill, with a small unstructured remainder, so changing thresholds moves actual work. Inputs use portable PCG64 integer mantissas; synthetic row zero is empty. "
             "Dense widths are 64/256. Natural and stable descending-degree row ordering are crossed with every matrix/width; two additional scrambled-order controls test 1024 mixed blocks and BCSSTK09 at width 64. "
             "The second cohort uses an independent matrix/B seed and reverses group/method order. Four independent dense B matrices cycle through host requests.", "",
             "Real matrices are the structural stiffness matrices [BCSSTK09](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc1/bcsstk09.html) (1083) "
             "and [BCSSTK10](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc1/bcsstk10.html) (1086), attributed to John Lewis / Harwell-Boeing / NIST Matrix Market. "
             "The original files are retained with hashes. They are cast to common FP32 and scaled globally by an exact power of two so the largest magnitude is below one; this preparation is common input generation outside every method's timers. "
             "Both cohorts use these same two real matrices with independent B inputs; they are not four independent physical matrices or a complete graph-learning application. Irregular 193-by-257 padding and non-square/zero controls are additionally retained in the quick campaign.", "",
             "## Complete timing and setup boundaries", "",
             "Prepared timings reuse encoded A/B and measure compute plus needed reduction/crop/unpermutation. Online-B additionally encodes/gathers FP32 B; online-both also re-encodes FP32 A format values with an unchanged sparsity pattern. "
             "FP32 row-major paths alias their source buffers. Natural-order native references write directly to the requested output where layout permits; dense GEMM uses row-major layouts. "
             "Device timings are CUDA Graph averages with five warmups and thirty observations: eight operations per graph, or one when m*k*n>=2^30. These repeat B0 and do not establish changing-B latency.", "",
             "Complete uncaptured host requests include pageable FP32 B H2D, online-B computation and full FP32 C D2H, with four B inputs cycling through thirty requests. "
             "CPU CSR-based reordering/format construction, native allocation/A upload/preprocessing/heuristic calibration and graph construction are measured separately. "
             "Natural CSR consumes the already provided canonical source arrays without sorting/copying them. Hybrid format construction uses CSR block keys/counts rather than materializing a full dense A; BSR/ELL retain the padding their vendor format requires. "
             "CPU conversion times include validation and sorting where needed; they are implementation costs, not optimal conversion lower bounds.", "",
             "The [reuse estimates](results/spmm-frontiers.json) add CPU conversion plus native plan setup divided by R=1/4/16/64/256/1024 to each measured steady operation. "
             "Device graph estimates additionally charge construction of the three measured graphs; uncaptured host estimates do not. These are component-based fixed-A models, not measured cold-process latency distributions or changing-pattern GPU timings. "
             "Input generation, validation/readbacks outside the declared host boundary and CPU numerical witnesses are excluded equally from timed operations.", "",
             f"Explicit owned device allocations request {min(owned):,} to {max(owned):,} buffer bytes. They include source/encoded buffers, padded/gathered B, dense block outputs, reduction/output buffers and supplied workspaces. "
             "Opaque library/graph allocations and other GPU contexts are additional; this is not a total peak-memory measurement. The CSV separately retains CPU format payload bytes, which may reference the original CSR arrays and are not a CPU allocator peak.", "",
             "## Fixed-error comparison with the strongest feasible reference", "",
             "Each group compares the fastest eligible hybrid with actual dense blocks against the fastest eligible CSR/BSR/ELL/dense library path. A ratio above one favors the hybrid. "
             "Groups with no feasible active hybrid remain infeasible rather than disappearing from the denominator. Results aggregate dimensions/order/seed; repeated timings are not independent input trials.", "",
             "| Boundary | Maximum error | Active hybrid feasible groups | Faster active hybrid groups | Median native / active-hybrid time where feasible |", "|---|---:|---:|---:|---:|"]
    for boundary in BOUNDARIES:
        for budget in (1e-5, .001, .01):
            stats = summary[boundary, budget]
            ratio = f"{statistics.median(stats['ratios']):.4f}" if stats["ratios"] else "unavailable"
            lines.append(f"| {boundary} | {budget:g} | {stats['active_feasible']}/{stats['groups']} | {stats['active_wins']}/{stats['groups']} | {ratio} |")
    lines += ["", "## Natural-order examples at error 0.001", "",
              "Ranges span the two cohorts. Online-B numbers include gather/encoding/reduction. A single best native label is not assumed across cohorts; all chosen methods remain in the raw frontier.", "",
              "| Matrix / width | Best native online-B, microseconds | Native / active-hybrid online-B | Native / active-hybrid host request |", "|---|---:|---:|---:|"]
    example_keys = sorted({(name, shape, width) for name, shape, width, boundary in by_example})
    def span(values):
        return f"{min(values):.4g}-{max(values):.4g}" if values else "infeasible"
    for name, shape, width in example_keys:
        online = by_example.get((name, shape, width, "online-b"), [])
        host = by_example.get((name, shape, width, "host-request"), [])
        lines.append(f"| {name}, {shape[0]}x{shape[1]} / {width} | {span([c['native_ms']*1000 for c in online if c['native_ms'] is not None])} | "
                     f"{span([c['active_hybrid_speedup'] for c in online if c['active_hybrid_speedup'] is not None])} | {span([c['active_hybrid_speedup'] for c in host if c['active_hybrid_speedup'] is not None])} |")
    lines += ["", "## Reuse and negative controls", "",
              "The table below summarizes modeled online-B costs at error 0.001. Native and active-hybrid candidates are reselected at each reuse count using complete setup plus steady operation cost. "
              "A short warmed kernel alone does not establish a benefit for a newly converted matrix.", "",
              "| Reuse R | Active hybrid feasible groups | Active hybrid faster after setup | Median native / active-hybrid amortized time |", "|---|---:|---:|---:|"]
    for reuse in (1, 4, 16, 64, 256, 1024):
        rows = [r for r in frontier["reuse"] if r["requests"] == reuse and r["boundary"] == "online-b" and r["maximum_error"] == .001]
        ratios = [r["active_hybrid_speedup"] for r in rows if r["active_hybrid_speedup"] is not None]
        lines.append(f"| {reuse} | {len(ratios)}/{len(rows)} | {sum(r>1 for r in ratios)}/{len(rows)} | " + (f"{statistics.median(ratios):.4f}" if ratios else "unavailable") + " |")
    lines += ["", f"Among hybrid configurations, {counts['hybrid_with_dense_blocks']:,} have actual dense blocks and {counts['hybrid_without_dense_blocks']:,} have none. "
              "A zero-block choice executes a CSR path and cannot substantiate Tensor Core acceleration. Degree/scrambled ordering can destroy block locality, increase ELL padding or leave no eligible dense blocks; those regressions and conversion costs are retained.", "",
              f"There are {counts['opaque_fp32_ell']:,} FP32-storage ELL configurations. On this library/device they are approximate and are not labeled strict FP32 arithmetic. "
              "The report does not certify their opaque internal quantizer. "
              "Their complete original-input error is still measured against FP64 and used for eligibility. Strict FP32 CSR/dense and independently decoded FP16/BF16 block references remain available. API storage dtype alone must not be mistaken for full arithmetic precision.", "",
              "Zero-matrix controls cover native CSR and both hybrid precisions, including all graph boundaries and no dense blocks. Wrong B shapes and null native plans are rejected. "
              "The initial 216-configuration CSR capability probe covers algorithm/layout/preprocessing combinations, including changed B after preprocessing; it is independently replayed and is not counted as study completion by itself. "
              "A preliminary flat/strided B-gather version was archived privately (one complete and one partial cohort) before improving coalesced packing, direct output and row-major dense references. Both published full cohorts execute the final source hashes; preliminary timings are not mixed into these results.", "",
              "## Independent verification and limits", "",
              "A separate dense NumPy oracle checks lossless CSR/dense-block decomposition; source CSR format preparation is independently decoded for every shape/order/format. "
             f"GPU encoded A/B and source buffers are checked byte-for-byte before timing, after device timings and after host requests ({counts['preparation_bytes_checked']:,} checked bytes). "
              "CPU CI reconstructs their complete expected bytes and checks all retained hashes, including padding and row permutations.", "",
              "Full actual outputs are checked against original and independently rounded-input FP64 products before timing, after the last actual graph output of each boundary, and for each distinct changing-B output hash. "
              "Every host output is read back and hash-linked to a check; distinct results from library reduction-order differences are checked separately. Intermediate graph outputs are not retained. Sixty-four actual output coordinates per check support independent CPU row-dot replay with FP32 accumulation envelopes. "
              "Complete witnesses cover every method on natural 128-square/width-64 inputs and selected CSR/block/dense/hybrid references on both real matrices at width 64. Other large complete output arrays are not retained; full large-case quality relies on source-bound runtime CPU checks plus retained samples, not a claim of full large-output CI replay. "
              "Rounded-product verification uses relative error 2e-5 as an experimental screening contract, not a universally correctly-rounded Tensor Core proof. The opaque FP32 ELL exception is explicitly flagged.", "",
              "Native runs use the recorded CUDA 13.3 toolchain on RTX 5060 Ti / SM120. "
              "The [NVIDIA cuSPARSE API](https://docs.nvidia.com/cuda/archive/13.2.0/cusparse/index.html#cusparsespmm) documents the algorithm/layout/preprocessing interfaces; actual runtime versions, flags, supported outcomes and outputs are retained. "
              "Other GPU contexts remain present and no clocks, power settings or services were changed. Results are exploratory shared-system observations and do not establish an architecture ceiling, optimal converter/kernel, scientific novelty or general graph-workload performance.", "",
              "## Reproduction", "", "```sh", "bash build_spmm.sh",
              "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python spmm_study.py --seed 20261125 --quick --output spmm-quick.json",
              "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python spmm_study.py --seed 20261125 --output spmm-a.json",
              "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python spmm_study.py --seed 20261225 --reverse --output spmm-b.json",
              "python verify_spmm.py", "python summarize_spmm.py --check", "```", "",
              "See the [complete CSV](results/spmm-summary.csv), [fixed-error/reuse frontiers](results/spmm-frontiers.json), "
              "[first cohort archive](results/spmm-a.json.gz) and [second cohort archive](results/spmm-b.json.gz) for all configurations, failures, timings, quality, memory, setup and source/data/binary bindings. "
              "The gzip files preserve the executed JSON bytes exactly. CPU tools accept these archives or newly reproduced uncompressed JSON; the latter takes precedence when present.", ""]
    return {"SPMM_REPORT.md": "\n".join(lines), "results/spmm-summary.csv": stream.getvalue(),
            "results/spmm-frontiers.json": json.dumps(frontier, indent=2, allow_nan=False) + "\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, content in outputs().items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Stale SpMM artifact: {name}")
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    print("SpMM derived artifacts match" if args.check else "Wrote SpMM report, complete CSV and fixed-error/reuse frontiers")


if __name__ == "__main__":
    main()
