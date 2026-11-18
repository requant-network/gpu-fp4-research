"""Derive English tables and CSV from the retained pilot observations."""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics

import numpy as np

from reference import PROFILES, SHAPES

ROOT = Path(__file__).resolve().parent
BOUNDARIES = ("prepacked", "online-a", "online-both")


def artifacts():
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((ROOT / "results").glob("pilot-*.json"))]
    if len(records) != 2 or not all(record.get("complete") for record in records):
        raise ValueError("Expected two complete pilot cohorts")
    lookup = {(record["seed"], tuple(case["shape"]), case["profile"], case["mode"]): case
              for record in records for case in record["cases"]}
    seeds = sorted(record["seed"] for record in records)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["seed", "m", "n", "k", "profile", "mode", "boundary", "median_ms", "p10_ms",
                     "p90_ms", "relative_frobenius_error", "maximum_absolute_error", "algorithm_candidate"])
    for record in records:
        for case in sorted(record["cases"], key=lambda c: (c["shape"], c["profile"], c["mode"])):
            for boundary in BOUNDARIES:
                values = case["timings_ms"][boundary]
                writer.writerow([record["seed"], *case["shape"], case["profile"], case["mode"], boundary,
                                 f"{statistics.median(values):.8f}", f"{np.quantile(values, .1):.8f}",
                                 f"{np.quantile(values, .9):.8f}",
                                 f"{case['checks']['relative_frobenius_error']:.10g}",
                                 f"{case['checks']['maximum_absolute_error']:.10g}", case["algorithm_candidate"]])
    count = sum(len(record["cases"]) for record in records)
    entries = sum(case["checks"]["checked_output_entries"] for record in records for case in record["cases"])
    lines = ["# Pilot results", "", "A research project by [Requant Network](https://requant.network).", "",
             "**Status.** This page presents the original bounded pilot. The [completed programme ledger](PROGRAM.md) indexes all fifteen investigations.", "",
             f"Two input cohorts contain {count} mode/shape/profile cases, {count*90:,} retained timing observations "
             f"and {entries:,} output entries checked both before and after timing. Twenty complete small-case "
             "numerical witnesses are replayable on CPU. Each observation times 32 repeated operations.", "",
             "These are exploratory shared-system observations. See [methods](METHODS.md) for timing boundaries, "
             "quantization recipes, candidate search and limitations. Every format uses FP32 source operands "
             "and FP32 output. Speed ratios below compare different achieved errors; they are **not quality-matched speedups**.", "",
             "## FP4 versus FP16", "",
             "Values are FP16 median time / FP4 median time, shown as the range across the two cohorts. "
             "Above 1 favors FP4. The last column is FP4 relative Frobenius error against the original operands.", "",
             "| M x N x K | Profile | Prepacked ratio | Online A ratio | Online A+B ratio | FP4 error |",
             "|---|---|---:|---:|---:|---:|"]
    wins = {boundary: 0 for boundary in BOUNDARIES}
    for shape in SHAPES:
        for profile in PROFILES:
            ratios = []
            for boundary in BOUNDARIES:
                values = []
                for seed in seeds:
                    fp4 = lookup[seed, shape, profile, "fp4"]
                    fp16 = lookup[seed, shape, profile, "fp16"]
                    ratio = statistics.median(fp16["timings_ms"][boundary]) / statistics.median(fp4["timings_ms"][boundary])
                    values.append(ratio)
                    wins[boundary] += int(ratio > 1)
                ratios.append(f"{min(values):.2f}-{max(values):.2f}")
            errors = [100 * lookup[seed, shape, profile, "fp4"]["checks"]["relative_frobenius_error"] for seed in seeds]
            lines.append(f"| {' x '.join(map(str, shape))} | {profile} | " + " | ".join(ratios) +
                         f" | {min(errors):.2f}-{max(errors):.2f}% |")
    lines += ["", "## Scope of the observations", "",
              "Out of 24 shape/profile/cohort comparisons, FP4 has lower median time than FP16 in "
              + ", ".join(f"{wins[boundary]}/24 {boundary}" for boundary in BOUNDARIES) + ".", "",
              "The baseline FP4 encoder, scale choice and fixed CUTLASS configuration are not an optimized "
              "FP4 stack. cuBLASLt comparators select among up to eight heuristic candidates. "
              "The workload repeatedly reuses the same buffers and inputs in captured graphs. "
              "These results do not establish model quality, streaming performance, request latency, "
              "the best achievable FP4 kernel, or an advantage at equal error.", "",
              "All five formats, errors and observation percentiles are available in "
              "[the full CSV](results/summary.csv). Raw cohort records and NPZ witnesses are in "
              "[results](results). Run `python verify_results.py` before interpreting the tables, "
              "and `python summarize.py --check` to check that derived files match the raw records.", ""]
    return {"results/summary.csv": output.getvalue(), "RESULTS.md": "\n".join(lines)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, contents in artifacts().items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != contents:
                raise ValueError(f"Derived artifact is stale: {name}")
        else:
            path.write_text(contents, encoding="utf-8", newline="\n")
    print("Derived tables match raw results" if args.check else "Wrote RESULTS.md and results/summary.csv")


if __name__ == "__main__":
    main()
