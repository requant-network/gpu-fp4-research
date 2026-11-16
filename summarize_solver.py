"""Derive accuracy-constrained solver comparisons without hiding failed solves."""
import argparse
import csv
import io
import json
import statistics

from solver_study import ROOT


def eligible(group, case, boundary):
    observations = case[boundary]
    return all(group["solutions"][o["solution"]]["accuracy"]["target_pass"]
               and o["trace"][-1][1:3] == [0, 0] and o["trace"][-1][4] for o in observations)


def median(case, boundary):
    return statistics.median(o["cuda_ms" if boundary == "resident_observations" else "host_ms"] for o in case[boundary])


def outputs():
    records = [json.loads((ROOT / f"results/solver-{name}.json").read_text(encoding="utf-8")) for name in ("a", "b")]
    if not all(r["complete"] and not r["configuration"]["quick"] for r in records):
        raise ValueError("Both completed full solver cohorts are required")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["seed", "kind", "matrix", "n", "nrhs", "condition_exponent", "scale_exponent", "method", "boundary",
                     "target_pass_all_observations", "median_ms", "p10_observation_ms", "p90_observation_ms", "maximum_residual", "maximum_forward_error",
                     "plan_setup_wall_ms", "owned_device_bytes", "native_fresh_speedup_if_feasible", "best_library_speedup_if_feasible"])
    frontiers = {"scope": "Offline empirical time to BOTH max per-RHS relative L2 residual <=1e-12 and forward error <=1e-8 on every retained observation. No feasible solution is reported for unmet accuracy. Factor reuse is separate from fresh solves. These choices use the accuracy oracle offline and are not a deployable selector.",
                 "choices": [], "reuse_amortization": []}
    counts = {"attempted": 0, "available": 0, "observations": 0, "witness_solutions": 0, "witness_entries": 0, "exact_residual_entries": 0}
    method_stats, condition_stats, real_rows, failures = {}, {}, [], []
    for record in records:
        seed = record["configuration"]["seed"]
        for group in record["groups"]:
            spec = group["specification"]
            cases = {case["method"]["name"]: case for case in group["cases"] if case["available"]}
            counts["attempted"] += len(group["cases"])
            counts["available"] += len(cases)
            counts["witness_solutions"] += len(group["solutions"])
            counts["witness_entries"] += len(group["solutions"]) * group["n"] * group["nrhs"]
            counts["exact_residual_entries"] += sum(len(w["exact_residual_samples"]) * group["nrhs"]
                                                     for w in group["solutions"].values() if w["exact_residual_samples"] is not None)
            for boundary in ("resident_observations", "host_observations"):
                fresh = [c for name, c in cases.items() if name != "cholesky64_reuse" and eligible(group, c, boundary)]
                builtin = [c for c in fresh if c["method"].get("mode") != 2]
                best = min(fresh, key=lambda c: median(c, boundary)) if fresh else None
                best_library = min(builtin, key=lambda c: median(c, boundary)) if builtin else None
                baseline = cases["cholesky64"]
                base_ok = eligible(group, baseline, boundary)
                frontiers["choices"].append({"seed": seed, "specification": spec, "boundary": boundary,
                                             "selected": None if best is None else best["method"]["name"],
                                             "median_ms": None if best is None else median(best, boundary),
                                             "fastest_library": None if best_library is None else best_library["method"]["name"]})
                for name, case in cases.items():
                    ok = eligible(group, case, boundary)
                    observations = case[boundary]
                    values = sorted(o["cuda_ms" if boundary == "resident_observations" else "host_ms"] for o in observations)
                    quality = [group["solutions"][o["solution"]]["accuracy"] for o in observations]
                    residuals = [q["max_relative_residual"] for q in quality if q["max_relative_residual"] is not None]
                    forwards = [q["max_forward_error"] for q in quality if q["max_forward_error"] is not None]
                    native_ratio = median(baseline, boundary) / median(case, boundary) if ok and base_ok and name != "cholesky64_reuse" else None
                    library_ratio = median(best_library, boundary) / median(case, boundary) if ok and best_library is not None and name != "cholesky64_reuse" else None
                    writer.writerow([seed, spec["kind"], spec.get("name", "walsh-spd"), group["n"], group["nrhs"], spec.get("exponent"), spec.get("scale_exponent", 0),
                                     name, boundary, ok, f"{median(case, boundary):.9f}", f"{values[2]:.9f}", f"{values[26]:.9f}",
                                     max(residuals, default=None), max(forwards, default=None), case["plan_setup_wall_ms"],
                                     case["before"]["metadata"]["owned_device_bytes"], native_ratio, library_ratio])
                    counts["observations"] += len(observations)
                    stats = method_stats.setdefault((name, boundary), {"total": 0, "pass": 0, "native_ratios": [], "library_ratios": []})
                    stats["total"] += 1
                    stats["pass"] += int(ok)
                    if native_ratio is not None:
                        stats["native_ratios"].append(native_ratio)
                    if library_ratio is not None:
                        stats["library_ratios"].append(library_ratio)
                    if spec["kind"] == "controlled" and boundary == "resident_observations":
                        cstats = condition_stats.setdefault((spec["exponent"], name), [0, 0])
                        cstats[0] += int(ok)
                        cstats[1] += 1
                    if name in ("adaptive", "equilibrated_adaptive") and boundary == "host_observations":
                        for observation in observations:
                            q = group["solutions"][observation["solution"]]["accuracy"]
                            if observation["trace"][-1][4] and not q["target_pass"]:
                                failures.append({"seed": seed, "specification": spec, "method": name, "variant": observation["variant"],
                                                 "gpu_residual": observation["trace"][-1][5], "forward_error": q["max_forward_error"],
                                                 "solution": observation["solution"], "witness_file": group["witness_file"]})
                reuse = cases["cholesky64_reuse"]
                if eligible(group, reuse, boundary):
                    setup = statistics.median(o["cuda_ms" if boundary == "resident_observations" else "host_ms"] for o in reuse["factor_observations"])
                    for requests in (1, 4, 16, 64):
                        amortized = setup / requests + median(reuse, boundary)
                        frontiers["reuse_amortization"].append({"seed": seed, "specification": spec, "boundary": boundary, "requests": requests,
                                                                "factor_setup_median_ms": setup, "solve_median_ms": median(reuse, boundary),
                                                                "amortized_ms": amortized,
                                                                "speedup_vs_fresh_cholesky": median(baseline, boundary) / amortized if base_ok else None})
            if spec["kind"] == "real" and spec["scale_exponent"] == 0:
                real_rows.append((seed, spec, group, cases))
    lines = ["# Adaptive precision for dense SPD linear systems", "",
             "**Status: study 7 is complete within the protocol below. Studies 8-15 remain part of the active fifteen-study program.**", "",
             f"Two cohorts attempted {counts['attempted']:,} configurations; {counts['available']:,} are available and {counts['attempted']-counts['available']} explicitly reject multi-RHS GMRES. "
             f"The campaign retains {counts['observations']:,} solve timing observations, {counts['witness_solutions']:,} complete output witnesses "
             f"({counts['witness_entries']:,} FP64 entries) and {counts['exact_residual_entries']:,} exact-integer residual samples. Every measured output is copied and linked to its complete retained witness.", "",
             "## Contract and inputs", "",
             "All methods start from the same stored FP64 A/B and return FP64 X. A solve passes only if every RHS has relative L2 residual <=1e-12 AND relative L2 forward error <=1e-8. "
             "Eligibility in the tables requires passing every retained observation for that boundary, including all four changing RHS variants for host requests. "
             "Forward error is measured against a mathematically exact manufactured solution of the stored system; it is not measured against another rounded GPU solver.", "",
             "Controlled matrices use a signed and row-permuted Walsh basis, dimensions 128/512/1024/2048, and exact spectral condition numbers 1, 2^12, 2^24, 2^36 and 2^40. "
             "Integer Walsh transforms of dyadic eigenvalues construct every A entry exactly in FP64. RHS vectors combine three basis directions, including smallest/largest eigenvalues; B and truth are exactly representable. "
             "The four RHS variants keep A unchanged. Cases have one or four RHS columns. The second cohort independently changes the matrix/RHS seed and reverses matrix/method order.", "",
             "Real inputs are [BCSSTK09](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc1/bcsstk09.html), a clamped plate stiffness matrix (1083), "
             "and [BCSSTK10](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc1/bcsstk10.html), a hot washer stiffness matrix (1086), supplied by John Lewis to the Harwell-Boeing collection and distributed by NIST Matrix Market. "
             "Each is tested at exact global scales 2^-20, 1 and 2^20 with one/four RHS columns. Truth consists of selected unit vectors and B of the corresponding original columns, so no RHS-rounding ambiguity is hidden. "
             "The two cohorts change real-matrix RHS directions, not the underlying two physical matrices. These are dense-storage precision tests with manufactured loads, not original physical-load accuracy tests or sparse-solver comparisons. "
             "Original compressed files, URLs, hashes and attribution are in [the data manifest](fixtures/solver-matrices.json).", "",
             "## Native methods and costs", "",
             "The native bridge directly calls CUDA 13.3 cuSOLVER on GPU 0 / SM120. Strong FP64 references include SPD Cholesky (Dpotrf/Dpotrs) and general LU. "
             "Other references request FP32/FP16/BF16/TF32 factorization with no refinement or classical iterative refinement, FP16 GMRES for one RHS, and the library's FP16 classical-refinement fallback. "
             "Equilibrated controls compute D=diag(A)^-1/2, solve DAD y=Db, and return x=Dy; their external residual is always evaluated against original A/B. "
             "Requested lowest-precision enums are configuration evidence, not proof that every opaque library operation uses Tensor Cores. No FP4 factorization capability is asserted.", "",
             "The custom adaptive policy attempts FP16 classical refinement with limit 4, FP32 with limit 12, then FP64 Cholesky. Each attempt starts from original inputs, charges restore/scaling and gates on native status/device info plus full original-system relative L2 residual <=1e-12. "
             "An equilibrated version applies Jacobi scaling to the two lower-precision attempts. The online policy never sees the manufactured truth, condition label or forward error. "
             "Raw negative iteration/fallback/nonconvergence codes are retained and are not interpreted as negative computational work.", "",
             "Resident timers include fresh factorization, solve, full FP64 original-system residual, scalar readback and host decisions. CUDA events span library host orchestration gaps; separate host wall times are retained. "
             "Complete host-request timers additionally include pageable FP64 A+B H2D and full FP64 X D2H, cycling through four RHS variants. "
             "Plan creation, allocation, CPU input generation, numerical replay and history extraction are outside solve timers; plan setup wall time and explicit owned device bytes remain in the CSV. "
             "Opaque cuSOLVER/cuBLAS handle/internal allocation costs are additional, so owned bytes are not total peak memory. Thirty observations follow five resident warmups, without changing clocks, services or power settings.", "",
             "## Accuracy-constrained outcomes", "",
             "A ratio above 1 favors the named method. Ratios use only cases where BOTH the candidate and comparator pass the full contract. Best-library comparisons include the fastest feasible direct/IR/fallback reference in the same group and boundary. "
             "Failed solves retain their timings in the CSV but cannot earn an accuracy-constrained speedup.", "",
             "| Method | Resident passing groups | Host passing groups | Median resident speedup vs fresh FP64 Cholesky | Median resident speedup vs best feasible library |", "|---|---:|---:|---:|---:|"]
    for name in [c["method"]["name"] for c in records[0]["groups"][0]["cases"]]:
        resident = method_stats.get((name, "resident_observations"))
        host = method_stats.get((name, "host_observations"))
        if resident is None:
            continue
        ratio = lambda values: f"{statistics.median(values):.4f}" if values else "unavailable"
        lines.append(f"| {name} | {resident['pass']}/{resident['total']} | {host['pass']}/{host['total']} | {ratio(resident['native_ratios'])} | {ratio(resident['library_ratios'])} |")
    lines += ["", "Complete host requests include changing B and both transfers. The ratios below retain the same strict feasibility condition; they are separate from resident timings.", "",
              "| Method | Median host speedup vs fresh FP64 Cholesky | Median host speedup vs best feasible library |", "|---|---:|---:|"]
    for name in ("ir_fp32", "ir_fp16", "library_fallback_fp16", "equilibrated_library_fp16", "adaptive", "equilibrated_adaptive"):
        stats = method_stats[name, "host_observations"]
        lines.append(f"| {name} | {ratio(stats['native_ratios'])} | {ratio(stats['library_ratios'])} |")
    lines += ["", "## Controlled conditioning and forward-error failures", "",
              "High conditioning can defeat even native FP64 at the fixed 1e-8 forward-error target. This is retained as an unmet accuracy outcome, not silently relaxed. "
              "The following resident success counts span dimensions, RHS counts and both seeds.", "",
              "| Condition number | Cholesky64 | FP32 IR | FP16 IR | Adaptive | Equilibrated adaptive |", "|---|---:|---:|---:|---:|---:|"]
    for exponent in (0, 12, 24, 36, 40):
        values = [condition_stats[exponent, name] for name in ("cholesky64", "ir_fp32", "ir_fp16", "adaptive", "equilibrated_adaptive")]
        lines.append(f"| 2^{exponent} | " + " | ".join(f"{ok}/{total}" for ok, total in values) + " |")
    lines += ["", f"There are {len(failures):,} changing-RHS adaptive observations whose online residual gate passes while the offline combined accuracy contract fails. "
              "Repeated observations of a RHS are not independent failure trials. The [complete failure list](results/solver-frontiers.json) keeps seed/specification, method, RHS variant and exact witness links. "
              "The tested residual-only gate therefore does not certify forward error. A trusted condition estimate or stronger certified stopping rule is future work, not an established capability.", "",
              "| Seed | Controlled n / condition | Method | RHS variant | GPU relative residual | Actual forward error |", "|---|---|---|---:|---:|---:|"]
    shown = set()
    for failure in failures:
        key = failure["seed"], failure["method"]
        if key in shown or failure["specification"]["kind"] != "controlled":
            continue
        shown.add(key)
        spec = failure["specification"]
        lines.append(f"| {failure['seed']} | {spec['n']} / 2^{spec['exponent']} | {failure['method']} | {failure['variant']} | {failure['gpu_residual']:.5g} | {failure['forward_error']:.5g} |")
    lines += ["", "## Real-matrix results at original scale", "",
              "Reported times are resident milliseconds. A failed candidate is labeled failed rather than included in an equal-accuracy comparison. Global-scale variants and complete host requests are in the CSV.", "",
              "| Seed | Matrix / RHS | Cholesky64 | FP32 IR | Equilibrated library FP16 | Adaptive | Equilibrated adaptive |", "|---|---|---:|---:|---:|---:|---:|"]
    for seed, spec, group, cases in real_rows:
        def cell(name):
            case = cases[name]
            return f"{median(case, 'resident_observations'):.4f}" + ("" if eligible(group, case, "resident_observations") else " (failed)")
        lines.append(f"| {seed} | {spec['name']} / {spec['nrhs']} | " + " | ".join(cell(name) for name in
                     ("cholesky64", "ir_fp32", "equilibrated_library_fp16", "adaptive", "equilibrated_adaptive")) + " |")
    lines += ["", "## Factor reuse and verification", "",
              "`cholesky64_reuse` factors A once and solves changing RHS with the retained factor. The factor setup is separately measured thirty times; host setup includes A+B upload. "
              "[Amortization records](results/solver-frontiers.json) report factor_setup/R + solve for R=1/4/16/64. The setup restores a B work buffer even though factorization does not use it, so its small copy cost is conservatively included. "
              "Reuse solves still charge the original-system residual and host gate; host requests upload only B. This separate fixed-A workload cannot be mixed into the fresh-factorization frontier.", "",
              "| Reused RHS request count | Median resident amortized speedup vs fresh Cholesky | Median host amortized speedup vs fresh Cholesky |", "|---|---:|---:|"]
    for requests in (1, 4, 16, 64):
        values = [[row["speedup_vs_fresh_cholesky"] for row in frontiers["reuse_amortization"]
                   if row["requests"] == requests and row["boundary"] == boundary and row["speedup_vs_fresh_cholesky"] is not None]
                  for boundary in ("resident_observations", "host_observations")]
        lines.append(f"| {requests} | " + " | ".join(ratio(v) for v in values) + " |")
    lines += ["", "Every unique actual output is retained in full; every timed observation names its exact solution hash/witness. CPU replay reconstructs all inputs and checks every output's residual and analytic forward error. "
              "Sixteen rows per system/RHS are also evaluated by arbitrary-precision integer dot products with exact binary64 input representations and one final rounding, exposing ordinary FP64 residual cancellation. "
              "Cross-platform CPU residual comparisons use a gamma_n dot-rounding bound, while exact integer samples must match bit-for-bit and accuracy pass/fail labels must remain unchanged. "
              "Runtime preparation checks compare every scaled matrix/RHS/diagonal entry against independent NumPy operations; CI checks their provenance/counts and independently replays complete solution witnesses.", "",
              "Native controls cover invalid dimensions/RHS/mode, null plan, unsupported multi-RHS GMRES, singular Cholesky and repeated IRS calls. "
              "A first quick attempt reused IRSInfos and produced ALLOC_FAILED on subsequent calls; its source/binary/raw outputs were archived privately. Final campaigns destroy/create this per-solve object inside the timed operation and include a four-call regression control. "
              "History is requested only for refinement and retained only for successful positive iteration counts with a defined prefix; unrefined/zero/negative-code calls do not claim initialized histories. "
              "The separate sixty-case capability preflight is replayed but is not counted as a performance study or assumed to meet this study's stricter targets.", "",
              "The implementation follows the [NVIDIA cuSOLVER API](https://docs.nvidia.com/cuda/archive/13.3.1/cusolver/index.html), with explicit refinement, precision, fallback and iteration limits. "
              "Library history entries are its own residual norms/iteration counts, not the external relative L2 metric. "
              "Library default stopping tolerances are left unchanged; the external original-system gate is a separate check. GPU contexts from other processes remain present, so measurements are exploratory shared-system observations. "
              "No scientific-novelty, architecture-ceiling, universal solver or certified forward-error claim is made.", "",
              "## Reproduction", "", "```sh", "bash build_solver.sh",
              "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python solver_study.py --seed 20261124 --quick --output solver-quick.json",
              "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python solver_study.py --seed 20261124 --output solver-a.json",
              "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python solver_study.py --seed 20261224 --reverse --output solver-b.json",
              "python verify_solver.py", "python summarize_solver.py --check", "```", "",
              "Use the recorded CUDA 13.3 toolchain and NumPy environment. The [complete table](results/solver-summary.csv), [first cohort](results/solver-a.json) "
              "and [second cohort](results/solver-b.json) retain setup, failures, raw iteration/status codes, histories, timing series, source/data/binary hashes and output witnesses.", ""]
    frontiers["residual_gate_forward_failures"] = failures
    frontiers["counts"] = counts
    return {"SOLVER_REPORT.md": "\n".join(lines), "results/solver-summary.csv": stream.getvalue(),
            "results/solver-frontiers.json": json.dumps(frontiers, indent=2, allow_nan=False) + "\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, content in outputs().items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Stale solver artifact: {name}")
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    print("Solver derived artifacts match" if args.check else "Wrote solver report, complete CSV and constrained frontiers")


if __name__ == "__main__":
    main()
