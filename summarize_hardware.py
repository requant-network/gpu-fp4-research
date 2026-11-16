"""Derive native-conversion comparisons without choosing a per-test winner."""
import argparse
import csv
import io
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
NAMES={0:"original scalar16",2:"original pair8",10:"native scalar16",11:"native float2/shuffle8",
       12:"native float4/shuffle4",13:"native float4/redux4"}


def median(case,boundary):return statistics.median(case["timings_ms"][boundary])


def outputs():
    records=[json.loads((ROOT/"results"/name).read_text(encoding="utf-8"))
             for name in ("hardware-cohort-a.json","hardware-cohort-b.json")]
    if not all(r.get("complete") and len(r["cases"])==340 for r in records):raise ValueError("Complete hardware cohorts required")
    groups=[]
    for record in records:
        groups.append({(tuple(c["shape"]),c["profile"],c["variant"],c["activation"],c["fused"]):c for c in record["cases"]})
    buffer=io.StringIO(newline="")
    writer=csv.writer(buffer,lineterminator="\n")
    writer.writerow(["seed","m","n","k","profile","partition","variant","activation","fused","boundary","median_ms","p10_ms","p90_ms"])
    for record in records:
        for case in record["cases"]:
            for boundary,values in case["timings_ms"].items():
                values=sorted(values)
                writer.writerow([record["seed"],*case["shape"],case["profile"],case["partition"],case["variant"],case["activation"],case["fused"],boundary,
                                 f"{statistics.median(values):.9f}",f"{values[2]:.9f}",f"{values[26]:.9f}"])
    total=sum(c["shape"][0]*c["shape"][1] for r in records for c in r["cases"])
    lines=["# Native FP4 conversion and memory access","",
           "**Status: this bounded native-instruction experiment is complete. The full architecture/configuration investigation remains open.**","",
           f"Two independent cohorts contain 680 configurations, 102,000 retained timing observations and {total:,} output entries checked before and after timing. "
           "All packed values and logical scale bytes match the independent CPU encoder. Eight small witnesses allow CPU-only numerical replay.","",
           "## What is native","",
           "GPU work is CUDA C++ compiled for SM120a. CUTLASS performs block-scaled FP4 Tensor Core GEMM; cuBLASLt supplies the other formats and an additional FP4 application baseline. "
           "Python/CuPy performs orchestration and memory management. Captured CUDA-event times exclude Python dispatch, while the separate application report retains changing-input host-request costs. "
           "A pure C++ request driver has not been measured.","",
           "The new encoders use `__nv_cvt_float2_to_fp4x2` with round-to-nearest, vector `float2`/`float4` loads and either subgroup shuffles or integer warp reduction of nonnegative absolute-value bit patterns. "
           "All retain the original scale calculation, range and byte layout. Comparisons therefore change implementation without changing numerical quality. "
           "Their advantage over original encoders is distinct from FP4-versus-FP16 comparisons, which do not guarantee equal accuracy.","",
           "## Fixed candidate comparisons","",
           "Ratios span the two cohorts on the normal profile. Above 1 favors the named candidate. All encoders use 128 threads per CUDA block and the same M=128 CUTLASS tile. "
           "The float2 and float4 candidates are fixed across shapes; this table does not select a winner using each measured test case.","",
           "| M x N x K | Original pair8 / native float2: prepare A | Original pair8 / native float4: prepare A | Original pair8 / native float4: online A | Native float4 separate / fused: online A |",
           "|---|---:|---:|---:|---:|"]
    shapes=sorted({tuple(c["shape"]) for c in records[0]["cases"]})
    for shape in shapes:
        ratios=[[],[],[],[]]
        for group in groups:
            old=group[shape,"normal",2,False,False]
            pair=group[shape,"normal",11,False,False]
            quad=group[shape,"normal",12,False,False]
            ratios[0].append(median(old,"prepare-a")/median(pair,"prepare-a"))
            ratios[1].append(median(old,"prepare-a")/median(quad,"prepare-a"))
            ratios[2].append(median(old,"online-a")/median(quad,"online-a"))
            ratios[3].append(median(group[shape,"normal",12,True,False],"online-a")/median(group[shape,"normal",12,True,True],"online-a"))
        lines.append(f"| {' x '.join(map(str,shape))} | "+" | ".join(f"{min(v):.2f}-{max(v):.2f}" for v in ratios)+" |")
    redux_ratios=[]
    for group in groups:
        key=((2048,2048,2048),"normal")
        redux_ratios.append(median(group[*key,12,False,False],"prepare-a")/median(group[*key,13,False,False],"prepare-a"))
    lines += ["",f"The native integer `REDUX.MAX` alternative is slower than the shuffle-based float4 encoder at 2048 cubed: shuffle/redux latency ratio "
              f"{min(redux_ratios):.3f}-{max(redux_ratios):.3f}. Both produce identical bytes. A dedicated instruction does not guarantee a faster complete kernel.","",
              "The complete [CSV](results/hardware-summary.csv) retains scalar/native/reduction alternatives, tiny and midpoint-tie inputs, outliers, separate/fused activation and all five boundaries, including regressions. "
              "Preparation-only acceleration need not produce the same complete online-GEMM acceleration. Warm graphs reuse operands, so this experiment does not establish streaming DRAM throughput.","",
              "## Emitted instructions and resources","",
              "`cuobjdump` of the measured binary confirms FP4 `OMMA.SF...E2M1` Tensor Core operations and `UTMALDG`/`UTMASTG` TMA traffic in GEMM. "
              "New encoders contain `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`; variant 13 also contains `REDUX.MAX`. "
              "Original encoders retain their software FP4 conversion path. Static instruction counts include branches and fallback code; they are not executed instruction counts.","",
              "| Encoder | Registers/thread | Static shared bytes | Local bytes | Theoretical thread occupancy at 128 threads |",
              "|---|---:|---:|---:|---:|"]
    for row in records[0]["resources"]:
        if row["block_threads"]==128:
            lines.append(f"| {NAMES[row['variant']]} | {row['registers_per_thread']} | {row['static_shared_bytes']} | {row['local_bytes']} | {row['theoretical_thread_occupancy']:.3f} |")
    lines += ["","Resources come from `cudaFuncGetAttributes` and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for the unactivated A encoder. "
              "This is a register/shared-memory capacity calculation, not measured active warps or achieved throughput. Fused activation can change resources. "
              "The [hardware inspection](results/native-inspection-hardware.json) retains every kernel's static instructions/resources and the exact measured binary hash. "
              "[Tile128](results/native-inspection-128.json) and [tile256](results/native-inspection-256.json) retain earlier emitted-code evidence.","",
              "## Profiler availability","",
              "Nsight Compute 2026.2.1 was actually invoked on a separate 2048-cubed encoder probe with LaunchStats, Occupancy and SpeedOfLight sections. "
              "The installed driver returned `ERR_NVGPUCTRPERM`: the current account cannot access performance counters. "
              "The exact failure, tool version, source and binary hashes are retained in [profiler evidence](results/native-profiler.json). "
              "No achieved occupancy, bandwidth, Tensor Core utilization or counter-derived bottleneck is reported. "
              "Profiling permissions were not changed. This limits profiling evidence but does not invalidate the separately timed, numerically checked campaigns.","",
              "## Limits and next architectural questions","",
              "These are shared-system measurements with other GPU contexts present; clocks, power settings and services were not changed. "
              "No claim is made that every available architectural feature is used or that the best achievable throughput has been reached. "
              "Explicit loading/pipeline controls, buffering/warp roles, traversal, cold-cache/streaming behavior and held-out configuration selection beyond the earlier encoder rule remain open. "
              "Relevant mechanisms must improve a particular workload under its correctness contract; enabling more mechanisms alone is not evidence of a useful speedup.","",
              "Run `python verify_hardware.py` and `python summarize_hardware.py --check`. GPU reproduction: after the extended builds, run "
              "`bash build_hardware.sh ../cutlass-fp4`; run `hardware_study.py --seed 20261118` and a separate reversed cohort with seed 20261218/output path. "
              "Run `inspect_native.py` separately for each library with the recorded CUDA toolkit.",""]
    return {"results/hardware-summary.csv":buffer.getvalue(),"HARDWARE_REPORT.md":"\n".join(lines)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    for name,content in outputs().items():
        path=ROOT/name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8")!=content:raise ValueError(f"Stale hardware artifact: {name}")
        else:path.write_text(content,encoding="utf-8",newline="\n")
    print("Hardware derived artifacts match" if args.check else "Wrote native hardware report and table")


if __name__=="__main__":main()
