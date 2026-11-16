"""Build an explicitly configured native SM120 FP4 pipeline from the frozen base."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
PIN="b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd"


def generated(base,tile,stages,schedule):
    replacements={
        "using ThreadBlockShape    = Shape<_128,_128,_128>;":f"using ThreadBlockShape    = Shape<_{tile},_128,_128>;",
        "cutlass::gemm::collective::KernelScheduleAuto":
            "cutlass::gemm::KernelTmaWarpSpecialized"+("Pingpong" if schedule=="ping" else "")+"Nvf4Sm120"}
    if stages!="auto":
        replacements["cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>"]=(
            f"cutlass::gemm::collective::StageCount<{stages}>")
    for old,new in replacements.items():
        if base.count(old)!=1:raise ValueError(f"Expected a unique frozen-base declaration: {old}")
        base=base.replace(old,new)
    return base


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cutlass",type=Path)
    parser.add_argument("--tile-m",type=int,choices=(128,256),default=128)
    parser.add_argument("--stages",choices=("auto","2","3","4"),default="2")
    parser.add_argument("--schedule",choices=("coop","ping"),default="ping")
    args=parser.parse_args();cutlass=args.cutlass.resolve()
    revision=subprocess.check_output(["git","-C",str(cutlass),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(cutlass),"status","--porcelain","--untracked-files=no"],text=True)
    if revision!=PIN or dirty:raise ValueError("A clean pinned CUTLASS checkout is required")
    label=f"m{args.tile_m}-s{args.stages}-{args.schedule}"
    header=ROOT/f"build/base-pipeline-{label}.cuh";header.parent.mkdir(exist_ok=True)
    header.write_text(generated((ROOT/"native/bridge.cu").read_text(encoding="utf-8"),args.tile_m,args.stages,args.schedule),encoding="utf-8",newline="\n")
    library=ROOT/f"build/libpipeline-{label}.so"
    nvcc="/usr/local/cuda-13.3/bin/nvcc"
    command=[nvcc,"-O3","-std=c++17","-arch=sm_120a","--expt-relaxed-constexpr","-shared","-Xcompiler=-fPIC",
             f"-I{cutlass/'include'}",f"-I{cutlass/'tools/util/include'}","-Inative","-Ibuild",f'-DRQ_BASE_SOURCE="{header.name}"',
             "native/pipeline_bridge.cu","-L/usr/local/cuda-13.3/lib64","-lcublasLt","-Xlinker","-rpath","-Xlinker","/usr/local/cuda-13.3/lib64","-o",str(library)]
    subprocess.run(command,cwd=ROOT,check=True)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    sources=("build_pipeline.py","native/pipeline_bridge.cu","native/bridge.cu","native/optimized_bridge.cu","native/application_bridge.cu","native/hardware_bridge.cu")
    record={"tile_m":args.tile_m,"stages":args.stages,"schedule":args.schedule,"cutlass_commit":revision,"architecture":"sm_120a",
            "generated_source_sha256":sha(header),"binary_sha256":sha(library),"source_sha256":{name:sha(ROOT/name) for name in sources},
            "nvcc":subprocess.check_output([nvcc,"--version"],text=True),"complete":True}
    (ROOT/f"results/pipeline-build-{label}.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print(f"Built native pipeline {label}",flush=True)


if __name__=="__main__":main()
