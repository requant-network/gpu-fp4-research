"""Build traversal/cache controls with the frozen m128/automatic/cooperative base."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from build_pipeline import ROOT,PIN,generated


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("cutlass",type=Path);args=parser.parse_args()
    cutlass=args.cutlass.resolve()
    revision=subprocess.check_output(["git","-C",str(cutlass),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(cutlass),"status","--porcelain","--untracked-files=no"],text=True)
    if revision!=PIN or dirty:raise ValueError("Clean pinned CUTLASS required")
    header=ROOT/"build/base-raster.cuh"
    header.write_text(generated((ROOT/"native/bridge.cu").read_text(encoding="utf-8"),128,"auto","coop"),encoding="utf-8",newline="\n")
    nvcc="/usr/local/cuda-13.3/bin/nvcc";library=ROOT/"build/libraster.so"
    subprocess.run([nvcc,"-O3","-std=c++17","-arch=sm_120a","--expt-relaxed-constexpr","-shared","-Xcompiler=-fPIC",
                    f"-I{cutlass/'include'}",f"-I{cutlass/'tools/util/include'}","-Inative","-Ibuild",'-DRQ_BASE_SOURCE="base-raster.cuh"',
                    "native/raster_bridge.cu","-L/usr/local/cuda-13.3/lib64","-lcublasLt","-Xlinker","-rpath","-Xlinker","/usr/local/cuda-13.3/lib64", "-o",str(library)],cwd=ROOT,check=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sources=("build_raster.py","build_pipeline.py","native/raster_bridge.cu","native/pipeline_bridge.cu","native/hardware_bridge.cu",
             "native/application_bridge.cu","native/optimized_bridge.cu","native/bridge.cu")
    record={"cutlass_commit":revision,"tile_m":128,"stages":"auto","schedule":"coop","generated_source_sha256":sha(header),
            "binary_sha256":sha(library),"source_sha256":{name:sha(ROOT/name) for name in sources},
            "nvcc":subprocess.check_output([nvcc,"--version"],text=True),"architecture":"sm_120a","complete":True}
    (ROOT/"results/raster-build.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print("Built native raster/cache control library",flush=True)


if __name__=="__main__":main()
