"""Correctness/resource preflight for explicit SM120 pipeline configurations."""
import argparse
import ctypes
import json
import re

import cupy as cp

from followup import ROOT,file_hash,oracle,validate,write_record
from gpu_study import environment
from hardware_study import HardwarePlan,load_hardware
from reference import inputs


def load_pipeline(label):
    if not re.fullmatch(r"m(?:128|256)-s(?:auto|2|3|4)-(?:coop|ping)",label):raise ValueError("Invalid pipeline label")
    prototype=load_hardware();library=ctypes.CDLL(str(ROOT/f"build/libpipeline-{label}.so"))
    for name in ("rq_error","rq_create","rq_destroy","rq_quantize","rq_run","rq_candidates","rq_algorithm_id","rq_buffer","rq_bytes",
                 "rq_cublas_version","rq_quantize_variant","rq_prepare_fused","rq_activate","rq_quantize_native","rq_fused_native"):
        target,original=getattr(library,name),getattr(prototype,name)
        target.argtypes,target.restype=original.argtypes,original.restype
    library.rq_pipeline_resources.argtypes=[ctypes.POINTER(ctypes.c_int)]
    library.rq_pipeline_resources.restype=ctypes.c_int
    return library


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("label");args=parser.parse_args()
    cp.cuda.Device(0).use();library=load_pipeline(args.label)
    build=json.loads((ROOT/f"results/pipeline-build-{args.label}.json").read_text(encoding="utf-8"))
    if build["binary_sha256"]!=file_hash(ROOT/f"build/libpipeline-{args.label}.so"):raise ValueError("Pipeline build/binary mismatch")
    record={"label":args.label,"environment":environment(0,library),"build":build,
            "source_sha256":{"pipeline_preflight.py":file_hash(ROOT/"pipeline_preflight.py")},
            "scope":"Correctness and theoretical-resource preflight only; no comparative timing or achieved occupancy claim","cases":[]}
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        for shape in ((128,128,128),(192,768,768),(2048,2048,2048)):
            for profile in ("normal","outliers"):
                a,b=inputs(20261120,shape,profile)
                plan=HardwarePlan(library,12,cp.asarray(a),cp.asarray(b),stream)
                try:
                    checks,_=validate(plan,oracle(a,b))
                    record["cases"].append({"shape":shape,"profile":profile,"checks":checks})
                    print(f"{args.label}: {shape} {profile} passed",flush=True)
                finally:plan.close()
        values=(ctypes.c_int*7)()
        if library.rq_pipeline_resources(values):raise RuntimeError(library.rq_error().decode())
        names=("stages","block_threads","dynamic_shared_bytes","registers_per_thread","static_shared_bytes","local_bytes","theoretical_blocks_per_sm")
        record["resources"]=dict(zip(names,values))
    record["complete"]=True;write_record(ROOT/f"results/pipeline-preflight-{args.label}.json",record)


if __name__=="__main__":main()
