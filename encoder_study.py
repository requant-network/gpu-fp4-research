"""Byte-identical cooperative FP4 encoders, fusion and tile comparisons on SM120."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

from followup import (ROOT, ExtendedPlan, file_hash, load_extended, oracle,
                      validate, write_record)
from gpu_study import capture, environment, gpu_snapshot, measure
from reference import SHAPES, digest, inputs

HELD_OUT = ((32,768,1280), (192,768,768), (384,1536,512), (768,1536,1024))
PROFILES = ("normal", "outliers", "tiny", "ties")
BOUNDARIES = ("prepare-a", "prepare-both", "prepacked", "online-a", "online-both")
VARIANTS = ("scalar", "lane16-128", "pair8-128", "pair8-256", "pair8-512")
SOURCES = ("reference.py", "gpu_study.py", "followup.py", "encoder_study.py", "native/bridge.cu",
           "native/optimized_bridge.cu", "build.sh", "build_extended.sh")


def dataset(seed, shape, profile):
    a,b = inputs(seed,shape,"outliers" if profile == "outliers" else "normal")
    if profile == "tiny":
        a *= np.float32(2**-10)
        b *= np.float32(2**-10)
    elif profile == "ties":
        palette=np.array([-6,-5,-3.5,-2.5,-1.75,-1.25,-.75,-.25,0,.25,.75,1.25,1.75,2.5,5,6],np.float32)
        a=np.resize(palette,a.shape)
        b=np.resize(np.roll(palette,5),b.shape)
        a[0]=0
        b[0]=0
    return a,b


def operation(plan,boundary):
    if boundary == "prepare-a":
        plan.prepare(1)
    elif boundary == "prepare-both":
        plan.prepare(3)
    else:
        plan.step(boundary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=20261117)
    parser.add_argument("--output",default="results/encoder-20261117.json")
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--reverse",action="store_true")
    args=parser.parse_args()
    if args.seed<0:
        parser.error("Seed must be nonnegative")
    cp.cuda.Device(0).use()
    libraries={tile:load_extended(tile) for tile in (128,256)}
    stream=cp.cuda.Stream(non_blocking=True)
    record={"schema_version":1,"utc":datetime.now(timezone.utc).isoformat(),"seed":args.seed,
            "quick":args.quick,"reverse":args.reverse,"environment":environment(0,libraries[128]),
            "source_sha256":{name:file_hash(ROOT/name) for name in SOURCES},
            "binary_sha256":{str(tile):file_hash(ROOT/f"build/libextended-{tile}.so") for tile in libraries},
            "generated_source_sha256":{str(tile):file_hash(ROOT/f"build/base-{tile}.cuh") for tile in libraries},
            "negative_controls":0,
            "warmups":2 if args.quick else 5,"samples":3 if args.quick else 30,
            "graph_repetitions":4 if args.quick else 32,"cases":[],
            "timing_scope":"device-resident captured repetitions; preparation, GEMM and fusion boundaries explicit; shared system"}
    shapes=(SHAPES[2],) if args.quick else SHAPES+HELD_OUT+((2048,2048,2048),)
    groups=[(shape,profile) for shape in shapes for profile in PROFILES
            if shape != (2048,2048,2048) or profile in ("normal","outliers")]
    if args.reverse:
        groups.reverse()
    destination=Path(args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with stream:
        for shape,profile in groups:
            a,b=dataset(args.seed,shape,profile)
            expected=oracle(a,b)
            da,db=cp.asarray(a),cp.asarray(b)
            configurations=[(128,variant,False,None) for variant in range(5)]
            configurations.append((256,3,False,None))
            # Bias/ReLU fusion is tested on ordinary and outlier inputs separately.
            if profile in ("normal","outliers"):
                configurations += [(128,3,False,"bias-relu"),(128,3,True,"bias-relu")]
            if args.reverse:
                configurations.reverse()
            bias=(np.arange(shape[2],dtype=np.int32)%33-16).astype(np.float32)/np.float32(64)
            dbias=cp.asarray(bias)
            activated=np.maximum(a+bias,np.float32(0))
            activated_reference=oracle(activated,b) if profile in ("normal","outliers") else None
            plans,entries,graphs,witness_outputs=[],[],[],{}
            try:
                for tile,variant,fused,activation in configurations:
                    plan=ExtendedPlan(libraries[tile],variant,da,db,stream,dbias if activation else None,fused)
                    plans.append(plan)
                    if not record["negative_controls"]:
                        for bad_mask,bad_variant in ((0,1),(4,1),(3,-1),(3,5)):
                            if libraries[tile].rq_quantize_variant(plan.pointer,bad_mask,bad_variant,stream.ptr)!=1:
                                raise RuntimeError("Invalid encoder configuration accepted")
                        if libraries[tile].rq_quantize_variant(None,3,1,stream.ptr)!=1:
                            raise RuntimeError("Null encoder plan accepted")
                        record["negative_controls"]=5
                    checks,output=validate(plan,activated_reference if activation else expected)
                    label=f"tile{tile}-{VARIANTS[variant]}"+("-fused" if fused else "-separate") if activation else f"tile{tile}-{VARIANTS[variant]}"
                    entry={"shape":list(shape),"profile":profile,"tile_m":tile,"variant":variant,
                           "label":label,"activation":activation,"fused":fused,
                           "partition":"held-out" if shape in HELD_OUT else "training",
                           "input_sha256":[digest(a),digest(b)],"checks":checks,
                           "timings_ms":{name:[] for name in BOUNDARIES},"gpu_before":gpu_snapshot(0)}
                    entries.append(entry)
                    if shape==(128,128,128):
                        witness_outputs[label]=output
                    for boundary in BOUNDARIES:
                        graphs.append((capture(lambda p=plan,name=boundary:operation(p,name),record["graph_repetitions"],stream),entry,boundary))
                for graph,_,_ in graphs:
                    for _ in range(record["warmups"]):
                        graph.launch(stream)
                stream.synchronize()
                for observation in range(record["samples"]):
                    offset=observation%len(graphs)
                    for graph,entry,boundary in graphs[offset:]+graphs[:offset]:
                        entry["timings_ms"][boundary].append(measure(graph,record["graph_repetitions"],stream))
                for plan,entry in zip(plans,entries):
                    checks,_=validate(plan,activated_reference if entry["activation"] else expected)
                    entry["post_timing_checks"]=checks
                    entry["gpu_after"]=gpu_snapshot(0)
                if witness_outputs:
                    witness=destination.parent/f"encoder-witness-{args.seed}-{profile}.npz"
                    np.savez_compressed(witness,a=a,b=b,bias=bias,**witness_outputs)
                    for entry in entries:
                        entry["witness"]={"file":witness.name,"array":entry["label"],"sha256":file_hash(witness)}
                record["cases"].extend(entries)
                write_record(destination,record)
                print(f"{shape} {profile}: {len(entries)} validated encoder/fusion/tile configurations",flush=True)
            finally:
                stream.synchronize()
                graphs=[]
                for plan in plans:
                    plan.close()
    record["complete"]=True
    record["environment_after"]=gpu_snapshot(0)
    write_record(destination,record)
    print(f"Completed {len(record['cases'])} cases",flush=True)


if __name__=="__main__":
    main()
