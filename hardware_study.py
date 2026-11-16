"""Native FP4 conversion/vector loads/warp reduction with exact-byte controls."""
import argparse
import ctypes
from datetime import datetime,timezone
from pathlib import Path

import cupy as cp
import numpy as np

from encoder_study import HELD_OUT,PROFILES,dataset
from followup import ROOT,ExtendedPlan,file_hash,load_extended,oracle,validate,write_record
from gpu_study import capture,environment,gpu_snapshot,measure
from reference import SHAPES,digest

VARIANTS=(0,2,10,11,12,13)
BOUNDARIES=("prepare-a","prepare-both","prepacked","online-a","online-both")
SOURCES=("hardware_study.py","reference.py","gpu_study.py","followup.py","encoder_study.py",
         "native/bridge.cu","native/optimized_bridge.cu","native/application_bridge.cu",
         "native/hardware_bridge.cu","build_hardware.sh","build_extended.sh")


def load_hardware():
    prototype=load_extended()
    library=ctypes.CDLL(str(ROOT/"build/libhardware.so"))
    for name in ("rq_error","rq_create","rq_destroy","rq_quantize","rq_run","rq_candidates","rq_algorithm_id",
                 "rq_buffer","rq_bytes","rq_cublas_version","rq_quantize_variant","rq_prepare_fused","rq_activate"):
        target,original=getattr(library,name),getattr(prototype,name)
        target.argtypes,target.restype=original.argtypes,original.restype
    pointer,integer=ctypes.c_void_p,ctypes.c_int
    library.rq_quantize_native.argtypes=[pointer,integer,integer,pointer]
    library.rq_quantize_native.restype=integer
    library.rq_fused_native.argtypes=[pointer,pointer,pointer,integer,pointer]
    library.rq_fused_native.restype=integer
    library.rq_native_resources.argtypes=[integer,integer,ctypes.POINTER(integer)]
    library.rq_native_resources.restype=integer
    return library


class HardwarePlan(ExtendedPlan):
    def prepare(self,mask=3):
        if self.variant<10:
            return super().prepare(mask)
        if self.bias is not None and mask&1:
            if self.fused:
                self.checked(self.library.rq_fused_native(self.pointer,self.source.data.ptr,
                    self.bias.data.ptr,self.variant,self.stream.ptr))
                mask&=~1
            else:
                self.checked(self.library.rq_activate(self.source.data.ptr,self.bias.data.ptr,
                    self.inputs[0].data.ptr,int(self.source.size),self.source.shape[1],self.stream.ptr))
        if mask:self.checked(self.library.rq_quantize_native(self.pointer,mask,self.variant,self.stream.ptr))


def resources(library):
    rows=[]
    names=("registers_per_thread","static_shared_bytes","local_bytes","kernel_max_threads","theoretical_active_blocks_per_sm",
           "sm_max_threads","sm_registers","warp_size","sm_count","l2_bytes","shared_bytes_per_sm","binary_version")
    for variant in VARIANTS:
        for threads in ((128,256,512) if variant==2 else (128,)):
            values=(ctypes.c_int*12)()
            if library.rq_native_resources(variant,threads,values):raise RuntimeError(library.rq_error().decode())
            row={"variant":variant,"block_threads":threads,**dict(zip(names,values))}
            row["theoretical_thread_occupancy"]=row["theoretical_active_blocks_per_sm"]*threads/row["sm_max_threads"]
            rows.append(row)
    return rows


def operation(plan,boundary):
    if boundary=="prepare-a":plan.prepare(1)
    elif boundary=="prepare-both":plan.prepare(3)
    else:plan.step(boundary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=20261118)
    parser.add_argument("--output",default="results/hardware-cohort-a.json")
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--reverse",action="store_true")
    args=parser.parse_args()
    cp.cuda.Device(0).use()
    library=load_hardware()
    record={"schema_version":1,"seed":args.seed,"quick":args.quick,"reverse":args.reverse,
            "utc":datetime.now(timezone.utc).isoformat(),"environment":environment(0,library),
            "source_sha256":{name:file_hash(ROOT/name) for name in SOURCES},
            "binary_sha256":file_hash(ROOT/"build/libhardware.so"),"resources":resources(library),
            "warmups":2 if args.quick else 5,"samples":3 if args.quick else 30,"graph_repetitions":4 if args.quick else 32,
            "scope":"Native CUDA FP4 packed conversion, vector loads and warp reductions; exact original quantization contract; resource occupancy is theoretical, not achieved",
            "negative_controls":0,"cases":[]}
    shapes=(SHAPES[2],) if args.quick else SHAPES+HELD_OUT+((2048,2048,2048),)
    groups=[(shape,profile) for shape in shapes for profile in PROFILES
            if shape!=(2048,2048,2048) or profile in ("normal","outliers")]
    if args.reverse:groups.reverse()
    destination=Path(args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        for shape,profile in groups:
            a,b=dataset(args.seed,shape,profile)
            bias=(np.arange(shape[2],dtype=np.int32)%33-16).astype(np.float32)/np.float32(64)
            da,db,dbias=cp.asarray(a),cp.asarray(b),cp.asarray(bias)
            expected=oracle(a,b)
            activation_reference=oracle(np.maximum(a+bias,np.float32(0)),b) if profile in ("normal","outliers") else None
            configurations=[(variant,False,False) for variant in VARIANTS]
            if activation_reference is not None:configurations += [(2,True,False),(2,True,True),(12,True,False),(12,True,True)]
            if args.reverse:configurations.reverse()
            plans,entries,graphs,witness_outputs=[],[],[],{}
            try:
                for variant,activation,fused in configurations:
                    plan=HardwarePlan(library,variant,da,db,stream,dbias if activation else None,fused)
                    plans.append(plan)
                    if variant>=10 and not record["negative_controls"]:
                        for mask,bad in ((0,10),(4,10),(3,9),(3,14)):
                            if library.rq_quantize_native(plan.pointer,mask,bad,stream.ptr)!=1:
                                raise RuntimeError("Invalid hardware encoder configuration accepted")
                        if library.rq_fused_native(plan.pointer,da.data.ptr+4,dbias.data.ptr,variant,stream.ptr)!=1:
                            raise RuntimeError("Unaligned native vector input accepted")
                        record["negative_controls"]=5
                    checks,output=validate(plan,activation_reference if activation else expected)
                    label=f"v{variant}"+("-fused" if fused else "-separate") if activation else f"v{variant}"
                    entry={"shape":list(shape),"profile":profile,"variant":variant,"activation":activation,"fused":fused,
                           "label":label,"partition":"held-out" if shape in HELD_OUT else "training",
                           "input_sha256":[digest(a),digest(b)],"checks":checks,
                           "timings_ms":{boundary:[] for boundary in BOUNDARIES},"gpu_before":gpu_snapshot(0)}
                    entries.append(entry)
                    if shape==(128,128,128):witness_outputs[label]=output
                    for boundary in BOUNDARIES:
                        graphs.append((capture(lambda p=plan,name=boundary:operation(p,name),record["graph_repetitions"],stream),entry,boundary))
                for graph,_,_ in graphs:
                    for _ in range(record["warmups"]):graph.launch(stream)
                stream.synchronize()
                for observation in range(record["samples"]):
                    offset=observation%len(graphs)
                    for graph,entry,boundary in graphs[offset:]+graphs[:offset]:
                        entry["timings_ms"][boundary].append(measure(graph,record["graph_repetitions"],stream))
                for plan,entry in zip(plans,entries):
                    entry["post_timing_checks"],_=validate(plan,activation_reference if entry["activation"] else expected)
                    entry["gpu_after"]=gpu_snapshot(0)
                if witness_outputs:
                    witness=destination.parent/f"hardware-witness-{args.seed}-{profile}.npz"
                    np.savez_compressed(witness,a=a,b=b,bias=bias,**witness_outputs)
                    for entry in entries:entry["witness"]={"file":witness.name,"array":entry["label"],"sha256":file_hash(witness)}
                record["cases"].extend(entries)
                write_record(destination,record)
                print(f"{shape} {profile}: {len(entries)} native architecture variants passed",flush=True)
            finally:
                stream.synchronize()
                graphs=[]
                for plan in plans:plan.close()
    record["complete"]=True
    record["environment_after"]=gpu_snapshot(0)
    write_record(destination,record)


if __name__=="__main__":main()
