"""Native FP4 pipeline/traversal/cache controls with independent numerical checks."""
import argparse
import ctypes
from datetime import datetime,timezone
import json
import re
from pathlib import Path

import cupy as cp
import numpy as np

from build_pipeline import generated
from encoder_study import HELD_OUT
from followup import ROOT,file_hash,oracle,write_record
from gpu_study import environment,gpu_snapshot
from hardware_study import HardwarePlan
from pipeline_preflight import load_pipeline
from reference import SHAPES,digest,inputs,relative_error

LABELS=tuple(f"m{tile}-s{stage}-{schedule}" for tile in (128,256) for stage in ("auto","2","3","4") for schedule in ("coop","ping"))
ORDERS=((0,0),(1,1),(2,1),(1,2),(2,2),(1,4),(2,4),(1,8),(2,8))
BOUNDARIES=("prepacked","online-a","online-both")
SOURCES=("pipeline_study.py","pipeline_preflight.py","build_pipeline.py","build_raster.py","reference.py","gpu_study.py","followup.py","hardware_study.py",
         "encoder_study.py","native/bridge.cu","native/optimized_bridge.cu","native/application_bridge.cu",
         "native/hardware_bridge.cu","native/pipeline_bridge.cu","native/raster_bridge.cu")
RESOURCE_NAMES=("stages","block_threads","dynamic_shared_bytes","registers_per_thread","static_shared_bytes","local_bytes","theoretical_blocks_per_sm")


def resource_query(library):
    values=(ctypes.c_int*7)()
    if library.rq_pipeline_resources(values):raise RuntimeError(library.rq_error().decode())
    return dict(zip(RESOURCE_NAMES,values))


def inventory():
    records=[]
    for label in LABELS:
        path=ROOT/f"results/pipeline-build-{label}.json"
        if path.exists():
            build=json.loads(path.read_text(encoding="utf-8"))
            for name,expected in build["source_sha256"].items():
                if file_hash(ROOT/name)!=expected:raise ValueError(f"Changed pipeline build source: {name}")
            if file_hash(ROOT/f"build/libpipeline-{label}.so")!=build["binary_sha256"]:raise ValueError("Pipeline binary differs from build")
            try:resources=resource_query(load_pipeline(label))
            except RuntimeError as error:
                records.append({"label":label,"status":"resource-query-failed","build":build,"diagnostic":str(error)})
            else:records.append({"label":label,"status":"available","build":build,"resources":resources})
        else:
            log=ROOT/f"build/pipeline-{label}-build.log"
            output=log.read_text(encoding="utf-8",errors="replace") if log.exists() else ""
            if "CalledProcessError" not in output or "returned non-zero exit status" not in output:
                raise RuntimeError(f"Pipeline build has no terminal outcome: {label}")
            _,stage,_=label.split("-")
            tile=int(label.split("-")[0][1:]);stage=stage[1:];schedule=label.split("-")[-1]
            import hashlib
            header=generated((ROOT/"native/bridge.cu").read_text(encoding="utf-8"),tile,stage,schedule)
            diagnostics=[re.sub(r"/[A-Za-z0-9_.+/-]+",lambda match:"<path>/"+match.group().rsplit("/",1)[-1],line.strip())
                         for line in output.splitlines() if re.search(r"error(?: #|:| :)",line)][:12]
            if not diagnostics:raise RuntimeError(f"Missing actual compiler diagnostic for {label}")
            records.append({"label":label,"status":"compile-failed","build_log_sha256":file_hash(log),"compiler_diagnostics":diagnostics,
                            "generated_source_sha256":hashlib.sha256(header.encode()).hexdigest(),
                            "diagnostic":"nvcc returned non-zero exit status; private compiler log retained"})
    return records


def load_raster():
    prototype=load_pipeline("m128-sauto-coop");library=ctypes.CDLL(str(ROOT/"build/libraster.so"))
    for name in ("rq_error","rq_create","rq_destroy","rq_quantize","rq_run","rq_candidates","rq_algorithm_id","rq_buffer","rq_bytes","rq_cublas_version",
                 "rq_quantize_variant","rq_prepare_fused","rq_activate","rq_quantize_native","rq_fused_native","rq_pipeline_resources"):
        target,original=getattr(library,name),getattr(prototype,name)
        target.argtypes,target.restype=original.argtypes,original.restype
    p,i=ctypes.c_void_p,ctypes.c_int
    library.rq_run_raster.argtypes=[p,i,i,p];library.rq_run_raster.restype=i
    library.rq_cache_pressure.argtypes=[p,ctypes.c_size_t,p];library.rq_cache_pressure.restype=i
    library.rq_record_external.argtypes=[p,p];library.rq_record_external.restype=i
    return library


class PipelinePlan(HardwarePlan):
    def __init__(self,library,a,b,stream,raster=None):
        self.raster=raster
        super().__init__(library,12,a,b,stream)

    def run(self):
        if self.raster is None:super().run()
        else:self.checked(self.library.rq_run_raster(self.pointer,*self.raster,self.stream.ptr))


def checks_current(plan,reference):
    plan.stream.synchronize()
    for operand in (0,1):
        np.testing.assert_array_equal(plan.buffer(operand),reference["encoded"][operand].ravel())
        np.testing.assert_array_equal(plan.buffer(operand+2),reference["scales"][operand].ravel())
    actual=cp.asnumpy(plan.output)
    ratio=float(np.max(np.abs(actual.astype(np.float64)-reference["expected"])/reference["bound"]))
    if not np.isfinite(actual).all() or ratio>1:raise ValueError(f"Pipeline arithmetic bound exceeded: {ratio}")
    return {"encoded_sha256":reference["hashes"],"arithmetic_max_bound_ratio":ratio,
            "relative_frobenius_error":relative_error(actual,reference["original"]),"output_sha256":digest(actual),"checked_output_entries":int(actual.size)},actual


def capture_instrumented(plan,control,scratch,boundary,cold):
    start,end=cp.cuda.Event(),cp.cuda.Event()
    plan.step(boundary);plan.stream.synchronize()
    plan.stream.begin_capture()
    try:
        if cold:plan.checked(control.rq_cache_pressure(scratch.data.ptr,scratch.size,plan.stream.ptr))
        else:plan.step(boundary)
        plan.checked(control.rq_record_external(start.ptr,plan.stream.ptr))
        plan.step(boundary)
        plan.checked(control.rq_record_external(end.ptr,plan.stream.ptr))
    except BaseException:
        plan.stream.end_capture();raise
    return plan.stream.end_capture(),start,end


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=20261120);parser.add_argument("--output",default="results/pipeline-cohort-a.json")
    parser.add_argument("--quick",action="store_true");parser.add_argument("--reverse",action="store_true");args=parser.parse_args()
    cp.cuda.Device(0).use();control=load_raster()
    families=inventory() if not args.quick else [{"label":label,"status":"available","build":json.loads((ROOT/f"results/pipeline-build-{label}.json").read_text()),
                                               "resources":resource_query(load_pipeline(label))} for label in ("m128-sauto-coop","m128-s2-ping")]
    raster_build=json.loads((ROOT/"results/raster-build.json").read_text(encoding="utf-8"))
    if file_hash(ROOT/"build/libraster.so")!=raster_build["binary_sha256"]:raise ValueError("Raster build/binary mismatch")
    for name,expected in raster_build["source_sha256"].items():
        if file_hash(ROOT/name)!=expected:raise ValueError("Raster source changed")
    properties=cp.cuda.runtime.getDeviceProperties(0);l2=int(properties["l2CacheSize"])
    if l2<=0:raise RuntimeError("Device did not report an L2 capacity")
    record={"schema_version":1,"seed":args.seed,"quick":args.quick,"reverse":args.reverse,"utc":datetime.now(timezone.utc).isoformat(),
            "environment":environment(0,control),"source_sha256":{name:file_hash(ROOT/name) for name in SOURCES},"families":families,"raster_build":raster_build,
            "cache_bytes":4*l2,"l2_bytes":l2,"samples":3 if args.quick else 30,"warmups":2 if args.quick else 5,"graph_repetitions":1,
            "scope":"External CUDA-event nodes surround one operation inside each graph; an identical unmeasured operation primes warm cases; a four-L2 read/write sweep precedes cache-pressure cases; both conditioning steps excluded from operation latency; no achieved-counter or guaranteed-cold-cache claim",
            "negative_controls":0,"cases":[]}
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    configurations=[(f["label"],None) for f in families if f["status"]=="available"]+[("raster",order) for order in ORDERS]
    shapes=((128,128,128),) if args.quick else SHAPES+HELD_OUT+((2048,2048,2048),)
    groups=[(shape,profile) for shape in shapes for profile in ("normal","outliers")]
    if args.reverse:groups.reverse();configurations.reverse()
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        scratch=cp.asarray(np.zeros(record["cache_bytes"]//4,np.float32))
        for shape,profile in groups:
            a,b=inputs(args.seed,shape,profile);reference=oracle(a,b)
            da,db=cp.asarray(a),cp.asarray(b);witness_outputs={};entries=[]
            for label,order in configurations:
                library=control if label=="raster" else load_pipeline(label)
                plan=PipelinePlan(library,da,db,stream,order)
                try:
                    if label=="raster" and not record["negative_controls"]:
                        for bad_raster,bad_swizzle in ((-1,1),(3,1),(1,0),(1,3),(0,16)):
                            if library.rq_run_raster(plan.pointer,bad_raster,bad_swizzle,stream.ptr)!=1:raise RuntimeError("Invalid raster configuration accepted")
                        record["negative_controls"]=5
                    plan.prepare();plan.run();checks,actual=checks_current(plan,reference)
                    identifier=label if order is None else f"raster-{order[0]}-{order[1]}"
                    if shape==(128,128,128):witness_outputs[identifier]=actual
                    entry={"shape":list(shape),"profile":profile,"partition":"held-out" if shape in HELD_OUT else "training","label":identifier,"family":label,
                           "raster":None if order is None else list(order),"input_sha256":[digest(a),digest(b)],"checks":checks,"resources":resource_query(library),
                           "timings_ms":{},"gpu_before":gpu_snapshot(0)}
                    graphs=[]
                    for boundary in BOUNDARIES:
                        for cold in (False,True):
                            key=boundary+("/cache-pressure" if cold else "/warm")
                            graphs.append((*capture_instrumented(plan,control,scratch,boundary,cold),key))
                            entry["timings_ms"][key]=[]
                    for graph,_,_,_ in graphs:
                        for _ in range(record["warmups"]):graph.launch(stream)
                    stream.synchronize()
                    for observation in range(record["samples"]):
                        offset=observation%len(graphs)
                        for graph,start,end,key in graphs[offset:]+graphs[:offset]:
                            graph.launch(stream);end.synchronize()
                            elapsed=float(cp.cuda.get_elapsed_time(start,end))
                            if not np.isfinite(elapsed) or elapsed<=0:raise RuntimeError("Invalid external-event latency")
                            entry["timings_ms"][key].append(elapsed)
                    entry["post_timing_checks"],_=checks_current(plan,reference);entry["gpu_after"]=gpu_snapshot(0);entries.append(entry)
                finally:stream.synchronize();graphs=[];plan.close()
            if witness_outputs:
                path=destination.parent/f"pipeline-witness-{args.seed}-{profile}.npz";np.savez_compressed(path,a=a,b=b,**witness_outputs)
                for entry in entries:entry["witness"]={"file":path.name,"array":entry["label"],"sha256":file_hash(path)}
            record["cases"].extend(entries);write_record(destination,record)
            print(f"{shape} {profile}: {len(entries)} pipeline/raster variants passed",flush=True)
    record["complete"]=True;record["environment_after"]=gpu_snapshot(0);write_record(destination,record)


if __name__=="__main__":main()
