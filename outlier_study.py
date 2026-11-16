"""Native FP4 column/block/residual correction with complete preparation costs."""
import argparse
import ctypes
from datetime import datetime,timezone
from pathlib import Path
import time

import cupy as cp
import numpy as np

from followup import ROOT,file_hash,write_record
from gpu_study import capture,environment,gpu_snapshot,measure,tune,verify_plan
from hardware_study import HardwarePlan,load_hardware
from outlier_reference import check_output,oracle
from reference import digest,inputs

SHAPES=((16,512,512),(128,128,128),(256,1024,1024),(32,768,1280),(192,768,768),(768,1536,1024),(2048,2048,2048))
HELD_OUT=((32,768,1280),(192,768,768),(768,1536,1024))
PROFILES=("normal","outliers","columns","blocks")
BOUNDARIES=("prepare-a","prepare-both","prepacked","online-a","online-both")
SOURCES=("outlier_study.py","outlier_reference.py","reference.py","gpu_study.py","followup.py","encoder_study.py","hardware_study.py",
         "prepare_digits.py","prepare_outlier_activations.py","native/bridge.cu","native/optimized_bridge.cu","native/application_bridge.cu",
         "native/hardware_bridge.cu","native/outlier_bridge.cu","build_outlier.sh","build_extended.sh")


def load_outlier():
    prototype=load_hardware()
    library=ctypes.CDLL(str(ROOT/"build/liboutlier.so"))
    for name in ("rq_error","rq_create","rq_destroy","rq_quantize","rq_run","rq_candidates","rq_algorithm_id","rq_buffer","rq_bytes",
                 "rq_cublas_version","rq_quantize_variant","rq_prepare_fused","rq_activate","rq_quantize_native","rq_fused_native"):
        target,original=getattr(library,name),getattr(prototype,name)
        target.argtypes,target.restype=original.argtypes,original.restype
    p,i,f=ctypes.c_void_p,ctypes.c_int,ctypes.c_float
    for name,args,result in (("create",[p,i,i,f],p),("destroy",[p],None),("prepare",[p,i,p],i),("run",[p,i,i,p],i),
                             ("candidates",[p,i],i),("gemm",[p,i,i,p],i),("buffer",[p,i],p)):
        function=getattr(library,"rq_correction_"+name)
        function.argtypes,function.restype=args,result
    return library


class CorrectedPlan(HardwarePlan):
    def __init__(self,library,method,budget,fraction,a,b,stream):
        self.correction=None
        super().__init__(library,12,a,b,stream)
        self.method,self.budget,self.fraction=method,budget,fraction
        self.correction=library.rq_correction_create(self.pointer,method,budget,fraction)
        if not self.correction:
            error=library.rq_error().decode();super().close();raise RuntimeError(error)
        self.algorithms=[0,0]

    def prepare(self,mask=3):self.checked(self.library.rq_correction_prepare(self.correction,mask,self.stream.ptr))

    def run(self):self.checked(self.library.rq_correction_run(self.correction,*self.algorithms,self.stream.ptr))

    def choose(self):
        results=[]
        for which in (0,1):
            rows=[]
            for candidate in range(self.library.rq_correction_candidates(self.correction,which)):
                graph=capture(lambda:self.checked(self.library.rq_correction_gemm(self.correction,which,candidate,self.stream.ptr)),16,self.stream)
                for _ in range(2):graph.launch(self.stream)
                self.stream.synchronize()
                rows.append({"candidate":candidate,"ms":[measure(graph,16,self.stream) for _ in range(3)]})
            self.algorithms[which]=min(rows,key=lambda r:np.median(r["ms"]))["candidate"]
            results.append(rows)
        return results

    def correction_array(self,index,shape,dtype):
        output=np.empty(shape,dtype)
        cp.cuda.runtime.memcpy(output.ctypes.data,self.library.rq_correction_buffer(self.correction,index),output.nbytes,cp.cuda.runtime.memcpyDeviceToHost)
        return output

    def close(self):
        if self.correction:self.library.rq_correction_destroy(self.correction);self.correction=None
        super().close()


def synthetic(seed,shape,profile):
    a,b=inputs(seed,shape,"outliers" if profile=="outliers" else "normal")
    if profile=="columns":
        a[:,::97]*=np.float32(32);b[:,::97]*=np.float32(32)
    elif profile=="blocks":
        a.reshape(-1,16)[::37]*=np.float32(32);b.reshape(-1,16)[::41]*=np.float32(32)
    return a,b


def configurations(k):
    return [(mode,0,128,.25) for mode in ("fp4","fp8","fp16","fp32")]+[("fp4",1,budget,.25) for budget in sorted({min(128,k),min(256,k)})]+[
        ("fp4",2,128,.25),("fp4",2,128,.75),("fp4",3,128,.25)]


def validate(plan,a,b,reference):
    if not isinstance(plan,CorrectedPlan):return verify_plan(plan,a,b,reference)
    plan.prepare();plan.run();plan.stream.synchronize()
    for operand,original in enumerate((a,b)):
        from reference import quantize
        raw,scales,_=quantize(original,"fp4")
        np.testing.assert_array_equal(plan.buffer(operand),raw.ravel())
        np.testing.assert_array_equal(plan.buffer(operand+2),scales.ravel())
    for index,expected in enumerate(reference["factors"]):
        actual=plan.correction_array(index,expected.shape,np.float16)
        np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))
    counts={}
    for key,expected in reference["metadata"].items():
        index=4 if key=="selected_columns" else 5 if key=="mask_a" else 6
        actual=plan.correction_array(index,expected.shape,expected.dtype)
        np.testing.assert_array_equal(actual,expected)
        counts[key]=int(actual.size) if key=="selected_columns" else int(np.count_nonzero(actual))
    output=cp.asnumpy(plan.output)
    checks=check_output(output,reference)
    checks.update(factor_sha256=reference["factor_sha256"],metadata_sha256=reference["metadata_sha256"],selected_counts=counts)
    return checks,output


def operation(plan,boundary):
    if boundary=="prepare-a":plan.prepare(1)
    elif boundary=="prepare-both":plan.prepare(3)
    else:plan.step(boundary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=20261119)
    parser.add_argument("--model-seed",type=int,choices=(20261116,20261216),default=20261116)
    parser.add_argument("--output",default="results/outlier-cohort-a.json")
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--reverse",action="store_true")
    args=parser.parse_args()
    cp.cuda.Device(0).use();library=load_outlier()
    data_names=[f"fixtures/{prefix}-{args.model_seed}.npz" for prefix in ("digits-model","outlier-activations")]
    record={"schema_version":1,"seed":args.seed,"model_seed":args.model_seed,"quick":args.quick,"reverse":args.reverse,"utc":datetime.now(timezone.utc).isoformat(),
            "environment":environment(0,library),"source_sha256":{name:file_hash(ROOT/name) for name in SOURCES},
            "data_sha256":{name:file_hash(ROOT/name) for name in data_names},"binary_sha256":file_hash(ROOT/"build/liboutlier.so"),
            "samples":3 if args.quick else 30,"warmups":2 if args.quick else 5,"graph_repetitions":4 if args.quick else 32,
            "scope":"FP4 plus two FP16 Tensor Core correction products; detection, selection and gather included in preparation; shared-system warm device-resident graphs",
            "negative_controls":0,"cases":[]}
    groups=[(shape,profile,"synthetic") for shape in ((128,128,128),) if args.quick for profile in PROFILES] if args.quick else [
        (shape,profile,"synthetic") for shape in SHAPES for profile in PROFILES]
    groups += [((batch,256,128),"digits-input","real" ) for batch in (16,64,192)]
    groups += [((batch,128,256),"digits-hidden","real") for batch in (16,64,192)]
    if args.quick:groups=[g for g in groups if g[2]=="synthetic" or g[0][0]==16]
    if args.reverse:groups.reverse()
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    model=np.load(ROOT/data_names[0],allow_pickle=False);activations=np.load(ROOT/data_names[1],allow_pickle=False)
    stream=cp.cuda.Stream(non_blocking=True)
    try:
        with stream:
            for shape,profile,kind in groups:
                a,b=synthetic(args.seed,shape,profile) if kind=="synthetic" else (
                    activations["x" if profile=="digits-input" else "hidden"][:shape[0]],model["w1" if profile=="digits-input" else "w2"])
                da,db=cp.asarray(a),cp.asarray(b)
                original=a.astype(np.float64)@b.astype(np.float64).T
                configs=configurations(shape[2]);configs=configs[::-1] if args.reverse else configs
                plans,graphs,entries,arrays=[],[],[],{"a":a,"b":b}
                try:
                    for mode,method,budget,fraction in configs:
                        start=time.perf_counter()
                        plan=CorrectedPlan(library,method,budget,fraction,da,db,stream) if method else HardwarePlan(library,12 if mode=="fp4" else 0,da,db,stream,mode=mode)
                        plans.append(plan);stream.synchronize();setup_ms=(time.perf_counter()-start)*1000
                        plan.prepare()
                        tuning=plan.choose() if method else tune(plan)
                        if method and not record["negative_controls"]:
                            for bad_method,bad_budget,bad_fraction in ((0,128,.25),(4,128,.25),(1,1,.25),(1,128,0)):
                                if library.rq_correction_create(plan.pointer,bad_method,bad_budget,bad_fraction):raise RuntimeError("Invalid correction plan accepted")
                            if library.rq_correction_prepare(plan.correction,0,stream.ptr)!=1:raise RuntimeError("Invalid correction mask accepted")
                            record["negative_controls"]=5
                        reference=oracle(a,b,method,budget,fraction) if method else original
                        checks,output=validate(plan,a,b,reference)
                        if output is None:output=cp.asnumpy(plan.output)
                        label=mode if not method else f"columns-{budget}" if method==1 else f"blocks-{fraction}" if method==2 else "residual"
                        arrays[label]=output
                        entry={"shape":list(shape),"profile":profile,"kind":kind,"partition":"held-out" if shape in HELD_OUT or shape[0]==192 and kind=="real" else "training",
                               "mode":mode,"method":method,"budget":budget,"fraction":fraction,"label":label,"input_sha256":[digest(a),digest(b)],"checks":checks,
                               "setup_wall_ms":setup_ms,"tuning":tuning,"timings_ms":{key:[] for key in BOUNDARIES},"gpu_before":gpu_snapshot(0)}
                        entries.append(entry)
                        for boundary in BOUNDARIES:graphs.append((capture(lambda p=plan,key=boundary:operation(p,key),record["graph_repetitions"],stream),entry,boundary))
                    for graph,_,_ in graphs:
                        for _ in range(record["warmups"]):graph.launch(stream)
                    stream.synchronize()
                    for observation in range(record["samples"]):
                        offset=observation%len(graphs)
                        for graph,entry,boundary in graphs[offset:]+graphs[:offset]:entry["timings_ms"][boundary].append(measure(graph,record["graph_repetitions"],stream))
                    for plan,entry in zip(plans,entries):
                        reference=oracle(a,b,entry["method"],entry["budget"],entry["fraction"]) if entry["method"] else original
                        entry["post_timing_checks"],_=validate(plan,a,b,reference)
                        entry["gpu_after"]=gpu_snapshot(0)
                    if shape==(128,128,128) or kind=="real" and shape[0]==16:
                        path=destination.parent/f"outlier-witness-{args.seed}-{profile}.npz";np.savez_compressed(path,**arrays)
                        for entry in entries:entry["witness"]={"file":path.name,"array":entry["label"],"sha256":file_hash(path)}
                    record["cases"].extend(entries);write_record(destination,record)
                    print(f"{shape} {profile}: {len(entries)} correction/reference configurations passed",flush=True)
                finally:
                    stream.synchronize();graphs=[]
                    for plan in plans:plan.close()
        record["complete"]=True;record["environment_after"]=gpu_snapshot(0);write_record(destination,record)
    finally:model.close();activations.close()


if __name__=="__main__":main()
