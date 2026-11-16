"""FP64 reconstruction from INT8 products with native/library-emulation controls."""
import argparse
import ctypes
from datetime import datetime,timezone
from pathlib import Path
import time

import cupy as cp
import numpy as np

from followup import ROOT,file_hash,write_record
from gpu_study import capture,environment,gpu_snapshot,measure
from precision_reference import arithmetic_reference,check_output,components,dataset,exact_product,exact_samples,gold,row_statistics
from reference import digest,relative_error

CONFIGURATIONS=(("fp32",0,0),("fp64",1,0),("emu64-performant",2,-1),("emu64-eager",2,0),("emu64-fixed31",2,31),("emu64-fixed55",2,55),
                ("requested-bf16x9",3,0))+tuple((f"slices-{count}",4,count) for count in (1,2,3,4,6,8))+(("adaptive-8",5,8),)
SHAPES=((16,512,512),(128,128,128),(256,512,512),(1024,1024,1024),(2048,2048,2048),(32,768,1280),(384,768,256),(128,128,8192))
HELD_OUT=((32,768,1280),(384,768,256))
PROFILES=("normal","cancellation","spread","subnormal")
BOUNDARIES=("prepacked","online-a","online-both")
SOURCES=("precision_study.py","precision_reference.py","native/precision_bridge.cu","build_precision.sh","reference.py","gpu_study.py","followup.py")


def load_precision():
    library=ctypes.CDLL(str(ROOT/"build/libprecision.so"));p,i=ctypes.c_void_p,ctypes.c_int
    library.hp_error.restype=ctypes.c_char_p;library.rq_cublas_version.restype=ctypes.c_size_t
    signatures={"create":([i,i,i,i,i,p,p,p],p),"destroy":([p],None),"analyze":([p,i,p],i),"encode":([p,i,p],i),"select":([p,p],i),
                "run":([p,i,i,p],i),"candidates":([p,i],i),"tune_prepare":([p,p],i),"tune_gemm":([p,i,i,p],i),
                "algorithm_info":([p,i,i,ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(i),ctypes.POINTER(i)],i),"buffer":([p,i],p),
                "emulation_config":([p,ctypes.POINTER(i)],i)}
    for name,(args,result) in signatures.items():
        function=getattr(library,"hp_"+name);function.argtypes,function.restype=args,result
    return library


class PrecisionPlan:
    def __init__(self,library,mode,parameter,a,b,stream):
        self.library,self.mode,self.parameter,self.inputs,self.stream=library,mode,parameter,(a,b),stream
        self.output=cp.empty((a.shape[0],b.shape[0]),cp.float64);self.selected=parameter if mode>=4 else 0;self.algorithms=[0,0]
        self.pointer=library.hp_create(mode,parameter,a.shape[0],b.shape[0],a.shape[1],a.data.ptr,b.data.ptr,self.output.data.ptr)
        if not self.pointer:raise RuntimeError(library.hp_error().decode())

    def checked(self,status):
        if status:raise RuntimeError(self.library.hp_error().decode())

    def prepare(self,mask=3,dynamic=False):
        self.checked(self.library.hp_analyze(self.pointer,mask,self.stream.ptr))
        if self.mode==5 and dynamic:
            old=self.selected;choice=self.library.hp_select(self.pointer,self.stream.ptr)
            if choice<0:raise RuntimeError(self.library.hp_error().decode())
            self.selected=choice
            if choice!=old:mask=3
        self.checked(self.library.hp_encode(self.pointer,mask,self.stream.ptr))

    def run(self):self.checked(self.library.hp_run(self.pointer,*self.algorithms,self.stream.ptr))

    def step(self,boundary):
        if boundary!="prepacked":self.prepare(1 if boundary=="online-a" else 3)
        self.run()

    def array(self,index,shape,dtype):
        output=np.empty(shape,dtype);pointer=self.library.hp_buffer(self.pointer,index)
        if not pointer:raise RuntimeError("Unavailable precision buffer")
        cp.cuda.runtime.memcpy(output.ctypes.data,pointer,output.nbytes,cp.cuda.runtime.memcpyDeviceToHost)
        return output

    def choose(self,repetitions):
        self.checked(self.library.hp_tune_prepare(self.pointer,self.stream.ptr));self.stream.synchronize();result=[]
        for which in ((0,1) if self.mode==5 else (0,)):
            rows=[]
            for candidate in range(self.library.hp_candidates(self.pointer,which)):
                graph=capture(lambda:self.checked(self.library.hp_tune_gemm(self.pointer,which,candidate,self.stream.ptr)),repetitions,self.stream)
                for _ in range(2):graph.launch(self.stream)
                self.stream.synchronize();values=[measure(graph,repetitions,self.stream) for _ in range(3)]
                flags,emulated,identifier=ctypes.c_uint64(),ctypes.c_int(),ctypes.c_int()
                self.checked(self.library.hp_algorithm_info(self.pointer,which,candidate,ctypes.byref(flags),ctypes.byref(emulated),ctypes.byref(identifier)))
                rows.append({"candidate":candidate,"algorithm_id":identifier.value,"numerical_flags":flags.value,"emulation_capability":emulated.value,"ms":values})
            eligible=[row for row in rows if self.mode<4 or which==1 or row["numerical_flags"]&4]
            if not eligible:raise RuntimeError("No IMMA Tensor Core candidate for component products")
            self.algorithms[which]=min(eligible,key=lambda row:np.median(row["ms"]))["candidate"];result.append(rows)
        return result

    def host_request(self,a,output):
        cp.cuda.runtime.memcpyAsync(self.inputs[0].data.ptr,a.ctypes.data,a.nbytes,cp.cuda.runtime.memcpyHostToDevice,self.stream.ptr)
        self.prepare(1,dynamic=True);self.run()
        cp.cuda.runtime.memcpyAsync(output.ctypes.data,self.output.data.ptr,output.nbytes,cp.cuda.runtime.memcpyDeviceToHost,self.stream.ptr)
        self.stream.synchronize()

    def close(self):
        if self.pointer:self.library.hp_destroy(self.pointer);self.pointer=None


def validate(plan,a,b,target,indices,cache=None,exact=None):
    plan.stream.synchronize();metadata={};hashes={};required=None
    if plan.mode>=4:
        for operand,x in enumerate((a,b)):
            exponent,req=row_statistics(x)
            for index,expected in ((operand+2,exponent),(operand+4,req)):
                np.testing.assert_array_equal(plan.array(index,expected.shape,np.int32),expected)
            metadata[f"operand_{operand}_exponents"],metadata[f"operand_{operand}_required"]=digest(exponent),digest(req)
            if plan.selected:
                parts,_,_,_=components(x,plan.selected)
                np.testing.assert_array_equal(plan.array(operand,parts.shape,np.int8),parts)
                hashes[f"operand_{operand}_components"]=digest(parts)
        required=int(max(row_statistics(a)[1].max(),row_statistics(b)[1].max(),1))
        if int(plan.array(6,(1,),np.int32)[0])!=required:raise ValueError("Incorrect required component count")
        if plan.mode==5 and plan.selected!=(required if required<=plan.parameter else 0):raise ValueError("Adaptive component selection failed")
    if not (plan.mode>=4 and plan.selected):
        dtype=np.float32 if plan.mode in (0,3) else np.float64
        for operand,x in enumerate((a,b)):
            actual=plan.array(operand,x.shape,dtype);np.testing.assert_array_equal(actual,x.astype(dtype))
            hashes[f"operand_{operand}_prepared"]=digest(actual)
    retained=int(plan.array(7,(1,),np.int32)[0]);actual=cp.asnumpy(plan.output)
    if cache is None:cache={}
    key=(plan.selected,retained)
    if key not in cache:cache[key]=arithmetic_reference(a,b,plan.mode,plan.selected,retained)
    expected,bound,_,_=cache[key]
    checks=check_output(actual,expected,bound,target)
    if exact is None:exact=exact_samples(a,b,indices)
    sampled=actual[indices[:,0],indices[:,1]]
    checks.update(prepared_sha256=hashes,metadata_sha256=metadata,required_components=required,selected_components=plan.selected,
                  retained_emulation_bits=retained,exact_sample_relative_error=relative_error(sampled,exact),exact_samples_sha256=digest(exact),exact_sample_values=sampled.tolist())
    if plan.mode>=4 and plan.selected:
        partial=plan.array(8,(a.shape[0],b.shape[0]),np.int32)
        aa,bb=components(a,plan.selected)[0][0],components(b,plan.selected)[0][0]
        exact_partial=np.array([np.dot(aa[i].astype(np.int64),bb[j].astype(np.int64)) for i,j in indices],np.int64)
        np.testing.assert_array_equal(partial[indices[:,0],indices[:,1]].astype(np.int64),exact_partial)
        checks["last_slice_product_samples_sha256"]=digest(exact_partial)
        checks["last_slice_product_samples"]=partial[indices[:,0],indices[:,1]].tolist()
    return checks,actual,exact


def transition_probe(library,seed,stream,destination):
    _,b=dataset(seed,(16,128,128),"normal");a,_=dataset(seed,(16,128,128),"normal")
    da,db=cp.asarray(a),cp.asarray(b);plan=PrecisionPlan(library,5,8,da,db,stream);rows=[];arrays={"b":b}
    try:
        plan.prepare(dynamic=True);plan.choose(2)
        for index,profile in enumerate(("normal","cancellation","spread","normal")):
            a,_=dataset(seed+index,(16,128,128),profile);output=np.empty((16,128),np.float64)
            plan.host_request(a,output);exact=exact_product(a,b)
            expected,bound,_,_=arithmetic_reference(a,b,5,plan.selected,-1)
            check_output(output,expected,bound,exact)
            rows.append({"profile":profile,"selected_components":plan.selected,"input_sha256":digest(a),"output_sha256":digest(output),"exact_relative_error":relative_error(output,exact)})
            arrays[f"a{index}"]=a;arrays[f"output{index}"]=output;arrays[f"exact{index}"]=exact
        path=destination.parent/f"precision-transitions-{seed}.npz";np.savez_compressed(path,**arrays)
        return {"rows":rows,"witness":{"file":path.name,"sha256":file_hash(path)},"scope":"Changed A inputs force component-count/fallback transitions; B stays resident; reselection and required B re-encoding included"}
    finally:plan.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=20261121);parser.add_argument("--model-seed",type=int,choices=(20261116,20261216),default=20261116)
    parser.add_argument("--output",default="results/precision-cohort-a.json");parser.add_argument("--quick",action="store_true");parser.add_argument("--reverse",action="store_true")
    args=parser.parse_args();cp.cuda.Device(0).use();library=load_precision();destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    data_names=[f"fixtures/{prefix}-{args.model_seed}.npz" for prefix in ("digits-model","outlier-activations")]
    record={"schema_version":1,"seed":args.seed,"model_seed":args.model_seed,"quick":args.quick,"reverse":args.reverse,"utc":datetime.now(timezone.utc).isoformat(),
            "environment":environment(0,library),"source_sha256":{name:file_hash(ROOT/name) for name in SOURCES},"data_sha256":{name:file_hash(ROOT/name) for name in data_names},
            "binary_sha256":file_hash(ROOT/"build/libprecision.so"),"samples":3 if args.quick else 30,"warmups":2 if args.quick else 5,"workspace_bytes":512*1024**2,
            "scope":"Common FP32 source inputs and FP64 outputs; INT8 7-bit components/exact integer diagonals/FP64 reconstruction; captured device times cache adaptive host decision for repeated identical inputs; host-request times charge GPU statistics, D2H selection, changed decision preparation, H2D A and D2H full output",
            "negative_controls":0,"unavailable":[],"cases":[]}
    shapes=((128,128,128),) if args.quick else SHAPES
    groups=[(shape,profile,"synthetic") for shape in shapes for profile in PROFILES]
    if not args.quick:
        groups += [((batch,256,128),"digits-input","real") for batch in (64,192)]+[((batch,128,256),"digits-hidden","real") for batch in (64,192)]
    configs=list(CONFIGURATIONS)
    if args.reverse:groups.reverse();configs.reverse()
    model=np.load(ROOT/data_names[0],allow_pickle=False);activations=np.load(ROOT/data_names[1],allow_pickle=False);stream=cp.cuda.Stream(non_blocking=True)
    try:
        with stream:
            record["adaptive_transition_probe"]=transition_probe(library,args.seed,stream,destination)
            for shape,profile,kind in groups:
                a,b=dataset(args.seed,shape,profile) if kind=="synthetic" else (activations["x" if profile=="digits-input" else "hidden"][:shape[0]],model["w1" if profile=="digits-input" else "w2"])
                target=gold(a,b,profile);da,db=cp.asarray(a),cp.asarray(b);stream.synchronize()
                rng=np.random.default_rng(np.random.SeedSequence([args.seed,*shape,97]));indices=np.column_stack((rng.integers(shape[0],size=64),rng.integers(shape[1],size=64))).astype(np.int32)
                outputs={"a":a,"b":b,"indices":indices,"exact_samples":exact_samples(a,b,indices)}
                if shape==(128,128,128):outputs["exact"]=exact_product(a,b)
                entries=[];repetitions=1 if shape[0]*shape[1]*shape[2]>=2**29 or shape[2]>=4096 else 8
                if args.quick:repetitions=2
                for label,mode,parameter in configs:
                    begin=time.perf_counter()
                    try:plan=PrecisionPlan(library,mode,parameter,da,db,stream)
                    except RuntimeError as error:
                        if mode!=3:raise
                        record["unavailable"].append({"shape":shape,"profile":profile,"label":label,"diagnostic":str(error)})
                        continue
                    try:
                        stream.synchronize();setup_ms=(time.perf_counter()-begin)*1000
                        if not record["negative_controls"]:
                            for bad in ((9,0,16),(4,0,16),(4,9,16),(1,1,16),(0,0,1),(2,80,16)):
                                if library.hp_create(bad[0],bad[1],bad[2],128,128,da.data.ptr,db.data.ptr,plan.output.data.ptr):raise RuntimeError("Invalid precision configuration accepted")
                            if library.hp_analyze(plan.pointer,0,stream.ptr)!=1:raise RuntimeError("Invalid precision mask accepted")
                            record["negative_controls"]=7
                        begin=time.perf_counter();plan.prepare(dynamic=True);stream.synchronize();initial_ms=(time.perf_counter()-begin)*1000
                        tuning=plan.choose(1 if repetitions==1 else 4)
                        validation_cache={};plan.prepare(dynamic=True);plan.run();checks,actual,_=validate(plan,a,b,target,indices,validation_cache,outputs["exact_samples"])
                        descriptor=None
                        if mode in (2,3):
                            values=(ctypes.c_int*5)();plan.checked(library.hp_emulation_config(plan.pointer,values))
                            descriptor=dict(zip(("strategy","special_values_mask","mantissa_control","max_mantissa_bits","mantissa_bit_offset"),values))
                        entry={"shape":shape,"profile":profile,"kind":kind,"partition":"held-out" if shape in HELD_OUT or kind=="real" and shape[0]==192 else "training",
                               "label":label,"mode":mode,"parameter":parameter,"input_sha256":[digest(a),digest(b)],"checks":checks,"tuning":tuning,"selected_algorithms":plan.algorithms,
                               "emulation_descriptor":descriptor,"setup_wall_ms":setup_ms,"initial_preparation_wall_ms":initial_ms,"graph_repetitions":repetitions,"timings_ms":{key:[] for key in BOUNDARIES},"host_request_ms":[],"gpu_before":gpu_snapshot(0)}
                        graphs=[(capture(lambda key=key:plan.step(key),repetitions,stream),key) for key in BOUNDARIES]
                        for graph,_ in graphs:
                            for _ in range(record["warmups"]):graph.launch(stream)
                        stream.synchronize()
                        for observation in range(record["samples"]):
                            offset=observation%len(graphs)
                            for graph,key in graphs[offset:]+graphs[:offset]:entry["timings_ms"][key].append(measure(graph,repetitions,stream))
                        entry["post_timing_checks"],_,_=validate(plan,a,b,target,indices,validation_cache,outputs["exact_samples"])
                        host_output=np.empty(plan.output.shape,np.float64)
                        for _ in range(record["samples"]):
                            begin=time.perf_counter();plan.host_request(a,host_output);entry["host_request_ms"].append((time.perf_counter()-begin)*1000)
                        entry["post_request_checks"],_,_=validate(plan,a,b,target,indices,validation_cache,outputs["exact_samples"])
                        entry["gpu_after"]=gpu_snapshot(0);entries.append(entry)
                        if shape==(128,128,128):outputs[label]=actual
                    finally:stream.synchronize();graphs=[];plan.close()
                if shape==(128,128,128):
                    path=destination.parent/f"precision-witness-{args.seed}-{profile}.npz";np.savez_compressed(path,**outputs)
                    for entry in entries:entry["witness"]={"file":path.name,"array":entry["label"],"sha256":file_hash(path)}
                record["cases"].extend(entries);write_record(destination,record)
                print(f"{shape} {profile}: {len(entries)} precision configurations passed",flush=True)
        record["complete"]=True;record["environment_after"]=gpu_snapshot(0);write_record(destination,record)
    finally:model.close();activations.close()


if __name__=="__main__":main()
