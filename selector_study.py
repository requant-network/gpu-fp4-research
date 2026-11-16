"""Native sampled error prediction with training-only calibration and full costs."""
import argparse
import ctypes
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import cupy as cp
import numpy as np

from followup import ROOT,ExtendedPlan,file_hash,load_extended,write_record
from gpu_study import capture,environment,gpu_snapshot,measure,tune
from precision_study import PrecisionPlan,load_precision
from precision_reference import exact_product,exact_samples,gold
from reference import MODES,digest,quantize,relative_error
from selector_reference import BUDGETS,HELD_SHAPES,PROFILES,SAMPLE_COUNT,TRAIN_SHAPES,choose,dataset,feature,fit_policy,probe_reference,sample_indices

SOURCES=("selector_study.py","selector_reference.py","native/selector_bridge.cu","build_selector.sh",
         "reference.py","gpu_study.py","followup.py","precision_study.py","precision_reference.py",
         "native/bridge.cu","native/optimized_bridge.cu","native/precision_bridge.cu","build_precision.sh")


def load_selector():
    prototype=load_extended(128);library=ctypes.CDLL(str(ROOT/"build/libselector.so"))
    for name in ("rq_error","rq_create","rq_destroy","rq_quantize","rq_run","rq_candidates","rq_algorithm_id","rq_buffer","rq_bytes","rq_cublas_version","rq_quantize_variant"):
        target,source=getattr(library,name),getattr(prototype,name);target.argtypes,target.restype=source.argtypes,source.restype
    ptr,integer=ctypes.c_void_p,ctypes.c_int
    library.rq_selector_probe.argtypes=[ptr,ptr,integer,ptr,ptr];library.rq_selector_probe.restype=integer
    library.rq_selector_promote.argtypes=[ptr,ptr,ptr];library.rq_selector_promote.restype=integer
    return library


class Candidate:
    def __init__(self,mode,libraries,a,b,indices,stream):
        self.mode,self.stream=mode,stream
        self.plan=PrecisionPlan(libraries[1],1,0,a,b,stream) if mode=="fp64" else ExtendedPlan(libraries[0],3,a,b,stream,mode=mode)
        self.output=self.plan.output if mode=="fp64" else cp.empty(self.plan.output.shape,cp.float64)
        self.indices=indices;self.samples=cp.empty((SAMPLE_COUNT,3),cp.float64) if mode!="fp64" else None

    def prepare(self,mask=3):self.plan.prepare(mask)

    def run(self):
        self.plan.run()
        if self.mode!="fp64":self.plan.checked(self.plan.library.rq_selector_promote(self.plan.pointer,self.output.data.ptr,self.stream.ptr))

    def probe(self):
        self.plan.checked(self.plan.library.rq_selector_probe(self.plan.pointer,self.indices.data.ptr,SAMPLE_COUNT,self.samples.data.ptr,self.stream.ptr))

    def direct(self,a,output):
        cp.cuda.runtime.memcpyAsync(self.plan.inputs[0].data.ptr,a.ctypes.data,a.nbytes,cp.cuda.runtime.memcpyHostToDevice,self.stream.ptr)
        self.prepare(1);self.run();self.copy_output(output)

    def copy_output(self,output):
        cp.cuda.runtime.memcpyAsync(output.ctypes.data,self.output.data.ptr,output.nbytes,cp.cuda.runtime.memcpyDeviceToHost,self.stream.ptr);self.stream.synchronize()

    def close(self):self.plan.close()


def cpu_reference(a,b,mode,target):
    hashes={};values=[]
    for operand,x in enumerate((a,b)):
        if mode=="fp64":raw=x.astype(np.float64);scales=np.empty(0,np.uint8);q=raw
        else:raw,scales,q=quantize(x,mode)
        hashes[f"operand_{operand}_encoded"]=digest(raw)
        if mode=="fp4":hashes[f"operand_{operand}_scales"]=digest(scales)
        values.append(q.astype(np.float64))
    x,y=values;expected=x@y.T;u=2**-53 if mode=="fp64" else 2**-24;gamma=(a.shape[1]+20)*u/(1-(a.shape[1]+20)*u)
    bound=8*gamma*(np.abs(x)@np.abs(y).T)+np.finfo(np.float64 if mode=="fp64" else np.float32).tiny*8*a.shape[1]
    return expected,bound,hashes,target


def check_actual(actual,reference,indices,exact):
    expected,bound,hashes,target=reference
    if actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError("Invalid selector output")
    ratio=float(np.max(np.abs(actual-expected)/bound))
    if ratio>1:raise ValueError(f"Selector arithmetic exceeded bound: {ratio}")
    return {"arithmetic_max_bound_ratio":ratio,"relative_frobenius_error":relative_error(actual,target),
            "exact_sample_relative_error":relative_error(actual[indices[:,0],indices[:,1]],exact),
            "exact_sample_values":actual[indices[:,0],indices[:,1]].tolist(),"checked_output_entries":int(actual.size),
            "output_sha256":digest(actual),"prepared_sha256":hashes}


def validate(candidate,a,b,reference,indices,exact):
    candidate.stream.synchronize()
    for operand,x in enumerate((a,b)):
        if candidate.mode=="fp64":np.testing.assert_array_equal(candidate.plan.array(operand,x.shape,np.float64),x.astype(np.float64))
        else:
            raw,scales,_=quantize(x,candidate.mode);np.testing.assert_array_equal(candidate.plan.buffer(operand),raw.view(np.uint8).ravel())
            if candidate.mode=="fp4":np.testing.assert_array_equal(candidate.plan.buffer(operand+2),scales.ravel())
    actual=cp.asnumpy(candidate.output)
    return check_actual(actual,reference,indices,exact),actual


def validate_probe(values,a,b,mode,indices):
    expected=probe_reference(a,b,mode,indices);original_abs=np.array([np.dot(np.abs(a[i].astype(np.float64)),np.abs(b[j].astype(np.float64))) for i,j in indices])
    gamma=(a.shape[1]+128)*2**-53/(1-(a.shape[1]+128)*2**-53)
    bounds=np.column_stack((original_abs,np.abs(expected[:,2]),np.abs(expected[:,2])))*8*gamma+np.finfo(np.float64).tiny
    if values.shape!=expected.shape or not np.isfinite(values).all() or np.any(np.abs(values-expected)>bounds):raise ValueError("Native selector probe disagrees with CPU dot products")


def selector_request(candidates,policy,a,output,budget):
    first=next(iter(candidates.values()));device_a=first.plan.inputs[0]
    cp.cuda.runtime.memcpyAsync(device_a.data.ptr,a.ctypes.data,a.nbytes,cp.cuda.runtime.memcpyHostToDevice,first.stream.ptr)
    for mode in MODES:candidates[mode].prepare(1);candidates[mode].probe()
    probes={mode:cp.asnumpy(candidates[mode].samples) for mode in MODES}
    shape=(a.shape[0],candidates["fp64"].output.shape[1],a.shape[1]);selected,estimates,nearest=choose(policy,shape,probes,budget)
    candidate=candidates[selected]
    if selected=="fp64":candidate.prepare(1)
    candidate.run();candidate.copy_output(output)
    return selected,estimates,nearest,probes


def transition_probe(libraries,policy,seed,stream,destination):
    shape=(16,128,128);initial,b=dataset(seed,shape,"normal");indices=sample_indices(seed,shape)
    da,db,di=cp.asarray(initial),cp.asarray(b),cp.asarray(indices);candidates={};rows=[];arrays={"b":b,"indices":indices}
    try:
        for mode in (*MODES,"fp64"):
            candidate=Candidate(mode,libraries,da,db,di,stream);candidates[mode]=candidate;candidate.prepare()
        for index,profile in enumerate(("normal","cancellation","spread","normal")):
            a,_=dataset(seed+index,shape,profile);exact=exact_product(a,b);arrays[f"a{index}"]=a;arrays[f"exact{index}"]=exact
            for budget in BUDGETS:
                output=np.empty(exact.shape,np.float64);selected,estimates,nearest,probes=selector_request(candidates,policy,a,output,budget)
                for mode in MODES:validate_probe(probes[mode],a,b,mode,indices)
                reference=cpu_reference(a,b,selected,exact);checks=check_actual(output,reference,indices,exact[indices[:,0],indices[:,1]])
                validate(candidates[selected],a,b,reference,indices,exact[indices[:,0],indices[:,1]])
                key=f"output{index}-{budget}";arrays[key]=output
                rows.append({"index":index,"profile":profile,"maximum_relative_error":budget,"selected":selected,"estimates":estimates,
                             "nearest_training_shape":nearest,"probe":{mode:probes[mode].tolist() for mode in MODES},"checks":checks,"array":key})
        path=destination.parent/f"selector-transitions-{seed}.npz";np.savez_compressed(path,**arrays)
        return {"rows":rows,"witness":{"file":path.name,"sha256":file_hash(path)}}
    finally:
        stream.synchronize()
        for candidate in candidates.values():candidate.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--phase",choices=("calibration","evaluation"),required=True)
    parser.add_argument("--seed",type=int,default=20261122);parser.add_argument("--model-seed",type=int,default=20261116)
    parser.add_argument("--policy");parser.add_argument("--output",required=True);parser.add_argument("--reverse",action="store_true");parser.add_argument("--quick",action="store_true")
    args=parser.parse_args()
    if args.seed<0 or args.model_seed<0 or args.phase=="evaluation" and not args.policy:parser.error("Nonnegative seeds and evaluation policy required")
    policy=json.loads(Path(args.policy).read_text()) if args.policy else None
    cp.cuda.Device(0).use();libraries=load_selector(),load_precision();stream=cp.cuda.Stream(non_blocking=True)
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    record={"schema_version":1,"utc":datetime.now(timezone.utc).isoformat(),"phase":args.phase,"seed":args.seed,"model_seed":args.model_seed,"quick":args.quick,"reverse":args.reverse,
            "environment":environment(0,libraries[0]),"source_sha256":{name:file_hash(ROOT/name) for name in SOURCES},
            "binary_sha256":{name:file_hash(ROOT/f"build/{name}") for name in ("libselector.so","libprecision.so")},
            "data_sha256":{f"fixtures/{prefix}-{args.model_seed}.npz":file_hash(ROOT/f"fixtures/{prefix}-{args.model_seed}.npz") for prefix in ("digits-model","outlier-activations")},
            "policy_sha256":file_hash(Path(args.policy)) if args.policy else None,"policy":policy,"negative_controls":0,
            "samples":3 if args.quick else 30,"warmups":2 if args.quick else 5,"probe_samples":SAMPLE_COUNT,"cases":[],"selections":[]}
    shapes=TRAIN_SHAPES if args.phase=="calibration" else TRAIN_SHAPES+HELD_SHAPES;profiles=PROFILES[:2] if args.phase=="calibration" else PROFILES
    groups=[(shape,profile,"synthetic") for shape in shapes for profile in profiles]
    model=np.load(ROOT/f"fixtures/digits-model-{args.model_seed}.npz",allow_pickle=False);activations=np.load(ROOT/f"fixtures/outlier-activations-{args.model_seed}.npz",allow_pickle=False)
    if args.phase=="evaluation":groups += [((batch,256,128),"digits-input","real") for batch in (64,192)]+[((batch,128,256),"digits-hidden","real") for batch in (64,192)]
    if args.quick:groups=[((128,128,128),profile,"synthetic") for profile in ("normal","cancellation","spread")]
    if args.reverse:groups.reverse()
    with stream:
        for shape,profile,kind in groups:
            a,b=dataset(args.seed,shape,profile) if kind=="synthetic" else (activations["x" if profile=="digits-input" else "hidden"][:shape[0]],model["w1" if profile=="digits-input" else "w2"])
            target=gold(a,b,profile);indices=sample_indices(args.seed,shape);exact=exact_samples(a,b,indices);da,db,di=cp.asarray(a),cp.asarray(b),cp.asarray(indices)
            host_output=np.empty((shape[0],shape[1]),np.float64);candidates={};entries={};references={};graphs=[];witness={"a":a,"b":b,"indices":indices,"exact_samples":exact}
            repetitions=1 if shape[0]*shape[1]*shape[2]>=2**29 else 8
            order=list((*MODES,"fp64"));order=order[::-1] if args.reverse else order
            try:
                for mode in order:
                    start=time.perf_counter();candidate=Candidate(mode,libraries,da,db,di,stream);candidates[mode]=candidate;candidate.prepare();stream.synchronize();setup=(time.perf_counter()-start)*1000
                    tuning=candidate.plan.choose(4) if mode=="fp64" else tune(candidate.plan)
                    candidate.prepare();candidate.run();reference=cpu_reference(a,b,mode,target);references[mode]=reference
                    checks,actual=validate(candidate,a,b,reference,indices,exact)
                    probe=None
                    if mode!="fp64":candidate.probe();probe=cp.asnumpy(candidate.samples);validate_probe(probe,a,b,mode,indices)
                    entry={"shape":list(shape),"profile":profile,"kind":kind,"mode":mode,"input_sha256":[digest(a),digest(b)],
                           "setup_and_initial_preparation_wall_ms":setup,"tuning":tuning,"graph_repetitions":repetitions,"checks":checks,
                           "probe":None if probe is None else probe.tolist(),"feature":None if probe is None else feature(probe,shape[2]),
                           "timings_ms":{key:[] for key in ("prepacked","online-a","online-both")},"host_request_ms":[]}
                    entries[mode]=entry
                    if shape==(128,128,128):witness[mode]=actual
                    if not record["negative_controls"] and mode!="fp64":
                        for pointer,count,result in ((None,64,candidate.samples.data.ptr),(candidate.plan.pointer,0,candidate.samples.data.ptr),(candidate.plan.pointer,1025,candidate.samples.data.ptr),(candidate.plan.pointer,64,None)):
                            if libraries[0].rq_selector_probe(pointer,di.data.ptr,count,result,stream.ptr)!=1:raise ValueError("Invalid probe accepted")
                        record["negative_controls"]=4
                    for boundary in entry["timings_ms"]:
                        def operation(c=candidate,key=boundary):
                            if key!="prepacked":c.prepare(1 if key=="online-a" else 3)
                            c.run()
                        graphs.append((capture(operation,repetitions,stream),entry,boundary))
                for graph,_,_ in graphs:
                    for _ in range(record["warmups"]):graph.launch(stream)
                stream.synchronize()
                for observation in range(record["samples"]):
                    offset=observation%len(graphs)
                    for graph,entry,boundary in graphs[offset:]+graphs[:offset]:entry["timings_ms"][boundary].append(measure(graph,repetitions,stream))
                for mode,candidate in candidates.items():
                    entries[mode]["post_timing_checks"],_=validate(candidate,a,b,references[mode],indices,exact)
                    for _ in range(record["warmups"]):candidate.direct(a,host_output)
                    for _ in range(record["samples"]):
                        start=time.perf_counter();candidate.direct(a,host_output);entries[mode]["host_request_ms"].append((time.perf_counter()-start)*1000)
                    entries[mode]["post_request_checks"],_=validate(candidate,a,b,references[mode],indices,exact)
                if policy:
                    for budget in BUDGETS:
                        row={"shape":list(shape),"profile":profile,"kind":kind,"maximum_relative_error":budget,"host_request_ms":[],"selected":[]}
                        for _ in range(record["warmups"]):selector_request(candidates,policy,a,host_output,budget)
                        for _ in range(record["samples"]):
                            start=time.perf_counter();selected,estimates,nearest,probes=selector_request(candidates,policy,a,host_output,budget);row["host_request_ms"].append((time.perf_counter()-start)*1000);row["selected"].append(selected)
                            if not row.get("probe"):
                                for mode in MODES:validate_probe(probes[mode],a,b,mode,indices)
                                row["probe"]={mode:probes[mode].tolist() for mode in MODES};row["estimates"],row["nearest_training_shape"]=estimates,nearest
                            else:
                                for mode in MODES:np.testing.assert_array_equal(probes[mode],np.array(row["probe"][mode]))
                        row["checks"]=check_actual(host_output,references[selected],indices,exact)
                        row["budget_met"]=max(row["checks"]["relative_frobenius_error"],row["checks"]["exact_sample_relative_error"])<=budget
                        record["selections"].append(row)
                        if shape==(128,128,128):witness[f"selected-{budget}"]=host_output.copy()
                if shape==(128,128,128):
                    witness["exact"]=exact_product(a,b);path=destination.parent/f"selector-{args.phase}-witness-{args.seed}-{profile}.npz";np.savez_compressed(path,**witness)
                    for entry in entries.values():entry["witness"]={"file":path.name,"array":entry["mode"],"sha256":file_hash(path)}
                    for row in record["selections"]:
                        if tuple(row["shape"])==shape and row["profile"]==profile:row["witness"]={"file":path.name,"array":f"selected-{row['maximum_relative_error']}","sha256":file_hash(path)}
                record["cases"].extend(entries.values());write_record(destination,record)
                print(f"{shape} {profile}: {len(entries)} validated references, {len(BUDGETS) if policy else 0} complete selector requests",flush=True)
            finally:
                stream.synchronize();graphs=[]
                for candidate in candidates.values():candidate.close()
        if policy:record["transition_probe"]=transition_probe(libraries,policy,args.seed,stream,destination)
    model.close();activations.close();record["complete"]=True;record["environment_after"]=gpu_snapshot(0);write_record(destination,record)
    if args.phase=="calibration" and not args.quick:write_record(destination.with_name("selector-policy.json"),fit_policy(record))


if __name__=="__main__":main()
