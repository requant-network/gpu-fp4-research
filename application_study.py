"""Complete real-data classifier quality, latency and reuse study on SM120."""
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
from prepare_digits import load_digits
from reference import check_output,digest,quantize,relative_error

CONFIGURATIONS=(("fp32","fp32","cutlass",False),("fp16","fp16","cutlass",False),
                ("bf16","bf16","cutlass",False),("fp8","fp8","cutlass",False),
                ("fp4","fp4","cutlass",True),("fp4","fp16","cutlass",False),
                ("fp4","fp4","cublaslt",True),("fp4","fp16","cublaslt",False))
BATCHES=(16,32,64,128,192,256)
SOURCES=("application_study.py","prepare_digits.py","reference.py","gpu_study.py","followup.py",
         "native/bridge.cu","native/optimized_bridge.cu","native/application_bridge.cu",
         "build_application.sh","build_extended.sh")


def load_application():
    prototype=load_extended()
    library=ctypes.CDLL(str(ROOT/"build/libapplication.so"))
    for name in ("rq_error","rq_create","rq_destroy","rq_quantize","rq_run","rq_candidates",
                 "rq_algorithm_id","rq_buffer","rq_bytes","rq_cublas_version","rq_quantize_variant",
                 "rq_prepare_fused","rq_activate"):
        target,original=getattr(library,name),getattr(prototype,name)
        target.argtypes,target.restype=original.argtypes,original.restype
    ptr,integer=ctypes.c_void_p,ctypes.c_int
    library.rq_library_fp4_create.argtypes=[ptr]
    library.rq_library_fp4_create.restype=ptr
    library.rq_library_fp4_destroy.argtypes=[ptr]
    library.rq_library_fp4_destroy.restype=None
    library.rq_library_fp4_count.argtypes=[ptr]
    library.rq_library_fp4_count.restype=integer
    library.rq_library_fp4_run.argtypes=[ptr,integer,ptr]
    library.rq_library_fp4_run.restype=integer
    library.rq_classify.argtypes=[ptr,ptr,ptr,integer,integer,integer,ptr]
    library.rq_classify.restype=integer
    return library


class BackendPlan(ExtendedPlan):
    def __init__(self,library,mode,backend,variant,a,b,stream,bias=None,fused=False):
        self.alternative=None
        super().__init__(library,variant,a,b,stream,bias,fused,mode)
        if mode=="fp4" and backend=="cublaslt":
            self.alternative=library.rq_library_fp4_create(self.pointer)
            if not self.alternative:
                error=library.rq_error().decode()
                super().close()
                raise RuntimeError(error)

    def run(self):
        if self.alternative:
            self.checked(self.library.rq_library_fp4_run(self.alternative,self.algorithm,self.stream.ptr))
        else:
            super().run()

    def choose(self):
        if not self.alternative:
            return tune(self)
        observations=[]
        for candidate in range(self.library.rq_library_fp4_count(self.alternative)):
            self.algorithm=candidate
            graph=capture(self.run,16,self.stream)
            for _ in range(2):graph.launch(self.stream)
            self.stream.synchronize()
            observations.append({"candidate":candidate,"ms":[measure(graph,16,self.stream) for _ in range(3)]})
        self.algorithm=min(observations,key=lambda row:np.median(row["ms"]))["candidate"]
        return observations

    def close(self):
        if self.alternative:
            self.library.rq_library_fp4_destroy(self.alternative)
            self.alternative=None
        super().close()


class Pipeline:
    def __init__(self,library,configuration,model,x,stream):
        self.library,self.stream,self.configuration=library,stream,configuration
        self.first,self.second=None,None
        self.x=cp.asarray(x)
        self.weights=[cp.asarray(model[name]) for name in ("w1","w2","b1","b2")]
        w1,w2,b1,b2=self.weights
        first,second,backend,fused=configuration
        rule=json.loads((ROOT/"results/encoder-selector.json").read_text(encoding="utf-8"))
        variant=lambda count:rule["at_or_below_variant"] if count<=rule["threshold"] else rule["above_variant"]
        try:
            self.first=BackendPlan(library,first,backend,variant(x.size),self.x,w1,stream)
            self.second=BackendPlan(library,second,backend,variant(x.shape[0]*w1.shape[0]),self.first.output,w2,
                                    stream,b1,fused and second=="fp4")
            self.labels=cp.empty(x.shape[0],cp.int32)
        except BaseException:
            self.close()
            raise

    def close(self):
        for plan in (self.second,self.first):
            if plan:plan.close()
        self.second=self.first=None

    def prepare_weights(self):
        self.first.prepare(2)
        self.second.prepare(2)

    def run(self):
        self.first.prepare(1)
        self.first.run()
        self.second.prepare(1)
        self.second.run()
        self.first.checked(self.library.rq_classify(self.second.output.data.ptr,self.weights[3].data.ptr,
                            self.labels.data.ptr,self.x.shape[0],128,10,self.stream.ptr))

    def set_input(self,x):
        if x.shape!=self.x.shape or x.dtype!=np.float32 or not x.flags.c_contiguous:
            raise ValueError("Request must be contiguous FP32 with the configured shape")
        cp.cuda.runtime.memcpyAsync(self.x.data.ptr,x.ctypes.data,x.nbytes,cp.cuda.runtime.memcpyHostToDevice,self.stream.ptr)

    def host_request(self,x):
        self.set_input(x)
        self.run()
        labels=np.empty(len(x),np.int32)
        cp.cuda.runtime.memcpyAsync(labels.ctypes.data,self.labels.data.ptr,labels.nbytes,
                                   cp.cuda.runtime.memcpyDeviceToHost,self.stream.ptr)
        self.stream.synchronize()
        return labels


def check_layer(plan,a,b,actual):
    quantized=[]
    hashes={}
    for operand,original in enumerate((a,b)):
        raw,scales,values=quantize(original,plan.mode)
        np.testing.assert_array_equal(plan.buffer(operand),raw.view(np.uint8).ravel())
        if plan.mode=="fp4":np.testing.assert_array_equal(plan.buffer(operand+2),scales.ravel())
        quantized.append(values)
        hashes[f"operand_{operand}_encoded"]=digest(raw)
        hashes[f"operand_{operand}_scales"]=digest(scales)
    ratio=check_output(actual,*quantized)
    return {"encoded_sha256":hashes,"arithmetic_max_bound_ratio":ratio,"output_sha256":digest(actual)}


def validate(pipeline,x,model):
    pipeline.set_input(x)
    pipeline.run()
    pipeline.stream.synchronize()
    first,second=cp.asnumpy(pipeline.first.output),cp.asnumpy(pipeline.second.output)
    labels=cp.asnumpy(pipeline.labels)
    first_check=check_layer(pipeline.first,x,model["w1"],first)
    active=np.maximum(first+model["b1"],np.float32(0))
    second_check=check_layer(pipeline.second,active,model["w2"],second)
    expected_labels=(second[:,:10]+model["b2"][:10]).argmax(axis=1).astype(np.int32)
    np.testing.assert_array_equal(labels,expected_labels)
    cpu_first=np.maximum(x.astype(np.float64)@model["w1"].astype(np.float64).T+model["b1"],0)
    cpu_scores=cpu_first@model["w2"][:10].astype(np.float64).T+model["b2"][:10]
    checks={"first":first_check,"second":second_check,"input_sha256":digest(x),
            "predictions_sha256":digest(labels),"scores_relative_error":relative_error(second[:,:10]+model["b2"][:10],cpu_scores)}
    return checks,{"input":x,"first":first,"second":second,"predictions":labels}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=20261116)
    parser.add_argument("--output",default="results/application-cohort-a.json")
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--reverse",action="store_true")
    args=parser.parse_args()
    cp.cuda.Device(0).use()
    library=load_application()
    x,y=load_digits()
    model_path=ROOT/f"fixtures/digits-model-{args.seed}.npz"
    with np.load(model_path,allow_pickle=False) as stored:model={name:stored[name] for name in stored.files}
    record={"schema_version":1,"utc":datetime.now(timezone.utc).isoformat(),"seed":args.seed,"quick":args.quick,
            "reverse":args.reverse,"environment":environment(0,library),
            "source_sha256":{name:file_hash(ROOT/name) for name in SOURCES},
            "data_sha256":{"fixtures/digits.csv.gz":file_hash(ROOT/"fixtures/digits.csv.gz"),
                            str(model_path.relative_to(ROOT)).replace('\\','/'):file_hash(model_path),
                            "results/encoder-selector.json":file_hash(ROOT/"results/encoder-selector.json")},
            "binary_sha256":file_hash(ROOT/"build/libapplication.so"),"samples":3 if args.quick else 30,
            "timing_scope":{"captured_device":"per inference; 32 captured repetitions; reusable prepared weights",
                            "uncaptured_device":"wall time including Python/native dispatch and stream completion",
                            "host_request":"wall time including changing FP32 input H2D, complete inference, host label allocation and D2H",
                            "cold_plan_cached_algorithm":"wall time including arrays/transfers, fresh native plans, weight preparation, inference, labels and native plan teardown; cached algorithm index; warm process/CuPy allocator"},
            "warmups":2 if args.quick else 5,"graph_repetitions":4 if args.quick else 32,
            "cases":[],"quality":[]}
    stream=cp.cuda.Stream(non_blocking=True)
    configurations=list(CONFIGURATIONS)
    batches=(16,) if args.quick else BATCHES
    if args.reverse:configurations.reverse()
    with stream:
        for batch in batches:
            for config in configurations:
                ids=model["test"][np.arange(batch)%len(model["test"])]
                current=np.ascontiguousarray(x[ids])
                started=time.perf_counter()
                pipeline=Pipeline(library,config,model,current,stream)
                stream.synchronize()
                allocation_ms=(time.perf_counter()-started)*1000
                try:
                    pipeline.prepare_weights()
                    pipeline.run()
                    stream.synchronize()
                    tuning=[pipeline.first.choose()]
                    pipeline.first.run()
                    pipeline.second.prepare(1)
                    tuning.append(pipeline.second.choose())
                    checks,witness=validate(pipeline,current,model)
                    entry={"batch":batch,"configuration":list(config),"allocation_plan_ms":allocation_ms,
                           "partition":"training" if batch in (16,64,256) else "held-out",
                           "tuning":tuning,"algorithm_candidates":[pipeline.first.algorithm,pipeline.second.algorithm],
                           "encoder_variants":[pipeline.first.variant,pipeline.second.variant],"checks":checks,
                           "timings_ms":{"captured_device":[],"uncaptured_device":[],"host_request":[],"cold_plan_cached_algorithm":[]},
                           "reuse_total_ms":{},"gpu_before":gpu_snapshot(0)}
                    if batch==16:
                        name=f"application-witness-{args.seed}-{'-'.join(str(v) for v in config)}.npz"
                        target=Path(args.output).parent/name
                        target.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(target,**witness)
                        entry["witness"]={"file":name,"sha256":file_hash(target)}
                    graph=capture(pipeline.run,record["graph_repetitions"],stream)
                    for _ in range(record["warmups"]):graph.launch(stream)
                    stream.synchronize()
                    for _ in range(record["samples"]):
                        entry["timings_ms"]["captured_device"].append(measure(graph,record["graph_repetitions"],stream))
                        started=time.perf_counter()
                        pipeline.run()
                        stream.synchronize()
                        entry["timings_ms"]["uncaptured_device"].append((time.perf_counter()-started)*1000)
                    requests=[]
                    entry["host_request_inputs"]=[]
                    for offset in (0,29,83,157):
                        request=np.ascontiguousarray(x[model["test"][(np.arange(batch)+offset)%len(model["test"])]] )
                        _,sample=validate(pipeline,request,model)
                        requests.append((request,sample["predictions"]))
                        entry["host_request_inputs"].append({"offset":offset,"input_sha256":digest(request),
                                                            "predictions_sha256":digest(sample["predictions"])})
                    for observation in range(record["samples"]):
                        request,expected=requests[observation%len(requests)]
                        started=time.perf_counter()
                        actual=pipeline.host_request(request)
                        entry["timings_ms"]["host_request"].append((time.perf_counter()-started)*1000)
                        np.testing.assert_array_equal(actual,expected)
                    pipeline.set_input(current)
                    for count in (1,8,64):
                        def repeated(repetitions=count):
                            pipeline.prepare_weights()
                            for _ in range(repetitions):pipeline.run()
                        reuse_graph=capture(repeated,1,stream)
                        for _ in range(record["warmups"]):reuse_graph.launch(stream)
                        stream.synchronize()
                        entry["reuse_total_ms"][str(count)]=[measure(reuse_graph,1,stream) for _ in range(record["samples"])]
                    entry["post_timing_checks"],_=validate(pipeline,current,model)
                    for _ in range(record["samples"]):
                        started=time.perf_counter()
                        fresh=Pipeline(library,config,model,current,stream)
                        try:
                            fresh.first.algorithm=pipeline.first.algorithm
                            fresh.second.algorithm=pipeline.second.algorithm
                            fresh.prepare_weights()
                            fresh.run()
                            actual=np.empty(batch,np.int32)
                            cp.cuda.runtime.memcpyAsync(actual.ctypes.data,fresh.labels.data.ptr,actual.nbytes,
                                cp.cuda.runtime.memcpyDeviceToHost,stream.ptr)
                            stream.synchronize()
                        finally:
                            fresh.close()
                        entry["timings_ms"]["cold_plan_cached_algorithm"].append((time.perf_counter()-started)*1000)
                        np.testing.assert_array_equal(actual,witness["predictions"])
                    entry["gpu_after"]=gpu_snapshot(0)
                    if batch==64 or args.quick:
                        quality={"configuration":list(config),"splits":{}}
                        for split in ("validation","test"):
                            labels=[]
                            split_ids=model[split]
                            for start in range(0,len(split_ids),batch):
                                chosen=split_ids[start:start+batch]
                                padded=np.resize(x[chosen],(batch,128)) if len(chosen)<batch else np.ascontiguousarray(x[chosen])
                                _,sample=validate(pipeline,np.ascontiguousarray(padded),model)
                                labels.extend(sample["predictions"][:len(chosen)].tolist())
                            predictions=np.array(labels,np.int32)
                            quality["splits"][split]={"count":len(labels),"correct":int(np.sum(predictions==y[split_ids])),
                                "accuracy":float(np.mean(predictions==y[split_ids])),"predictions":labels}
                        record["quality"].append(quality)
                    record["cases"].append(entry)
                    write_record(args.output,record)
                    print(f"batch={batch} {'/'.join(map(str,config))}: arithmetic, dynamic requests and timings passed",flush=True)
                finally:
                    stream.synchronize()
                    pipeline.close()
    record["complete"]=True
    record["environment_after"]=gpu_snapshot(0)
    write_record(args.output,record)


if __name__=="__main__":main()
