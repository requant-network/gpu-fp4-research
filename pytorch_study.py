"""Eager and fully compiled PyTorch classifiers with matched FP32 accumulations."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import triton

from prepare_digits import load_digits
from reference import check_output,digest,quantize

ROOT=Path(__file__).resolve().parent
MODES=("fp32","fp16","bf16","fp8")
BATCHES=(16,32,64,128,192,256)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def multiplication(a,b,mode,one):
    if mode=="fp8":
        return torch._scaled_mm(a,b.T,scale_a=one,scale_b=one,out_dtype=torch.float32)
    if mode in ("fp16","bf16"):
        return torch.mm(a,b.T,out_dtype=torch.float32)
    return torch.mm(a,b.T)


def network(mode):
    dtype={"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16,"fp8":torch.float8_e4m3fn}[mode]
    def forward(x,w1,w2,b1,b2,one):
        first=multiplication(x.to(dtype),w1,mode,one)
        hidden=torch.relu(first+b1)
        second=multiplication(hidden.to(dtype),w2,mode,one)
        return torch.argmax(second[:,:10]+b2[:10],dim=1)
    return forward,dtype


def validated(function,mode,device_x,request,parameters,model):
    device_x.copy_(torch.from_numpy(request))
    actual=function(device_x,*parameters).cpu().numpy()
    w1,w2,b1,b2,one=parameters
    dtype=w1.dtype
    first=multiplication(device_x.to(dtype),w1,mode,one)
    hidden=torch.relu(first+b1)
    second=multiplication(hidden.to(dtype),w2,mode,one)
    first_cpu,second_cpu=first.cpu().numpy(),second.cpu().numpy()
    quantized=[]
    for values,weights in ((request,model["w1"]),(np.maximum(first_cpu+model["b1"],np.float32(0)),model["w2"])):
        quantized.append((quantize(values,mode)[2],quantize(weights,mode)[2]))
    bounds=[check_output(first_cpu,*quantized[0]),check_output(second_cpu,*quantized[1])]
    expected=(second_cpu[:,:10]+model["b2"][:10]).argmax(axis=1)
    np.testing.assert_array_equal(actual,expected)
    return {"arithmetic_max_bound_ratio":bounds,"input_sha256":digest(request),
            "predictions_sha256":digest(actual)}, {"input":request,"first":first_cpu,"second":second_cpu,"predictions":actual}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=20261116)
    parser.add_argument("--output",default="results/pytorch-cohort-a.json")
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--reverse",action="store_true")
    args=parser.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch._dynamo.config.recompile_limit=64
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.set_float32_matmul_precision("highest")
    x,y=load_digits()
    model_path=ROOT/f"fixtures/digits-model-{args.seed}.npz"
    with np.load(model_path,allow_pickle=False) as arrays:model={k:arrays[k] for k in arrays.files}
    props=torch.cuda.get_device_properties(0)
    record={"schema_version":1,"seed":args.seed,"quick":args.quick,"reverse":args.reverse,
            "utc":datetime.now(timezone.utc).isoformat(),"torch":torch.__version__,"torch_git":torch.version.git_version,
            "torch_cuda":torch.version.cuda,"triton":triton.__version__,"numpy":np.__version__,
            "gpu_name":props.name,"compute_capability":[props.major,props.minor],"gpu_exclusive":False,
            "sm_count":props.multi_processor_count,
            "compiler_notes":"On this 36-SM GPU the framework reports insufficient SMs for GEMM max-autotuning; the complete graph still compiles and uses its available GEMM backend. TF32 is disabled for the strict FP32 comparison.",
            "compiler_mode":"max-autotune-no-cudagraphs; fullgraph=True; dynamic=False; one compile thread",
            "recompile_limit":64,
            "setup_adjustments":["The initial complete-grid attempt exceeded the default eight-specialization limit; the bounded 24 shape/precision variants use an explicit limit of 64. Partial first-attempt data is excluded."],
            "source_sha256":{name:sha(ROOT/name) for name in ("pytorch_study.py","prepare_digits.py","reference.py")},
            "data_sha256":{"fixtures/digits.csv.gz":sha(ROOT/"fixtures/digits.csv.gz"),
                           f"fixtures/digits-model-{args.seed}.npz":sha(model_path)},
            "warmups":2 if args.quick else 5,"samples":3 if args.quick else 30,
            "graph_repetitions":4 if args.quick else 32,"cases":[],"quality":[],
            "timing_scope":"captured GPU repetitions, uncaptured dispatch-to-completion, changing input H2D and label D2H host requests; prepared weights; FP32 GEMM outputs"}
    destination=Path(args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    def save():destination.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    stream=torch.cuda.Stream()
    batches=(16,) if args.quick else BATCHES
    modes=list(MODES)
    if args.reverse:modes.reverse()
    with torch.inference_mode(),torch.cuda.stream(stream):
        for mode in modes:
            eager,dtype=network(mode)
            compiled=torch.compile(eager,mode="max-autotune-no-cudagraphs",fullgraph=True,dynamic=False)
            parameters=(torch.from_numpy(model["w1"]).cuda().to(dtype),torch.from_numpy(model["w2"]).cuda().to(dtype),
                        torch.from_numpy(model["b1"]).cuda(),torch.from_numpy(model["b2"]).cuda(),torch.ones((),device="cuda"))
            for batch in batches:
                current=np.ascontiguousarray(x[model["test"][np.arange(batch)%len(model["test"])]] )
                device_x=torch.from_numpy(current).cuda()
                for engine,function in (("eager",eager),("compiled",compiled)):
                    started=time.perf_counter()
                    function(device_x,*parameters)
                    stream.synchronize()
                    first_call_ms=(time.perf_counter()-started)*1000
                    checks,witness=validated(function,mode,device_x,current,parameters,model)
                    entry={"batch":batch,"mode":mode,"engine":engine,
                           "partition":"training" if batch in (16,64,256) else "held-out",
                           "compile_and_first_call_ms":first_call_ms,"checks":checks,
                           "timings_ms":{"captured_device":[],"uncaptured_device":[],"host_request":[]}}
                    if batch==16:
                        target=destination.parent/f"pytorch-witness-{args.seed}-{mode}-{engine}.npz"
                        np.savez_compressed(target,**witness)
                        entry["witness"]={"file":target.name,"sha256":sha(target)}
                    for _ in range(record["warmups"]):function(device_x,*parameters)
                    stream.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):
                        for _ in range(record["graph_repetitions"]):output=function(device_x,*parameters)
                    for _ in range(record["warmups"]):graph.replay()
                    stream.synchronize()
                    for _ in range(record["samples"]):
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record(stream)
                        graph.replay()
                        end.record(stream)
                        end.synchronize()
                        entry["timings_ms"]["captured_device"].append(begin.elapsed_time(end)/record["graph_repetitions"])
                        started=time.perf_counter()
                        output=function(device_x,*parameters)
                        stream.synchronize()
                        entry["timings_ms"]["uncaptured_device"].append((time.perf_counter()-started)*1000)
                    requests=[]
                    for offset in (0,29,83,157):
                        request=np.ascontiguousarray(x[model["test"][(np.arange(batch)+offset)%len(model["test"])]] )
                        _,sample=validated(function,mode,device_x,request,parameters,model)
                        requests.append((request,sample["predictions"]))
                    for observation in range(record["samples"]):
                        request,expected=requests[observation%len(requests)]
                        started=time.perf_counter()
                        device_x.copy_(torch.from_numpy(request))
                        actual=function(device_x,*parameters).cpu().numpy()
                        stream.synchronize()
                        entry["timings_ms"]["host_request"].append((time.perf_counter()-started)*1000)
                        np.testing.assert_array_equal(actual,expected)
                    entry["post_timing_checks"],_=validated(function,mode,device_x,current,parameters,model)
                    if batch==64 or args.quick:
                        quality={"mode":mode,"engine":engine,"splits":{}}
                        for split in ("validation","test"):
                            ids=model[split]
                            labels=[]
                            for start in range(0,len(ids),batch):
                                chosen=ids[start:start+batch]
                                request=np.ascontiguousarray(np.resize(x[chosen],(batch,128)))
                                _,sample=validated(function,mode,device_x,request,parameters,model)
                                labels.extend(sample["predictions"][:len(chosen)].tolist())
                            predictions=np.array(labels,np.int64)
                            quality["splits"][split]={"count":len(labels),"correct":int(np.sum(predictions==y[ids])),
                                "accuracy":float(np.mean(predictions==y[ids])),"predictions":labels}
                        record["quality"].append(quality)
                    record["cases"].append(entry)
                    save()
                    print(f"{mode}/{engine} batch={batch}: matched arithmetic, dynamic requests and timings",flush=True)
    record["complete"]=True
    save()


if __name__=="__main__":main()
