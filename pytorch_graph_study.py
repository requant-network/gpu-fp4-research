"""Practical dynamic-shape/reduce-overhead CUDA-Graph reference for host latency."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch._inductor.cudagraph_trees import get_container

from prepare_digits import load_digits
from pytorch_study import BATCHES,MODES,ROOT,network,sha,validated


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=20261116)
    parser.add_argument("--output",default="results/pytorch-graphs-cohort-a.json")
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--reverse",action="store_true")
    args=parser.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch._dynamo.config.recompile_limit=64
    torch._inductor.config.triton.cudagraph_trees=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.set_float32_matmul_precision("highest")
    x,y=load_digits()
    model_path=ROOT/f"fixtures/digits-model-{args.seed}.npz"
    with np.load(model_path,allow_pickle=False) as arrays:model={k:arrays[k] for k in arrays.files}
    record={"schema_version":1,"seed":args.seed,"quick":args.quick,"reverse":args.reverse,
            "utc":datetime.now(timezone.utc).isoformat(),"torch":torch.__version__,"torch_cuda":torch.version.cuda,
            "gpu_name":torch.cuda.get_device_name(0),"gpu_exclusive":False,
            "compiler_mode":"reduce-overhead; fullgraph=True; dynamic=True; internal CUDA Graphs enabled; weights marked static",
            "source_sha256":{name:sha(ROOT/name) for name in ("pytorch_graph_study.py","pytorch_study.py","prepare_digits.py","reference.py")},
            "data_sha256":{"fixtures/digits.csv.gz":sha(ROOT/"fixtures/digits.csv.gz"),f"fixtures/digits-model-{args.seed}.npz":sha(model_path)},
            "warmups":2 if args.quick else 5,"samples":3 if args.quick else 30,"cases":[],"quality":[],
            "timing_scope":"wall-time uncaptured calls and changing-input H2D/label-D2H requests; prepared weights; internal graph management and step markers included"}
    destination=Path(args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    def save():destination.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    stream=torch.cuda.Stream()
    modes=list(MODES)
    if args.reverse:modes.reverse()
    with torch.inference_mode(),torch.cuda.stream(stream):
        for mode in modes:
            function,dtype=network(mode)
            parameters=(torch.from_numpy(model["w1"]).cuda().to(dtype),torch.from_numpy(model["w2"]).cuda().to(dtype),
                        torch.from_numpy(model["b1"]).cuda(),torch.from_numpy(model["b2"]).cuda(),torch.ones((),device="cuda"))
            for parameter in parameters:torch._dynamo.mark_static_address(parameter)
            compiled=torch.compile(function,mode="reduce-overhead",dynamic=True,fullgraph=True)
            def run(*values):
                torch.compiler.cudagraph_mark_step_begin()
                return compiled(*values)
            for batch in ((16,) if args.quick else BATCHES):
                current=np.ascontiguousarray(x[model["test"][np.arange(batch)%len(model["test"])]] )
                device_x=torch.from_numpy(current).cuda()
                started=time.perf_counter()
                result=run(device_x,*parameters)
                stream.synchronize()
                first_call=(time.perf_counter()-started)*1000
                del result
                for _ in range(record["warmups"]):run(device_x,*parameters)
                stream.synchronize()
                checks,witness=validated(run,mode,device_x,current,parameters,model)
                manager=get_container(0).tree_manager
                roots=[] if manager is None else list(manager.get_roots())
                live_graphs=sum(isinstance(root.graph,torch.cuda.CUDAGraph) for root in roots)
                if not live_graphs:
                    raise RuntimeError("Requested internal CUDA Graph execution was not actually established")
                entry={"batch":batch,"mode":mode,"engine":"compiled_dynamic_graphs",
                       "partition":"training" if batch in (16,64,256) else "held-out",
                       "compile_and_first_call_ms":first_call,"checks":checks,
                       "internal_cuda_graph_roots":live_graphs,
                       "timings_ms":{"uncaptured_device":[],"host_request":[]}}
                if batch==16:
                    target=destination.parent/f"pytorch-graphs-witness-{args.seed}-{mode}.npz"
                    np.savez_compressed(target,**witness)
                    entry["witness"]={"file":target.name,"sha256":sha(target)}
                for _ in range(record["samples"]):
                    started=time.perf_counter()
                    result=run(device_x,*parameters)
                    stream.synchronize()
                    entry["timings_ms"]["uncaptured_device"].append((time.perf_counter()-started)*1000)
                    del result
                requests=[]
                for offset in (0,29,83,157):
                    request=np.ascontiguousarray(x[model["test"][(np.arange(batch)+offset)%len(model["test"])]] )
                    _,sample=validated(run,mode,device_x,request,parameters,model)
                    requests.append((request,sample["predictions"]))
                for observation in range(record["samples"]):
                    request,expected=requests[observation%len(requests)]
                    started=time.perf_counter()
                    device_x.copy_(torch.from_numpy(request))
                    actual=run(device_x,*parameters).cpu().numpy()
                    stream.synchronize()
                    entry["timings_ms"]["host_request"].append((time.perf_counter()-started)*1000)
                    np.testing.assert_array_equal(actual,expected)
                entry["post_timing_checks"],_=validated(run,mode,device_x,current,parameters,model)
                if batch==64 or args.quick:
                    quality={"mode":mode,"engine":"compiled_dynamic_graphs","splits":{}}
                    for split in ("validation","test"):
                        ids=model[split]
                        labels=[]
                        for start in range(0,len(ids),batch):
                            chosen=ids[start:start+batch]
                            request=np.ascontiguousarray(np.resize(x[chosen],(batch,128)))
                            _,sample=validated(run,mode,device_x,request,parameters,model)
                            labels.extend(sample["predictions"][:len(chosen)].tolist())
                        predicted=np.asarray(labels,np.int64)
                        quality["splits"][split]={"count":len(labels),"correct":int(np.sum(predicted==y[ids])),
                            "accuracy":float(np.mean(predicted==y[ids])),"predictions":labels}
                    record["quality"].append(quality)
                record["cases"].append(entry)
                save()
                print(f"{mode}/dynamic-graphs batch={batch}: arithmetic, labels and host timings passed",flush=True)
    record["complete"]=True
    save()


if __name__=="__main__":main()
