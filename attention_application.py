"""Complete transformer quality and latency with native or forced fused attention."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel

from attention_model import DigitAttention
from attention_native import NativeAttention,load
from attention_reference import attention,network
from attention_study import BACKENDS,CONFIGS,DTYPES,benchmark,configure,environment,sha,validate_native
from prepare_digits import load_digits
from reference import digest,relative_error
from sage_reference import load as load_sage,model_function as sage_model_function

ROOT=Path(__file__).resolve().parent
BATCHES=(4,16,32)
SOURCES=("attention_application.py","attention_model.py","attention_native.py","attention_reference.py","attention_validation.py","attention_study.py",
         "prepare_digits.py","reference.py","native/attention_bridge.cu","native/bridge.cu","native/optimized_bridge.cu","build_attention.sh","sage_reference.py")


class ModelNative:
    def __init__(self,libraries,batch,modes):
        self.buffers=tuple(torch.empty((batch,2,64,64),device="cuda",dtype=torch.float32) for _ in range(3));self.plan=NativeAttention(libraries,*self.buffers,modes,"none")

    def __call__(self,q,k,v):
        for destination,source in zip(self.buffers,(q,k,v)):destination.copy_(source)
        return self.plan()

    def close(self):self.plan.close()


def library_attention(dtype,backend):
    def forward(q,k,v):
        with sdpa_kernel(backend):return F.scaled_dot_product_attention(q.to(dtype),k.to(dtype),v.to(dtype),dropout_p=0.).float()
    return forward


def validate_model(model,device_x,host_x,weights,attend,indices):
    logits,trace=model(device_x,attend,trace=True);torch.cuda.synchronize();arrays={name:value.cpu().numpy().copy() for name,value in trace.items()};actual=logits.cpu().numpy().copy();qkv=[arrays[name] for name in ("q","k","v")]
    def injected(q,k,v):
        for expected,name in zip((q,k,v),("q","k","v")):np.testing.assert_allclose(arrays[name],expected,rtol=2e-4,atol=2e-4)
        return arrays["attended"].astype(np.float64)
    expected=network(host_x,weights,injected);np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-4)
    if isinstance(attend,ModelNative):
        attention_checks,buffers,_=validate_native(attend.plan,qkv,attention(*qkv),indices)
        arrays.update(scores=buffers[3],probabilities=buffers[4],padded_output=buffers[5])
    else:
        target=attention(*qkv);attention_checks={"relative_frobenius_error":relative_error(arrays["attended"],target)}
    predictions=actual.argmax(axis=1)
    return {"logits_sha256":digest(actual),"predictions_sha256":digest(predictions),"logit_arithmetic_relative_error":relative_error(actual,expected),"attention":attention_checks},dict(input=host_x,logits=actual,predictions=predictions,indices=indices,**arrays)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--model-seed",type=int,default=20261123);parser.add_argument("--output",required=True);parser.add_argument("--reverse",action="store_true");parser.add_argument("--quick",action="store_true");args=parser.parse_args()
    if args.model_seed<0:parser.error("Nonnegative model seed required")
    configure();libraries=load();sage_api,sage_hash=load_sage();x,y=load_digits();model_path=ROOT/f"fixtures/attention-model-{args.model_seed}.npz";reference_path=ROOT/f"fixtures/attention-model-reference-{args.model_seed}.npz"
    with np.load(model_path,allow_pickle=False) as data:weights={name:data[name] for name in data.files if name not in ("train","validation","test")};splits={name:data[name] for name in ("train","validation","test")}
    with np.load(reference_path,allow_pickle=False) as data:original={name:data[name] for name in data.files}
    model=DigitAttention().cuda().eval();model.load_state_dict({name:torch.from_numpy(value).cuda() for name,value in weights.items()})
    record={"schema_version":1,"utc":datetime.now(timezone.utc).isoformat(),"model_seed":args.model_seed,"quick":args.quick,"reverse":args.reverse,"environment":environment(libraries),
            "source_sha256":{name:sha(ROOT/name) for name in SOURCES},"data_sha256":{str(path.relative_to(ROOT)).replace("\\","/"):sha(path) for path in (model_path,reference_path,ROOT/"fixtures/digits.csv.gz")},
            "binary_sha256":sha(ROOT/"build/libattention.so"),"sage_build_sha256":sage_hash,"samples":3 if args.quick else 30,"warmups":2 if args.quick else 5,"cases":[],"quality":[],"unavailable":[],
            "timing_scope":"Complete single-block transformer, FP32 projections/norms/MLP/weights, only attention backend changes; FP32 input and logits. Device graphs, uncaptured resident host clock and changing-image H2D/label-D2H requests are separate. Reused plans and weights."}
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    def save():destination.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    configurations=[("native",a,b) for a,b in CONFIGS]+[(name,dtype,None) for name,_ in BACKENDS for dtype,_ in DTYPES]+[("sage3",dtype,None) for dtype in ("fp16","bf16")]
    if args.quick:configurations=[("native","fp4","fp4"),("native","fp16","fp16"),("math","fp32",None),("flash","fp16",None),("sage3","fp16",None),("sage3","bf16",None)]
    if args.reverse:configurations.reverse()
    with torch.inference_mode():
        for engine,first,second in configurations:
            label=f"native-{first}-{second}" if engine=="native" else f"{engine}-{first}"
            for batch in ((4,) if args.quick else BATCHES):
                module=None;started=time.perf_counter();ids=splits["test"][np.arange(batch)%len(splits["test"])];host=np.ascontiguousarray(x[ids]);device_x=torch.from_numpy(host).cuda()
                try:
                    if engine=="native":module=ModelNative(libraries,batch,(first,second));attend=module
                    elif engine=="sage3":attend=sage_model_function(sage_api,dict(DTYPES)[first])
                    else:attend=library_attention(dict(DTYPES)[first],dict(BACKENDS)[engine])
                    def forward():return model(device_x,attend)
                    forward();torch.cuda.synchronize();tuning=module.plan.tune() if module else None
                except (RuntimeError,NotImplementedError) as exc:
                    if module:module.close()
                    record["unavailable"].append({"label":label,"batch":batch,"diagnostic":str(exc)});save();continue
                try:
                    setup=(time.perf_counter()-started)*1000;rng=np.random.default_rng(np.random.SeedSequence([args.model_seed,batch,307]));indices=rng.integers(batch*2*64*64,size=64)
                    checks,witness=validate_model(model,device_x,host,weights,attend,indices);initial=witness["logits"]
                    entry={"label":label,"engine":engine,"batch":batch,"qk_mode":first if module else None,"pv_mode":second,"checks":checks,"setup_and_tuning_wall_ms":setup,
                           "tuning":tuning,"selected_algorithms":module.plan.algorithms.copy() if module else None,"weight_bytes":sum(value.nbytes for value in weights.values()),"graph_repetitions":4,"timings_ms":{}}
                    if module:entry["owned_native_device_buffer_bytes"]=libraries[0].att_owned_bytes(module.plan.pointer)
                    torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();memory_output=forward();torch.cuda.synchronize();entry["extra_peak_torch_allocated_bytes"]=torch.cuda.max_memory_allocated()-base;del memory_output
                    entry["timings_ms"]["captured_device"],timed=benchmark(forward,4,record["samples"],record["warmups"]);np.testing.assert_array_equal(timed,initial)
                    entry["post_timing_logits_sha256"]=digest(timed)
                    for _ in range(record["warmups"]):forward()
                    torch.cuda.synchronize();values=[]
                    for _ in range(record["samples"]):
                        start=time.perf_counter();output=forward();torch.cuda.synchronize();values.append((time.perf_counter()-start)*1000)
                    entry["timings_ms"]["uncaptured_resident"]=values
                    requests=[]
                    for offset in (0,29,83,157):
                        chosen=splits["test"][(np.arange(batch)+offset)%len(splits["test"])];request=np.ascontiguousarray(x[chosen]);device_x.copy_(torch.from_numpy(request));predictions=forward().argmax(dim=1).cpu().numpy().copy();requests.append((request,predictions))
                    values=[]
                    for observation in range(record["samples"]):
                        request,expected=requests[observation%len(requests)];start=time.perf_counter();device_x.copy_(torch.from_numpy(request));predictions=forward().argmax(dim=1).cpu().numpy();torch.cuda.synchronize();values.append((time.perf_counter()-start)*1000);np.testing.assert_array_equal(predictions,expected)
                    entry["timings_ms"]["host_request"]=values;entry["request_checks"]=[{"input_sha256":digest(a),"predictions_sha256":digest(b)} for a,b in requests]
                    device_x.copy_(torch.from_numpy(host));entry["post_request_checks"],_=validate_model(model,device_x,host,weights,attend,indices)
                    if batch==4:
                        path=destination.parent/f"attention-model-witness-{args.model_seed}-{label}.npz";np.savez_compressed(path,**witness);entry["witness"]={"file":path.name,"sha256":sha(path)}
                    if batch==16 or args.quick:
                        quality={"label":label,"batch":batch,"splits":{}};quality_arrays={}
                        for split in ("validation","test"):
                            selected=splits[split][:32] if args.quick else splits[split];all_logits=[]
                            for start in range(0,len(selected),batch):
                                current=selected[start:start+batch];request=np.ascontiguousarray(x[np.resize(current,batch)]);device_x.copy_(torch.from_numpy(request));all_logits.append(forward().cpu().numpy()[:len(current)].copy())
                            logits=np.concatenate(all_logits);predictions=logits.argmax(axis=1);quality_arrays[f"{split}_logits"]=logits;quality_arrays[f"{split}_predictions"]=predictions
                            quality["splits"][split]={"count":len(selected),"correct":int(np.sum(predictions==y[selected])),"logits_sha256":digest(logits),"predictions_sha256":digest(predictions),
                                "logit_relative_error_vs_original":relative_error(logits,original[f"{split}_logits"][:len(selected)]),"changed_predictions_vs_original":int(np.sum(predictions!=original[f"{split}_predictions"][:len(selected)]))}
                        path=destination.parent/f"attention-model-quality-{args.model_seed}-{label}.npz";np.savez_compressed(path,**quality_arrays);quality["witness"]={"file":path.name,"sha256":sha(path)};record["quality"].append(quality)
                    record["cases"].append(entry);save();print(f"model {args.model_seed} {label} batch {batch}: complete transformer validated",flush=True)
                finally:
                    torch.cuda.synchronize()
                    if module:module.close()
    record["complete"]=True;save()


if __name__=="__main__":main()
