"""Probe forced native SDPA backends before designing mixed-precision attention."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel

from reference import digest,inputs,relative_error

ROOT=Path(__file__).resolve().parent
SHAPES=((1,1,128,64),(2,4,512,64),(1,4,1024,128),(1,2,256,32))
MASKS=("none","causal","padding")


def arrays(seed,shape):
    batch,heads,length,depth=shape;rows=batch*heads*length
    q,k=inputs(seed,(rows,rows,depth),"normal");v,_=inputs(seed+1,(rows,rows,depth),"normal")
    return tuple(x.reshape(shape) for x in (q,k,v))


def mask(length,kind):
    if kind=="none":return np.ones((length,length),bool)
    if kind=="causal":return np.tril(np.ones((length,length),bool))
    if kind!="padding":raise ValueError("Unsupported attention mask")
    allowed=np.ones((length,length),bool);allowed[:,-length//4:]=False;allowed[0]=False
    return allowed


def reference(q,k,v,allowed):
    q,k,v=(x.astype(np.float64) for x in (q,k,v));scores=(q@k.swapaxes(-1,-2))/np.sqrt(q.shape[-1])
    scores=np.where(allowed,scores,-np.inf);maximum=np.max(scores,axis=-1,keepdims=True)
    maximum=np.where(np.isfinite(maximum),maximum,0.);exponent=np.exp(scores-maximum)
    total=np.sum(exponent,axis=-1,keepdims=True);probabilities=exponent/np.where(total>0,total,1.)
    return probabilities@v


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",default="results/attention-preflight.json");args=parser.parse_args()
    torch.cuda.set_device(0);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.set_float32_matmul_precision("highest")
    props=torch.cuda.get_device_properties(0);destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    record={"schema_version":1,"study":5,"scope":"Native-backend availability and numerical preflight only; not a complete mixed-precision attention or model-quality study.",
            "utc":datetime.now(timezone.utc).isoformat(),"seed":20261123,"torch":torch.__version__,"torch_cuda":torch.version.cuda,"torch_git":torch.version.git_version,
            "numpy":np.__version__,"gpu_name":props.name,"compute_capability":[props.major,props.minor],"gpu_exclusive":False,"warmups":2,"samples":5,
            "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("attention_preflight.py","reference.py")},"cases":[]}
    def save():destination.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    backends=(("math",SDPBackend.MATH),("flash",SDPBackend.FLASH_ATTENTION),("efficient",SDPBackend.EFFICIENT_ATTENTION),("cudnn",SDPBackend.CUDNN_ATTENTION))
    with torch.inference_mode():
        for shape in SHAPES:
            host=arrays(record["seed"],shape)
            for kind in MASKS:
                allowed=mask(shape[2],kind);target=reference(*host,allowed);device_mask=torch.from_numpy(allowed).cuda() if kind=="padding" else None
                for name,dtype in (("fp32",torch.float32),("fp16",torch.float16),("bf16",torch.bfloat16)):
                    operands=tuple(torch.from_numpy(x).cuda().to(dtype) for x in host)
                    rounded=tuple(x.float().cpu().numpy() for x in operands);arithmetic=reference(*rounded,allowed)
                    for label,backend in backends:
                        entry={"shape":list(shape),"mask":kind,"dtype":name,"requested_backend":label,"input_sha256":[digest(x) for x in host]}
                        try:
                            with warnings.catch_warnings(record=True) as captured,sdpa_kernel(backend):
                                def run():return F.scaled_dot_product_attention(*operands,attn_mask=device_mask,is_causal=kind=="causal",dropout_p=0.)
                                output=run();torch.cuda.synchronize();actual=output.float().cpu().numpy();del output
                                if not np.isfinite(actual).all():raise ValueError("Nonfinite attention output")
                                algorithm_error=relative_error(actual,arithmetic)
                                if algorithm_error>({"fp32":5e-5,"fp16":.005,"bf16":.03}[name]):raise ValueError("Attention arithmetic preflight failed")
                                if kind=="padding" and np.any(actual[:,:,0]!=0):raise ValueError("Fully masked attention row must return zeros")
                                for _ in range(record["warmups"]):output=run();del output
                                torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();values=[]
                                for _ in range(record["samples"]):
                                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();output=run();end.record();end.synchronize();values.append(start.elapsed_time(end));del output
                                peak=torch.cuda.max_memory_allocated()-base
                                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:output=run()
                                torch.cuda.synchronize();del output
                                entry.update({"status":"available","warnings":[str(w.message) for w in captured],"relative_error_vs_original_fp64":relative_error(actual,target),
                                    "relative_error_vs_rounded_input_fp64":algorithm_error,"output_sha256":digest(actual),"timings_ms":values,
                                    "extra_peak_torch_allocated_bytes":peak,"dispatch_operations":sorted({event.key for event in profile.key_averages() if "attention" in event.key})})
                                if shape==(1,1,128,64):
                                    path=destination.parent/f"attention-preflight-{name}-{label}-{kind}.npz";np.savez_compressed(path,q=host[0],k=host[1],v=host[2],allowed=allowed,output=actual)
                                    entry["witness"]={"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
                        except (RuntimeError,NotImplementedError) as exc:
                            entry.update({"status":"unavailable","diagnostic":str(exc)})
                        record["cases"].append(entry);save()
                print(f"{shape} {kind}: forced native backends tested",flush=True)
    if not any(c["status"]=="available" and c["requested_backend"]=="flash" for c in record["cases"]):raise ValueError("No usable FlashAttention baseline")
    record["complete"]=True;save()


if __name__=="__main__":main()
