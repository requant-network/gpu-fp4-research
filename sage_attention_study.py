"""Full-cost specialized FP4 attention reference on the frozen main input grid."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from attention_reference import MASKS,arrays,attention
from attention_native import load as native_load
from attention_study import SHAPES,HELD_OUT,benchmark,configure,environment,host_requests,sample_checks,sha
from reference import digest
from sage_reference import function,load

ROOT=Path(__file__).resolve().parent
SOURCES=("sage_attention_study.py","sage_reference.py","attention_study.py","attention_native.py","attention_reference.py","attention_validation.py","reference.py")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--seed",type=int,default=20261123);parser.add_argument("--output",required=True);parser.add_argument("--reverse",action="store_true");parser.add_argument("--quick",action="store_true");args=parser.parse_args()
    if args.seed<0:parser.error("Nonnegative seed required")
    configure();api,build_hash=load();destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    record={"schema_version":1,"utc":datetime.now(timezone.utc).isoformat(),"seed":args.seed,"reverse":args.reverse,"quick":args.quick,"environment":environment(native_load()),
            "source_sha256":{name:sha(ROOT/name) for name in SOURCES},"sage_build_sha256":build_hash,"samples":3 if args.quick else 30,"warmups":2 if args.quick else 5,
            "cases":[],"unavailable":[],"scope":"Unmodified pinned fused FP4 reference; smoothing, quantization, preparation and FP32 output promotion included. Prepared requests clone K because the upstream implementation centers it in place. No arbitrary-mask or unsupported-dimension fallback is accepted."}
    def save():destination.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    groups=[(shape,kind,profile) for shape in ((SHAPES[0],) if args.quick else SHAPES) for kind in MASKS for profile in ("normal","peaky")]
    if args.reverse:groups.reverse()
    with torch.inference_mode():
        for shape,kind,profile in groups:
            host=arrays(args.seed,shape)
            if profile=="peaky":host=host[0]*np.float32(4),host[1]*np.float32(4),host[2]
            operands=tuple(torch.from_numpy(x).cuda() for x in host);target=attention(*host,kind)
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,*shape,271]));indices=rng.integers(target.size,size=64)
            for name,dtype in (("fp16",torch.float16),("bf16",torch.bfloat16)):
                label=f"sage3-{name}";started=time.perf_counter()
                try:
                    online=function(api,operands,dtype,kind);prepared=function(api,operands,dtype,kind,True);actual=online().cpu().numpy().copy();torch.cuda.synchronize()
                except (RuntimeError,NotImplementedError) as exc:
                    record["unavailable"].append({"shape":list(shape),"mask":kind,"profile":profile,"label":label,"diagnostic":str(exc)});save();continue
                if actual.shape!=shape or not np.isfinite(actual).all():raise ValueError("Invalid specialized FP4 output")
                checks=sample_checks(actual,target,indices);repetitions=1 if shape[0]*shape[1]*shape[2]>=4096 else 4
                entry={"shape":list(shape),"mask":kind,"profile":profile,"label":label,"engine":"sage3","partition":"held-out" if shape in HELD_OUT else "training",
                       "input_sha256":[digest(x) for x in host],"checks":checks,"setup_and_first_call_wall_ms":(time.perf_counter()-started)*1000,"graph_repetitions":repetitions,"timings_ms":{}}
                torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();memory_output=online();torch.cuda.synchronize();entry["extra_peak_torch_allocated_bytes"]=torch.cuda.max_memory_allocated()-base;del memory_output
                entry["timings_ms"]["prepared"],first=benchmark(prepared,repetitions,record["samples"],record["warmups"])
                entry["timings_ms"]["online"],second=benchmark(online,repetitions,record["samples"],record["warmups"])
                entry["host_request_ms"],last=host_requests(online,operands,host,record["samples"],record["warmups"])
                for output in (first,second,last):np.testing.assert_array_equal(output,actual)
                for original,device in zip(host,operands):np.testing.assert_array_equal(original,device.cpu().numpy())
                entry["post_timing_output_sha256"],entry["post_request_output_sha256"]=digest(second),digest(last)
                if shape==SHAPES[0]:
                    path=destination.parent/f"sage-attention-witness-{args.seed}-{profile}-{kind}-{name}.npz";np.savez_compressed(path,q=host[0],k=host[1],v=host[2],output=actual,indices=indices);entry["witness"]={"file":path.name,"sha256":sha(path)}
                record["cases"].append(entry);save()
            print(f"{shape} {kind}/{profile}: pinned specialized FP4 reference attempted",flush=True)
    record["complete"]=True;save()


if __name__=="__main__":main()
