"""Mixed QK/PV native attention versus forced fused library backends on SM120."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel

from attention_native import NativeAttention,load
from attention_reference import MASKS,allowed,arrays,attention,encoding,padded
from attention_validation import validate_arrays
from reference import digest,relative_error

ROOT=Path(__file__).resolve().parent
SHAPES=((1,1,128,64),(2,4,512,64),(1,4,1024,128),(1,2,256,32),(1,2,193,48),(1,1,768,96))
HELD_OUT=SHAPES[-2:]
CONFIGS=tuple((a,b) for a in ("fp4","fp8","fp16") for b in ("fp4","fp8","fp16"))+(("bf16","bf16"),("fp32","fp32"))
SOURCES=("attention_study.py","attention_reference.py","attention_native.py","attention_validation.py","reference.py",
         "native/attention_bridge.cu","native/bridge.cu","native/optimized_bridge.cu","build_attention.sh")
BACKENDS=(("math",SDPBackend.MATH),("flash",SDPBackend.FLASH_ATTENTION),("efficient",SDPBackend.EFFICIENT_ATTENTION),("cudnn",SDPBackend.CUDNN_ATTENTION))
DTYPES=(("fp32",torch.float32),("fp16",torch.float16),("bf16",torch.bfloat16))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,record):path.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def configure():
    torch.cuda.set_device(0);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False


def environment(libraries):
    props=torch.cuda.get_device_properties(0)
    return {"torch":torch.__version__,"torch_cuda":torch.version.cuda,"torch_git":torch.version.git_version,"numpy":np.__version__,
            "gpu_name":props.name,"compute_capability":[props.major,props.minor],"sm_count":props.multi_processor_count,"gpu_exclusive":False,
            "cublaslt_version":libraries[0].rq_cublas_version(),"nvcc":subprocess.check_output(["/usr/local/cuda-13.3/bin/nvcc","--version"],text=True),
            "cutlass_commit":"b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd","tf32":False,"fp16_reduced_reduction":False,"bf16_reduced_reduction":False}


def sample_checks(output,target,indices):
    values=output.ravel()[indices];expected=target.ravel()[indices]
    return {"relative_frobenius_error":relative_error(output,target),"sample_values":values.tolist(),"sample_relative_error":relative_error(values,expected),"output_sha256":digest(output)}


def validate_native(plan,host,target,indices,previous=None):
    torch.cuda.synchronize();q,k,v=host;batch,heads,length,depth=q.shape;sp=(length+127)//128*128;dp=(depth+127)//128*128
    shapes=((batch*heads,sp,dp),(batch*heads,sp,dp),(batch*heads,dp,sp),(batch*heads,sp,sp),(batch*heads,sp,sp),(batch*heads,sp,dp))
    buffers=[plan.read(index,shape) for index,shape in enumerate(shapes)];actual=plan.output.cpu().numpy().copy()
    byte_arrays={f"stage{stage}-buffer{index}":plan.encoded(stage,index) for stage in (0,1) for index in range(4)}
    fingerprint={**{f"buffer{i}":digest(x) for i,x in enumerate(buffers)},**{name:digest(x) for name,x in byte_arrays.items()},"output":digest(actual)}
    if previous is not None and fingerprint==previous["snapshot_sha256"]:return {**previous,"validation_reused_unchanged_bytes":True},buffers,actual
    qp,kp,vt=padded(q,k,v)
    for x,y in zip(buffers[:3],(qp,kp,vt)):np.testing.assert_array_equal(x,y)
    for stage,(a,b) in enumerate(((qp,kp),(buffers[4],vt))):
        for operand,x in enumerate((a,b)):
            raw,scales,_=encoding(x,plan.modes[stage]);np.testing.assert_array_equal(byte_arrays[f"stage{stage}-buffer{operand}"],raw.view(np.uint8).ravel())
            np.testing.assert_array_equal(byte_arrays[f"stage{stage}-buffer{operand+2}"],scales.ravel())
    checks=validate_arrays(q,k,v,plan.modes,plan.mask,*buffers[3:],actual,reference_output=target)
    checks.update(sample_checks(actual,target,indices));checks["snapshot_sha256"]=fingerprint;checks["validation_reused_unchanged_bytes"]=False
    return checks,buffers,actual


def framework_function(operands,dtype,backend,kind,device_mask,prepared=False):
    cached=tuple(x.to(dtype) for x in operands) if prepared else None
    def forward():
        with sdpa_kernel(backend):return F.scaled_dot_product_attention(*(cached if prepared else (x.to(dtype) for x in operands)),attn_mask=device_mask,is_causal=kind=="causal",dropout_p=0.).float()
    return forward


def benchmark(function,repetitions,samples,warmups):
    for _ in range(warmups):function()
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repetitions):captured_output=function()
    for _ in range(warmups):graph.replay()
    torch.cuda.synchronize();values=[]
    for _ in range(samples):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();graph.replay();end.record();end.synchronize();values.append(start.elapsed_time(end)/repetitions)
    torch.cuda.synchronize();actual=captured_output.cpu().numpy().copy();del graph
    return values,actual


def host_requests(function,operands,host,samples,warmups):
    def request():
        for x,y in zip(operands,host):x.copy_(torch.from_numpy(y))
        output=function().cpu().numpy().copy();torch.cuda.synchronize();return output
    for _ in range(warmups):request()
    values=[]
    for _ in range(samples):start=time.perf_counter();actual=request();values.append((time.perf_counter()-start)*1000)
    return values,actual


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--seed",type=int,default=20261123);parser.add_argument("--output",required=True)
    parser.add_argument("--quick",action="store_true");parser.add_argument("--reverse",action="store_true");args=parser.parse_args()
    if args.seed<0:parser.error("Nonnegative seed required")
    configure();libraries=load();destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    record={"schema_version":1,"utc":datetime.now(timezone.utc).isoformat(),"seed":args.seed,"quick":args.quick,"reverse":args.reverse,"environment":environment(libraries),
            "source_sha256":{name:sha(ROOT/name) for name in SOURCES},"binary_sha256":sha(ROOT/"build/libattention.so"),"samples":3 if args.quick else 30,"warmups":2 if args.quick else 5,
            "negative_controls":0,"cases":[],"unavailable":[],"timing_scope":"Prepared Q/K/V versus fresh native padding/encoding or framework casts; dynamic softmax and PV always included. Host requests include all Q/K/V H2D and complete FP32 output D2H. Plans/masks reused; identical inputs per timing case."}
    shapes=((1,1,128,64),(1,2,193,48)) if args.quick else SHAPES
    groups=[(shape,kind,profile) for shape in shapes for kind in MASKS for profile in ("normal","peaky")]
    if args.reverse:groups.reverse()
    with torch.inference_mode():
        for shape,kind,profile in groups:
            host=arrays(args.seed,shape)
            if profile=="peaky":host=(host[0]*np.float32(4),host[1]*np.float32(4),host[2])
            operands=tuple(torch.from_numpy(x).cuda() for x in host);target=attention(*host,kind)
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,*shape,271]));indices=rng.integers(target.size,size=64)
            device_mask=torch.from_numpy(allowed(shape[2],kind)).cuda() if kind=="padding" else None
            repetitions=1 if shape[0]*shape[1]*shape[2]>=4096 else 4
            configurations=[("native",a,b) for a,b in CONFIGS]+[(backend,dtype,None) for backend,_ in BACKENDS for dtype,_ in DTYPES]
            if args.reverse:configurations.reverse()
            for engine,first,second in configurations:
                plan=None;label=f"native-{first}-{second}" if engine=="native" else f"{engine}-{first}";started=time.perf_counter();rounded=None
                try:
                    if engine=="native":
                        plan=NativeAttention(libraries,*operands,(first,second),kind);tuning=plan.tune();function=plan;prepared=plan.run
                        if not record["negative_controls"]:
                            ptr=operands[0].data_ptr();out=plan.output.data_ptr()
                            for a,b,batch,heads,length,depth,mask in ((-1,0,*shape,0),(0,5,*shape,0),(0,0,0,1,128,64,0),(0,0,1,1,0,64,0),(0,0,*shape,3)):
                                invalid=libraries[0].att_create(a,b,batch,heads,length,depth,mask,ptr,ptr,ptr,out)
                                if invalid: libraries[0].att_destroy(invalid);raise ValueError("Invalid attention configuration accepted")
                            if libraries[0].att_prepare(None,torch.cuda.current_stream().cuda_stream)!=1:raise ValueError("Null attention plan accepted")
                            record["negative_controls"]=6
                    else:
                        dtype=dict(DTYPES)[first];backend=dict(BACKENDS)[engine];function=framework_function(operands,dtype,backend,kind,device_mask);prepared=framework_function(operands,dtype,backend,kind,device_mask,True)
                        with warnings.catch_warnings(record=True) as notices:actual=function();torch.cuda.synchronize()
                        rounded=tuple(x.to(dtype).float().cpu().numpy() for x in operands);tuning=None
                except (RuntimeError,NotImplementedError) as exc:
                    if plan is not None:plan.close()
                    record["unavailable"].append({"shape":list(shape),"mask":kind,"profile":profile,"label":label,"diagnostic":str(exc)});save(destination,record);continue
                try:
                    setup=(time.perf_counter()-started)*1000;function();torch.cuda.synchronize()
                    if plan:
                        checks,buffers,actual=validate_native(plan,host,target,indices)
                    else:
                        actual=function().cpu().numpy();algorithm_error=relative_error(actual,attention(*rounded,kind))
                        if not np.isfinite(actual).all() or algorithm_error>{"fp32":5e-5,"fp16":.005,"bf16":.03}[first]:raise ValueError("Framework attention arithmetic failed")
                        if kind=="padding" and np.any(actual[:,:,0]):raise ValueError("Fully masked framework output is nonzero")
                        checks={**sample_checks(actual,target,indices),"algorithm_relative_error":algorithm_error,"checked_output_entries":int(actual.size)}
                    entry={"shape":list(shape),"mask":kind,"profile":profile,"label":label,"engine":engine,"qk_mode":first if plan else None,"pv_mode":second,
                           "partition":"held-out" if shape in HELD_OUT else "training","input_sha256":[digest(x) for x in host],"checks":checks,"setup_and_tuning_wall_ms":setup,
                           "tuning":tuning,"selected_algorithms":plan.algorithms.copy() if plan else None,"graph_repetitions":repetitions,"timings_ms":{}}
                    if plan:entry["owned_native_device_buffer_bytes"]=libraries[0].att_owned_bytes(plan.pointer)
                    else:
                        torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();memory_output=function();torch.cuda.synchronize()
                        entry["extra_peak_torch_allocated_bytes"]=torch.cuda.max_memory_allocated()-base;del memory_output
                    entry["timings_ms"]["prepared"],prepared_actual=benchmark(prepared,repetitions,record["samples"],record["warmups"])
                    entry["timings_ms"]["online"],timed_actual=benchmark(function,repetitions,record["samples"],record["warmups"])
                    np.testing.assert_array_equal(prepared_actual,actual);np.testing.assert_array_equal(timed_actual,actual)
                    if plan:entry["post_timing_checks"],_,_=validate_native(plan,host,target,indices,checks)
                    else:entry["post_timing_output_sha256"]=digest(timed_actual)
                    entry["host_request_ms"],last=host_requests(function,operands,host,record["samples"],record["warmups"])
                    np.testing.assert_array_equal(last,actual)
                    if plan:entry["post_request_checks"],_,_=validate_native(plan,host,target,indices,checks)
                    else:entry["post_request_output_sha256"]=digest(last)
                    if shape==(1,1,128,64):
                        path=destination.parent/f"attention-witness-{args.seed}-{profile}-{kind}-{label}.npz";witness={"q":host[0],"k":host[1],"v":host[2],"output":actual,"indices":indices}
                        if plan:witness.update(scores=buffers[3],probabilities=buffers[4],padded_output=buffers[5])
                        np.savez_compressed(path,**witness);entry["witness"]={"file":path.name,"sha256":sha(path)}
                    record["cases"].append(entry);save(destination,record)
                finally:
                    torch.cuda.synchronize()
                    if plan:plan.close()
            print(f"{shape} {kind}/{profile}: native combinations and forced fused references checked",flush=True)
    record["complete"]=True;save(destination,record)


if __name__=="__main__":main()
