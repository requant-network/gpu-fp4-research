"""CPU-only verification of native and compiled classifier evidence."""
import json
from pathlib import Path

import numpy as np

from prepare_digits import load_digits
from reference import check_output,digest,quantize
from verify_results import require,sha256

ROOT=Path(__file__).resolve().parent
BATCHES={16,32,64,128,192,256}
CONFIGURATIONS={("fp32","fp32","cutlass",False),("fp16","fp16","cutlass",False),
                ("bf16","bf16","cutlass",False),("fp8","fp8","cutlass",False),
                ("fp4","fp4","cutlass",True),("fp4","fp16","cutlass",False),
                ("fp4","fp4","cublaslt",True),("fp4","fp16","cublaslt",False)}
NATIVE_SOURCES={"application_study.py","prepare_digits.py","reference.py","gpu_study.py","followup.py",
                "native/bridge.cu","native/optimized_bridge.cu","native/application_bridge.cu",
                "build_application.sh","build_extended.sh"}


def timing_check(values):
    require(len(values)==30 and all(isinstance(v,(float,int)) and not isinstance(v,bool)
                                   and np.isfinite(v) and v>0 for v in values),"Invalid application timing observations")


def model(seed):
    with np.load(ROOT/f"fixtures/digits-model-{seed}.npz",allow_pickle=False) as arrays:
        return {name:arrays[name] for name in arrays.files}


def simulate(x,parameters,modes):
    qx=quantize(x,modes[0])[2]
    qw1=quantize(parameters["w1"],modes[0])[2]
    first=(qx.astype(np.float64)@qw1.astype(np.float64).T).astype(np.float32)
    hidden=np.maximum(first+parameters["b1"],np.float32(0))
    qh=quantize(hidden,modes[1])[2]
    qw2=quantize(parameters["w2"],modes[1])[2]
    second=(qh.astype(np.float64)@qw2.astype(np.float64).T).astype(np.float32)
    return (second[:,:10]+parameters["b2"][:10]).argmax(axis=1)


def provenance(record,sources):
    require(set(record["source_sha256"])==set(sources),"Application source coverage mismatch")
    for name,expected in record["source_sha256"].items():
        require(sha256(ROOT/name)==expected,f"Application executed source changed: {name}")
    for name,expected in record["data_sha256"].items():
        require(name.startswith(("fixtures/","results/")) and ".." not in Path(name).parts,"Invalid data path")
        require(sha256(ROOT/name)==expected,"Application input data binding changed")


def replay_witness(path,seed,batch,modes,entry,torch_reference=False):
    x,_=load_digits()
    parameters=model(seed)
    expected_x=np.ascontiguousarray(x[parameters["test"][np.arange(batch)%len(parameters["test"])]] )
    witness=entry["witness"]
    require(Path(witness["file"]).name==witness["file"],"Invalid application witness path")
    target=path.parent/witness["file"]
    require(sha256(target)==witness["sha256"],"Application witness hash changed")
    with np.load(target,allow_pickle=False) as arrays:
        require(set(arrays.files)=={"input","first","second","predictions"},"Unexpected application witness arrays")
        np.testing.assert_array_equal(arrays["input"],expected_x)
        require(entry["checks"]["input_sha256"]==digest(expected_x),"Witness input binding mismatch")
        first,second,predictions=arrays["first"],arrays["second"],arrays["predictions"]
        qx=quantize(expected_x,modes[0])[2]
        qw1=quantize(parameters["w1"],modes[0])[2]
        check_output(first,qx,qw1)
        hidden=np.maximum(first+parameters["b1"],np.float32(0))
        qh=quantize(hidden,modes[1])[2]
        qw2=quantize(parameters["w2"],modes[1])[2]
        check_output(second,qh,qw2)
        np.testing.assert_array_equal(predictions,(second[:,:10]+parameters["b2"][:10]).argmax(axis=1))
        require(digest(predictions)==entry["checks"]["predictions_sha256"],"Witness prediction binding mismatch")
        if not torch_reference:
            require(digest(first)==entry["checks"]["first"]["output_sha256"] and
                    digest(second)==entry["checks"]["second"]["output_sha256"] and
                    digest(predictions)==entry["checks"]["predictions_sha256"],"Native witness output binding mismatch")
    return 1


def quality_check(record,parameters,x,y,native,engines=("eager","compiled")):
    seen=set()
    for quality in record["quality"]:
        key=tuple(quality["configuration"]) if native else (quality["mode"],quality["engine"])
        require(key not in seen,"Duplicate application quality row")
        seen.add(key)
        modes=key[:2] if native else (key[0],key[0])
        require(set(quality["splits"])=={"validation","test"},"Quality split coverage incomplete")
        for split,result in quality["splits"].items():
            ids=parameters[split]
            predicted=np.asarray(result["predictions"],np.int64)
            require(predicted.shape==(len(ids),) and np.all(predicted>=0) and np.all(predicted<10),"Invalid prediction coverage")
            correct=int(np.sum(predicted==y[ids]))
            require(result["count"]==len(ids) and result["correct"]==correct and result["accuracy"]==correct/len(ids),
                    "Application accuracy does not match retained predictions")
            np.testing.assert_array_equal(predicted,simulate(x[ids],parameters,modes),
                                          err_msg=f"Independent complete CPU classifier replay failed: {key} {split}")
    expected=CONFIGURATIONS if native else {(mode,engine) for mode in ("fp32","fp16","bf16","fp8") for engine in engines}
    require(seen==expected,"Application quality configuration coverage incomplete")


def verify_native():
    x,y=load_digits()
    identities,witnesses=[],0
    for name,seed,reverse in (("application-cohort-a.json",20261116,False),("application-cohort-b.json",20261216,True)):
        path=ROOT/"results"/name
        record=json.loads(path.read_text(encoding="utf-8"))
        require(record.get("complete") is True and not record["quick"],"Native application campaign incomplete")
        require(record["seed"]==seed and record["reverse"]==reverse,"Native model cohort/order mismatch")
        require((record["warmups"],record["samples"],record["graph_repetitions"])==(5,30,32),"Native application observation counts changed")
        provenance(record,NATIVE_SOURCES)
        require(record["environment"]["gpu_exclusive"] is False,"Native shared-system qualification missing")
        identities.append((record["source_sha256"],record["binary_sha256"]))
        parameters=model(seed)
        seen=set()
        for entry in record["cases"]:
            batch,config=entry["batch"],tuple(entry["configuration"])
            require(batch in BATCHES and config in CONFIGURATIONS and (batch,config) not in seen,"Native case grid mismatch")
            seen.add((batch,config))
            require(entry["partition"]==("training" if batch in (16,64,256) else "held-out"),"Native batch partition changed")
            current=np.ascontiguousarray(x[parameters["test"][np.arange(batch)%len(parameters["test"])]] )
            for checks in (entry["checks"],entry["post_timing_checks"]):
                require(checks["input_sha256"]==digest(current),"Native request input hash mismatch")
                for layer in ("first","second"):
                    ratio=checks[layer]["arithmetic_max_bound_ratio"]
                    require(np.isfinite(ratio) and 0<=ratio<=1,"Native layer arithmetic check failed")
                require(np.isfinite(checks["scores_relative_error"]) and checks["scores_relative_error"]>=0,"Invalid native score error")
            expected={"captured_device","uncaptured_device","host_request","cold_plan_cached_algorithm"}
            require(set(entry["timings_ms"])==expected,"Native request timing boundaries incomplete")
            for values in entry["timings_ms"].values():timing_check(values)
            require(set(entry["reuse_total_ms"])=={"1","8","64"},"Reuse experiment incomplete")
            for values in entry["reuse_total_ms"].values():timing_check(values)
            require([row["offset"] for row in entry["host_request_inputs"]]==[0,29,83,157],"Dynamic request coverage incomplete")
            for row in entry["host_request_inputs"]:
                request=np.ascontiguousarray(x[parameters["test"][(np.arange(batch)+row["offset"])%len(parameters["test"])]] )
                require(digest(request)==row["input_sha256"],"Dynamic host request input hash mismatch")
            if batch==16:witnesses+=replay_witness(path,seed,batch,config[:2],entry)
        require(seen=={(batch,config) for batch in BATCHES for config in CONFIGURATIONS},"Native application grid incomplete")
        quality_check(record,parameters,x,y,True)
    require(identities[0]==identities[1],"Native application source/binary differed between cohorts")
    print(f"Native application: 96 timing cases, {witnesses} numerical witnesses, all validation/test predictions independently replayed")


def verify_torch():
    x,y=load_digits()
    identities,witnesses=[],0
    expected={(batch,mode,engine) for batch in BATCHES for mode in ("fp32","fp16","bf16","fp8") for engine in ("eager","compiled")}
    for name,seed,reverse in (("pytorch-cohort-a.json",20261116,False),("pytorch-cohort-b.json",20261216,True)):
        path=ROOT/"results"/name
        record=json.loads(path.read_text(encoding="utf-8"))
        require(record.get("complete") is True and not record["quick"],"PyTorch campaign incomplete")
        require(record["seed"]==seed and record["reverse"]==reverse,"PyTorch cohort/order mismatch")
        require((record["warmups"],record["samples"],record["graph_repetitions"])==(5,30,32),"PyTorch observation counts changed")
        provenance(record,{"pytorch_study.py","prepare_digits.py","reference.py"})
        require(record["gpu_exclusive"] is False and record["compute_capability"]==[12,0],"PyTorch hardware qualification missing")
        identities.append(record["source_sha256"])
        seen=set()
        for entry in record["cases"]:
            key=(entry["batch"],entry["mode"],entry["engine"])
            require(key in expected and key not in seen,"PyTorch case grid mismatch")
            seen.add(key)
            require(set(entry["timings_ms"])=={"captured_device","uncaptured_device","host_request"},"PyTorch timer scope mismatch")
            for values in entry["timings_ms"].values():timing_check(values)
            for checks in (entry["checks"],entry["post_timing_checks"]):
                require(len(checks["arithmetic_max_bound_ratio"])==2 and
                        all(np.isfinite(v) and 0<=v<=1 for v in checks["arithmetic_max_bound_ratio"]),"PyTorch arithmetic check failed")
            if entry["batch"]==16:witnesses+=replay_witness(path,seed,16,(entry["mode"],entry["mode"]),entry,True)
        require(seen==expected,"PyTorch campaign grid incomplete")
        quality_check(record,model(seed),x,y,False)
    require(identities[0]==identities[1],"PyTorch source changed between cohorts")
    print(f"PyTorch: 96 timing cases, {witnesses} numerical witnesses, eager/compiled prediction replay passed")


def verify_graphs():
    x,y=load_digits()
    identities,witnesses=[],0
    expected={(batch,mode) for batch in BATCHES for mode in ("fp32","fp16","bf16","fp8")}
    for name,seed,reverse in (("pytorch-graphs-cohort-a.json",20261116,False),("pytorch-graphs-cohort-b.json",20261216,True)):
        path=ROOT/"results"/name
        record=json.loads(path.read_text(encoding="utf-8"))
        require(record.get("complete") is True and not record["quick"],"Internal-graph reference incomplete")
        require(record["seed"]==seed and record["reverse"]==reverse and record["samples"]==30 and record["warmups"]==5,
                "Internal-graph cohort/protocol mismatch")
        provenance(record,{"pytorch_graph_study.py","pytorch_study.py","prepare_digits.py","reference.py"})
        require(record["gpu_exclusive"] is False,"Internal-graph hardware qualification missing")
        identities.append(record["source_sha256"])
        seen=set()
        for entry in record["cases"]:
            key=(entry["batch"],entry["mode"])
            require(key in expected and key not in seen,"Internal-graph case grid mismatch")
            seen.add(key)
            require(entry["engine"]=="compiled_dynamic_graphs" and entry["internal_cuda_graph_roots"]>=1,
                    "Internal CUDA Graph execution evidence missing")
            require(set(entry["timings_ms"])=={"uncaptured_device","host_request"},"Internal-graph timer scope changed")
            for values in entry["timings_ms"].values():timing_check(values)
            for checks in (entry["checks"],entry["post_timing_checks"]):
                require(len(checks["arithmetic_max_bound_ratio"])==2 and
                        all(np.isfinite(v) and 0<=v<=1 for v in checks["arithmetic_max_bound_ratio"]),"Internal-graph arithmetic check failed")
            if entry["batch"]==16:
                witnesses+=replay_witness(path,seed,16,(entry["mode"],entry["mode"]),entry,True)
        require(seen==expected,"Internal-graph case coverage incomplete")
        quality_check(record,model(seed),x,y,False,("compiled_dynamic_graphs",))
    require(identities[0]==identities[1],"Internal-graph reference sources changed")
    print(f"Internal CUDA Graph reference: 48 timing cases, {witnesses} numerical witnesses, complete CPU prediction replay passed")


if __name__=="__main__":
    verify_native()
    verify_torch()
    verify_graphs()
