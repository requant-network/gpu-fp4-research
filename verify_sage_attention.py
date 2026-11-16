"""CPU replay and provenance checks for the pinned specialized FP4 reference."""
import json
from pathlib import PurePosixPath
import re

import numpy as np

from attention_reference import MASKS,arrays,attention
from reference import digest,relative_error
from verify_attention import SHAPES,close,sample_check
from verify_followups import ROOT
from verify_results import require,sha256

SOURCES={"sage_attention_study.py","sage_reference.py","attention_study.py","attention_native.py","attention_reference.py","attention_validation.py","reference.py"}


def verify():
    build=json.loads((ROOT/"results/sage-build.json").read_text(encoding="utf-8"))
    require(build.get("complete") is True and build["build_exit_code"]==0 and build["sage_commit"]=="d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5" and build["cutlass_commit"]=="b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd","Pinned specialized FP4 build incomplete")
    for name,expected in build["source_sha256"].items():require(sha256(ROOT/name)==expected,"Executed reference builder changed")
    require(build["upstream_source_sha256"] and build["installed_sha256"] and re.fullmatch(r"[0-9a-f]{64}",build["wheel_sha256"]),"Specialized FP4 build provenance missing")
    for field in ("upstream_source_sha256","installed_sha256"):
        for name,digest_value in build[field].items():require(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts and re.fullmatch(r"[0-9a-f]{64}",digest_value),"Invalid upstream/installed provenance")
    total=missing=witnesses=0;identities=[]
    for suffix,seed,reverse in (("a",20261123,False),("b",20261223,True)):
        record=json.loads((ROOT/f"results/sage-attention-cohort-{suffix}.json").read_text(encoding="utf-8"))
        require(record.get("complete") is True and not record["quick"] and (record["seed"],record["reverse"])==(seed,reverse),"Incomplete specialized FP4 cohort")
        require(record["samples"]==30 and record["warmups"]==5 and record["environment"]["gpu_exclusive"] is False,"Reference protocol changed")
        require(record["sage_build_sha256"]==sha256(ROOT/"results/sage-build.json") and set(record["source_sha256"])==SOURCES,"Specialized FP4 provenance changed")
        for name,expected in record["source_sha256"].items():require(sha256(ROOT/name)==expected,f"Executed specialized reference changed: {name}")
        identities.append(record["source_sha256"]);expected={(shape,kind,profile,label) for shape in SHAPES for kind in MASKS for profile in ("normal","peaky") for label in ("sage3-fp16","sage3-bf16")};seen=set();available=set();cache={}
        for row in record["unavailable"]:
            key=tuple(row["shape"]),row["mask"],row["profile"],row["label"]
            require(key in expected and key not in seen and row["diagnostic"],"Invalid specialized reference rejection");seen.add(key);missing+=1
        for row in record["cases"]:
            shape,kind,profile=tuple(row["shape"]),row["mask"],row["profile"];key=shape,kind,profile,row["label"];group=shape,kind,profile
            require(key in expected and key not in seen,"Unexpected specialized reference case");seen.add(key);available.add(key)
            require(kind!="padding" and shape[-1] in (64,128),"Unsupported specialized reference silently fell back")
            if cache.get("group")!=group:
                host=arrays(seed,shape)
                if profile=="peaky":host=host[0]*np.float32(4),host[1]*np.float32(4),host[2]
                rng=np.random.default_rng(np.random.SeedSequence([seed,*shape,271]));target=attention(*host,kind);indices=rng.integers(target.size,size=64);cache={"group":group,"host":host,"target":target,"indices":indices}
            host,target,indices=cache["host"],cache["target"],cache["indices"]
            require(row["input_sha256"]==[digest(x) for x in host] and row["partition"]==("held-out" if shape in SHAPES[-2:] else "training"),"Specialized FP4 input/split changed")
            sample_check(row["checks"],target,indices)
            require(row["post_timing_output_sha256"]==row["checks"]["output_sha256"] and row["post_request_output_sha256"]==row["checks"]["output_sha256"],"Specialized FP4 repeated preparation mutated results")
            require(set(row["timings_ms"])=={"prepared","online"} and row["setup_and_first_call_wall_ms"]>0 and row["extra_peak_torch_allocated_bytes"]>=0,"Specialized FP4 cost boundary missing")
            require(row["graph_repetitions"]==(1 if shape[0]*shape[1]*shape[2]>=4096 else 4),"Reference graph repetitions changed")
            for values in list(row["timings_ms"].values())+[row["host_request_ms"]]:require(len(values)==30 and all(v>0 and np.isfinite(v) for v in values),"Invalid specialized FP4 observations")
            if "witness" in row:
                path=ROOT/"results"/row["witness"]["file"];require(sha256(path)==row["witness"]["sha256"],"Specialized FP4 witness changed")
                with np.load(path,allow_pickle=False) as data:
                    for name,value in zip(("q","k","v"),host):np.testing.assert_array_equal(data[name],value)
                    np.testing.assert_array_equal(data["indices"],indices);actual=data["output"]
                    require(actual.shape==shape and np.isfinite(actual).all() and digest(actual)==row["checks"]["output_sha256"],"Specialized FP4 output changed")
                    np.testing.assert_array_equal(actual.ravel()[indices],np.array(row["checks"]["sample_values"]))
                    close(relative_error(actual,target),row["checks"]["relative_frobenius_error"],"Specialized FP4 measured error changed");witnesses+=1
            total+=1
        require(seen==expected,"Specialized FP4 input grid incomplete")
        require(available=={key for key in expected if key[0][-1] in (64,128) and key[1]!="padding"},"Required specialized FP4 comparison unavailable")
    require(identities[0]==identities[1],"Specialized reference source changed between cohorts")
    print(f"Specialized FP4: {total} cases, {missing} explicit unsupported cases, {witnesses} complete output witnesses")


if __name__=="__main__":verify()
