"""CPU replay of native pipeline, traversal, cache and fusion-resource evidence."""
import hashlib
import json
import re

import numpy as np

from build_pipeline import PIN,generated
from reference import SHAPES,digest,inputs,quantize,relative_error
from verify_followups import ROOT,HELD_OUT
from verify_results import require,sha256

LABELS={f"m{tile}-s{stage}-{schedule}" for tile in (128,256) for stage in ("auto","2","3","4") for schedule in ("coop","ping")}
ORDERS={(0,0),(1,1),(2,1),(1,2),(2,2),(1,4),(2,4),(1,8),(2,8)}
BOUNDARIES={f"{boundary}/{condition}" for boundary in ("prepacked","online-a","online-both") for condition in ("warm","cache-pressure")}
SOURCES={"pipeline_study.py","pipeline_preflight.py","build_pipeline.py","build_raster.py","reference.py","gpu_study.py","followup.py","hardware_study.py",
         "encoder_study.py","native/bridge.cu","native/optimized_bridge.cu","native/application_bridge.cu","native/hardware_bridge.cu","native/pipeline_bridge.cu","native/raster_bridge.cu"}


def source_bindings(record):
    for name,expected in record["source_sha256"].items():require(sha256(ROOT/name)==expected,f"Executed pipeline source changed: {name}")


def check_build(build,label):
    source_bindings(build)
    require(build["complete"] is True and build["cutlass_commit"]==PIN and build["architecture"]=="sm_120a","Invalid pipeline build")
    tile,stage,schedule=label.split("-")
    text=generated((ROOT/"native/bridge.cu").read_text(encoding="utf-8"),int(tile[1:]),stage[1:],schedule)
    require(hashlib.sha256(text.encode()).hexdigest()==build["generated_source_sha256"],"Generated pipeline header changed")
    require(re.fullmatch("[0-9a-f]{64}",build["binary_sha256"]) is not None,"Invalid pipeline binary digest")


def check_resources(row):
    require(row["stages"]>=1 and row["block_threads"]>0 and row["block_threads"]%32==0,"Invalid pipeline stages/threads")
    require(row["dynamic_shared_bytes"]>0 and row["registers_per_thread"]>0 and row["theoretical_blocks_per_sm"]>0,"Invalid pipeline resource capacity")
    require(row["static_shared_bytes"]>=0 and row["local_bytes"]>=0,"Invalid pipeline storage")


def verify():
    counts,witnesses,cases_total=0,0,0;identities=[]
    for suffix,seed,reverse in (("a",20261120,False),("b",20261220,True)):
        record=json.loads((ROOT/f"results/pipeline-cohort-{suffix}.json").read_text(encoding="utf-8"))
        require(record.get("complete") is True and not record["quick"] and (record["seed"],record["reverse"])==(seed,reverse),"Incomplete pipeline replication")
        require((record["samples"],record["warmups"],record["graph_repetitions"])==(30,5,1),"Pipeline instrumentation protocol changed")
        require(record["cache_bytes"]==4*record["l2_bytes"] and record["l2_bytes"]>0,"Cache pressure coverage changed")
        require(record["negative_controls"]==5 and record["environment"]["gpu_exclusive"] is False,"Missing pipeline controls/qualification")
        require(set(record["source_sha256"])==SOURCES,"Pipeline source coverage changed");source_bindings(record)
        families=record["families"]
        require(len(families)==16 and {r["label"] for r in families}==LABELS,"Pipeline build coverage changed")
        available=set()
        for family in families:
            label=family["label"]
            require(family["status"] in ("available","compile-failed","resource-query-failed"),"Unknown pipeline availability state")
            if family["status"]=="compile-failed":
                tile,stage,schedule=label.split("-")
                text=generated((ROOT/"native/bridge.cu").read_text(encoding="utf-8"),int(tile[1:]),stage[1:],schedule)
                require(hashlib.sha256(text.encode()).hexdigest()==family["generated_source_sha256"],"Failed build header changed")
                require(family["compiler_diagnostics"] and re.fullmatch("[0-9a-f]{64}",family["build_log_sha256"]) is not None,"Missing actual compiler failure")
            else:
                check_build(family["build"],label)
                if family["status"]=="available":available.add(label);check_resources(family["resources"])
                else:require(bool(family["diagnostic"]),"Missing resource-query failure")
        require("m128-sauto-coop" in available and "m128-s2-ping" in available,"Missing required pipeline controls")
        source_bindings(record["raster_build"])
        identities.append((record["source_sha256"],families,record["raster_build"]))
        expected_labels=available|{f"raster-{a}-{b}" for a,b in ORDERS}
        groups={(shape,profile) for shape in SHAPES+HELD_OUT+((2048,2048,2048),) for profile in ("normal","outliers")}
        expected={(shape,profile,label) for shape,profile in groups for label in expected_labels}
        seen,cache=set(),{}
        for case in record["cases"]:
            shape,profile=tuple(case["shape"]),case["profile"];key=(shape,profile,case["label"])
            require(key in expected and key not in seen,"Unexpected/duplicate pipeline configuration");seen.add(key)
            require(case["partition"]==("held-out" if shape in HELD_OUT else "training"),"Pipeline held-out split changed")
            require(case["family"] in available|{"raster"},"Unavailable pipeline measured")
            require(case["raster"] is None if case["family"]!="raster" else tuple(case["raster"]) in ORDERS,"Invalid traversal control")
            require(case["label"]==case["family"] if case["raster"] is None else
                    case["label"]==f"raster-{case['raster'][0]}-{case['raster'][1]}","Traversal label does not match the executed arguments")
            check_resources(case["resources"])
            if cache.get("group")!=(shape,profile):
                a,b=inputs(seed,shape,profile);values=[];hashes={}
                for operand,x in enumerate((a,b)):
                    raw,scale,q=quantize(x,"fp4");values.append(q)
                    hashes[f"operand_{operand}_encoded"],hashes[f"operand_{operand}_scales"]=digest(raw),digest(scale)
                cache={"group":(shape,profile),"a":a,"b":b,"values":values,"hashes":hashes}
            a,b=cache["a"],cache["b"]
            require(case["input_sha256"]==[digest(a),digest(b)],"Pipeline input regeneration failed")
            for checks in (case["checks"],case["post_timing_checks"]):
                require(checks["encoded_sha256"]==cache["hashes"],"Pipeline encoded operands changed")
                require(0<=checks["arithmetic_max_bound_ratio"]<=1 and checks["checked_output_entries"]==shape[0]*shape[1],"Pipeline arithmetic check failed")
                require(np.isfinite(checks["relative_frobenius_error"]) and checks["relative_frobenius_error"]>=0,"Invalid pipeline error")
            require(set(case["timings_ms"])==BOUNDARIES,"Pipeline timing/condition coverage changed")
            for observations in case["timings_ms"].values():
                require(len(observations)==30 and all(np.isfinite(v) and v>0 for v in observations),"Invalid instrumented observations")
            if shape==(128,128,128):
                witness=case["witness"];path=ROOT/"results"/witness["file"]
                require(witness["file"]==f"pipeline-witness-{seed}-{profile}.npz" and witness["array"]==case["label"],"Unexpected pipeline witness")
                require(sha256(path)==witness["sha256"],"Pipeline witness hash changed")
                with np.load(path,allow_pickle=False) as arrays:
                    np.testing.assert_array_equal(arrays["a"],a);np.testing.assert_array_equal(arrays["b"],b)
                    actual=arrays[witness["array"]]
                    require(digest(actual)==case["checks"]["output_sha256"],"Pipeline witness output changed")
                    x,y=(v.astype(np.float64) for v in cache["values"]);gamma=shape[2]*2**-24/(1-shape[2]*2**-24)
                    bound=8*gamma*(np.abs(x)@np.abs(y).T)+np.finfo(np.float32).tiny
                    require(np.isfinite(actual).all() and np.all(np.abs(actual-x@y.T)<=bound),"Pipeline witness arithmetic failed")
                    error=relative_error(actual,a.astype(np.float64)@b.astype(np.float64).T)
                    require(np.isclose(error,case["checks"]["relative_frobenius_error"],rtol=2e-6,atol=1e-12),"Pipeline witness error changed")
                    witnesses+=actual.size
            counts+=shape[0]*shape[1];cases_total+=1
        require(seen==expected,"Pipeline case/condition grid incomplete")
    require(identities[0]==identities[1],"Pipeline source/binary/availability changed between cohorts")
    timer=json.loads((ROOT/"results/pipeline-timer-calibration.json").read_text(encoding="utf-8"));source_bindings(timer)
    require(timer["complete"] is True and timer["binary_sha256"]==identities[0][2]["binary_sha256"],"Timer calibration binary changed")
    require((timer["samples"],timer["warmups"])==(30,5) and set(timer["timings_ms"])=={"warm","cache-pressure"},"Incomplete timer calibration")
    for values in timer["timings_ms"].values():
        require(len(values)==30 and all(np.isfinite(v) and v>=0 for v in values),"Invalid empty event intervals")
    fusion=json.loads((ROOT/"results/fusion-resources.json").read_text(encoding="utf-8"));source_bindings(fusion)
    require(fusion["complete"] is True and fusion["binary_sha256"]==identities[0][2]["binary_sha256"],"Fusion-resource binary changed")
    require(len(fusion["rows"])==30 and {(r["variant"],r["fused"],r["block_threads"]) for r in fusion["rows"]}==
            {(v,f,t) for v in (2,10,11,12,13) for f in (False,True) for t in (128,256,512)},"Fusion resource grid changed")
    for row in fusion["rows"]:
        occupancy=row["theoretical_blocks_per_sm"]*row["block_threads"]/row["sm_max_threads"]
        require(row["registers_per_thread"]>0 and 0<occupancy<=1 and occupancy==row["theoretical_thread_occupancy"],"Invalid theoretical fusion capacity")
    print(f"Pipeline: {cases_total:,} cases, {counts:,} outputs checked before/after, {witnesses:,} CPU-replayed entries; full build/traversal/cache/resource coverage checked")


if __name__=="__main__":verify()
