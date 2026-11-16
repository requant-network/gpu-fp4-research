"""CPU-only replay of native packed-conversion/vector-load experiments."""
import json

import numpy as np

from reference import SHAPES,digest,quantize,relative_error
from verify_followups import ROOT,HELD_OUT,PROFILES,BOUNDARIES,regenerated
from verify_results import require,sha256

VARIANTS=(0,2,10,11,12,13)
SOURCES={"hardware_study.py","reference.py","gpu_study.py","followup.py","encoder_study.py",
         "native/bridge.cu","native/optimized_bridge.cu","native/application_bridge.cu",
         "native/hardware_bridge.cu","build_hardware.sh","build_extended.sh"}


def expected_keys():
    keys=set()
    for shape in SHAPES+HELD_OUT+((2048,2048,2048),):
        for profile in PROFILES:
            if shape==(2048,2048,2048) and profile not in ("normal","outliers"):continue
            for variant in VARIANTS:keys.add((shape,profile,variant,False,False))
            if profile in ("normal","outliers"):
                for variant in (2,12):
                    for fused in (False,True):keys.add((shape,profile,variant,True,fused))
    return keys


def verify():
    outputs,witnesses,identities=0,0,[]
    for name,seed,reverse in (("hardware-cohort-a.json",20261118,False),("hardware-cohort-b.json",20261218,True)):
        record=json.loads((ROOT/"results"/name).read_text(encoding="utf-8"))
        require(record.get("complete") is True and not record["quick"],"Incomplete hardware campaign")
        require((record["seed"],record["reverse"])==(seed,reverse),"Hardware replication mismatch")
        require((record["warmups"],record["samples"],record["graph_repetitions"])==(5,30,32),"Hardware timing protocol changed")
        require(record["negative_controls"]==5 and record["environment"]["gpu_exclusive"] is False,"Missing controls/qualification")
        require(set(record["source_sha256"])==SOURCES,"Hardware source coverage changed")
        for source,expected in record["source_sha256"].items():
            require(sha256(ROOT/source)==expected,f"Executed hardware source changed: {source}")
        identities.append((record["source_sha256"],record["binary_sha256"],record["resources"]))
        resources=record["resources"]
        require({(r["variant"],r["block_threads"]) for r in resources}==
                {(v,128) for v in VARIANTS}|{(2,256),(2,512)},"Resource-query coverage changed")
        for row in resources:
            require(row["binary_version"]==120 and row["sm_count"]==36 and row["warp_size"]==32,"Wrong architecture resource query")
            require(row["registers_per_thread"]>0 and row["static_shared_bytes"]>=0 and row["local_bytes"]>=0,"Invalid native resources")
            expected=row["theoretical_active_blocks_per_sm"]*row["block_threads"]/row["sm_max_threads"]
            require(0<expected<=1 and expected==row["theoretical_thread_occupancy"],"Invalid theoretical occupancy")
        seen,cache=set(),{}
        for case in record["cases"]:
            shape,profile=tuple(case["shape"]),case["profile"]
            key=(shape,profile,case["variant"],case["activation"],case["fused"])
            require(key in expected_keys() and key not in seen,"Unexpected/duplicate hardware configuration")
            seen.add(key)
            require(case["partition"]==("held-out" if shape in HELD_OUT else "training"),"Hardware split changed")
            if cache.get("group")!=(shape,profile):
                a,b=regenerated(seed,shape,profile)
                bias=(np.arange(shape[2],dtype=np.int32)%33-16).astype(np.float32)/np.float32(64)
                cache={"group":(shape,profile),"a":a,"b":b,"bias":bias,"variants":{}}
            a,b=cache["a"],cache["b"]
            require(case["input_sha256"]==[digest(a),digest(b)],"Hardware input regeneration failed")
            active=case["activation"]
            if active not in cache["variants"]:
                aa=np.maximum(a+cache["bias"],np.float32(0)) if active else a
                hashes,values={},[]
                for operand,original in enumerate((aa,b)):
                    raw,scales,value=quantize(original,"fp4")
                    hashes[f"operand_{operand}_encoded"],hashes[f"operand_{operand}_scales"]=digest(raw),digest(scales)
                    values.append(value)
                cache["variants"][active]=(hashes,values,aa)
            hashes,values,aa=cache["variants"][active]
            for checks in (case["checks"],case["post_timing_checks"]):
                require(checks["encoded_sha256"]==hashes,"Hardware encoded bytes/scales differ")
                require(0<=checks["arithmetic_max_bound_ratio"]<=1,"Hardware arithmetic bound failed")
                require(checks["checked_output_entries"]==shape[0]*shape[1],"Wrong hardware output count")
                require(np.isfinite(checks["relative_frobenius_error"]) and checks["relative_frobenius_error"]>=0,"Invalid numerical error")
            require(set(case["timings_ms"])==BOUNDARIES,"Hardware timing boundaries incomplete")
            for observations in case["timings_ms"].values():
                require(len(observations)==30 and all(np.isfinite(v) and v>0 for v in observations),"Invalid hardware observations")
            if shape==(128,128,128):
                witness=case["witness"]
                require(witness["file"]==f"hardware-witness-{seed}-{profile}.npz","Unexpected hardware witness path")
                path=ROOT/"results"/witness["file"]
                require(sha256(path)==witness["sha256"],"Hardware witness hash differs")
                with np.load(path,allow_pickle=False) as arrays:
                    for label,expected in (("a",a),("b",b),("bias",cache["bias"])):
                        np.testing.assert_array_equal(arrays[label],expected)
                    require(witness["array"]==case["label"],"Wrong hardware witness array")
                    actual=arrays[witness["array"]]
                    require(digest(actual)==case["checks"]["output_sha256"],"Hardware output digest differs")
                    x,y=(v.astype(np.float64) for v in values)
                    expected=x@y.T
                    gamma=shape[2]*2**-24/(1-shape[2]*2**-24)
                    bound=8*gamma*(np.abs(x)@np.abs(y).T)+np.finfo(np.float32).tiny
                    require(actual.shape==expected.shape and np.isfinite(actual).all() and np.all(np.abs(actual-expected)<=bound),"Hardware witness arithmetic failed")
                    original=aa.astype(np.float64)@b.astype(np.float64).T
                    require(np.isclose(relative_error(actual,original),case["checks"]["relative_frobenius_error"],rtol=2e-6,atol=1e-12),"Hardware error metric differs")
                    witnesses+=actual.size
            outputs+=shape[0]*shape[1]
        require(seen==expected_keys(),"Hardware case grid incomplete")
    require(identities[0]==identities[1],"Hardware source/binary/resources changed across cohorts")
    for name in ("native-inspection-128.json","native-inspection-256.json","native-inspection-hardware.json"):
        record=json.loads((ROOT/"results"/name).read_text(encoding="utf-8"))
        require(record["source_sha256"]=={"inspect_native.py":sha256(ROOT/"inspect_native.py")},"Inspection source changed")
        require(set(record["architectures"])=={"sm_120","sm_120a"},"Inspection architecture changed")
        kernels=record["kernels"]
        require(kernels and all(k["instruction_total"]==sum(k["opcodes"].values()) for k in kernels),"Invalid instruction counts")
        opcodes={o for k in kernels for o in k["opcodes"]}
        require(any("OMMA" in o and "E2M1" in o for o in opcodes) and any("UTMALDG" in o for o in opcodes),"Missing native FP4/TMA evidence")
        if name=="native-inspection-hardware.json":
            require(record["binary_sha256"]==identities[0][1],"Inspected hardware binary differs from measured binary")
            require(any("F2FP" in o and "E2M1" in o for o in opcodes) and "REDUX.MAX" in opcodes,"Missing hardware conversion/reduction evidence")
        else:
            encoder=json.loads((ROOT/"results/encoder-cohort-a.json").read_text(encoding="utf-8"))
            tile="256" if "256" in name else "128"
            require(record["binary_sha256"]==encoder["binary_sha256"][tile],"Inspected encoder binary differs")
    profiler=json.loads((ROOT/"results/native-profiler.json").read_text(encoding="utf-8"))
    require(profiler.get("complete") is True and profiler["binary_sha256"]==identities[0][1],"Profiler binary changed")
    for source,expected in profiler["source_sha256"].items():
        require(sha256(ROOT/source)==expected,"Profiler source changed")
    require(profiler["counters_available"] is False and len(profiler["probes"])==1,"Retained profiler availability changed")
    probe=profiler["probes"][0]
    require(probe["variant"]==2 and probe["returncode"]!=0 and probe["counter_permission_denied"] is True
            and "ERR_NVGPUCTRPERM" in probe["output"],"Missing actual profiler permission failure")
    print(f"Hardware: 680 cases, {outputs:,} outputs checked before/after, {witnesses:,} CPU-replayed entries; native instruction/resource evidence checked")


if __name__=="__main__":verify()
