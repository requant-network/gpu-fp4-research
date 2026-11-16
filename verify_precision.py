"""CPU-only integer-gold replay of native/component/emulated precision evidence."""
import json

import numpy as np

from precision_reference import arithmetic_reference,check_output,components,dataset,exact_product,exact_samples,gold,row_statistics
from reference import digest,relative_error
from verify_results import require,sha256
from verify_followups import ROOT

CONFIGS={("fp32",0,0),("fp64",1,0),("emu64-performant",2,-1),("emu64-eager",2,0),("emu64-fixed31",2,31),("emu64-fixed55",2,55),("requested-bf16x9",3,0)}|{
    (f"slices-{count}",4,count) for count in (1,2,3,4,6,8)}|{("adaptive-8",5,8)}
SHAPES=((16,512,512),(128,128,128),(256,512,512),(1024,1024,1024),(2048,2048,2048),(32,768,1280),(384,768,256),(128,128,8192))
HELD_OUT=((32,768,1280),(384,768,256));PROFILES=("normal","cancellation","spread","subnormal")
SOURCES={"precision_study.py","precision_reference.py","native/precision_bridge.cu","build_precision.sh","reference.py","gpu_study.py","followup.py"}


def keys():
    groups=[(shape,profile,"synthetic") for shape in SHAPES for profile in PROFILES]
    groups += [((batch,256,128),"digits-input","real") for batch in (64,192)]+[((batch,128,256),"digits-hidden","real") for batch in (64,192)]
    return {(shape,profile,kind,*config) for shape,profile,kind in groups for config in CONFIGS}


def verify_transitions(record,seed):
    probe=record["adaptive_transition_probe"];witness=probe["witness"];path=ROOT/"results"/witness["file"]
    require(witness["file"]==f"precision-transitions-{seed}.npz" and sha256(path)==witness["sha256"],"Precision transition witness changed")
    require([row["profile"] for row in probe["rows"]]==["normal","cancellation","spread","normal"],"Missing changed-input transitions")
    with np.load(path,allow_pickle=False) as arrays:
        _,b=dataset(seed,(16,128,128),"normal");np.testing.assert_array_equal(arrays["b"],b)
        for index,row in enumerate(probe["rows"]):
            a,_=dataset(seed+index,(16,128,128),row["profile"]);np.testing.assert_array_equal(arrays[f"a{index}"],a)
            required=int(max(row_statistics(a)[1].max(),row_statistics(b)[1].max(),1));selected=required if required<=8 else 0
            require(row["selected_components"]==selected and row["input_sha256"]==digest(a),"Incorrect changed-input decision")
            actual,exact=arrays[f"output{index}"],exact_product(a,b)
            np.testing.assert_array_equal(arrays[f"exact{index}"],exact)
            require(digest(actual)==row["output_sha256"],"Transition output changed")
            expected,bound,_,_=arithmetic_reference(a,b,5,selected,-1);check_output(actual,expected,bound,exact)
            require(np.isclose(relative_error(actual,exact),row["exact_relative_error"],rtol=2e-12,atol=1e-30),"Transition exact error changed")


def verify():
    identities=[];outputs=0;witness_entries=0;sample_entries=0;cases_count=0;positive_emulation=0
    expected=keys();require(len(expected)==504,"Precision grid declaration changed")
    for suffix,seed,model_seed,reverse in (("a",20261121,20261116,False),("b",20261221,20261216,True)):
        record=json.loads((ROOT/f"results/precision-cohort-{suffix}.json").read_text(encoding="utf-8"))
        require(record.get("complete") is True and not record["quick"] and (record["seed"],record["model_seed"],record["reverse"])==(seed,model_seed,reverse),"Incomplete precision replication")
        require(record["samples"]==30 and record["warmups"]==5 and record["negative_controls"]==7,"Precision protocol/controls changed")
        require(record["workspace_bytes"]==512*1024**2 and record["environment"]["gpu_exclusive"] is False,"Precision workspace/qualification changed")
        require(set(record["source_sha256"])==SOURCES,"Precision source coverage changed")
        require(set(record["data_sha256"])=={f"fixtures/{prefix}-{model_seed}.npz" for prefix in ("digits-model","outlier-activations")},"Precision data coverage changed")
        for mapping in ("source_sha256","data_sha256"):
            for name,expected_hash in record[mapping].items():require(sha256(ROOT/name)==expected_hash,f"Precision executed provenance changed: {name}")
        identities.append((record["source_sha256"],record["binary_sha256"]));verify_transitions(record,seed)
        seen,cache=set(),{}
        for missing in record["unavailable"]:
            key=(tuple(missing["shape"]),missing["profile"],"synthetic" if missing["profile"] in PROFILES else "real","requested-bf16x9",3,0)
            require(key in expected and key not in seen and missing["diagnostic"],"Invalid unavailable precision control");seen.add(key)
        with np.load(ROOT/f"fixtures/digits-model-{model_seed}.npz",allow_pickle=False) as model,np.load(
                ROOT/f"fixtures/outlier-activations-{model_seed}.npz",allow_pickle=False) as activations:
            for case in record["cases"]:
                shape,profile,kind=tuple(case["shape"]),case["profile"],case["kind"];mode,parameter=case["mode"],case["parameter"]
                key=(shape,profile,kind,case["label"],mode,parameter)
                require(key in expected and key not in seen,"Unexpected/duplicate precision case");seen.add(key)
                require(case["partition"]==("held-out" if shape in HELD_OUT or kind=="real" and shape[0]==192 else "training"),"Precision split changed")
                repetitions=1 if shape[0]*shape[1]*shape[2]>=2**29 or shape[2]>=4096 else 8
                require(case["graph_repetitions"]==repetitions,"Precision graph repetitions changed")
                if cache.get("group")!=(shape,profile,kind):
                    a,b=dataset(seed,shape,profile) if kind=="synthetic" else (activations["x" if profile=="digits-input" else "hidden"][:shape[0]],model["w1" if profile=="digits-input" else "w2"])
                    rng=np.random.default_rng(np.random.SeedSequence([seed,*shape,97]));indices=np.column_stack((rng.integers(shape[0],size=64),rng.integers(shape[1],size=64))).astype(np.int32)
                    cache={"group":(shape,profile,kind),"a":a,"b":b,"indices":indices,"exact":exact_samples(a,b,indices),"encodings":{}}
                a,b,indices,exact=cache["a"],cache["b"],cache["indices"],cache["exact"]
                require(case["input_sha256"]==[digest(a),digest(b)],"Precision inputs changed")
                required=int(max(row_statistics(a)[1].max(),row_statistics(b)[1].max(),1)) if mode>=4 else None
                selected=parameter if mode==4 else required if mode==5 and required<=parameter else 0
                if selected not in cache["encodings"]:
                    hashes={};metadata={};parts=[]
                    if selected:
                        for operand,x in enumerate((a,b)):
                            pieces,exponent,req,_=components(x,selected);parts.append(pieces)
                            hashes[f"operand_{operand}_components"]=digest(pieces)
                            metadata[f"operand_{operand}_exponents"],metadata[f"operand_{operand}_required"]=digest(exponent),digest(req)
                    cache["encodings"][selected]=(hashes,metadata,parts)
                hashes,metadata,parts=cache["encodings"][selected]
                if not selected:
                    hashes={f"operand_{operand}_prepared":digest(x.astype(np.float32 if mode in (0,3) else np.float64)) for operand,x in enumerate((a,b))}
                    metadata={}
                    if mode>=4:
                        for operand,x in enumerate((a,b)):
                            exponent,req=row_statistics(x)
                            metadata[f"operand_{operand}_exponents"],metadata[f"operand_{operand}_required"]=digest(exponent),digest(req)
                for checks in (case["checks"],case["post_timing_checks"],case["post_request_checks"]):
                    require((checks["required_components"],checks["selected_components"])==(required,selected),"Precision component selection changed")
                    require(checks["prepared_sha256"]==hashes and checks["metadata_sha256"]==metadata,"Precision factors/statistics changed")
                    require(0<=checks["arithmetic_max_bound_ratio"]<=1 and checks["checked_output_entries"]==shape[0]*shape[1],"Precision arithmetic failed")
                    require(np.isfinite(checks["relative_frobenius_error"]) and checks["relative_frobenius_error"]>=0,"Invalid precision error")
                    values=np.array(checks["exact_sample_values"],np.float64)
                    require(values.shape==(64,) and np.isfinite(values).all() and checks["exact_samples_sha256"]==digest(exact),"Missing exact product samples")
                    require(np.isclose(relative_error(values,exact),checks["exact_sample_relative_error"],rtol=2e-12,atol=1e-30),"Exact integer-gold sample error changed")
                    if selected:
                        partial=np.array([np.dot(parts[0][0,i].astype(np.int64),parts[1][0,j].astype(np.int64)) for i,j in indices],np.int64)
                        np.testing.assert_array_equal(np.array(checks["last_slice_product_samples"],np.int64),partial)
                        require(checks["last_slice_product_samples_sha256"]==digest(partial),"INT8 product samples changed")
                    if mode==2:positive_emulation+=checks["retained_emulation_bits"]>0
                    else:require(checks["retained_emulation_bits"]==-1,"Unexpected emulation bit result")
                descriptor=case["emulation_descriptor"]
                if mode in (2,3):
                    require(descriptor=={"strategy":1 if mode==2 and parameter==-1 else 2,"special_values_mask":0,
                        "mantissa_control":1 if mode==2 and parameter>0 else 0,"max_mantissa_bits":parameter if mode==2 and parameter>0 else 0,"mantissa_bit_offset":0},"Emulation configuration changed")
                else:require(descriptor is None,"Unexpected emulation descriptor")
                require(set(case["timings_ms"])=={"prepacked","online-a","online-both"},"Precision timing boundaries missing")
                for values in list(case["timings_ms"].values())+[case["host_request_ms"]]:
                    require(len(values)==30 and all(np.isfinite(v) and v>0 for v in values),"Invalid precision timing observations")
                require(case["setup_wall_ms"]>0 and case["initial_preparation_wall_ms"]>0,"Missing precision setup/preparation cost")
                require(len(case["tuning"])==(2 if mode==5 else 1),"Missing precision algorithm tuning")
                for which,rows in enumerate(case["tuning"]):
                    require(1<=len(rows)<=8 and [r["candidate"] for r in rows]==list(range(len(rows))),"Invalid precision heuristic candidates")
                    require(all(len(r["ms"])==3 and all(np.isfinite(v) and v>0 for v in r["ms"]) for r in rows),"Invalid precision tuning observations")
                    chosen=case["selected_algorithms"][which];require(0<=chosen<len(rows),"Invalid selected precision algorithm")
                    if mode>=4 and which==0:require(rows[chosen]["numerical_flags"]&4,"Component kernel is not reported as IMMA")
                if shape==(128,128,128):
                    witness=case["witness"];path=ROOT/"results"/witness["file"]
                    require(witness["file"]==f"precision-witness-{seed}-{profile}.npz" and witness["array"]==case["label"] and sha256(path)==witness["sha256"],"Precision witness changed")
                    with np.load(path,allow_pickle=False) as arrays:
                        np.testing.assert_array_equal(arrays["a"],a);np.testing.assert_array_equal(arrays["b"],b)
                        np.testing.assert_array_equal(arrays["indices"],indices);np.testing.assert_array_equal(arrays["exact_samples"],exact)
                        np.testing.assert_array_equal(arrays["exact"],exact_product(a,b))
                        actual=arrays[witness["array"]];require(digest(actual)==case["checks"]["output_sha256"],"Precision witness output hash changed")
                        expected_value,bound,_,_=arithmetic_reference(a,b,mode,selected,case["checks"]["retained_emulation_bits"])
                        check_output(actual,expected_value,bound,gold(a,b,profile))
                        np.testing.assert_array_equal(actual[indices[:,0],indices[:,1]],np.array(case["checks"]["exact_sample_values"]))
                        witness_entries+=actual.size
                outputs+=shape[0]*shape[1];sample_entries+=64;cases_count+=1
        require(seen==expected,"Precision candidate/distribution/shape grid incomplete")
    require(identities[0]==identities[1] and positive_emulation>0,"Precision binary changed or no actual FP64 emulation observed")
    print(f"Precision: {cases_count:,} cases, {outputs:,} outputs checked at three checkpoints, {witness_entries:,} full CPU-replayed entries, {sample_entries:,} retained exact integer-gold samples")


if __name__=="__main__":verify()
