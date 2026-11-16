"""CPU-only replay of outlier corrections, real activations and retained witnesses."""
import json

import numpy as np

from outlier_reference import check_output,factors,oracle
from prepare_digits import load_digits
from reference import check_output as check_product,digest,inputs,quantize,relative_error
from verify_results import require,sha256
from verify_followups import ROOT

SHAPES=((16,512,512),(128,128,128),(256,1024,1024),(32,768,1280),(192,768,768),(768,1536,1024),(2048,2048,2048))
HELD_OUT=((32,768,1280),(192,768,768),(768,1536,1024))
PROFILES=("normal","outliers","columns","blocks")
BOUNDARIES={"prepare-a","prepare-both","prepacked","online-a","online-both"}
SOURCES={"outlier_study.py","outlier_reference.py","reference.py","gpu_study.py","followup.py","encoder_study.py","hardware_study.py",
         "prepare_digits.py","prepare_outlier_activations.py","native/bridge.cu","native/optimized_bridge.cu","native/application_bridge.cu",
         "native/hardware_bridge.cu","native/outlier_bridge.cu","build_outlier.sh","build_extended.sh"}


def regenerated(seed,shape,profile):
    a,b=inputs(seed,shape,"outliers" if profile=="outliers" else "normal")
    if profile=="columns":a[:,::97]*=np.float32(32);b[:,::97]*=np.float32(32)
    if profile=="blocks":a.reshape(-1,16)[::37]*=np.float32(32);b.reshape(-1,16)[::41]*=np.float32(32)
    return a,b


def configurations(k):
    return {(mode,0,128,.25) for mode in ("fp4","fp8","fp16","fp32")}|{("fp4",1,budget,.25) for budget in (128,min(256,k))}|{
        ("fp4",2,128,.25),("fp4",2,128,.75),("fp4",3,128,.25)}


def expected_keys():
    groups=[(shape,profile,"synthetic") for shape in SHAPES for profile in PROFILES]
    groups += [((batch,256,128),"digits-input","real") for batch in (16,64,192)]
    groups += [((batch,128,256),"digits-hidden","real") for batch in (16,64,192)]
    return {(shape,profile,kind,*config) for shape,profile,kind in groups for config in configurations(shape[2])}


def verify_activations():
    record=json.loads((ROOT/"results/outlier-activation-preparation.json").read_text(encoding="utf-8"))
    require(set(record["source_sha256"])=={"prepare_outlier_activations.py","prepare_digits.py","reference.py"},"Activation source coverage changed")
    require(set(record["data_sha256"])=={"fixtures/digits.csv.gz"}|{
        f"fixtures/{prefix}-{seed}.npz" for prefix in ("digits-model","outlier-activations") for seed in (20261116,20261216)},"Activation data coverage changed")
    for mapping in ("source_sha256","data_sha256"):
        for name,expected in record[mapping].items():require(sha256(ROOT/name)==expected,"Real-activation provenance changed")
    x,_=load_digits()
    for seed in (20261116,20261216):
        with np.load(ROOT/f"fixtures/digits-model-{seed}.npz",allow_pickle=False) as model,np.load(
                ROOT/f"fixtures/outlier-activations-{seed}.npz",allow_pickle=False) as arrays:
            expected=x[model["validation"][:192]]
            hidden=np.maximum(expected.astype(np.float64)@model["w1"].astype(np.float64).T+model["b1"],0).astype(np.float32)
            np.testing.assert_array_equal(arrays["x"],expected)
            np.testing.assert_array_equal(arrays["hidden"],hidden)


def verify():
    verify_activations()
    outputs,replayed,identities=0,0,[]
    expected=expected_keys()
    require(len(expected)==299,"Unexpected declared correction grid")
    for suffix,seed,model_seed,reverse in (("a",20261119,20261116,False),("b",20261219,20261216,True)):
        record=json.loads((ROOT/f"results/outlier-cohort-{suffix}.json").read_text(encoding="utf-8"))
        require(record.get("complete") is True and not record["quick"],"Incomplete correction campaign")
        require((record["seed"],record["model_seed"],record["reverse"])==(seed,model_seed,reverse),"Correction replication changed")
        require((record["samples"],record["warmups"],record["graph_repetitions"])==(30,5,32),"Correction timing protocol changed")
        require(record["negative_controls"]==5 and record["environment"]["gpu_exclusive"] is False,"Missing controls/qualification")
        require(record["environment"]["compute_capability"]==[12,0] and record["environment"]["sm_count"]==36,"Wrong correction target GPU")
        require(set(record["source_sha256"])==SOURCES,"Correction source coverage changed")
        require(set(record["data_sha256"])=={f"fixtures/{prefix}-{model_seed}.npz" for prefix in ("digits-model","outlier-activations")},"Correction data coverage changed")
        for mapping in ("source_sha256","data_sha256"):
            for name,expected_hash in record[mapping].items():require(sha256(ROOT/name)==expected_hash,f"Executed correction provenance changed: {name}")
        identities.append((record["source_sha256"],record["binary_sha256"]))
        seen,cache=set(),{}
        with np.load(ROOT/f"fixtures/digits-model-{model_seed}.npz",allow_pickle=False) as model,np.load(
                ROOT/f"fixtures/outlier-activations-{model_seed}.npz",allow_pickle=False) as activations:
            for case in record["cases"]:
                shape,profile,kind=tuple(case["shape"]),case["profile"],case["kind"]
                config=case["mode"],case["method"],case["budget"],case["fraction"]
                key=(shape,profile,kind,*config)
                require(key in expected and key not in seen,"Unexpected/duplicate correction case")
                seen.add(key)
                require(case["partition"]==("held-out" if shape in HELD_OUT or shape[0]==192 and kind=="real" else "training"),"Correction split changed")
                if cache.get("group")!=(shape,profile,kind):
                    a,b=regenerated(seed,shape,profile) if kind=="synthetic" else (
                        activations["x" if profile=="digits-input" else "hidden"][:shape[0]],model["w1" if profile=="digits-input" else "w2"])
                    cache={"group":(shape,profile,kind),"a":a,"b":b}
                a,b=cache["a"],cache["b"]
                require(case["input_sha256"]==[digest(a),digest(b)],"Correction inputs changed")
                method=case["method"]
                if method:
                    arrays,metadata,_=factors(a,b,method,case["budget"],case["fraction"])
                    hashes=[digest(v) for v in arrays]
                    metadata_hashes={key:digest(value) for key,value in metadata.items()}
                    counts={key:int(v.size) if key=="selected_columns" else int(np.count_nonzero(v)) for key,v in metadata.items()}
                else:
                    values=[];hashes={}
                    for operand,x in enumerate((a,b)):
                        raw,scales,q=quantize(x,case["mode"]);values.append(q)
                        hashes[f"operand_{operand}_encoded"],hashes[f"operand_{operand}_scales"]=digest(raw),digest(scales)
                for checks in (case["checks"],case["post_timing_checks"]):
                    require(0<=checks["arithmetic_max_bound_ratio"]<=1 and checks["checked_output_entries"]==shape[0]*shape[1],"Correction arithmetic check failed")
                    require(np.isfinite(checks["relative_frobenius_error"]) and checks["relative_frobenius_error"]>=0,"Invalid correction error")
                    if method:
                        require(checks["factor_sha256"]==hashes and checks["metadata_sha256"]==metadata_hashes and checks["selected_counts"]==counts,"Correction factors/selection changed")
                    else:require(checks["encoded_sha256"]==hashes,"Correction reference encoding changed")
                require(np.isfinite(case["setup_wall_ms"]) and case["setup_wall_ms"]>0,"Invalid setup time")
                require(set(case["timings_ms"])==BOUNDARIES,"Correction timing boundaries incomplete")
                for observations in case["timings_ms"].values():
                    require(len(observations)==30 and all(np.isfinite(v) and v>0 for v in observations),"Invalid correction observations")
                tuning=case["tuning"] if method else [case["tuning"]]
                require(len(tuning)==(2 if method else 1),"Incomplete correction tuning")
                for rows in tuning:
                    require(1<=len(rows)<=8 and [r["candidate"] for r in rows]==list(range(len(rows))),"Invalid tuning candidates")
                    require(all(len(r["ms"])==3 and all(np.isfinite(v) and v>0 for v in r["ms"]) for r in rows),"Invalid tuning observations")
                if shape==(128,128,128) or kind=="real" and shape[0]==16:
                    witness=case["witness"];path=ROOT/"results"/witness["file"]
                    require(witness["file"]==f"outlier-witness-{seed}-{profile}.npz" and witness["array"]==case["label"],"Unexpected correction witness")
                    require(sha256(path)==witness["sha256"],"Correction witness hash differs")
                    with np.load(path,allow_pickle=False) as arrays:
                        np.testing.assert_array_equal(arrays["a"],a);np.testing.assert_array_equal(arrays["b"],b)
                        actual=arrays[witness["array"]]
                        require(digest(actual)==case["checks"]["output_sha256"],"Correction witness output changed")
                        if method:check_output(actual,oracle(a,b,method,case["budget"],case["fraction"]))
                        else:check_product(actual,*values)
                        original=a.astype(np.float64)@b.astype(np.float64).T
                        require(np.isclose(relative_error(actual,original),case["checks"]["relative_frobenius_error"],rtol=2e-6,atol=1e-12),"Correction witness error differs")
                        replayed+=actual.size
                outputs+=shape[0]*shape[1]
        require(seen==expected,"Correction grid incomplete")
    require(identities[0]==identities[1],"Correction sources/binary changed across cohorts")
    print(f"Outliers: 598 cases, {outputs:,} outputs checked before/after, {replayed:,} CPU-replayed entries; all factors, selection, sources and real activations checked")


if __name__=="__main__":verify()
