"""CPU-only replay of additive encoder/fusion/tile studies and real-data models."""
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_digits import DATA_SHA, forward, load_digits
from reference import SHAPES, digest, inputs, quantize, relative_error
from verify_results import require, sha256

ROOT=Path(__file__).resolve().parent
HELD_OUT=((32,768,1280),(192,768,768),(384,1536,512),(768,1536,1024))
PROFILES=("normal","outliers","tiny","ties")
BOUNDARIES={"prepare-a","prepare-both","prepacked","online-a","online-both"}
SOURCES={"reference.py","gpu_study.py","followup.py","encoder_study.py","native/bridge.cu",
         "native/optimized_bridge.cu","build.sh","build_extended.sh"}


def regenerated(seed,shape,profile):
    a,b=inputs(seed,shape,"outliers" if profile=="outliers" else "normal")
    if profile=="tiny":
        return a*np.float32(2**-10),b*np.float32(2**-10)
    if profile=="ties":
        levels=np.array([-6,-5,-3.5,-2.5,-1.75,-1.25,-.75,-.25,0,.25,.75,1.25,1.75,2.5,5,6],np.float32)
        a=np.resize(levels,shape[:1]+shape[2:])
        b=np.resize(np.roll(levels,5),(shape[1],shape[2]))
        a[0]=0
        b[0]=0
    return a,b


def expected_keys():
    keys=set()
    for shape in SHAPES+HELD_OUT+((2048,2048,2048),):
        for profile in PROFILES:
            if shape==(2048,2048,2048) and profile not in ("normal","outliers"):
                continue
            for variant in range(5):
                keys.add((shape,profile,128,variant,False,False))
            keys.add((shape,profile,256,3,False,False))
            if profile in ("normal","outliers"):
                keys.add((shape,profile,128,3,True,False))
                keys.add((shape,profile,128,3,True,True))
    return keys


def verify_encoders():
    checked_entries,witness_entries=0,0
    identities=[]
    for name,seed,reverse in (("encoder-cohort-a.json",20261117,False),("encoder-cohort-b.json",20261217,True)):
        path=ROOT/"results"/name
        record=json.loads(path.read_text(encoding="utf-8"))
        require(record.get("complete") is True and not record["quick"],"Incomplete encoder campaign")
        require(record["seed"]==seed and record["reverse"]==reverse,"Encoder replication mismatch")
        require((record["warmups"],record["samples"],record["graph_repetitions"])==(5,30,32),"Encoder timing protocol changed")
        require(record["negative_controls"]==5,"Missing native encoder controls")
        require(record["environment"]["gpu_exclusive"] is False,"Missing shared-system qualification")
        require(set(record["source_sha256"])==SOURCES,"Encoder source coverage changed")
        for source,expected in record["source_sha256"].items():
            require(sha256(ROOT/source)==expected,f"Encoder executed source changed: {source}")
        base=(ROOT/"native/bridge.cu").read_text(encoding="utf-8")
        for tile in (128,256):
            generated=base.replace('using ThreadBlockShape    = Shape<_128,_128,_128>;',
                                   f'using ThreadBlockShape    = Shape<_{tile},_128,_128>;')
            require(hashlib.sha256(generated.encode()).hexdigest()==record["generated_source_sha256"][str(tile)],
                    "Generated native source differs from declared transformation")
        identities.append((record["source_sha256"],record["binary_sha256"],record["generated_source_sha256"]))
        seen,cache=set(),{}
        for case in record["cases"]:
            shape,profile=tuple(case["shape"]),case["profile"]
            key=(shape,profile,case["tile_m"],case["variant"],bool(case["activation"]),case["fused"])
            require(key in expected_keys() and key not in seen,"Unexpected/duplicate encoder configuration")
            seen.add(key)
            require(case["partition"]==("held-out" if shape in HELD_OUT else "training"),"Held-out split changed")
            group=(shape,profile)
            if cache.get("group")!=group:
                a,b=regenerated(seed,shape,profile)
                bias=(np.arange(shape[2],dtype=np.int32)%33-16).astype(np.float32)/np.float32(64)
                cache={"group":group,"a":a,"b":b,"bias":bias,"variants":{}}
            a,b=cache["a"],cache["b"]
            require(case["input_sha256"]==[digest(a),digest(b)],"Encoder input regeneration failed")
            active=bool(case["activation"])
            if active not in cache["variants"]:
                aa=np.maximum(a+cache["bias"],np.float32(0)) if active else a
                quantized=[]
                hashes={}
                for operand,original in enumerate((aa,b)):
                    raw,scales,values=quantize(original,"fp4")
                    hashes[f"operand_{operand}_encoded"]=digest(raw)
                    hashes[f"operand_{operand}_scales"]=digest(scales)
                    quantized.append(values)
                cache["variants"][active]=(hashes,quantized,aa)
            hashes,quantized,aa=cache["variants"][active]
            for checks in (case["checks"],case["post_timing_checks"]):
                require(checks["encoded_sha256"]==hashes,"Encoder packed bytes/scales differ")
                require(0<=checks["arithmetic_max_bound_ratio"]<=1,"Encoder arithmetic bound failed")
                require(checks["checked_output_entries"]==shape[0]*shape[1],"Wrong checked output count")
                require(np.isfinite(checks["relative_frobenius_error"]) and checks["relative_frobenius_error"]>=0,
                        "Invalid encoder numerical error")
            require(set(case["timings_ms"])==BOUNDARIES,"Encoder timing boundaries incomplete")
            for observations in case["timings_ms"].values():
                require(len(observations)==30 and all(np.isfinite(v) and v>0 for v in observations),"Bad encoder observations")
            if shape==(128,128,128):
                witness=case["witness"]
                require(witness["file"]==f"encoder-witness-{seed}-{profile}.npz","Unexpected encoder witness path")
                witness_path=path.parent/witness["file"]
                require(sha256(witness_path)==witness["sha256"],"Encoder witness hash mismatch")
                with np.load(witness_path,allow_pickle=False) as arrays:
                    np.testing.assert_array_equal(arrays["a"],a)
                    np.testing.assert_array_equal(arrays["b"],b)
                    np.testing.assert_array_equal(arrays["bias"],cache["bias"])
                    actual=arrays[witness["array"]]
                    require(digest(actual)==case["checks"]["output_sha256"],"Encoder witness output mismatch")
                    x,y=(q.astype(np.float64) for q in quantized)
                    expected=x@y.T
                    gamma=shape[2]*2**-24/(1-shape[2]*2**-24)
                    bound=8*gamma*(np.abs(x)@np.abs(y).T)+np.finfo(np.float32).tiny
                    require(np.isfinite(actual).all() and np.all(np.abs(actual-expected)<=bound),"Encoder witness arithmetic failed")
                    original=aa.astype(np.float64)@b.astype(np.float64).T
                    require(np.isclose(relative_error(actual,original),case["checks"]["relative_frobenius_error"],rtol=2e-6,atol=1e-12),
                            "Encoder witness error metric mismatch")
                    witness_entries+=actual.size
            checked_entries+=shape[0]*shape[1]
        require(seen==expected_keys(),"Encoder case grid incomplete")
    require(identities[0]==identities[1],"Encoder source/binary changed across cohorts")
    print(f"Encoder: 592 cases, {checked_entries:,} runtime-checked outputs, {witness_entries:,} CPU-replayed output entries")


def verify_models():
    x,y=load_digits()
    for seed in (20261116,20261216):
        record=json.loads((ROOT/f"results/digits-training-{seed}.json").read_text(encoding="utf-8"))
        require(record["complete"] and record["epochs"]==80 and record["dataset_sha256"]==DATA_SHA,"Wrong model training protocol")
        for name,expected in record["source_sha256"].items():
            require(sha256(ROOT/name)==expected,"Training source changed")
        require(record["model_file"]==f"digits-model-{seed}.npz","Unexpected model filename")
        path=ROOT/"fixtures"/record["model_file"]
        require(sha256(path)==record["model_sha256"],"Model hash mismatch")
        with np.load(path,allow_pickle=False) as arrays:
            parameters=[arrays[name] for name in ("w1","b1","w2","b2")]
            require([p.shape for p in parameters]==[(256,128),(256,),(128,256),(128,)],"Wrong model dimensions")
            require(all(p.dtype==np.float32 and np.isfinite(p).all() for p in parameters),"Invalid model parameters")
            indices=np.concatenate([arrays[name] for name in ("train","validation","test")])
            np.testing.assert_array_equal(np.sort(indices),np.arange(1797))
            for name in ("train","validation","test"):
                ids=arrays[name]
                require(digest(ids)==record["split_indices_sha256"][name] and len(ids)==record["split_sizes"][name],"Model split changed")
                accuracy=float(np.mean(forward(x[ids],*parameters).argmax(axis=1)==y[ids]))
                require(accuracy==record["accuracy"][name] and accuracy>=.9,"Frozen model accuracy changed")
    print("Models: two independent, disjoint-split digits classifiers replayed on CPU")


if __name__=="__main__":
    verify_encoders()
    verify_models()
