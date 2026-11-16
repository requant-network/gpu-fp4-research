"""CPU replay of training-only error calibration, held-out decisions and witnesses."""
import json

import numpy as np

from precision_reference import exact_product,exact_samples,gold
from reference import MODES,digest,quantize,relative_error
from selector_reference import BUDGETS,HELD_SHAPES,PROFILES,SAMPLE_COUNT,TRAIN_SHAPES,choose,dataset,feature,fit_policy,probe_reference,sample_indices
from verify_followups import ROOT
from verify_results import require,sha256

SOURCES={"selector_study.py","selector_reference.py","native/selector_bridge.cu","build_selector.sh",
         "reference.py","gpu_study.py","followup.py","precision_study.py","precision_reference.py",
         "native/bridge.cu","native/optimized_bridge.cu","native/precision_bridge.cu","build_precision.sh"}


def close(actual,expected,message):
    require(np.isclose(actual,expected,rtol=2e-12,atol=1e-30),message)


def factors(a,b,mode):
    hashes={};values=[]
    for operand,x in enumerate((a,b)):
        if mode=="fp64":raw=x.astype(np.float64);q=raw;scales=np.empty(0,np.uint8)
        else:raw,scales,q=quantize(x,mode)
        hashes[f"operand_{operand}_encoded"]=digest(raw)
        if mode=="fp4":hashes[f"operand_{operand}_scales"]=digest(scales)
        values.append(q.astype(np.float64))
    return hashes,values


def checks(check,shape,hashes,exact):
    require(check["prepared_sha256"]==hashes and check["checked_output_entries"]==shape[0]*shape[1],"Incorrect selector preparation/coverage")
    require(0<=check["arithmetic_max_bound_ratio"]<=1,"Failed selector arithmetic")
    require(np.isfinite(check["relative_frobenius_error"]) and check["relative_frobenius_error"]>=0,"Invalid selector error")
    values=np.array(check["exact_sample_values"],np.float64)
    require(values.shape==(SAMPLE_COUNT,) and np.isfinite(values).all(),"Invalid exact selector samples")
    close(relative_error(values,exact),check["exact_sample_relative_error"],"Exact selector sample error changed")


def check_probe(probe,a,b,mode,indices):
    values=np.array(probe,np.float64);expected=probe_reference(a,b,mode,indices)
    original_abs=np.array([np.dot(np.abs(a[i].astype(np.float64)),np.abs(b[j].astype(np.float64))) for i,j in indices])
    gamma=(a.shape[1]+128)*2**-53/(1-(a.shape[1]+128)*2**-53)
    bound=8*gamma*np.column_stack((original_abs,expected[:,2],expected[:,2]))+np.finfo(np.float64).tiny
    require(values.shape==expected.shape and np.isfinite(values).all() and np.all(np.abs(values-expected)<=bound),"Selector probe changed")


def replay(actual,values,target,mode,check,indices):
    x,y=values;k=x.shape[1];u=2**-53 if mode=="fp64" else 2**-24;gamma=(k+20)*u/(1-(k+20)*u)
    bound=8*gamma*(np.abs(x)@np.abs(y).T)+np.finfo(np.float64 if mode=="fp64" else np.float32).tiny*8*k
    require(actual.shape==target.shape and np.isfinite(actual).all() and np.all(np.abs(actual-x@y.T)<=bound),"Selector witness arithmetic failed")
    require(digest(actual)==check["output_sha256"],"Selector witness output changed")
    np.testing.assert_array_equal(actual[indices[:,0],indices[:,1]],np.array(check["exact_sample_values"]))
    close(relative_error(actual,target),check["relative_frobenius_error"],"Selector witness full error changed")
    return actual.size


def decision(row,policy,shape):
    selected,estimates,nearest=choose(policy,shape,row["probe"],row["maximum_relative_error"])
    require(row["nearest_training_shape"]==nearest,"Held-out cost model changed")
    for mode in MODES:close(row["estimates"][mode],estimates[mode],"Held-out error prediction changed")
    return selected


def transitions(record,policy,seed):
    probe=record["transition_probe"];witness=probe["witness"];path=ROOT/"results"/witness["file"]
    require(witness["file"]==f"selector-transitions-{seed}.npz" and sha256(path)==witness["sha256"],"Changing-input witness changed")
    require(len(probe["rows"])==16,"Missing input/budget transitions");entries=0;seen=set();shape=(16,128,128)
    with np.load(path,allow_pickle=False) as arrays:
        _,b=dataset(seed,shape,"normal");np.testing.assert_array_equal(arrays["b"],b)
        indices=sample_indices(seed,shape);np.testing.assert_array_equal(arrays["indices"],indices)
        for row in probe["rows"]:
            index,budget=row["index"],row["maximum_relative_error"];require(index in range(4) and budget in BUDGETS and (index,budget) not in seen,"Duplicate transition")
            seen.add((index,budget));profile=("normal","cancellation","spread","normal")[index];require(row["profile"]==profile,"Transition profile changed")
            a,_=dataset(seed+index,shape,profile);np.testing.assert_array_equal(arrays[f"a{index}"],a)
            exact=exact_product(a,b);np.testing.assert_array_equal(arrays[f"exact{index}"],exact)
            for mode in MODES:check_probe(row["probe"][mode],a,b,mode,indices)
            selected=decision(row,policy,shape);require(row["selected"]==selected,"Changing-input choice changed")
            hashes,values=factors(a,b,selected);checks(row["checks"],shape,hashes,exact[indices[:,0],indices[:,1]])
            entries+=replay(arrays[row["array"]],values,exact,selected,row["checks"],indices)
    return entries


def verify():
    calibration=json.loads((ROOT/"results/selector-calibration.json").read_text(encoding="utf-8"))
    policy=json.loads((ROOT/"results/selector-policy.json").read_text(encoding="utf-8"));expected_policy=fit_policy(calibration)
    for mode in MODES:close(policy["multipliers"][mode],expected_policy["multipliers"][mode],"Training-only error calibration changed")
    require({k:v for k,v in policy.items() if k!="multipliers"}=={k:v for k,v in expected_policy.items() if k!="multipliers"},"Training-only cost calibration changed")
    identities=[];total=outputs=witness_entries=selection_count=violations=0
    for filename,phase,seed,model_seed,reverse in (("selector-calibration.json","calibration",20261122,20261116,False),
            ("selector-cohort-a.json","evaluation",20261122,20261116,False),("selector-cohort-b.json","evaluation",20261222,20261216,True)):
        record=json.loads((ROOT/"results"/filename).read_text(encoding="utf-8"))
        require(record.get("complete") is True and not record["quick"] and (record["phase"],record["seed"],record["model_seed"],record["reverse"])==(phase,seed,model_seed,reverse),"Incomplete selector cohort")
        require(record["samples"]==30 and record["warmups"]==5 and record["probe_samples"]==64 and record["negative_controls"]==4,"Selector protocol changed")
        require(record["environment"]["gpu_exclusive"] is False and set(record["source_sha256"])==SOURCES,"Selector provenance/qualification missing")
        require(set(record["data_sha256"])=={f"fixtures/{prefix}-{model_seed}.npz" for prefix in ("digits-model","outlier-activations")},"Selector data coverage missing")
        for key in ("source_sha256","data_sha256"):
            for name,expected in record[key].items():require(sha256(ROOT/name)==expected,f"Selector executed provenance changed: {name}")
        identities.append((record["source_sha256"],record["binary_sha256"]))
        if phase=="calibration":require(record["policy"] is None and record["policy_sha256"] is None and not record["selections"],"Calibration used evaluation policy")
        else:
            require(record["policy"]==policy and record["policy_sha256"]==sha256(ROOT/"results/selector-policy.json"),"Evaluation policy changed")
            require(calibration["utc"]<record["utc"],"Policy was not fitted before evaluation");witness_entries+=transitions(record,policy,seed)
        shapes=TRAIN_SHAPES if phase=="calibration" else TRAIN_SHAPES+HELD_SHAPES;profiles=PROFILES[:2] if phase=="calibration" else PROFILES
        groups={(shape,profile,"synthetic") for shape in shapes for profile in profiles}
        if phase=="evaluation":groups|={((batch,256,128),"digits-input","real") for batch in (64,192)}|{((batch,128,256),"digits-hidden","real") for batch in (64,192)}
        expected={(shape,profile,kind,mode) for shape,profile,kind in groups for mode in (*MODES,"fp64")};seen=set();cache={};case_map={}
        with np.load(ROOT/f"fixtures/digits-model-{model_seed}.npz",allow_pickle=False) as model,np.load(ROOT/f"fixtures/outlier-activations-{model_seed}.npz",allow_pickle=False) as activations:
            for case in record["cases"]:
                shape,profile,kind,mode=tuple(case["shape"]),case["profile"],case["kind"],case["mode"];key=shape,profile,kind,mode
                require(key in expected and key not in seen,"Unexpected selector reference case");seen.add(key);case_map[key]=case
                group=shape,profile,kind
                if group not in cache:
                    a,b=dataset(seed,shape,profile) if kind=="synthetic" else (activations["x" if profile=="digits-input" else "hidden"][:shape[0]],model["w1" if profile=="digits-input" else "w2"])
                    indices=sample_indices(seed,shape);cache[group]=(a,b,indices,exact_samples(a,b,indices))
                a,b,indices,exact=cache[group];require(case["input_sha256"]==[digest(a),digest(b)],"Selector inputs changed")
                hashes,values=factors(a,b,mode)
                for check in (case["checks"],case["post_timing_checks"],case["post_request_checks"]):checks(check,shape,hashes,exact)
                if mode!="fp64":check_probe(case["probe"],a,b,mode,indices);close(case["feature"],feature(case["probe"],shape[2]),"Probe feature changed")
                else:require(case["probe"] is None and case["feature"] is None,"Unexpected FP64 probe")
                require(case["graph_repetitions"]==(1 if shape[0]*shape[1]*shape[2]>=2**29 else 8),"Selector graph repetitions changed")
                require(set(case["timings_ms"])=={"prepacked","online-a","online-both"} and case["setup_and_initial_preparation_wall_ms"]>0,"Missing selector cost boundary")
                for observations in list(case["timings_ms"].values())+[case["host_request_ms"]]:require(len(observations)==30 and all(np.isfinite(v) and v>0 for v in observations),"Invalid selector timings")
                tuning=case["tuning"][0] if mode=="fp64" else case["tuning"]
                require(1<=len(tuning)<=8 and [r["candidate"] for r in tuning]==list(range(len(tuning))),"Missing selector tuning")
                require(all(len(r["ms"])==3 and all(np.isfinite(v) and v>0 for v in r["ms"]) for r in tuning),"Invalid selector tuning")
                if shape==(128,128,128):
                    witness=case["witness"];path=ROOT/"results"/witness["file"]
                    require(witness["file"]==f"selector-{phase}-witness-{seed}-{profile}.npz" and sha256(path)==witness["sha256"] and witness["array"]==mode,"Selector witness changed")
                    with np.load(path,allow_pickle=False) as arrays:
                        for name,value in (("a",a),("b",b),("indices",indices),("exact_samples",exact),("exact",exact_product(a,b))):np.testing.assert_array_equal(arrays[name],value)
                        witness_entries+=replay(arrays[mode],values,gold(a,b,profile),mode,case["checks"],indices)
                total+=1;outputs+=shape[0]*shape[1]
        require(seen==expected,"Selector reference grid incomplete")
        selected_seen=set()
        for row in record["selections"]:
            shape,profile,kind=tuple(row["shape"]),row["profile"],row["kind"];group=shape,profile,kind;budget=row["maximum_relative_error"]
            require(group in groups and budget in BUDGETS and (*group,budget) not in selected_seen,"Unexpected selector decision");selected_seen.add((*group,budget))
            selected=decision(row,policy,shape);require(row["selected"]==[selected]*30,"Evaluation choice does not follow frozen policy")
            for mode in MODES:np.testing.assert_array_equal(np.array(row["probe"][mode]),np.array(case_map[*group,mode]["probe"]))
            a,b,indices,exact=cache[group];hashes,values=factors(a,b,selected);checks(row["checks"],shape,hashes,exact)
            require(row["budget_met"]==(max(row["checks"]["relative_frobenius_error"],row["checks"]["exact_sample_relative_error"])<=budget),"Budget outcome changed")
            require(len(row["host_request_ms"])==30 and all(np.isfinite(v) and v>0 for v in row["host_request_ms"]),"Missing complete selection timings")
            if shape==(128,128,128):
                witness=row["witness"];path=ROOT/"results"/witness["file"];require(sha256(path)==witness["sha256"],"Selected output witness changed")
                with np.load(path,allow_pickle=False) as arrays:witness_entries+=replay(arrays[witness["array"]],values,gold(a,b,profile),selected,row["checks"],indices)
            selection_count+=1;violations+=not row["budget_met"];outputs+=shape[0]*shape[1]
        if phase=="evaluation":require(selected_seen=={(*group,budget) for group in groups for budget in BUDGETS},"Selector evaluation grid incomplete")
    require(identities[0]==identities[1]==identities[2],"Selector sources/binaries changed between calibration and evaluation")
    print(f"Selector: {total} reference cases, {selection_count} decisions ({violations} budget violations retained), {outputs:,} output entries, {witness_entries:,} CPU-replayed witness entries")


if __name__=="__main__":verify()
