"""CPU-only complete-model witnesses, attention arithmetic and full quality replay."""
import json

import numpy as np

from attention_reference import attention,encoding,network
from attention_validation import validate_arrays
from prepare_digits import load_digits
from reference import digest,relative_error
from verify_attention import CONFIGS,close
from verify_followups import ROOT
from verify_results import require,sha256

SOURCES={"attention_application.py","attention_model.py","attention_native.py","attention_reference.py","attention_validation.py","attention_study.py",
         "prepare_digits.py","reference.py","native/attention_bridge.cu","native/bridge.cu","native/optimized_bridge.cu","build_attention.sh","sage_reference.py"}


def verify():
    x,y=load_digits();identities=[];count=quality_count=witnesses=0
    labels={f"native-{a}-{b}" for a,b in CONFIGS}|{f"{backend}-{mode}" for backend in ("math","flash","efficient","cudnn") for mode in ("fp32","fp16","bf16")}|{"sage3-fp16","sage3-bf16"}
    for suffix,seed,reverse in (("a",20261123,False),("b",20261223,True)):
        record=json.loads((ROOT/f"results/attention-application-{suffix}.json").read_text(encoding="utf-8"))
        require(record.get("complete") is True and not record["quick"] and (record["model_seed"],record["reverse"])==(seed,reverse),"Incomplete complete-model attention cohort")
        require((record["samples"],record["warmups"])==(30,5) and record["environment"]["gpu_exclusive"] is False,"Model attention protocol changed")
        require(set(record["source_sha256"])==SOURCES and set(record["data_sha256"])=={"fixtures/digits.csv.gz",f"fixtures/attention-model-{seed}.npz",f"fixtures/attention-model-reference-{seed}.npz"},"Model attention source/data coverage changed")
        for key in ("source_sha256","data_sha256"):
            for name,expected in record[key].items():require(sha256(ROOT/name)==expected,f"Executed complete-model provenance changed: {name}")
        identities.append((record["source_sha256"],record["binary_sha256"]))
        core=json.loads((ROOT/f"results/attention-cohort-{suffix}.json").read_text(encoding="utf-8"));require(core["binary_sha256"]==record["binary_sha256"],"Model used another native binary")
        require(record["sage_build_sha256"]==sha256(ROOT/"results/sage-build.json"),"Model used another SageAttention3 build")
        seen=set();available=set();expected={(label,batch) for label in labels for batch in (4,16,32)}
        for row in record["unavailable"]:
            key=row["label"],row["batch"];require(key in expected and key not in seen and row["diagnostic"],"Invalid model backend rejection");seen.add(key)
        with np.load(ROOT/f"fixtures/attention-model-{seed}.npz",allow_pickle=False) as model,np.load(ROOT/f"fixtures/attention-model-reference-{seed}.npz",allow_pickle=False) as original:
            weights={name:model[name] for name in model.files if name not in ("train","validation","test")}
            for c in record["cases"]:
                label,batch=c["label"],c["batch"];key=label,batch;require(key in expected and key not in seen,"Unexpected model attention case");seen.add(key);available.add(key)
                require(c["graph_repetitions"]==4 and c["weight_bytes"]==sum(w.nbytes for w in weights.values()) and c["setup_and_tuning_wall_ms"]>0,"Model setup/weight boundary changed")
                require(set(c["timings_ms"])=={"captured_device","uncaptured_resident","host_request"},"Complete-model timing boundary missing")
                for observations in c["timings_ms"].values():require(len(observations)==30 and all(v>0 and np.isfinite(v) for v in observations),"Invalid complete-model timings")
                require(c["extra_peak_torch_allocated_bytes"]>=0 and c["post_timing_logits_sha256"]==c["checks"]["logits_sha256"],"Model timed output/allocator result changed")
                require(c["post_request_checks"]["logits_sha256"]==c["checks"]["logits_sha256"] and c["post_request_checks"]["predictions_sha256"]==c["checks"]["predictions_sha256"],"Model changing requests corrupted repeated input")
                require(len(c["request_checks"])==4,"Missing changing-image requests")
                for row,offset in zip(c["request_checks"],(0,29,83,157)):
                    ids=model["test"][(np.arange(batch)+offset)%len(model["test"])];require(row["input_sha256"]==digest(x[ids]),"Model request input changed")
                if c["engine"]=="native":
                    require((c["qk_mode"],c["pv_mode"]) in CONFIGS and label==f"native-{c['qk_mode']}-{c['pv_mode']}","Model native modes changed")
                    require(c["owned_native_device_buffer_bytes"]>0 and len(c["tuning"])==2 and len(c["selected_algorithms"])==2,"Missing model native resources/tuning")
                if batch==4:
                    witness=c["witness"];path=ROOT/"results"/witness["file"];require(witness["file"]==f"attention-model-witness-{seed}-{label}.npz" and sha256(path)==witness["sha256"],"Model numerical witness changed")
                    with np.load(path,allow_pickle=False) as data:
                        ids=model["test"][np.arange(batch)%len(model["test"])];np.testing.assert_array_equal(data["input"],x[ids])
                        logits=data["logits"];require(digest(logits)==c["checks"]["logits_sha256"],"Model witness logits changed")
                        predictions=logits.argmax(axis=1);np.testing.assert_array_equal(data["predictions"],predictions);require(digest(predictions)==c["checks"]["predictions_sha256"],"Model witness labels changed")
                        def injected(q,k,v):
                            for name,value in zip(("q","k","v"),(q,k,v)):np.testing.assert_allclose(data[name],value,rtol=2e-4,atol=2e-4)
                            return data["attended"].astype(np.float64)
                        expected_logits=network(data["input"],weights,injected);np.testing.assert_allclose(logits,expected_logits,rtol=2e-4,atol=2e-4)
                        close(relative_error(logits,expected_logits),c["checks"]["logit_arithmetic_relative_error"],"Model arithmetic error changed")
                        if c["engine"]=="native":
                            checks=validate_arrays(data["q"],data["k"],data["v"],(c["qk_mode"],c["pv_mode"]),"none",data["scores"],data["probabilities"],data["padded_output"],data["attended"])
                            require(checks["encoded_sha256"]==c["checks"]["attention"]["encoded_sha256"] and checks["output_sha256"]==c["checks"]["attention"]["output_sha256"],"Model attention encodings/output changed")
                        elif c["engine"]=="sage3":
                            close(relative_error(data["attended"],attention(data["q"],data["k"],data["v"])),c["checks"]["attention"]["relative_frobenius_error"],"Specialized FP4 model error changed")
                        else:
                            dtype=label.split("-")[-1];rounded=[encoding(data[name],dtype)[2] for name in ("q","k","v")]
                            require(relative_error(data["attended"],attention(*rounded))<={"fp32":5e-5,"fp16":.005,"bf16":.03}[dtype],"Complete model fused attention arithmetic failed")
                        witnesses+=1
                count+=1
            require(seen==expected,"Model batch/method attempts incomplete")
            for label in {f"native-{a}-{b}" for a,b in CONFIGS}|{"math-fp32","efficient-fp32","flash-fp16","cudnn-fp16","sage3-fp16","sage3-bf16"}:
                require(all((label,batch) in available for batch in (4,16,32)),"Required full-model reference missing")
            quality_seen=set()
            for row in record["quality"]:
                label=row["label"];require(label not in quality_seen and (label,16) in available and row["batch"]==16,"Unexpected quality case");quality_seen.add(label)
                path=ROOT/"results"/row["witness"]["file"];require(sha256(path)==row["witness"]["sha256"],"Full quality witness changed")
                with np.load(path,allow_pickle=False) as data:
                    for split in ("validation","test"):
                        ids=model[split];logits=data[f"{split}_logits"];predictions=data[f"{split}_predictions"];q=row["splits"][split]
                        require(logits.shape==(len(ids),10) and np.isfinite(logits).all(),"Quality inference grid incomplete");np.testing.assert_array_equal(logits.argmax(axis=1),predictions)
                        require((q["count"],q["correct"],q["logits_sha256"],q["predictions_sha256"])==(len(ids),int(np.sum(predictions==y[ids])),digest(logits),digest(predictions)),"Quality statistics changed")
                        require(q["changed_predictions_vs_original"]==int(np.sum(predictions!=original[f"{split}_predictions"])),"Quality reference changed")
                        close(relative_error(logits,original[f"{split}_logits"]),q["logit_relative_error_vs_original"],"Complete quality logit error changed");quality_count+=len(ids)
            require(quality_seen=={label for label,batch in available if batch==16},"Available complete-model quality missing")
    require(identities[0]==identities[1],"Model source/binary changed between cohorts")
    print(f"Attention models: {count} complete inference configurations, {quality_count:,} quality predictions, {witnesses} full numerical witnesses")


if __name__=="__main__":verify()
