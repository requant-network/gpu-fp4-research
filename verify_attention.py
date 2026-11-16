"""CPU-only replay of native attention, fused controls and retained witnesses."""
import json

import numpy as np

from attention_reference import MASKS,allowed,arrays,attention,encoding,padded
from attention_validation import validate_arrays
from reference import digest,relative_error
from verify_followups import ROOT
from verify_results import require,sha256

SHAPES=((1,1,128,64),(2,4,512,64),(1,4,1024,128),(1,2,256,32),(1,2,193,48),(1,1,768,96))
CONFIGS=tuple((a,b) for a in ("fp4","fp8","fp16") for b in ("fp4","fp8","fp16"))+(("bf16","bf16"),("fp32","fp32"))
SOURCES={"attention_study.py","attention_reference.py","attention_native.py","attention_validation.py","reference.py",
         "native/attention_bridge.cu","native/bridge.cu","native/optimized_bridge.cu","build_attention.sh"}


def close(a,b,message):require(np.isclose(a,b,rtol=2e-7,atol=1e-12),message)


def sample_check(check,target,indices):
    values=np.array(check["sample_values"],np.float64)
    require(values.shape==(64,) and np.isfinite(values).all(),"Missing attention output samples")
    close(relative_error(values,target.ravel()[indices]),check["sample_relative_error"],"Attention sample error changed")
    require(np.isfinite(check["relative_frobenius_error"]) and check["relative_frobenius_error"]>=0,"Invalid attention error")


def native_check(check,shape,target,indices,hashes,pad_hashes):
    batch,heads,length,depth=shape;sp=(length+127)//128*128;dp=(depth+127)//128*128;sample_check(check,target,indices)
    require(0<=check["qk_max_bound_ratio"]<=1 and 0<=check["pv_max_bound_ratio"]<=1 and 0<=check["softmax_max_absolute_error"]<=4e-6,"Attention arithmetic screen failed")
    require(check["checked_output_entries"]==batch*heads*(2*sp*sp+sp*dp+length*depth),"Native attention output coverage changed")
    for name in ("q","k","v"):require(check["encoded_sha256"][name]==hashes[name],"Attention source encoding changed")
    snapshot=check["snapshot_sha256"]
    for index,expected in enumerate(pad_hashes):require(snapshot[f"buffer{index}"]==expected,"Attention padding/transposition changed")
    require(snapshot["output"]==check["output_sha256"] and snapshot["buffer3"]==check["scores_sha256"] and snapshot["buffer4"]==check["probabilities_sha256"] and snapshot["buffer5"]==check["padded_output_sha256"],"Native attention snapshot changed")
    for stage,names in enumerate((("q","k"),("p","v"))):
        for operand,name in enumerate(names):
            require(snapshot[f"stage{stage}-buffer{operand}"]==check["encoded_sha256"][name][0] and snapshot[f"stage{stage}-buffer{operand+2}"]==check["encoded_sha256"][name][1],"Attention encoded snapshot changed")


def verify_preflight():
    record=json.loads((ROOT/"results/attention-preflight.json").read_text(encoding="utf-8"));require(record.get("complete") is True and len(record["cases"])==144,"Incomplete forced-backend preflight")
    require(set(record["source_sha256"])=={"attention_preflight.py","reference.py"},"Preflight source coverage changed")
    for name,expected in record["source_sha256"].items():require(sha256(ROOT/name)==expected,"Executed backend preflight changed")
    seen=set();witnesses=0
    for c in record["cases"]:
        shape=tuple(c["shape"]);key=(shape,c["mask"],c["dtype"],c["requested_backend"])
        require(shape in SHAPES[:4] and c["mask"] in MASKS and c["dtype"] in ("fp32","fp16","bf16") and c["requested_backend"] in ("math","flash","efficient","cudnn") and key not in seen,"Unexpected preflight case");seen.add(key)
        if c["status"]=="unavailable":require(bool(c["diagnostic"]),"Missing preflight rejection");continue
        require(c["status"]=="available" and len(c["timings_ms"])==5 and all(v>0 and np.isfinite(v) for v in c["timings_ms"]),"Invalid preflight observations")
        require(c["dispatch_operations"] and c["extra_peak_torch_allocated_bytes"]>=0,"Missing native dispatch/allocator observations")
        if "witness" in c:
            path=ROOT/"results"/c["witness"]["file"];require(sha256(path)==c["witness"]["sha256"],"Preflight witness changed")
            host=arrays(record["seed"],shape)
            with np.load(path,allow_pickle=False) as data:
                for name,x in zip(("q","k","v"),host):np.testing.assert_array_equal(data[name],x)
                np.testing.assert_array_equal(data["allowed"],allowed(shape[2],c["mask"]))
                actual=data["output"];require(digest(actual)==c["output_sha256"],"Preflight output changed")
                rounded=[encoding(x,c["dtype"])[2] for x in host]
                close(relative_error(actual,attention(*host,c["mask"])),c["relative_error_vs_original_fp64"],"Preflight full error changed")
                close(relative_error(actual,attention(*rounded,c["mask"])),c["relative_error_vs_rounded_input_fp64"],"Preflight arithmetic error changed");witnesses+=1
    require(len(seen)==144 and witnesses>0,"Preflight grid/witness coverage incomplete")
    return witnesses


def verify():
    preflight_witnesses=verify_preflight();identities=[];count=outputs=witness_entries=missing=0
    for suffix,seed,reverse in (("a",20261123,False),("b",20261223,True)):
        record=json.loads((ROOT/f"results/attention-cohort-{suffix}.json").read_text(encoding="utf-8"))
        require(record.get("complete") is True and not record["quick"] and (record["seed"],record["reverse"])==(seed,reverse),"Incomplete attention cohort")
        require((record["samples"],record["warmups"],record["negative_controls"])==(30,5,6) and record["environment"]["gpu_exclusive"] is False,"Attention protocol/qualification changed")
        require(set(record["source_sha256"])==SOURCES,"Attention executed source coverage changed")
        for name,expected in record["source_sha256"].items():require(sha256(ROOT/name)==expected,f"Executed attention source changed: {name}")
        identities.append((record["source_sha256"],record["binary_sha256"]))
        labels={f"native-{a}-{b}" for a,b in CONFIGS}|{f"{engine}-{mode}" for engine in ("math","flash","efficient","cudnn") for mode in ("fp32","fp16","bf16")}
        groups={(shape,kind,profile) for shape in SHAPES for kind in MASKS for profile in ("normal","peaky")};expected={(*group,label) for group in groups for label in labels};seen=set();available=set();cache={}
        for c in record["unavailable"]:
            key=(tuple(c["shape"]),c["mask"],c["profile"],c["label"]);require(key in expected and key not in seen and c["diagnostic"],"Invalid attention rejection");seen.add(key);missing+=1
        for c in record["cases"]:
            shape,kind,profile=tuple(c["shape"]),c["mask"],c["profile"];group=shape,kind,profile;key=(*group,c["label"])
            require(key in expected and key not in seen,"Unexpected attention case");seen.add(key);available.add(key)
            if cache.get("group")!=group:
                host=arrays(seed,shape)
                if profile=="peaky":host=host[0]*np.float32(4),host[1]*np.float32(4),host[2]
                target=attention(*host,kind);rng=np.random.default_rng(np.random.SeedSequence([seed,*shape,271]));indices=rng.integers(target.size,size=64)
                pad=padded(*host);cache={"group":group,"host":host,"target":target,"indices":indices,"pad":pad,"encodings":{}}
            host,target,indices,pad=cache["host"],cache["target"],cache["indices"],cache["pad"]
            require(c["input_sha256"]==[digest(x) for x in host] and c["partition"]==("held-out" if shape in SHAPES[-2:] else "training"),"Attention inputs/split changed")
            require(c["graph_repetitions"]==(1 if shape[0]*shape[1]*shape[2]>=4096 else 4),"Attention graph repetitions changed")
            require(set(c["timings_ms"])=={"prepared","online"} and c["setup_and_tuning_wall_ms"]>0,"Missing attention timing/setup boundary")
            for observations in list(c["timings_ms"].values())+[c["host_request_ms"]]:require(len(observations)==30 and all(v>0 and np.isfinite(v) for v in observations),"Invalid attention timing observations")
            check=c["checks"]
            if c["engine"]=="native":
                modes=c["qk_mode"],c["pv_mode"];require(modes in CONFIGS and c["label"]==f"native-{modes[0]}-{modes[1]}","Attention modes changed")
                hashes={}
                for name,x,mode in (("q",pad[0],modes[0]),("k",pad[1],modes[0]),("v",pad[2],modes[1])):
                    if (name,mode) not in cache["encodings"]:
                        raw,scales,_=encoding(x,mode);cache["encodings"][name,mode]=[digest(raw),digest(scales)]
                    hashes[name]=cache["encodings"][name,mode]
                for ck in (check,c["post_timing_checks"],c["post_request_checks"]):native_check(ck,shape,target,indices,hashes,[digest(x) for x in pad])
                require(c["post_timing_checks"]["snapshot_sha256"]==check["snapshot_sha256"] and c["post_request_checks"]["snapshot_sha256"]==check["snapshot_sha256"],"Native attention changed across repeated inputs")
                require(c["owned_native_device_buffer_bytes"]>sum(x.nbytes for x in pad),"Missing native owned-buffer accounting")
                require(len(c["tuning"])==2 and len(c["selected_algorithms"])==2,"Missing independent QK/PV tuning")
                for stage,rows in enumerate(c["tuning"]):require(1<=len(rows)<=8 and all(len(row["ms"])==3 and all(v>0 for v in row["ms"]) for row in rows) and c["selected_algorithms"][stage] in range(len(rows)),"Invalid attention tuning")
            else:
                sample_check(check,target,indices);require(check["checked_output_entries"]==target.size and check["algorithm_relative_error"]<={"fp32":5e-5,"fp16":.005,"bf16":.03}[c["label"].split("-")[-1]],"Fused attention arithmetic failed")
                require(c["post_timing_output_sha256"]==check["output_sha256"] and c["post_request_output_sha256"]==check["output_sha256"] and c["extra_peak_torch_allocated_bytes"]>=0,"Fused attention output/allocator coverage changed")
            if shape==(1,1,128,64):
                witness=c["witness"];path=ROOT/"results"/witness["file"];require(sha256(path)==witness["sha256"],"Attention witness changed")
                with np.load(path,allow_pickle=False) as data:
                    for name,x in zip(("q","k","v"),host):np.testing.assert_array_equal(data[name],x)
                    np.testing.assert_array_equal(data["indices"],indices);actual=data["output"];require(digest(actual)==check["output_sha256"],"Attention output hash changed")
                    np.testing.assert_array_equal(actual.ravel()[indices],np.array(check["sample_values"]))
                    close(relative_error(actual,target),check["relative_frobenius_error"],"Attention full error changed")
                    if c["engine"]=="native":
                        replay=validate_arrays(*host,modes,kind,data["scores"],data["probabilities"],data["padded_output"],actual)
                        require(replay["encoded_sha256"]==check["encoded_sha256"],"Attention witness encoding changed");witness_entries+=replay["checked_output_entries"]
                    else:
                        mode=c["label"].split("-")[-1];rounded=[encoding(x,mode)[2] for x in host]
                        close(relative_error(actual,attention(*rounded,kind)),check["algorithm_relative_error"],"Fused witness arithmetic changed");witness_entries+=actual.size
            count+=1;outputs+=check["checked_output_entries"]
        require(seen==expected,"Attention shape/mask/profile/method attempts incomplete")
        for group in groups:
            for label in {f"native-{a}-{b}" for a,b in CONFIGS}|{"math-fp32","efficient-fp32","cudnn-fp16"}|({"flash-fp16"} if group[1]!="padding" else set()):require((*group,label) in available,"Required native or strong reference missing")
    require(identities[0]==identities[1],"Attention sources/binary changed between cohorts")
    inspection=json.loads((ROOT/"results/attention-native-inspection.json").read_text(encoding="utf-8"))
    require(inspection["binary_sha256"]==identities[0][1] and "sm_120a" in inspection["architectures"],"Attention emitted-code provenance changed")
    for name,expected_hash in inspection["source_sha256"].items():require(sha256(ROOT/name)==expected_hash,"Executed instruction inspector changed")
    opcodes={opcode for kernel in inspection["kernels"] for opcode in kernel["opcodes"]}
    require(any("MMA" in opcode and "E2M1" in opcode for opcode in opcodes) and any("UTMA" in opcode for opcode in opcodes),"Missing native FP4 Tensor Core/TMA instructions")
    dependencies=json.loads((ROOT/"results/attention-dependencies.json").read_text(encoding="utf-8"))
    require(dependencies["recorded_after_campaign"] is True and dependencies["binary_sha256"]==identities[0][1] and dependencies["packages"]["torch"]==record["environment"]["torch"] and dependencies["packages"]["numpy"]==record["environment"]["numpy"] and dependencies["packages"]["triton"] and dependencies["cudnn_runtime_version"]>0,"Attention dependency record changed")
    for name,expected_hash in dependencies["source_sha256"].items():require(sha256(ROOT/name)==expected_hash,"Executed dependency inspector changed")
    print(f"Attention: {count} cases, {missing} retained rejected attempts, {outputs:,} checked intermediates/outputs, {witness_entries:,} CPU-replayed entries; {preflight_witnesses} backend witnesses")


if __name__=="__main__":verify()
