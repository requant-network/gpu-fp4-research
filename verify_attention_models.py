"""CPU-only replay of frozen transformer training splits, selection and inference."""
import json

import numpy as np

from attention_reference import network
from prepare_digits import load_digits
from reference import digest,relative_error
from verify_followups import ROOT
from verify_results import require,sha256


def verify():
    x,y=load_digits();total=0
    for seed in (20261123,20261223):
        record=json.loads((ROOT/f"results/attention-training-{seed}.json").read_text(encoding="utf-8"))
        require(record.get("complete") is True and record["seed"]==seed and record["epochs"]==60 and len(record["history"])==60,"Incomplete transformer training")
        require(record["gpu_exclusive"] is False and set(record["source_sha256"])=={"prepare_attention_models.py","attention_model.py","prepare_digits.py","reference.py"},"Training scope/provenance changed")
        for key in ("source_sha256","data_sha256"):
            for name,expected in record[key].items():require(sha256(ROOT/name)==expected,f"Transformer training provenance changed: {name}")
        best=max(record["history"],key=lambda row:(row["validation_accuracy"],-row["validation_loss"]))
        require(record["selected_epoch"]==best["epoch"] and [r["epoch"] for r in record["history"]]==list(range(1,61)),"Checkpoint used another selection rule")
        for row in record["history"]:require(0<=row["validation_accuracy"]<=1 and np.isfinite(row["validation_loss"]) and row["validation_loss"]>0 and row["mean_training_loss"]>0,"Invalid transformer history")
        require(record["model_file"]==f"attention-model-{seed}.npz" and record["reference_file"]==f"attention-model-reference-{seed}.npz","Unexpected transformer artifact path")
        require(sha256(ROOT/"fixtures"/record["model_file"])==record["model_sha256"] and sha256(ROOT/"fixtures"/record["reference_file"])==record["reference_sha256"],"Frozen model/reference changed")
        rng=np.random.default_rng(seed);splits=[[],[],[]]
        for label in range(10):
            ids=np.flatnonzero(y==label);rng.shuffle(ids);a,b=int(.6*len(ids)),int(.8*len(ids))
            for target,part in zip(splits,(ids[:a],ids[a:b],ids[b:])):target.extend(part)
        expected_shapes={"position":(1,64,128)}
        for name,rows,columns in (("embed",128,1),("qkv",384,128),("projection",128,128),("ff1",256,128),("ff2",128,256),("classifier",10,128)):
            expected_shapes[name+".weight"]=(rows,columns);expected_shapes[name+".bias"]=(rows,)
        for name in ("norm1","norm2","norm3"):
            for key in ("weight","bias"):expected_shapes[f"{name}.{key}"]=(128,)
        with np.load(ROOT/"fixtures"/record["model_file"],allow_pickle=False) as model,np.load(ROOT/"fixtures"/record["reference_file"],allow_pickle=False) as reference:
            require(set(model.files)==set(expected_shapes)|{"train","validation","test"},"Unexpected model parameters")
            for name,shape in expected_shapes.items():require(model[name].shape==shape and model[name].dtype==np.float32 and np.isfinite(model[name]).all(),"Invalid frozen transformer parameter")
            for name,ids in zip(("train","validation","test"),splits):np.testing.assert_array_equal(model[name],np.array(sorted(ids),np.int64))
            weights={name:model[name] for name in expected_shapes}
            for split in ("validation","test"):
                ids=model[split];actual=reference[f"{split}_logits"];predictions=reference[f"{split}_predictions"]
                expected=np.concatenate([network(x[ids[start:start+32]],weights) for start in range(0,len(ids),32)])
                require(actual.shape==(len(ids),10) and np.isfinite(actual).all() and relative_error(actual,expected)<5e-5,"Transformer CPU inference disagrees with GPU")
                np.testing.assert_array_equal(actual.argmax(axis=1),predictions);np.testing.assert_array_equal(expected.argmax(axis=1),predictions)
                quality=record["quality"][split];require(quality=={"correct":int(np.sum(predictions==y[ids])),"count":len(ids),"predictions_sha256":digest(predictions)},"Transformer quality summary changed")
                total+=len(ids)
    print(f"Transformer models: both validation-selected cohorts replayed, {total} full CPU inference cases")


if __name__=="__main__":verify()
