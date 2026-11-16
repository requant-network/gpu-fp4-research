"""Freeze validation inputs/hidden activations for reproducible correction studies."""
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_digits import load_digits
from reference import digest

ROOT=Path(__file__).resolve().parent


def main():
    x,_=load_digits()
    records=[]
    for seed in (20261116,20261216):
        model=ROOT/f"fixtures/digits-model-{seed}.npz"
        with np.load(model,allow_pickle=False) as arrays:
            indices=arrays["validation"][:192]
            inputs=x[indices]
            hidden=np.maximum(inputs.astype(np.float64)@arrays["w1"].astype(np.float64).T+arrays["b1"],0).astype(np.float32)
            if inputs.shape!=(192,128) or hidden.shape!=(192,256) or not np.isfinite(hidden).all():raise ValueError("Invalid real activations")
        path=ROOT/f"fixtures/outlier-activations-{seed}.npz"
        np.savez_compressed(path,x=inputs,hidden=hidden,indices=indices)
        records.append({"model_seed":seed,"arrays_sha256":{k:digest(v) for k,v in (("x",inputs),("hidden",hidden),("indices",indices))}})
    names=["fixtures/digits.csv.gz"]+[f"fixtures/{prefix}-{s}.npz" for s in (20261116,20261216) for prefix in ("digits-model","outlier-activations")]
    record={"complete":True,"records":records,"contract":"First 192 validation examples; hidden activation formed in FP64 then rounded once to FP32; frozen arrays avoid platform BLAS drift",
            "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("prepare_outlier_activations.py","prepare_digits.py","reference.py")},
            "data_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}}
    (ROOT/"results/outlier-activation-preparation.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print("Froze two independent models' validation input/hidden activations")


if __name__=="__main__":main()
