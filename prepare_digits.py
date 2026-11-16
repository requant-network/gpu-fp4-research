"""Train a small real-data ReLU classifier for complete low-precision inference studies."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from reference import digest, inputs

ROOT=Path(__file__).resolve().parent
DATA_SHA="09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22"


def load_digits():
    path=ROOT/"fixtures/digits.csv.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=DATA_SHA:
        raise ValueError("Pinned digits data hash mismatch")
    with gzip.open(path,"rt",encoding="ascii") as stream:
        raw=np.loadtxt(stream,delimiter=",",dtype=np.float32)
    if raw.shape!=(1797,65) or not np.isfinite(raw).all():
        raise ValueError("Unexpected dataset shape/values")
    x=np.zeros((1797,128),np.float32)
    x[:,:64]=raw[:,:64]/np.float32(16)
    y=raw[:,64].astype(np.int64)
    if np.any(x<0) or np.any(x>1) or not np.array_equal(raw[:,64],y) or np.any(y<0) or np.any(y>9):
        raise ValueError("Invalid digit values or labels")
    return x,y


def forward(x,w1,b1,w2,b2):
    hidden=np.maximum(x@w1.T+b1,np.float32(0))
    return hidden@w2[:10].T+b2[:10]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=20261116)
    parser.add_argument("--epochs",type=int,default=80)
    args=parser.parse_args()
    if args.seed<0 or args.epochs<1:
        parser.error("Seed must be nonnegative and epochs positive")
    x,y=load_digits()
    rng=np.random.default_rng(args.seed)
    splits=[[],[],[]]
    for label in range(10):
        indices=np.flatnonzero(y==label)
        rng.shuffle(indices)
        first,second=int(.6*len(indices)),int(.8*len(indices))
        for target,part in zip(splits,(indices[:first],indices[first:second],indices[second:])):
            target.extend(part)
    train,validation,test=(np.array(sorted(part),np.int64) for part in splits)
    _,w1=inputs(args.seed,(128,256,128),"normal")
    _,small=inputs(args.seed+1,(128,128,256),"normal")
    w1*=np.float32(np.sqrt(2/64))
    w2=np.zeros((128,256),np.float32)
    w2[:10]=small[:10]*np.float32(np.sqrt(2/256))
    b1,b2=np.zeros(256,np.float32),np.zeros(128,np.float32)
    parameters=[w1,b1,w2,b2]
    moments=[np.zeros_like(p) for p in parameters]
    variances=[np.zeros_like(p) for p in parameters]
    history,step=[],0
    for epoch in range(args.epochs):
        order=rng.permutation(train)
        for start in range(0,len(order),128):
            indices=order[start:start+128]
            batch=x[indices]
            pre=batch@w1.T+b1
            hidden=np.maximum(pre,np.float32(0))
            logits=hidden@w2[:10].T+b2[:10]
            maximum=logits.max(axis=1,keepdims=True)
            probabilities=np.exp(logits-maximum)
            probabilities/=probabilities.sum(axis=1,keepdims=True)
            probabilities[np.arange(len(indices)),y[indices]]-=1
            probabilities/=np.float32(len(indices))
            gw2=np.zeros_like(w2)
            gb2=np.zeros_like(b2)
            gw2[:10]=probabilities.T@hidden
            gb2[:10]=probabilities.sum(axis=0)
            gradient=(probabilities@w2[:10])*(pre>0)
            gradients=[gradient.T@batch,gradient.sum(axis=0),gw2,gb2]
            step+=1
            for parameter,moment,variance,grad in zip(parameters,moments,variances,gradients):
                moment*=np.float32(.9)
                moment+=np.float32(.1)*grad
                variance*=np.float32(.999)
                variance+=np.float32(.001)*grad*grad
                corrected_m=moment/np.float32(1-.9**step)
                corrected_v=variance/np.float32(1-.999**step)
                parameter-=np.float32(.003)*corrected_m/(np.sqrt(corrected_v)+np.float32(1e-8))
        if epoch%10==0 or epoch==args.epochs-1:
            history.append({"epoch":epoch+1,"validation_accuracy":float(np.mean(
                forward(x[validation],*parameters).argmax(axis=1)==y[validation]))})
    output=ROOT/f"fixtures/digits-model-{args.seed}.npz"
    np.savez_compressed(output,w1=w1,b1=b1,w2=w2,b2=b2,train=train,validation=validation,test=test)
    accuracies={name:float(np.mean(forward(x[indices],*parameters).argmax(axis=1)==y[indices]))
                for name,indices in (("train",train),("validation",validation),("test",test))}
    if accuracies["validation"]<.9 or accuracies["test"]<.9:
        raise RuntimeError(f"Classifier did not meet the predeclared useful-baseline threshold: {accuracies}")
    record={"seed":args.seed,"epochs":args.epochs,"numpy":np.__version__,"dataset_sha256":DATA_SHA,
            "model_file":output.name,"model_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
            "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                              for name in ("prepare_digits.py","reference.py")},
            "split_indices_sha256":{name:digest(indices) for name,indices in (("train",train),("validation",validation),("test",test))},
            "split_sizes":{"train":len(train),"validation":len(validation),"test":len(test)},
            "accuracy":accuracies,"history":history,
            "selection_policy":f"Fixed {args.epochs}-epoch final model; no test-based checkpoint selection",
            "complete":True}
    destination=ROOT/f"results/digits-training-{args.seed}.json"
    destination.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(accuracies),flush=True)


if __name__=="__main__":
    main()
