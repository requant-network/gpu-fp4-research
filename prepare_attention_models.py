"""Train two frozen transformer cohorts; select checkpoints using validation only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attention_model import DigitAttention
from prepare_digits import load_digits
from reference import digest

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--seed",type=int,required=True);parser.add_argument("--epochs",type=int,default=60);args=parser.parse_args()
    if args.seed<0 or args.epochs<1:parser.error("Positive epochs and nonnegative seed required")
    torch.cuda.set_device(0);torch.set_num_threads(1);torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision("highest")
    x,y=load_digits();rng=np.random.default_rng(args.seed);splits=[[],[],[]]
    for label in range(10):
        ids=np.flatnonzero(y==label);rng.shuffle(ids);a,b=int(.6*len(ids)),int(.8*len(ids))
        for target,part in zip(splits,(ids[:a],ids[a:b],ids[b:])):target.extend(part)
    train,validation,test=(np.array(sorted(ids),np.int64) for ids in splits);device_x=torch.from_numpy(x).cuda();device_y=torch.from_numpy(y).cuda()
    model=DigitAttention().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    best=None;state=None;selected_epoch=None;history=[]
    for epoch in range(args.epochs):
        model.train();order=rng.permutation(train);losses=[]
        for start in range(0,len(order),64):
            ids=torch.from_numpy(order[start:start+64]).cuda();optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(device_x[ids]),device_y[ids]);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            logits=model(device_x[validation]);accuracy=float((logits.argmax(dim=1)==device_y[validation]).float().mean());validation_loss=float(F.cross_entropy(logits,device_y[validation]))
        history.append({"epoch":epoch+1,"mean_training_loss":float(np.mean(losses)),"validation_accuracy":accuracy,"validation_loss":validation_loss})
        score=accuracy,-validation_loss
        if best is None or score>best:best=score;selected_epoch=epoch+1;state={name:value.detach().cpu().numpy().copy() for name,value in model.state_dict().items()}
        if epoch%5==4 or epoch==0:print(f"seed {args.seed} epoch {epoch+1}: validation accuracy {accuracy:.4f}",flush=True)
    path=ROOT/f"fixtures/attention-model-{args.seed}.npz";np.savez_compressed(path,train=train,validation=validation,test=test,**state)
    model.load_state_dict({name:torch.from_numpy(value).cuda() for name,value in state.items()});quality={};witness={}
    with torch.inference_mode():
        for name,ids in (("validation",validation),("test",test)):
            logits=model(device_x[ids]).cpu().numpy();predictions=logits.argmax(axis=1);quality[name]={"correct":int(np.sum(predictions==y[ids])),"count":len(ids),"predictions_sha256":digest(predictions)}
            witness[f"{name}_logits"]=logits;witness[f"{name}_predictions"]=predictions
    witness_path=ROOT/f"fixtures/attention-model-reference-{args.seed}.npz";np.savez_compressed(witness_path,**witness)
    props=torch.cuda.get_device_properties(0);sha=lambda file:hashlib.sha256(file.read_bytes()).hexdigest()
    record={"schema_version":1,"utc":datetime.now(timezone.utc).isoformat(),"seed":args.seed,"epochs":args.epochs,"selected_epoch":selected_epoch,
            "checkpoint_rule":"Maximum validation accuracy, then minimum validation cross-entropy; test labels do not select the checkpoint.",
            "torch":torch.__version__,"torch_cuda":torch.version.cuda,"gpu_name":props.name,"gpu_exclusive":False,
            "source_sha256":{name:sha(ROOT/name) for name in ("prepare_attention_models.py","attention_model.py","prepare_digits.py","reference.py")},
            "data_sha256":{"fixtures/digits.csv.gz":sha(ROOT/"fixtures/digits.csv.gz")},"model_file":path.name,"model_sha256":sha(path),
            "reference_file":witness_path.name,"reference_sha256":sha(witness_path),"quality":quality,"history":history,"complete":True}
    (ROOT/f"results/attention-training-{args.seed}.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"seed {args.seed}: selected epoch {selected_epoch}, test accuracy {quality['test']['correct']/quality['test']['count']:.4f}",flush=True)


if __name__=="__main__":main()
