"""Pinned FP4 fused reference with explicit masks, dimensions and input ownership."""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parent


def load():
    path=ROOT/"results/sage-build.json";record=json.loads(path.read_text(encoding="utf-8"))
    if not record.get("complete"):raise RuntimeError("Pinned SageAttention3 build is unavailable")
    target=ROOT/"build/sage-python"
    for name,expected in record["installed_sha256"].items():
        if hashlib.sha256((target/name).read_bytes()).hexdigest()!=expected:raise ValueError("Installed SageAttention3 differs from its build manifest")
    sys.path.insert(0,str(target))
    from sageattn3 import sageattn3_blackwell
    return sageattn3_blackwell,hashlib.sha256(path.read_bytes()).hexdigest()


def function(api,operands,dtype,kind="none",prepared=False):
    if kind not in ("none","causal"):raise NotImplementedError("Pinned SageAttention3 does not implement the arbitrary padding mask")
    if operands[0].shape[-1] not in (64,128):raise NotImplementedError("Pinned SageAttention3 kernel supports only head dimensions 64/128")
    cached=tuple(x.to(dtype) for x in operands) if prepared else None
    def forward():
        q,k,v=cached if prepared else tuple(x.to(dtype) for x in operands)
        if prepared:k=k.clone()
        return api(q,k,v,is_causal=kind=="causal").float()
    return forward


def model_function(api,dtype):
    def forward(q,k,v):return api(q.to(dtype),k.to(dtype),v.to(dtype),is_causal=False).float()
    return forward
