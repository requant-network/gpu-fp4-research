"""Record post-campaign attention dependency versions without private paths."""
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import torch

ROOT=Path(__file__).resolve().parent


def main():
    packages={}
    for name in ("torch","triton","numpy","nvidia-cudnn-cu13","nvidia-cublas-cu13","ninja","packaging","einops","setuptools","wheel"):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    nvidia={dist.metadata["Name"]:dist.version for dist in importlib.metadata.distributions() if dist.metadata.get("Name","").lower().startswith("nvidia-")}
    record={"recorded_after_campaign":True,"utc":datetime.now(timezone.utc).isoformat(),"python_version":platform.python_version(),"packages":packages,"nvidia_distributions":nvidia,
            "package_metadata_note":"Null means no distribution metadata with that name, not an absent runtime library or system tool.","torch_cuda":torch.version.cuda,
            "cudnn_runtime_version":torch.backends.cudnn.version(),"torch_cxx11_abi":torch._C._GLIBCXX_USE_CXX11_ABI,
            "source_sha256":{"inspect_attention_environment.py":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "binary_sha256":hashlib.sha256((ROOT/"build/libattention.so").read_bytes()).hexdigest()}
    (ROOT/"results/attention-dependencies.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"packages":packages,"cudnn_runtime_version":record["cudnn_runtime_version"]}))


if __name__=="__main__":main()
