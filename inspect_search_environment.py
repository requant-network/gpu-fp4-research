"""Record actual isolated GPU search dependencies and loaded library fingerprints."""
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import cupy as cp
import faiss

from search_native import library

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT/"results/search-environment.json"
    if output.exists():
        raise FileExistsError("Refusing to replace an executed environment fingerprint")
    cp.cuda.Device(0).use()
    library()
    properties = cp.cuda.runtime.getDeviceProperties(0)
    packages = {}
    for dist in importlib.metadata.distributions():
        files = list(dist.files or [])
        record = next((path for path in files if str(path).endswith(".dist-info/RECORD")),None)
        packages[dist.metadata["Name"]] = {"version":dist.version,"record_sha256":None if record is None else sha(Path(dist.locate_file(record)))}
    loaded = []
    seen = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split()
        if len(fields) < 6 or ".so" not in fields[-1]:
            continue
        path = Path(fields[-1])
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        if any(token in path.name.lower() for token in ("cuvs","cublas","cudart","cuda.so","rapids_logger","faiss","rmm","nvrtc","nvjitlink")):
            loaded.append({"name":path.name,"bytes":path.stat().st_size,"sha256":sha(path)})
    record = {"scope":"Actual Python package RECORD and loaded GPU-library fingerprints; not a complete vector-search experiment.",
              "captured_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"python":sys.version,"gpu_index":0,"gpu_exclusive":False,
              "gpu_name":properties["name"].decode(),"compute_capability":[properties["major"],properties["minor"]],
              "multiprocessors":properties["multiProcessorCount"],"runtime_api_version":cp.cuda.runtime.runtimeGetVersion(),
              "driver_api_version":cp.cuda.runtime.driverGetVersion(),"faiss_compile_options":faiss.get_compile_options(),
              "packages":packages,"loaded_libraries":loaded,
              "cuda_sdk_components":json.loads(Path("/usr/local/cuda-13.3/version.json").read_text()),
              "nvcc":subprocess.check_output(["/usr/local/cuda-13.3/bin/nvcc","--version"],text=True).strip(),
              "nvidia_smi":subprocess.check_output(["nvidia-smi","--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip(),
              "source_sha256":{"inspect_search_environment.py":sha(Path(__file__))}}
    output.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print(f"Recorded {len(packages)} installed packages and {len(loaded)} loaded GPU-library fingerprints")


if __name__ == "__main__":
    main()
