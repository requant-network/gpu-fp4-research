"""Probe native encoder hardware counters separately from latency campaigns."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe",type=int,choices=(2,10,11,12,13))
    args=parser.parse_args()
    if args.probe is not None:
        import cupy as cp
        import numpy as np
        from hardware_study import HardwarePlan,load_hardware
        cp.cuda.Device(0).use()
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            a=cp.asarray(np.full((2048,2048),.25,dtype=np.float32))
            b=cp.asarray(np.full((2048,2048),.5,dtype=np.float32))
            plan=HardwarePlan(load_hardware(),args.probe,a,b,stream)
            try:
                plan.prepare(1)
                stream.synchronize()
            finally:plan.close()
        return
    tool="/usr/local/cuda-13.3/bin/ncu"
    record={"schema_version":1,"binary_sha256":hashlib.sha256((ROOT/"build/libhardware.so").read_bytes()).hexdigest(),
            "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                             for name in ("profile_native.py","hardware_study.py")},
            "tool_version":subprocess.check_output([tool,"--version"],text=True),
            "sections":["LaunchStats","Occupancy","SpeedOfLight"],"shape":[2048,2048,2048],
            "scope":"Single-launch hardware-counter probes; constant inputs; profiling observations are excluded from all timing campaigns",
            "probes":[]}
    for variant in (2,10,11,12,13):
        expression="cooperative_fp4" if variant==2 else "native_scalar16" if variant==10 else "native_cooperative"
        command=[tool,"--csv","--kernel-name",f"regex:.*{expression}.*","--launch-count","1"]
        for section in record["sections"]:command += ["--section",section]
        command += [sys.executable,str(Path(__file__)),"--probe",str(variant)]
        result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
        output=result.stdout.replace(str(ROOT),"<project>")
        denied="ERR_NVGPUCTRPERM" in output
        record["probes"].append({"variant":variant,"returncode":result.returncode,"counter_permission_denied":denied,"output":output})
        print(f"variant {variant}: returncode={result.returncode}, permission_denied={denied}",flush=True)
        if denied:break
        if result.returncode:raise RuntimeError(output)
    record["counters_available"]=not any(p["counter_permission_denied"] for p in record["probes"])
    record["complete"]=True
    (ROOT/"results/native-profiler.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":main()
