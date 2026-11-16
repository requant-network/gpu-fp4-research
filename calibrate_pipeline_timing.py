"""Measure empty external-event intervals under the same cache conditioning."""
import cupy as cp
import numpy as np

from followup import ROOT,file_hash,write_record
from pipeline_study import PipelinePlan,load_raster
from reference import inputs


def main():
    cp.cuda.Device(0).use();library=load_raster();stream=cp.cuda.Stream(non_blocking=True)
    l2=int(cp.cuda.runtime.getDeviceProperties(0)["l2CacheSize"])
    record={"source_sha256":{name:file_hash(ROOT/name) for name in ("calibrate_pipeline_timing.py","pipeline_study.py")},
            "binary_sha256":file_hash(ROOT/"build/libraster.so"),"cache_bytes":4*l2,"samples":30,"warmups":5,
            "scope":"No-operation interval between external event nodes; identical small prepacked operation primes warm condition, four-L2 sweep conditions pressure; calibration is descriptive and not subtracted from campaign latencies",
            "timings_ms":{}}
    with stream:
        a,b=inputs(20261120,(128,128,128),"normal")
        plan=PipelinePlan(library,cp.asarray(a),cp.asarray(b),stream)
        scratch=cp.asarray(np.zeros(l2,np.float32));plan.prepare();plan.run();stream.synchronize()
        try:
            for condition in ("warm","cache-pressure"):
                start,end=cp.cuda.Event(),cp.cuda.Event()
                stream.begin_capture()
                if condition=="warm":plan.run()
                else:plan.checked(library.rq_cache_pressure(scratch.data.ptr,scratch.size,stream.ptr))
                plan.checked(library.rq_record_external(start.ptr,stream.ptr));plan.checked(library.rq_record_external(end.ptr,stream.ptr))
                graph=stream.end_capture()
                for _ in range(5):graph.launch(stream)
                stream.synchronize();observations=[]
                for _ in range(30):
                    graph.launch(stream);end.synchronize();value=float(cp.cuda.get_elapsed_time(start,end))
                    if not np.isfinite(value) or value<0:raise RuntimeError("Invalid empty timing interval")
                    observations.append(value)
                record["timings_ms"][condition]=observations
        finally:stream.synchronize();plan.close()
    record["complete"]=True;write_record(ROOT/"results/pipeline-timer-calibration.json",record)
    print({key:float(np.median(values))*1000 for key,values in record["timings_ms"].items()})


if __name__=="__main__":main()
