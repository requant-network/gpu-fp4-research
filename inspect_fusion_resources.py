"""Record actual kernel attributes for separate/fused encoders, without counters."""
import ctypes

import cupy as cp

from followup import ROOT,file_hash,write_record
from pipeline_study import load_raster


def main():
    cp.cuda.Device(0).use();library=load_raster()
    library.rq_fusion_resources.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
    library.rq_fusion_resources.restype=ctypes.c_int
    record={"source_sha256":{"inspect_fusion_resources.py":file_hash(ROOT/"inspect_fusion_resources.py"),
                             "pipeline_study.py":file_hash(ROOT/"pipeline_study.py")},
            "binary_sha256":file_hash(ROOT/"build/libraster.so"),
            "scope":"CUDA function attributes and theoretical resident blocks, not achieved occupancy; same frozen encoder sources recompiled in traversal-control library",
            "rows":[]}
    names=("registers_per_thread","static_shared_bytes","local_bytes","theoretical_blocks_per_sm","sm_max_threads","sm_registers","binary_version")
    for variant in (2,10,11,12,13):
        for fused in (0,1):
            for threads in (128,256,512):
                values=(ctypes.c_int*7)()
                if library.rq_fusion_resources(variant,fused,threads,values):raise RuntimeError(library.rq_error().decode())
                row={"variant":variant,"fused":bool(fused),"block_threads":threads,**dict(zip(names,values))}
                row["theoretical_thread_occupancy"]=row["theoretical_blocks_per_sm"]*threads/row["sm_max_threads"]
                record["rows"].append(row)
    record["complete"]=True;write_record(ROOT/"results/fusion-resources.json",record)
    print("Recorded 30 separate/fused encoder resource configurations")


if __name__=="__main__":main()
