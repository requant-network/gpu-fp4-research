"""Native quantized search pipeline and boundary controls, not the full study."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from reference import digest
from search_native import Search,library
from search_reference import reference_search
from search_validation import check_native

ROOT = Path(__file__).resolve().parent
SOURCES = ("search_kernel_preflight.py","search_native.py","search_validation.py","search_reference.py","reference.py",
           "native/search_bridge.cu","native/search_select.cpp","build_search.sh","native/hardware_bridge.cu",
           "native/application_bridge.cu","native/optimized_bridge.cu","native/bridge.cu")


def inputs(n,d,batch):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261128,n,d,batch])))
    database = rng.integers(-2048,2049,(n,d)).astype(np.float32)/4096
    queries = rng.integers(-2048,2049,(batch,d)).astype(np.float32)/4096
    return database,queries


def main():
    target = ROOT/"results/search-kernel-preflight.json"
    if target.exists():
        raise FileExistsError("Refusing to replace a native search capability record")
    sha = lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    environment = ROOT/"results/search-environment.json"
    assert environment.is_file()
    lib = library()
    record = {"study":10,"complete":False,"scope":"Native pipeline and corner-case probe only: no independent full cohorts, real data, strong indexed cost comparison or completed-study claim.",
              "gpu_index":0,"gpu_exclusive":False,"source_sha256":{name:sha(ROOT/name) for name in SOURCES},
              "binary_sha256":sha(ROOT/"build/libsearch.so"),"selection_object_sha256":sha(ROOT/"build/search-select.o"),
              "environment_file":environment.name,"environment_sha256":sha(environment),
              "timing_scope":"Five complete resident native operation observations after one warmup, uncaptured and captured twice per graph; query encoding/norms/GEMM/distance/filter/select/refine are included. No transfers or CPU oracle work is timed. Setup and graph construction are separate.","groups":[]}
    configurations = [(n,d,batch,p,norm,64,True,"fraction30") for n,d,batch in ((4096,128,1),(65536,256,32)) for p in range(6) for norm in (0,1)]
    configurations += [(4096,128,1,p,1,10,False,"fraction30") for p in range(6)]
    configurations += [(4096,128,1,p,1,64,True,mask) for p in (0,4) for mask in ("empty","four")]
    configurations += [(65536,256,32,p,1,512,True,"fraction30") for p in (0,5)]
    for gi,(n,d,batch,precision,norm,count,rerank,mask) in enumerate(configurations):
        database,queries = inputs(n,d,batch)
        allowed = np.arange(n)%10<3 if mask == "fraction30" else (np.arange(n)<(4 if mask == "four" else 0))
        truth = [reference_search(database,q,10,allowed) for q in (queries,-queries)]
        group = {"n":n,"d":d,"batch":batch,"precision":precision,"norm_mode":norm,"candidates":count,"k":10,"refine":rerank,"mask":mask,
                 "database_sha256":digest(database),"queries_sha256":digest(queries),"allowed_sha256":digest(allowed),"available":False}
        arrays = {}
        plan = None
        started = time.perf_counter_ns()
        try:
            plan = Search(lib,database,queries,allowed,precision,count,10,norm,rerank)
            group["setup_wall_ms"] = (time.perf_counter_ns()-started)/1e6
            group["available"] = True
            plan.run()
            group["original"] = check_native(plan,database,queries,allowed,precision,norm,*truth[0],rerank,arrays,"original")
            original = plan.result()
            timings = []
            for captured in (False,True):
                if captured:
                    started = time.perf_counter_ns()
                    plan.call("sq_capture",2)
                    group["graph_setup_ms"] = (time.perf_counter_ns()-started)/1e6
                plan.call("sq_time",int(captured))
                observations = []
                for _ in range(5):
                    plan.call("sq_time",int(captured))
                    observations.append({key:plan.metadata()[key] for key in ("cuda_ms","wall_ms")})
                actual = plan.result()
                np.testing.assert_array_equal(actual[0],original[0])
                np.testing.assert_array_equal(actual[1],original[1])
                timings.append({"captured":captured,"observations":observations,"actual_ids_sha256":digest(actual[1]),"actual_distances_sha256":digest(actual[0])})
            group["timings"] = timings
            changed = np.ascontiguousarray(-queries)
            plan.load(changed)
            try:
                plan.snapshot(13)
                raise AssertionError("Reading stale output after changing queries was not rejected")
            except RuntimeError:
                group["stale_output_rejected"] = True
            plan.call("sq_time",1)
            group["changed"] = check_native(plan,database,changed,allowed,precision,norm,*truth[1],rerank,arrays,"changed")
            group["metadata"] = plan.metadata()
            full = batch*n <= 262144
            group["full_dot_score_witnesses"] = full
            if not full:
                indices = np.unique(np.linspace(0,batch*n-1,64,dtype=np.int64))
                arrays["dot_indices"] = indices
                for variant in ("original","changed"):
                    for kind in ("dot","score"):
                        arrays[variant+"_"+kind] = arrays[variant+"_"+kind].ravel()[indices].copy()
        except RuntimeError as error:
            if group["available"]:
                raise
            group["error"] = str(error)
            group["setup_wall_ms"] = (time.perf_counter_ns()-started)/1e6
        finally:
            if plan is not None:
                plan.close()
        witness = f"search-kernel-preflight-{gi:02d}.npz"
        path = ROOT/"results"/witness
        if path.exists():
            raise FileExistsError("Refusing to replace a native pipeline witness")
        np.savez_compressed(path,**arrays)
        group.update(witness_file=witness,witness_sha256=sha(path))
        record["groups"].append(group)
        print(gi+1,len(configurations),n,d,batch,precision,norm,count,mask,group["available"],flush=True)
    record["complete"] = True
    target.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Native search pipeline probe complete",flush=True)


if __name__ == "__main__":
    main()
