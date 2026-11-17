"""Actual Faiss/cuVS GPU capability probe; not a completed vector-search study."""
import gc
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import time

import cupy as cp
import faiss
import numpy as np
from cuvs.neighbors import brute_force, cagra, filters, ivf_flat, ivf_pq

from reference import digest
from search_reference import evaluate, reference_search, refine

ROOT = Path(__file__).resolve().parent
SOURCES = ("search_preflight.py", "search_reference.py", "reference.py")


def main():
    path = ROOT/"results/search-preflight.json"
    if path.exists():
        raise FileExistsError("Refusing to replace an executed search capability record")
    cp.cuda.Device(0).use()
    assert cp.cuda.Device(0).compute_capability == "120"
    rng = np.random.Generator(np.random.PCG64(20261127))
    data = rng.integers(-2048, 2049, (8192, 128)).astype(np.float32)/4096
    queries = rng.integers(-2048, 2049, (4, 128)).astype(np.float32)/4096
    arrays = {"database": data, "queries": queries}
    record = {"study": 10, "complete": False, "gpu_index": 0, "gpu_exclusive": False,
              "scope": "One synthetic finite-FP32 database and four independent queries. Actual GPU exact/indexed/filter capability and CPU-replayed recall only; no Tensor Core proposal, full cost/memory comparison or study completion is claimed.",
              "timing_scope": "Three API wall intervals after one warmup, with GPU synchronization. cuVS queries and results are device arrays; Faiss NumPy APIs include their implicit query/result transfers. No cross-method performance comparison is supported by these mixed probe boundaries.",
              "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},
              "packages": {d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
              "faiss_compile_options": faiss.get_compile_options(), "runtime_api_version": cp.cuda.runtime.runtimeGetVersion(),
              "driver_api_version": cp.cuda.runtime.driverGetVersion(), "compute_capability": [12, 0],
              "database_sha256": digest(data), "queries_sha256": digest(queries),
              "api_signatures": {name:str(inspect.signature(function)) for name,function in (("ivf_pq.search",ivf_pq.search),("ivf_flat.search",ivf_flat.search),("cagra.search",cagra.search),("brute_force.search",brute_force.search))}, "cases": []}
    gpu_data, gpu_queries = cp.asarray(data), cp.asarray(queries)
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(64 << 20)
    configurations = [(family, precision, fraction, candidates, probes)
                      for family in ("faiss-flat", "cuvs-brute", "cuvs-ivf-flat", "cuvs-ivf-pq", "cuvs-cagra")
                      for precision in (("fp32", "fp16") if family in ("faiss-flat", "cuvs-brute") else ("fp32",))
                      for fraction in ((1.,) if family == "faiss-flat" else (1., .5, .1))
                      for candidates in ((10, 64) if family == "cuvs-ivf-pq" else (10,))
                      for probes in ((8, 64) if family == "cuvs-ivf-flat" else (16,))]
    for ci,(family, precision, fraction, candidate_count, probes) in enumerate(configurations):
        allowed = np.arange(len(data)) % 10 < round(fraction*10)
        prefix = f"case{ci}"
        arrays[prefix+"_allowed"] = allowed
        gold_distances, gold_ids = reference_search(data, queries, 10, allowed)
        arrays[prefix+"_truth_ids"], arrays[prefix+"_truth_distances"] = gold_ids, gold_distances
        case = dict(family=family, precision=precision, allowed_fraction=fraction, candidates=candidate_count, n_probes=probes, prefix=prefix, available=False)
        record["cases"].append(case)
        index = None
        started = time.perf_counter_ns()
        try:
            bitset = np.packbits(allowed, bitorder="little").view(np.uint32)
            device_bitset = cp.asarray(bitset)
            prefilter = filters.from_bitset(device_bitset)
            if family == "faiss-flat":
                config = faiss.GpuIndexFlatConfig()
                config.device = 0
                config.useFloat16 = precision == "fp16"
                config.use_cuvs = True
                index = faiss.GpuIndexFlatL2(resources, data.shape[1], config)
                index.add(data)
                search = lambda:index.search(queries, 10)
            elif family == "cuvs-brute":
                typed_data = gpu_data.astype(cp.float16) if precision == "fp16" else gpu_data
                typed_queries = gpu_queries.astype(cp.float16) if precision == "fp16" else gpu_queries
                index = brute_force.build(typed_data)
                search = lambda:brute_force.search(index, typed_queries, 10, prefilter=prefilter)
            elif family == "cuvs-ivf-flat":
                index = ivf_flat.build(ivf_flat.IndexParams(n_lists=64, kmeans_n_iters=10, kmeans_trainset_fraction=1.), gpu_data)
                params = ivf_flat.SearchParams(n_probes=probes)
                search = lambda:ivf_flat.search(params,index,gpu_queries,10,filter=prefilter)
            elif family == "cuvs-ivf-pq":
                index = ivf_pq.build(ivf_pq.IndexParams(n_lists=64,pq_dim=16,pq_bits=8,kmeans_n_iters=10,kmeans_trainset_fraction=1.),gpu_data)
                params = ivf_pq.SearchParams(n_probes=16)
                if fraction == 1.:
                    search = lambda:ivf_pq.search(params,index,gpu_queries,candidate_count)
                else:
                    search = lambda:ivf_pq.search(params,index,gpu_queries,candidate_count,filter=prefilter)
            else:
                index = cagra.build(cagra.IndexParams(intermediate_graph_degree=64,graph_degree=32),gpu_data)
                params = cagra.SearchParams(itopk_size=64)
                search = lambda:cagra.search(params,index,gpu_queries,10,filter=prefilter)
            cp.cuda.runtime.deviceSynchronize()
            case["setup_wall_ms"] = (time.perf_counter_ns()-started)/1e6
            search()
            cp.cuda.runtime.deviceSynchronize()
            observations = []
            for _ in range(3):
                started = time.perf_counter_ns()
                reported, ids = search()
                cp.cuda.runtime.deviceSynchronize()
                observations.append((time.perf_counter_ns()-started)/1e6)
            reported = reported if isinstance(reported, np.ndarray) else cp.asnumpy(cp.asarray(reported))
            raw_ids = ids if isinstance(ids, np.ndarray) else cp.asnumpy(cp.asarray(ids))
            ids = raw_ids.astype(np.int64)
            ids[raw_ids == np.iinfo(raw_ids.dtype).max] = -1
            metrics, actual_distances = evaluate(data,queries,ids,gold_distances,gold_ids,allowed)
            assert metrics["invalid_or_filtered_ids"] == 0 and metrics["duplicate_ids"] == 0, metrics
            if family == "faiss-flat" or family == "cuvs-brute" or (family == "cuvs-ivf-flat" and probes == 64):
                assert metrics["tie_aware_recall"] >= .95, metrics
            refined_distances, refined_ids = refine(data,queries,ids,10,allowed)
            refined_metrics, _ = evaluate(data,queries,refined_ids,gold_distances,gold_ids,allowed)
            for name,value in (("raw_ids",raw_ids),("ids",ids),("reported_distances",reported),("actual_distances",actual_distances),("refined_ids",refined_ids),("refined_distances",refined_distances)):
                arrays[prefix+"_"+name] = value
            case.update(available=True, api_wall_ms=observations, candidate_set_quality=metrics, refined_quality=refined_metrics,
                        ids_sha256=digest(ids), refined_ids_sha256=digest(refined_ids), reported_distances_sha256=digest(reported))
        except (RuntimeError,TypeError,ValueError) as error:
            case.update(error=str(error), elapsed_to_error_wall_ms=(time.perf_counter_ns()-started)/1e6)
        finally:
            del index
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        print(ci+1,len(configurations),family,precision,fraction,candidate_count,case["available"],case.get("error",""),flush=True)
    npz = ROOT/"results/search-preflight.npz"
    np.savez_compressed(npz,**arrays)
    record.update(complete=True,witness_file=npz.name,witness_sha256=hashlib.sha256(npz.read_bytes()).hexdigest())
    path.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("GPU search capability probe complete",flush=True)


if __name__ == "__main__":
    main()
