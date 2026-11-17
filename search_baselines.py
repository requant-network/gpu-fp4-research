"""Resident GPU Faiss/cuVS references with explicit upload, filtering and refinement."""
import ctypes as ct
import gc
from pathlib import Path
import time

import cupy as cp
from cuvs.common import Resources
from cuvs.neighbors import brute_force,cagra,filters,ivf_flat,ivf_pq,refine
import faiss
import libcuvs
import numpy as np
import rmm

from search_native import validate
from search_memory import faiss_reservation_bytes

_global_ids = cp.ElementwiseKernel("raw T local, raw int64 mapping, int64 population, bool remap","int64 result",
    "const long long id = static_cast<long long>(local[i]); result = id >= 0 && id < population ? (remap ? mapping[id] : id) : -1;",
    "requant_search_global_ids")
class Baseline:
    def __init__(self,database,queries,allowed,method,k=10):
        self.stream = self.index = self.resources = self.faiss_resources = self.graph = self.prefilter = None
        self.arrays = {}
        try:
            self._initialize(database,queries,allowed,method,k)
        except Exception:
            self.close()
            raise

    def _initialize(self,database,queries,allowed,method,k):
        validate(database,database.shape,np.float32)
        validate(queries,queries.shape,np.float32)
        validate(allowed,(len(database),),np.bool_)
        if database.ndim != 2 or queries.ndim != 2 or database.shape[1] != queries.shape[1]:
            raise ValueError("Invalid baseline search dimensions")
        if not np.isfinite(database).all() or not np.isfinite(queries).all():
            raise ValueError("Finite search inputs are required")
        cp.cuda.Device(0).use()
        # cuVS 26.10 caches filtered CAGRA payloads until static destruction.
        # Their stored stream must outlive individual plans. PTDS has CUDA-context
        # lifetime, unlike an owned stream destroyed after each comparison.
        self.stream = cp.cuda.Stream.ptds if method["family"] == "cagra" else cp.cuda.Stream(non_blocking=True)
        self.index = self.resources = self.faiss_resources = self.graph = None
        self.arrays = {}
        self.method,self.k,self.n,self.d,self.query_shape = method,k,len(database),database.shape[1],queries.shape
        self.count = method.get("candidates",k)
        if self.count < k or self.count > np.count_nonzero(allowed):
            raise ValueError("The candidate count must fit the allowed population")
        self.family = method["family"]
        self.rerank = method.get("refine",False)
        if self.rerank and self.family != "ivf-pq":
            raise ValueError("External candidate refinement is used only for the PQ reference")
        if self.family == "ivf-pq" and not allowed.all():
            raise ValueError("The installed standalone cuVS IVF-PQ Python API has no filter argument; see retained capability probe")
        self.compact = self.family == "faiss" and not allowed.all()
        started = time.perf_counter_ns()
        selected = np.flatnonzero(allowed) if self.compact else np.arange(self.n,dtype=np.int64)
        host_database = np.ascontiguousarray(database[selected]) if self.compact else database
        self.compaction_cpu_ms = (time.perf_counter_ns()-started)/1e6
        self.population = len(host_database)
        self.faiss_temp_bytes = faiss_reservation_bytes(self.population,len(queries)) if self.family == "faiss" else 0
        self.free_before = cp.cuda.runtime.memGetInfo()[0]
        with self.stream:
            self.arrays["queries"] = cp.asarray(queries)
            self.arrays["database"] = cp.asarray(host_database)
            self.arrays["mapping"] = cp.asarray(selected) if self.compact else cp.empty(0,dtype=cp.int64)
            mask = np.ones(len(host_database),dtype=bool) if self.compact else allowed
            self.arrays["bitset"] = cp.asarray(np.packbits(np.pad(mask,(0,(-len(mask))%32)),bitorder="little").view(np.uint32))
            self.arrays["raw_distances"] = cp.empty((len(queries),self.count),dtype=cp.float32)
            self.arrays["raw_ids"] = cp.empty((len(queries),self.count),dtype=cp.uint32 if self.family == "cagra" else cp.int64)
            self.arrays["ids"] = cp.empty((len(queries),k),dtype=cp.int64)
            self.arrays["distances"] = cp.empty((len(queries),k),dtype=cp.float32) if self.rerank else self.arrays["raw_distances"]
        self.prefilter = filters.from_bitset(self.arrays["bitset"]) if not mask.all() else None
        self.stream.synchronize()
        precision = method.get("precision","fp32")
        if self.family == "faiss":
            if self.count != k:
                raise ValueError("The exact Faiss control returns k neighbours directly")
            self.faiss_resources = faiss.StandardGpuResources()
            self.faiss_resources.setTempMemory(self.faiss_temp_bytes)
            # Initialize the RAFT handle before assigning its user stream (Faiss 1.16.0).
            self.faiss_resources.getDefaultStream(0)
            self.faiss_resources.setDefaultStream(0,faiss.cast_integer_to_cudastream_t(self.stream.ptr))
            config = faiss.GpuIndexFlatConfig()
            config.device,config.useFloat16,config.use_cuvs = 0,precision == "fp16",method.get("use_cuvs",False)
            self.index = faiss.GpuIndexFlatL2(self.faiss_resources,self.d,config)
            self.index.add_c(self.population,faiss.cast_integer_to_float_ptr(self.arrays["database"].data.ptr))
            self.query_pointer = faiss.cast_integer_to_float_ptr(self.arrays["queries"].data.ptr)
            self.distance_pointer = faiss.cast_integer_to_float_ptr(self.arrays["raw_distances"].data.ptr)
            self.id_pointer = faiss.cast_integer_to_idx_t_ptr(self.arrays["raw_ids"].data.ptr)
            self.stream.synchronize()
            # FlatIndex owns its copied vectors; add has completed on the configured stream.
            self.arrays.pop("database")
        else:
            self.resources = Resources()
            api = ct.CDLL(str(Path(libcuvs.__file__).parent/"lib64/libcuvs_c.so"))
            api.cuvsStreamSet.argtypes = [ct.c_size_t,ct.c_void_p]
            api.cuvsStreamSet.restype = ct.c_int
            if api.cuvsStreamSet(self.resources.get_c_obj(),self.stream.ptr) != 1:
                raise RuntimeError("Could not assign the reference CUDA stream through the C API")
            api.cuvsResourcesSetWorkspacePool.argtypes = [ct.c_size_t,ct.c_size_t]
            api.cuvsResourcesSetWorkspacePool.restype = ct.c_int
            if precision == "fp16" and self.family == "brute":
                with self.stream:
                    self.arrays["typed_database"] = self.arrays["database"].astype(cp.float16)
                    self.arrays["typed_queries"] = cp.empty(queries.shape,dtype=cp.float16)
            else:
                self.arrays["typed_database"] = self.arrays["database"]
                self.arrays["typed_queries"] = self.arrays["queries"]
            self.n_lists = method.get("n_lists",max(64,min(1024,self.n//1024)))
            self.n_probes = min(method.get("n_probes",16),self.n_lists)
            if self.family == "brute":
                self.index = brute_force.build(self.arrays["typed_database"],resources=self.resources)
            elif self.family == "ivf-flat":
                params = ivf_flat.IndexParams(n_lists=self.n_lists,kmeans_n_iters=20,kmeans_trainset_fraction=min(1.,max(.1,16384/self.n)))
                self.index = ivf_flat.build(params,self.arrays["database"],resources=self.resources)
                self.search_parameters = ivf_flat.SearchParams(n_probes=self.n_probes)
            elif self.family == "ivf-pq":
                params = ivf_pq.IndexParams(n_lists=self.n_lists,pq_dim=self.d//8,pq_bits=8,kmeans_n_iters=20,kmeans_trainset_fraction=min(1.,max(.1,16384/self.n)))
                self.index = ivf_pq.build(params,self.arrays["database"],resources=self.resources)
                self.search_parameters = ivf_pq.SearchParams(n_probes=self.n_probes,lut_dtype=np.dtype(method.get("lut_dtype","float32")).type,internal_distance_dtype=np.float32)
            elif self.family == "cagra":
                self.index = cagra.build(cagra.IndexParams(intermediate_graph_degree=64,graph_degree=32),self.arrays["database"],resources=self.resources)
                self.search_parameters = cagra.SearchParams(itopk_size=method.get("itopk",64),search_width=method.get("search_width",1))
            else:
                raise ValueError("Unknown GPU reference family")
            self.resources.sync()
            if method.get("workspace_pool",True) and api.cuvsResourcesSetWorkspacePool(self.resources.get_c_obj(),64 << 20) != 1:
                raise RuntimeError("Could not configure the reference's per-resource workspace pool")
            # These index families own the vectors/codes; PQ refinement still needs FP32 originals.
            if self.family in ("ivf-flat","ivf-pq") and not self.rerank:
                self.arrays.pop("database")
                self.arrays.pop("typed_database")
            elif self.family == "brute" and precision == "fp16":
                self.arrays.pop("database")
        self.start,self.end = cp.cuda.Event(),cp.cuda.Event()
        self.repetitions,self.cuda_ms,self.wall_ms = 1,0.,0.
        self.stream.synchronize()

    def operation(self):
        a = self.arrays
        with self.stream:
            if self.family == "faiss":
                self.index.search_c(self.query_shape[0],self.query_pointer,self.count,self.distance_pointer,self.id_pointer)
            else:
                if a["typed_queries"] is not a["queries"]:
                    a["typed_queries"][...] = a["queries"]
                common = dict(neighbors=a["raw_ids"],distances=a["raw_distances"],resources=self.resources)
                if self.family == "brute":
                    brute_force.search(self.index,a["typed_queries"],self.count,prefilter=self.prefilter,**common)
                elif self.family == "ivf-flat":
                    ivf_flat.search(self.search_parameters,self.index,a["queries"],self.count,filter=self.prefilter,**common)
                elif self.family == "ivf-pq":
                    ivf_pq.search(self.search_parameters,self.index,a["queries"],self.count,**common)
                else:
                    cagra.search(self.search_parameters,self.index,a["queries"],self.count,filter=self.prefilter,**common)
            if self.rerank:
                refine(a["database"],a["queries"],a["raw_ids"],self.k,indices=a["ids"],distances=a["distances"],resources=self.resources)
            else:
                _global_ids(a["raw_ids"],a["mapping"],np.int64(self.population),self.compact,a["ids"])

    def run(self):
        self.operation()
        self.stream.synchronize()

    def load(self,queries):
        validate(queries,self.query_shape,np.float32)
        if not np.isfinite(queries).all():
            raise ValueError("Finite queries are required")
        self.arrays["queries"].set(queries,stream=self.stream)
        self.stream.synchronize()

    def capture(self,repetitions):
        if self.graph is not None or repetitions not in range(1,33):
            raise ValueError("Invalid/repeated reference graph capture")
        self.stream.begin_capture()
        try:
            for _ in range(repetitions):
                self.operation()
            self.graph = self.stream.end_capture()
        except Exception:
            try:
                self.stream.end_capture()
            except RuntimeError:
                pass
            raise
        self.repetitions = repetitions

    def time(self,captured):
        if captured and self.graph is None:
            raise ValueError("Capture the reference first")
        started = time.perf_counter_ns()
        self.start.record(self.stream)
        if captured:
            self.graph.launch(stream=self.stream)
        else:
            self.operation()
        self.end.record(self.stream)
        self.end.synchronize()
        divisor = self.repetitions if captured else 1
        self.cuda_ms = cp.cuda.get_elapsed_time(self.start,self.end)/divisor
        self.wall_ms = (time.perf_counter_ns()-started)/1e6/divisor

    def result(self):
        return cp.asnumpy(self.arrays["distances"],stream=self.stream),cp.asnumpy(self.arrays["ids"],stream=self.stream)

    def metadata(self):
        unique = {}
        for value in self.arrays.values():
            unique[value.data.ptr] = max(unique.get(value.data.ptr,0),value.nbytes)
        result = {"cuda_ms":self.cuda_ms,"wall_ms":self.wall_ms,"python_owned_array_bytes":sum(unique.values()),
                  "array_bytes":{key:value.nbytes for key,value in self.arrays.items()},"graph_repetitions":self.repetitions,
                  "cpu_filter_compaction_ms":self.compaction_cpu_ms,"static_filter_compaction":self.compact,
                  "faiss_temporary_reservation_bytes":self.faiss_temp_bytes,
                  "cuvs_initial_workspace_pool_bytes":64 << 20 if self.family != "faiss" and self.method.get("workspace_pool",True) else 0,
                  "shared_device_free_bytes_before":self.free_before,"shared_device_free_bytes_now":cp.cuda.runtime.memGetInfo()[0]}
        if self.family == "faiss":
            info = self.faiss_resources.getMemoryInfo()
            result["faiss_memory_info"] = {str(device):{str(key):list(value) for key,value in entries.items()} for device,entries in info.items()}
        result["rmm_device_resource"] = type(rmm.mr.get_current_device_resource()).__name__
        return result

    def close(self):
        if self.stream is not None:
            self.stream.synchronize()
        self.graph = self.index = self.resources = self.faiss_resources = None
        self.arrays.clear()
        self.prefilter = None
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
