"""Reproducible vector-search inputs and the declared comparison grid (CPU only)."""
import hashlib
import json
from pathlib import Path

import numpy as np

from reference import digest

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "uniform-small":(16384,128,"uniform"),
    "clustered":(65536,256,"clustered"),
    "shifted":(262144,128,"shifted"),
    "uniform-million":(1000000,128,"uniform"),
    "sift-million":(1000000,128,"sift"),
}
COHORTS = {"a":20261130,"b":20261230}
BATCHES = (1,32,128)
FILTERS = (100,10,1)


def load_sift():
    root = ROOT/"fixtures/search-sift"
    manifest = json.loads((root/"manifest.json").read_text())
    data,queries = [],None
    for entry in manifest["files"]:
        path = root/entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("SIFT fixture file checksum mismatch")
        with np.load(path,allow_pickle=False) as archive:
            vectors = archive["vectors"]
        if digest(vectors) != entry["vectors_sha256"]:
            raise ValueError("SIFT vector checksum mismatch")
        if entry["kind"] == "database":
            data.append(vectors)
        else:
            queries = vectors
    return np.concatenate(data).astype(np.float32),queries.astype(np.float32)


def inputs(name,cohort,variants=4,max_batch=128):
    if name not in DATASETS or cohort not in COHORTS or variants not in range(1,5) or max_batch not in range(1,129):
        raise ValueError("Unknown search dataset, cohort or query geometry")
    n,d,profile = DATASETS[name]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([COHORTS[cohort],n,d,list(DATASETS).index(name)])))
    count = variants*max_batch
    if profile == "sift":
        database,source_queries = load_sift()
        # Disjoint genuine query rows for the two cohorts; database remains canonical SIFT1M.
        start = 0 if cohort == "a" else 5000
        selected = start+rng.permutation(5000)[:count]
        queries = np.ascontiguousarray(source_queries[selected])
        provenance = {"query_source_ids":selected.tolist(),"fixture_manifest_sha256":hashlib.sha256((ROOT/"fixtures/search-sift/manifest.json").read_bytes()).hexdigest()}
    else:
        database = np.empty((n,d),dtype=np.float32)
        centers = rng.integers(-64,65,(64,d),dtype=np.int32).astype(np.float32)/16 if profile == "clustered" else None
        for start in range(0,n,65536):
            size = min(65536,n-start)
            noise = rng.integers(-2048,2049,(size,d),dtype=np.int32).astype(np.float32)
            if profile == "uniform":
                database[start:start+size] = noise/4096
            elif profile == "clustered":
                database[start:start+size] = centers[rng.integers(0,64,size)]+noise/262144
            else:
                database[start:start+size] = np.float32(32)+noise/65536
        anchors = rng.integers(32,n,count)
        noise = rng.integers(-128,129,(count,d),dtype=np.int32).astype(np.float32)
        divisor = 1024 if profile == "uniform" else 262144
        queries = np.ascontiguousarray(database[anchors]+noise/divisor)
        # Genuine equal-distance alternatives exercise the explicit ID/tie contract.
        database[:16] = database[32]
        queries[0] = database[32]
        provenance = {"seed":COHORTS[cohort],"profile":profile,"anchor_ids_sha256":digest(anchors)}
    return database,queries.reshape(variants,max_batch,d),provenance


def allow_mask(n,percent,cohort):
    if percent not in FILTERS or cohort not in COHORTS:
        raise ValueError("Invalid static global filter")
    return np.ascontiguousarray((np.arange(n,dtype=np.int64)*37+COHORTS[cohort])%100 < percent)


def methods(n,allowed_count,percent):
    """All declared precisions, candidate budgets, indexed controls and compact filters."""
    configurations = []
    for precision in range(6):
        for candidates,rerank in ((10,False),(32,True),(128,True),(512,True)):
            configurations.append(dict(family="native",precision=precision,candidates=min(candidates,allowed_count),refine=rerank,norm_mode=1,compact=False))
    configurations.append(dict(family="native",precision=0,candidates=min(128,allowed_count),refine=True,norm_mode=0,compact=False))
    if percent != 100:
        for precision in (0,2):
            for candidates in (32,128):
                configurations.append(dict(family="native",precision=precision,candidates=min(candidates,allowed_count),refine=True,norm_mode=1,compact=True))
    configurations.extend(dict(family="faiss",precision=p,use_cuvs=use) for p in ("fp32","fp16") for use in (False,True))
    configurations.extend(dict(family="brute",precision=p) for p in ("fp32","fp16"))
    lists = max(64,min(1024,n//1024))
    configurations.extend(dict(family="ivf-flat",n_lists=lists,n_probes=probes) for probes in sorted({min(16,lists),lists}))
    configurations.extend(dict(family="ivf-pq",n_lists=lists,n_probes=min(probes,lists),candidates=count,refine=rerank,lut_dtype="float32")
                          for probes in (16,64) for count,rerank in ((10,False),(128,True)))
    configurations.extend(dict(family="cagra",itopk=itopk,search_width=1) for itopk in (64,256))
    return configurations
