"""Complete resident/host vector-search comparison on one SM120 GPU.

Run each cohort separately; --smoke exercises the same execution and validation
paths on the first dataset, four queries, two variants and five observations.
The full protocol uses all five datasets, three batches, three static filters,
four query variants, five warmups and thirty observations per boundary.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import time

import cupy as cp
import numpy as np

from reference import digest
from search_baselines import Baseline
from search_cases import BATCHES,COHORTS,DATASETS,FILTERS,allow_mask,inputs,methods
from search_native_full import Native,library
from search_reference import evaluate,reference_search
from search_stage_full import capture_stage

ROOT = Path(__file__).resolve().parent
SOURCES = ("search_study.py","search_cases.py","search_stage_full.py","search_baselines.py","search_memory.py","search_native_full.py",
           "native/search_full_bridge.cu","build_search_full.sh","search_native.py","search_reference.py","search_validation.py","reference.py",
           "native/search_bridge.cu","native/search_select.cpp","build_search.sh","native/hardware_bridge.cu",
           "native/application_bridge.cu","native/optimized_bridge.cu","native/bridge.cu")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    temporary = path.with_name(path.name+".part")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def error_text(error):
    # Keep the actual API diagnostic, excluding machine-specific stack paths/addresses.
    text = str(error).split("Obtained ")[0].strip()
    text = re.sub(r"(?:/[^\s:]+)+/((?:raft|rmm|cuvs)/[^\s:]+)",r"<vendor>/\1",text)
    text = re.sub(r"/(?:home|pyenv|__w)/[^\s:]+", "<vendor-path>",text)
    return type(error).__name__+": "+text[:1200]


def quality(database,queries,distances,ids,gold,truth,allowed):
    metrics,exact = evaluate(database,queries,ids,gold,truth,allowed)
    if ids.shape != truth.shape or distances.shape != ids.shape:
        raise AssertionError("Search output geometry mismatch")
    good = (ids >= 0) & (ids < len(database))
    finite = np.isfinite(distances) & good
    metrics.update(missing_ids=int(np.count_nonzero(ids == -1)),
                   nonfinite_reported_valid_ids=int(np.count_nonzero(good & ~np.isfinite(distances))),
                   max_absolute_distance_error=float(np.max(abs(distances[finite].astype(np.float64)-exact[finite]))) if finite.any() else None)
    metrics["ids_valid"] = metrics["duplicate_ids"] == metrics["invalid_or_filtered_ids"] == 0
    return metrics


def truth_record(directory,name,cohort,database,queries,provenance,sources):
    path = directory/(name+"-truth.json")
    npz = path.with_suffix(".npz")
    if path.exists():
        record = json.loads(path.read_text())
        assert record["source_sha256"] == sources and record["complete"]
        assert record["database_sha256"] == digest(database) and record["queries_sha256"] == digest(queries)
        assert sha(npz) == record["witness_sha256"]
        with np.load(npz,allow_pickle=False) as archive:
            arrays = {key:archive[key] for key in archive.files}
        return record,arrays
    arrays,filters = {},[]
    started = time.perf_counter_ns()
    for percent in FILTERS:
        allowed = allow_mask(len(database),percent,cohort)
        one = time.perf_counter_ns()
        distances,ids = reference_search(database,queries.reshape(-1,database.shape[1]),10,allowed)
        arrays[f"f{percent}_distances"] = distances.reshape(*queries.shape[:2],10)
        arrays[f"f{percent}_ids"] = ids.reshape(*queries.shape[:2],10)
        filters.append(dict(percent=percent,allowed_count=int(allowed.sum()),allowed_sha256=digest(allowed),cpu_oracle_ms=(time.perf_counter_ns()-one)/1e6))
        print("TRUTH",cohort,name,percent,filters[-1]["cpu_oracle_ms"],flush=True)
    np.savez_compressed(npz,**arrays)
    record = dict(complete=True,cohort=cohort,dataset=name,database_shape=list(database.shape),queries_shape=list(queries.shape),
                  database_sha256=digest(database),queries_sha256=digest(queries),provenance=provenance,filters=filters,
                  cpu_oracle="All database vectors, direct FP64 squared differences, streamed top-k; ID breaks exact ties",
                  total_cpu_oracle_ms=(time.perf_counter_ns()-started)/1e6,source_sha256=sources,
                  witness_file=npz.name,witness_sha256=sha(npz))
    write_json(path,record)
    return record,arrays


def run_case(lib,database,queries,allowed,method,truth_arrays,percent,batch,observations,warmups,arrays,prefix):
    plan = None
    record = dict(method=method,available=False,valid=False,graph_available=False,outputs={},boundaries=[],stage_witness=None)
    started = time.perf_counter_ns()
    try:
        plan = Native(lib,database,queries[0,:batch].copy(),allowed,method) if method["family"] == "native" else Baseline(database,queries[0,:batch].copy(),allowed,method)
        cp.cuda.runtime.deviceSynchronize()
        record.update(available=True,setup_wall_ms=(time.perf_counter_ns()-started)/1e6)
        started = time.perf_counter_ns()
        plan.run()
        record["first_run_wall_ms"] = (time.perf_counter_ns()-started)/1e6

        def remember(variant):
            reported,ids = plan.result()
            key = digest(ids)+"-"+digest(reported)
            output_key = f"v{variant}-"+key
            if output_key not in record["outputs"]:
                name = prefix+"_o"+str(len(record["outputs"]))
                arrays[name+"_ids"],arrays[name+"_distances"] = ids,reported
                metrics = quality(database,queries[variant,:batch],reported,ids,truth_arrays[f"f{percent}_distances"][variant,:batch],truth_arrays[f"f{percent}_ids"][variant,:batch],allowed)
                record["outputs"][output_key] = dict(prefix=name,variant=variant,ids_sha256=digest(ids),distances_sha256=digest(reported),quality=metrics)
            return output_key

        remember(0)
        if method["family"] == "native":
            stage,stage_record = capture_stage(plan,queries[0,:batch].copy())
            stage_prefix = prefix+"_stage"
            for key,value in stage.items():
                arrays[stage_prefix+"_"+key] = value
            record["stage_witness"] = dict(prefix=stage_prefix,**stage_record)
        # Prove that the graph contains the complete changing-query operation.
        started = time.perf_counter_ns()
        try:
            plan.capture(1)
            record["graph_capture_wall_ms"] = (time.perf_counter_ns()-started)/1e6
            record["graph_available"] = True
            for variant in range(len(queries)):
                plan.load(queries[variant,:batch].copy())
                plan.time(True)
                captured = remember(variant)
                plan.run()
                ordinary = remember(variant)
                # Algorithms may choose different true-distance ties; equality is a control,
                # and a mismatch keeps the graph out of performance comparisons.
                if captured != ordinary:
                    record["graph_available"] = False
                    record["graph_error"] = "Changed-query graph output differs from uncaptured output; retained as ineligible"
                    break
        except Exception as error:
            record["graph_available"] = False
            record["graph_error"] = error_text(error)
            record["graph_capture_to_error_wall_ms"] = (time.perf_counter_ns()-started)/1e6
        boundaries = ["resident-uncaptured","host-service"]
        if record["graph_available"]:
            boundaries.insert(1,"resident-captured")
        for boundary in boundaries:
            plan.load(queries[0,:batch].copy())
            for _ in range(warmups):
                if boundary == "host-service":
                    plan.load(queries[0,:batch].copy())
                    plan.run()
                    plan.result()
                else:
                    plan.time(boundary == "resident-captured")
            rows = []
            for observation in range(observations):
                variant = observation%len(queries)
                q = queries[variant,:batch].copy()
                if boundary == "host-service":
                    started = time.perf_counter_ns()
                    plan.load(q)
                    plan.run()
                    plan.result()
                    wall_ms = (time.perf_counter_ns()-started)/1e6
                    cuda_ms = None
                else:
                    plan.load(q)
                    plan.time(boundary == "resident-captured")
                    meta = plan.metadata()
                    cuda_ms,wall_ms = meta["cuda_ms"],meta["wall_ms"]
                rows.append(dict(variant=variant,cuda_ms=cuda_ms,wall_ms=wall_ms,output_key=remember(variant)))
            record["boundaries"].append(dict(name=boundary,observations=rows))
        record["metadata"] = plan.metadata()
        record["valid"] = all(item["quality"]["ids_valid"] for item in record["outputs"].values())
    except Exception as error:
        record.update(error=error_text(error),elapsed_to_error_wall_ms=(time.perf_counter_ns()-started)/1e6)
        if not record["available"]:
            record["unavailable_scope"] = "Construction/API capability; no performance or quality claim"
        else:
            record["valid"] = False
            # Context failures cannot be hidden by continuing to produce records.
            cp.cuda.runtime.deviceSynchronize()
    finally:
        if plan is not None:
            plan.close()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort",choices=COHORTS,required=True)
    parser.add_argument("--dataset",choices=DATASETS)
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    cp.cuda.Device(0).use()
    assert cp.cuda.Device(0).compute_capability == "120"
    lib = library()
    sources = {name:sha(ROOT/name) for name in SOURCES}
    binary_sha = sha(ROOT/"build/libsearch-full.so")
    directory = ROOT/"results"/(("search-smoke-" if args.smoke else "search-full-")+args.cohort)
    directory.mkdir(exist_ok=True)
    names = [args.dataset] if args.dataset else (["uniform-small"] if args.smoke else list(DATASETS))
    if args.cohort == "b":
        names.reverse()
    variants,max_batch,observations,warmups = (2,4,5,2) if args.smoke else (4,128,30,5)
    batches = (1,4) if args.smoke else BATCHES
    for name in names:
        database,queries,provenance = inputs(name,args.cohort,variants,max_batch)
        truth,truth_arrays = truth_record(directory,name,args.cohort,database,queries,provenance,sources)
        geometry = [(batch,percent) for batch in batches for percent in FILTERS]
        if args.cohort == "b":
            geometry.reverse()
        for batch,percent in geometry:
            path = directory/f"{name}-b{batch}-f{percent}.json"
            if path.exists():
                old = json.loads(path.read_text())
                assert old["complete"] and old["source_sha256"] == sources and old["binary_sha256"] == binary_sha
                assert sha(path.with_suffix(".npz")) == old["witness_sha256"]
                print("RESUME",path.name,flush=True)
                continue
            allowed = allow_mask(len(database),percent,args.cohort)
            grid = methods(len(database),int(allowed.sum()),percent)
            if args.cohort == "b":
                grid.reverse()
            record = dict(study=10,complete=False,smoke=args.smoke,cohort=args.cohort,dataset=name,batch=batch,allowed_percent=percent,
                          source_sha256=sources,binary_sha256=binary_sha,environment_file="../search-environment.json",
                          environment_sha256=sha(ROOT/"results/search-environment.json"),gpu_index=0,gpu_exclusive=False,
                          runtime_api_version=cp.cuda.runtime.runtimeGetVersion(),driver_api_version=cp.cuda.runtime.driverGetVersion(),
                          truth_file=name+"-truth.json",truth_record_sha256=sha(directory/(name+"-truth.json")),
                          database_sha256=truth["database_sha256"],queries_sha256=truth["queries_sha256"],allowed_sha256=digest(allowed),
                          observations_per_boundary=observations,warmups_per_boundary=warmups,query_variants=variants,
                          timing_contract={"resident":"Current FP32 query already on device; encoding, query norms, search/selection/filter/global-ID mapping and optional direct FP32 refinement included. CUDA events include CPU launch gaps in uncaptured operation. Captured graphs repeat one complete operation.",
                                           "host_service":"Python wall interval: pageable FP32 query H2D, all uncaptured computation, synchronization and complete F32 distances/int64 global IDs D2H. Host allocation/dispatch included; database preparation/index training excluded and separately charged.",
                                           "setup":"Constructor wall interval includes static CPU compaction/padding, uploads, database encoding/norms, allocations and library/native tuning or index training. First run and capture are separately recorded.",
                                           "memory":"Named owned buffers and available library allocation accounting; resource reservations and shared-device free-memory observations are separate. No physical peak or exclusive-device claim."},
                          started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=[])
            arrays = {}
            for index,method in enumerate(grid):
                case = run_case(lib,database,queries,allowed,method,truth_arrays,percent,batch,observations,warmups,arrays,"c"+str(index))
                record["cases"].append(case)
                print(args.cohort,name,batch,percent,index+1,len(grid),method,case["available"],case["valid"],case.get("error",""),flush=True)
            npz = path.with_suffix(".npz")
            np.savez_compressed(npz,**arrays)
            record.update(complete=True,witness_file=npz.name,witness_sha256=sha(npz),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_json(path,record)
    print("Search cohort collection complete",args.cohort,"smoke" if args.smoke else "full",flush=True)


if __name__ == "__main__":
    main()
