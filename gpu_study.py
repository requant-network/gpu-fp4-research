"""Bounded SM120 low-precision GEMM campaign with independent CPU controls."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import cupy as cp
import numpy as np

from reference import (MODES, PROFILES, SHAPES, check_output, digest, inputs,
                       quantize, relative_error)

ROOT = Path(__file__).resolve().parent
BOUNDARIES = ("prepacked", "online-a", "online-both")
SOURCES = ("reference.py", "gpu_study.py", "native/bridge.cu", "build.sh")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_snapshot(device):
    fields = ("name", "driver_version", "memory.total", "memory.used", "utilization.gpu",
              "temperature.gpu", "power.draw", "power.limit", "clocks.sm", "clocks.mem")
    output = subprocess.check_output(["nvidia-smi", "-i", str(device),
        "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"], text=True)
    return dict(zip(fields, [s.strip() for s in output.strip().split(",")]))


def environment(device, library):
    props = cp.cuda.runtime.getDeviceProperties(device)
    record = {"gpu": gpu_snapshot(device), "compute_capability": [props["major"], props["minor"]],
              "sm_count": props["multiProcessorCount"], "vram_bytes": props["totalGlobalMem"],
              "python": platform.python_version(), "os": platform.system(), "kernel": platform.release(),
              "numpy": np.__version__, "cupy": cp.__version__,
              "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
              "cuda_driver": cp.cuda.runtime.driverGetVersion(),
              "cublaslt_version": library.rq_cublas_version(),
              "gpu_exclusive": False,
              "measurement_class": "exploratory shared-system observations; other GPU contexts remain present"}
    for package in ("nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12"):
        try:
            record[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    record["nvcc"] = subprocess.check_output(["/usr/local/cuda-13.3/bin/nvcc", "--version"], text=True)
    return record


def load_library():
    library = ctypes.CDLL(str(ROOT / "build/libstudy.so"))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    library.rq_error.restype = ctypes.c_char_p
    library.rq_create.argtypes = [integer] * 4 + [ptr] * 3
    library.rq_create.restype = ptr
    library.rq_destroy.argtypes = [ptr]
    library.rq_destroy.restype = None
    for name in ("rq_quantize", "rq_run"):
        getattr(library, name).argtypes = [ptr, integer, ptr]
        getattr(library, name).restype = integer
    library.rq_candidates.argtypes = [ptr]
    library.rq_candidates.restype = integer
    library.rq_algorithm_id.argtypes = [ptr, integer]
    library.rq_algorithm_id.restype = integer
    library.rq_buffer.argtypes = [ptr, integer]
    library.rq_buffer.restype = ptr
    library.rq_bytes.argtypes = [ptr, integer]
    library.rq_bytes.restype = ctypes.c_size_t
    library.rq_cublas_version.restype = ctypes.c_size_t
    return library


class Plan:
    def __init__(self, library, mode, a, b, stream):
        self.library, self.mode, self.stream = library, mode, stream
        self.inputs = (a, b)
        self.output = cp.empty((a.shape[0], b.shape[0]), cp.float32)
        self.pointer = library.rq_create(MODES.index(mode), a.shape[0], b.shape[0], a.shape[1],
                                         a.data.ptr, b.data.ptr, self.output.data.ptr)
        if not self.pointer:
            raise RuntimeError(library.rq_error().decode())
        self.algorithm = 0

    def close(self):
        if self.pointer:
            self.library.rq_destroy(self.pointer)
            self.pointer = None

    def checked(self, status):
        if status:
            raise RuntimeError(self.library.rq_error().decode())

    def prepare(self, mask=3):
        self.checked(self.library.rq_quantize(self.pointer, mask, self.stream.ptr))

    def run(self):
        self.checked(self.library.rq_run(self.pointer, self.algorithm, self.stream.ptr))

    def step(self, boundary):
        if boundary != "prepacked":
            self.prepare(1 if boundary == "online-a" else 3)
        self.run()

    def buffer(self, index):
        size = self.library.rq_bytes(self.pointer, index)
        output = np.empty(size, np.uint8)
        if size:
            cp.cuda.runtime.memcpy(output.ctypes.data, self.library.rq_buffer(self.pointer, index), size,
                                   cp.cuda.runtime.memcpyDeviceToHost)
        return output


def capture(function, repetitions, stream):
    function()
    stream.synchronize()
    stream.begin_capture()
    try:
        for _ in range(repetitions):
            function()
    except BaseException:
        stream.end_capture()
        raise
    return stream.end_capture()


def measure(graph, repetitions, stream):
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record(stream)
    graph.launch(stream)
    end.record(stream)
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end)) / repetitions


def tune(plan):
    results = []
    for candidate in range(plan.library.rq_candidates(plan.pointer)):
        plan.algorithm = candidate
        graph = capture(plan.run, 16, plan.stream)
        for _ in range(2):
            graph.launch(plan.stream)
        plan.stream.synchronize()
        observations = [measure(graph, 16, plan.stream) for _ in range(3)]
        results.append({"candidate": candidate, "algorithm_id": plan.library.rq_algorithm_id(plan.pointer, candidate),
                        "ms": observations})
    plan.algorithm = min(results, key=lambda row: np.median(row["ms"]))["candidate"]
    return results


def verify_plan(plan, a, b, expected):
    plan.prepare()
    plan.run()
    plan.stream.synchronize()
    quantized = []
    hashes = {}
    for operand, original in enumerate((a, b)):
        raw, scales, decoded = quantize(original, plan.mode)
        actual_raw = plan.buffer(operand)
        np.testing.assert_array_equal(actual_raw, raw.view(np.uint8).ravel(), err_msg=f"{plan.mode} operand {operand}")
        if plan.mode == "fp4":
            np.testing.assert_array_equal(plan.buffer(operand + 2), scales.ravel())
        quantized.append(decoded)
        hashes[f"operand_{operand}_encoded"] = digest(raw)
        hashes[f"operand_{operand}_scales"] = digest(scales)
    actual = cp.asnumpy(plan.output)
    ratio = check_output(actual, *quantized)
    return {"encoded_sha256": hashes, "arithmetic_max_bound_ratio": ratio,
            "relative_frobenius_error": relative_error(actual, expected),
            "maximum_absolute_error": float(np.abs(actual.astype(np.float64) - expected).max()),
            "output_sha256": digest(actual), "checked_output_entries": int(actual.size)}, actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="results/pilot-20261116.json")
    parser.add_argument("--quick", action="store_true", help="Correctness preflight; not the timing campaign")
    parser.add_argument("--reverse", action="store_true", help="Reverse case and initial mode order")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("Seed must be nonnegative")
    cp.cuda.Device(args.device).use()
    library = load_library()
    scratch = cp.empty((128, 128), cp.float32)
    for invalid in ((0, 1, 127, 128), (9, 128, 128, 128), (0, -1, 128, 128)):
        unexpected = library.rq_create(*invalid, *([scratch.data.ptr] * 3))
        if unexpected:
            library.rq_destroy(unexpected)
            raise RuntimeError("Native invalid-input control unexpectedly accepted")
    del scratch
    stream = cp.cuda.Stream(non_blocking=True)
    record = {"schema_version": 1, "utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
              "quick": args.quick, "reverse": args.reverse,
              "environment": environment(args.device, library), "source_sha256": {
                  name: file_hash(ROOT / name) for name in SOURCES},
              "binary_sha256": file_hash(ROOT / "build/libstudy.so"),
              "cutlass_revision": "b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd",
              "warmups": 2 if args.quick else 5, "samples": 3 if args.quick else 30,
              "graph_repetitions": 4 if args.quick else 32, "native_invalid_controls": 3,
              "timing_scope": "CUDA events around a graph of repeated device-resident operations; per-operation milliseconds; allocation, tuning, capture, transfers and CPU validation excluded",
              "cases": []}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cases = [(shape, profile) for shape in (SHAPES[2:3] if args.quick else SHAPES) for profile in PROFILES]
    if args.reverse:
        cases.reverse()
    started = time.perf_counter()
    with stream:
        for shape, profile in cases:
            a, b = inputs(args.seed, shape, profile)
            expected = a.astype(np.float64) @ b.astype(np.float64).T
            da, db = cp.asarray(a), cp.asarray(b)
            plans = []
            try:
                modes = tuple(reversed(MODES)) if args.reverse else MODES
                for mode in modes:
                    plan = Plan(library, mode, da, db, stream)
                    plans.append(plan)
                    plan.prepare()
                stream.synchronize()
                entries, graphs = [], []
                for plan in plans:
                    tuning = tune(plan)
                    checks, output = verify_plan(plan, a, b, expected)
                    entry = {"shape": list(shape), "profile": profile, "mode": plan.mode,
                             "input_sha256": [digest(a), digest(b)], "algorithm_candidate": plan.algorithm,
                             "tuning": tuning, "checks": checks,
                             "timings_ms": {boundary: [] for boundary in BOUNDARIES},
                             "gpu_before": gpu_snapshot(args.device)}
                    if shape == (128, 128, 128):
                        witness = destination.parent / f"witness-{args.seed}-{profile}-{plan.mode}.npz"
                        np.savez_compressed(witness, a=a, b=b, output=output)
                        entry["witness"] = {"file": witness.name, "sha256": file_hash(witness)}
                    entries.append(entry)
                    for boundary in BOUNDARIES:
                        graph = capture(lambda p=plan, name=boundary: p.step(name), record["graph_repetitions"], stream)
                        graphs.append((graph, entry, boundary))
                for graph, _, _ in graphs:
                    for _ in range(record["warmups"]):
                        graph.launch(stream)
                stream.synchronize()
                # Rotate the first measured configuration to reduce fixed-order bias.
                for observation in range(record["samples"]):
                    offset = observation % len(graphs)
                    for graph, entry, boundary in graphs[offset:] + graphs[:offset]:
                        entry["timings_ms"][boundary].append(measure(graph, record["graph_repetitions"], stream))
                for plan, entry in zip(plans, entries):
                    plan.run()
                    stream.synchronize()
                    after, _ = verify_plan(plan, a, b, expected)
                    if after["encoded_sha256"] != entry["checks"]["encoded_sha256"]:
                        raise RuntimeError("Post-timing encoded operands changed")
                    entry["post_timing_checks"] = after
                    entry["output_bitwise_repeatable"] = after["output_sha256"] == entry["checks"]["output_sha256"]
                    entry["gpu_after"] = gpu_snapshot(args.device)
                record["cases"].extend(entries)
                destination.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                print(f"{shape} {profile}: {len(entries)} modes validated and measured", flush=True)
            finally:
                stream.synchronize()
                graphs = []
                for plan in plans:
                    plan.close()
    record["elapsed_seconds"] = time.perf_counter() - started
    record["environment_after"] = gpu_snapshot(args.device)
    record["complete"] = True
    destination.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(record['cases'])} validated mode/shape/profile cases to {destination}")


if __name__ == "__main__":
    main()
