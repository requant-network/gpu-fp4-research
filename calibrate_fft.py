"""Measure the empty-graph timer floor with the same external-event boundary."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT/"results/fft-timer.json"
    if output.exists():
        raise FileExistsError("Refusing to replace executed timer calibration")
    nvcc = "/usr/local/cuda-13.3/bin/nvcc"
    subprocess.run([nvcc, "-O3", "-std=c++17", "-arch=sm_120", "-Xcompiler", "-fPIC", "-shared", "native/fft_timer.cu", "-o", "build/libfft-timer.so"], cwd=ROOT, check=True)
    lib = ct.CDLL(str(ROOT/"build/libfft-timer.so"))
    lib.fft_timer_control.argtypes = [ct.c_int]
    lib.fft_timer_control.restype = ct.c_char_p
    cases = [json.loads(lib.fft_timer_control(repetitions)) for repetitions in (1, 8)]
    assert all("error" not in case and len(case["observations"]) == 30 for case in cases)
    record = {"scope": "Empty graph with 1 or 8 chained empty nodes; same external CUDA-event and native host boundary as the complete FFT graph. Diagnostic floor only, not subtracted or a model of FFT graph-node scheduling.",
              "gpu_index": 0, "gpu_exclusive": False, "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("calibrate_fft.py", "native/fft_timer.cu")},
              "binary_sha256": hashlib.sha256((ROOT/"build/libfft-timer.so").read_bytes()).hexdigest(),
              "nvcc": subprocess.check_output([nvcc, "--version"], text=True).strip(), "cases": cases}
    output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print("FFT external-event timer calibration complete")


if __name__ == "__main__":
    main()
