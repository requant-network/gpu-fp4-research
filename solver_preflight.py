"""Record an actual cuSOLVER IRS capability probe; not a completed solver study."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent


def main():
    result=subprocess.run([str(ROOT/"build/solver-preflight")],text=True,capture_output=True,check=True);record=json.loads(result.stdout)
    sha=lambda file:hashlib.sha256(file.read_bytes()).hexdigest()
    record.update({"study":7,"scope":"One-shot capability/numerical probe on two 128-dimensional constructed SPD problems, not full timings, real-matrix evaluation or a completed study.","gpu_exclusive":False,
        "source_sha256":{name:sha(ROOT/name) for name in ("solver_preflight.py","native/solver_preflight.cu","build_solver_preflight.sh")},"binary_sha256":sha(ROOT/"build/solver-preflight"),
        "nvcc":subprocess.check_output(["/usr/local/cuda-13.3/bin/nvcc","--version"],text=True)})
    (ROOT/"results/solver-preflight.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Solver preflight: {len(record['cases'])} attempted configurations, {sum('solution' in c for c in record['cases'])} returned solution arrays")


if __name__=="__main__":main()
