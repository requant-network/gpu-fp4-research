"""Build pinned SageAttention3 in a project-local target without changing Torch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
SAGE_PIN="d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"
CUTLASS_PIN="b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd"


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--sage",required=True);parser.add_argument("--cutlass",required=True);parser.add_argument("--python",required=True);args=parser.parse_args()
    sage,cutlass=(Path(p).resolve() for p in (args.sage,args.cutlass));python=Path(args.python).absolute()
    for path,pin in ((sage,SAGE_PIN),(cutlass,CUTLASS_PIN)):
        if subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()!=pin or subprocess.check_output(["git","-C",str(path),"status","--porcelain","--untracked-files=no"],text=True).strip():raise ValueError("Pinned clean upstream checkout required")
    source=sage/"sageattention3_blackwell";link=source/"csrc/cutlass";link.parent.mkdir(exist_ok=True)
    if not link.exists():link.symlink_to(cutlass,target_is_directory=True)
    if link.resolve()!=cutlass:raise ValueError("Sage CUTLASS dependency differs from the declared pin")
    wheel_dir=ROOT/"build/sage-wheels";wheel_dir.mkdir(parents=True,exist_ok=True);log_path=ROOT/"build/sage-build.log"
    env={**os.environ,"CUDA_HOME":"/usr/local/cuda-13.3","CUDA_VISIBLE_DEVICES":"0","MAX_JOBS":"2","FAHOPPER_FORCE_BUILD":"TRUE"}
    command=[str(python),"-m","pip","wheel","--no-deps","--no-build-isolation",str(source),"--wheel-dir",str(wheel_dir)]
    with log_path.open("w",encoding="utf-8") as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,text=True,env=env)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    tracked=subprocess.check_output(["git","-C",str(sage),"ls-files","-z"]).decode().strip("\0").split("\0")
    record={"schema_version":1,"sage_commit":SAGE_PIN,"cutlass_commit":CUTLASS_PIN,"source_sha256":{"build_sage_reference.py":sha(Path(__file__))},
            "upstream_source_sha256":{name:sha(sage/name) for name in tracked},"cuda_root":"/usr/local/cuda-13.3","visible_device":"0","max_jobs":2,
            "build_log_sha256":sha(log_path),"build_exit_code":result.returncode,"complete":False,
            "scope":"Unmodified pinned SageAttention3, upstream compiler flags; pinned CUTLASS link, no dependency resolution or shared environment installation."}
    if result.returncode:
        diagnostic=log_path.read_text(encoding="utf-8")[-16000:]
        for path,label in ((sage,"<sage-source>"),(cutlass,"<cutlass-source>"),(ROOT,"<experiment>"),(python.parent.parent,"<torch-environment>")):diagnostic=diagnostic.replace(str(path),label)
        record["diagnostic_tail"]=diagnostic
    else:
        wheels=list(wheel_dir.glob("sageattn3-*.whl"))
        if len(wheels)!=1:raise ValueError("Expected one pinned SageAttention3 wheel")
        subprocess.run([str(python),"-m","pip","install","--no-deps","--no-compile","--target",str(ROOT/"build/sage-python"),str(wheels[0])],check=True)
        record["wheel_sha256"]=sha(wheels[0]);record["installed_sha256"]={str(path.relative_to(ROOT/"build/sage-python")):sha(path) for path in (ROOT/"build/sage-python").rglob("*") if path.is_file()}
        record["complete"]=True
    (ROOT/"results/sage-build.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Pinned SageAttention3 build {'complete' if record['complete'] else 'failed'}",flush=True)
    if not record["complete"]:raise SystemExit(result.returncode)


if __name__=="__main__":main()
