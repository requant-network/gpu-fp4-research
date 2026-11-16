"""Record actual SM120 SASS and static resources of an executed native library."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parent


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",default="build/libextended-128.so")
    parser.add_argument("--output",default="results/native-inspection-128.json")
    args=parser.parse_args()
    library=ROOT/args.library
    tool="/usr/local/cuda-13.3/bin/cuobjdump"
    resource=subprocess.check_output([tool,"--dump-resource-usage",str(library)],text=True)
    sass=subprocess.check_output([tool,"--dump-sass",str(library)],text=True)
    functions={}
    for match in re.finditer(r" Function ([^\n]+):\n([^\n]+)",resource):
        symbol,fields=match.groups()
        functions[symbol]={"symbol":symbol,"resources":{key:int(value) for key,value in
                           re.findall(r"([A-Z]+(?:\[\d+\])?):(\d+)",fields)},"opcodes":{},"examples":[]}
    symbol=None
    counts={}
    for line in sass.splitlines():
        function=re.search(r"Function\s*:\s*(\S+)",line)
        if function:
            symbol=function.group(1)
            counts.setdefault(symbol,Counter())
            continue
        instruction=re.search(r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]*)\s",line)
        if symbol and instruction:
            mnemonic=instruction.group(1)
            counts[symbol][mnemonic]+=1
            if symbol in functions and any(tag in mnemonic for tag in ("MMA","UTMA","F2FP","REDUX","LDGSTS")):
                examples=functions[symbol]["examples"]
                if len(examples)<12 and not any(mnemonic==entry["opcode"] for entry in examples):
                    examples.append({"opcode":mnemonic,"assembly":line.strip()})
    for symbol,function in functions.items():
        function["opcodes"]=dict(sorted(counts.get(symbol,{}).items()))
        function["instruction_total"]=sum(counts.get(symbol,{}).values())
    if not functions or not any("MMA" in opcode for function in functions.values() for opcode in function["opcodes"]):
        raise RuntimeError("Expected actual Tensor Core instructions in this native library")
    record={"library":args.library,"binary_sha256":sha(library.read_bytes()),
            "tool_version":subprocess.check_output([tool,"--version"],text=True),
            "resource_dump_sha256":sha(resource.encode()),"sass_dump_sha256":sha(sass.encode()),
            "source_sha256":{"inspect_native.py":sha(Path(__file__).read_bytes())},
            "architectures":sorted(set(re.findall(r"arch = (\S+)",resource))),
            "kernels":list(functions.values()),
            "scope":"Static emitted instruction/resource evidence, not dynamic execution counts, achieved occupancy or an architectural throughput ceiling"}
    destination=ROOT/args.output
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    for kernel in record["kernels"]:
        name="CUTLASS GEMM" if "device_kernel" in kernel["symbol"] else kernel["symbol"][:72]
        interesting={key:value for key,value in kernel["opcodes"].items() if any(tag in key for tag in ("MMA","UTMA","F2FP","REDUX","LDGSTS"))}
        print(name,kernel["resources"].get("REG"),kernel["instruction_total"],interesting)


if __name__=="__main__":main()
