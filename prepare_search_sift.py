"""Fetch checksum-pinned SIFT1M mirror files and store lossless uint8 shards."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/"
FILES = (("database","sift_base.fvecs",1000000,"21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816"),
         ("queries","sift_query.fvecs",10000,"f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(cache,name,rows,expected_sha):
    size = rows*129*4
    target = cache/name
    source_url = BASE+name
    if not target.exists():
        def part(start):
            stop = min(size,start+4*1024*1024)
            path = cache/f"{name}.range-{start}-{stop-1}"
            if not path.exists():
                for attempt in range(3):
                    try:
                        request = urllib.request.Request(source_url,headers={"Range":f"bytes={start}-{stop-1}"})
                        with urllib.request.urlopen(request,timeout=30) as response:
                            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop-1}/{size}":
                                raise ValueError("The mirror did not honor the exact bounded HTTP range")
                            payload = bytearray()
                            while len(payload) < stop-start:
                                chunk = response.read(min(64*1024,stop-start-len(payload)))
                                if not chunk:
                                    raise EOFError("Truncated SIFT mirror range")
                                payload.extend(chunk)
                        partial = path.with_name(path.name+".part")
                        partial.write_bytes(payload)
                        partial.replace(path)
                        break
                    except (TimeoutError,OSError):
                        if attempt == 2:
                            raise
            if path.stat().st_size != stop-start:
                raise ValueError("Invalid cached SIFT range byte extent")
            return path
        starts = list(range(0,size,4*1024*1024))
        paths = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            jobs = {executor.submit(part,start):start for start in starts}
            for completed,job in enumerate(as_completed(jobs),1):
                paths[jobs[job]] = job.result()
                print(f"Fetched {name}: {completed}/{len(starts)} ranges",flush=True)
        temporary = target.with_suffix(".part")
        with temporary.open("wb") as output:
            for start in starts:
                output.write(paths[start].read_bytes())
        if sha(temporary) != expected_sha:
            raise ValueError("The complete SIFT source differs from its published SHA256")
        temporary.replace(target)
    if target.stat().st_size != size or sha(target) != expected_sha:
        raise ValueError("Cached SIFT source differs from its pinned bytes")
    return target,{"source_url":source_url,"source_rows":rows,"dimension":128,"bytes":size,"sha256":expected_sha,
                   "format":"Per-row little-endian int32 dimension followed by 128 little-endian FP32 values"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=Path.home()/"requant-research/search-data")
    args = parser.parse_args()
    output = ROOT/"fixtures/search-sift"
    if output.exists():
        raise FileExistsError("Refusing to replace canonical search data")
    args.cache.mkdir(parents=True,exist_ok=True)
    downloaded = [(kind,*fetch(args.cache,name,rows,expected_sha)) for kind,name,rows,expected_sha in FILES]
    output.mkdir(parents=True)
    files = []
    for kind,path,info in downloaded:
        rows = info["source_rows"]
        data = np.memmap(path,dtype="<f4",mode="r",shape=(rows,129))
        chunk = 65536 if kind == "database" else rows
        for part,start in enumerate(range(0,rows,chunk)):
            stop = min(rows,start+chunk)
            block = data[start:stop]
            if not np.all(block.view("<i4")[:,0] == 128):
                raise ValueError("Invalid per-row SIFT dimension header")
            values = np.ascontiguousarray(block[:,1:])
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 255):
                raise ValueError("SIFT descriptors exceed the lossless uint8 domain")
            compact = values.astype(np.uint8)
            if not np.array_equal(compact.astype(np.float32).view(np.uint32),values.view(np.uint32)):
                raise ValueError("Source FP32 descriptor values are not bitwise representable as uint8")
            target = output/f"{kind}-{part:03d}.npz"
            np.savez_compressed(target,vectors=compact)
            files.append({"kind":kind,"file":target.name,"start":start,"stop":stop,"shape":[stop-start,128],
                          "sha256":sha(target),"vectors_sha256":hashlib.sha256(compact.tobytes()).hexdigest()})
    record = {"dataset":"ANN_SIFT1M mirrored by qbo-odp/sift1m, checksum-pinned original .fvecs files",
              "scope":"Numerical SIFT descriptors only. Original FP32 values are checked bit-for-bit after lossless uint8 storage. Original dataset attribution/rights and the mirror's Apache-2.0 declaration remain separate from the repository code license.",
              "source_reference":"https://github.com/facebookresearch/faiss/blob/v1.16.0/contrib/datasets.py",
              "mirror_reference":"https://huggingface.co/datasets/qbo-odp/sift1m",
              "inputs":{kind:info for kind,_,info in downloaded},"files":files,"source_sha256":{"prepare_search_sift.py":sha(Path(__file__))}}
    (output/"manifest.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print(f"Stored {len(files)} lossless SIFT descriptor shards",flush=True)


if __name__ == "__main__":
    main()
