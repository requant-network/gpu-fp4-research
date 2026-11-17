"""Verify every SIFT shard and reconstruct both pinned original .fvecs hashes."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    directory = ROOT/"fixtures/search-sift"
    record = json.loads((directory/"manifest.json").read_text())
    for name,expected in record["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    rows = 0
    for kind,information in record["inputs"].items():
        source_hash = hashlib.sha256()
        previous = 0
        for part in sorted((item for item in record["files"] if item["kind"] == kind),key=lambda item:item["start"]):
            path = directory/part["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == part["sha256"]
            with np.load(path,allow_pickle=False) as z:
                values = z["vectors"]
            assert values.dtype == np.uint8 and values.shape == tuple(part["shape"])
            assert part["start"] == previous and part["stop"] == part["start"]+len(values)
            assert values.shape[1] == 128 and hashlib.sha256(values.tobytes()).hexdigest() == part["vectors_sha256"]
            restored = np.empty((len(values),129),dtype="<f4")
            restored.view("<i4")[:,0] = 128
            restored[:,1:] = values
            source_hash.update(restored.tobytes())
            previous = part["stop"]
            rows += len(values)
        assert previous == information["source_rows"]
        assert source_hash.hexdigest() == information["sha256"],kind
    assert rows == 1010000
    print(f"SIFT data: {rows:,} descriptors in {len(record['files'])} shards; both complete original source-file SHA256 values reconstructed exactly")


if __name__ == "__main__":
    main()
