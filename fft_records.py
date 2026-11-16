"""Byte-preserving publication of large FFT witnesses in bounded Git blobs."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def witness_bytes(path):
    path = Path(path)
    if path.is_file():
        return path.read_bytes()
    manifest = json.loads((path.parent/"fft-artifacts.json").read_text())
    entry = manifest[path.name]
    parts = []
    for part in entry["parts"]:
        relative = Path(part["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid FFT artifact part path")
        raw = (path.parent/relative).read_bytes()
        if len(raw) != part["bytes"] or hashlib.sha256(raw).hexdigest() != part["sha256"]:
            raise ValueError("Corrupt FFT artifact part")
        parts.append(raw)
    raw = b"".join(parts)
    if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("Corrupt assembled FFT witness")
    return raw


def load_witness(path, expected_sha256):
    raw = witness_bytes(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("FFT witness SHA256 does not match its executed record")
    return np.load(io.BytesIO(raw), allow_pickle=False)


def pack(directory):
    directory = Path(directory)
    entries = {}
    output = directory/"fft-artifacts"
    output.mkdir(exist_ok=True)
    for path in sorted(directory.glob("fft-*.npz")):
        if path.stat().st_size < 40*1024*1024:
            continue
        raw = path.read_bytes()
        full_sha = hashlib.sha256(raw).hexdigest()
        parts = []
        for index, start in enumerate(range(0, len(raw), 16*1024*1024)):
            part = raw[start:start+16*1024*1024]
            name = f"fft-artifacts/{full_sha}-{index:03d}.bin"
            target = directory/name
            if target.exists() and target.read_bytes() != part:
                raise ValueError("Existing content-addressed artifact differs")
            if not target.exists():
                target.write_bytes(part)
            parts.append({"file": name, "bytes": len(part), "sha256": hashlib.sha256(part).hexdigest()})
        entries[path.name] = {"bytes": len(raw), "sha256": full_sha, "parts": parts}
    (directory/"fft-artifacts.json").write_bytes((json.dumps(entries, indent=2)+"\n").encode())
    print(f"Packed {len(entries)} large FFT witnesses without changing their original bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    pack(parser.parse_args().pack)
