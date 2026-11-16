"""Fetch a pinned external NVIDIA MathDx SDK and record the used header corpus."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = "https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/cuda13/nvidia-mathdx-26.06.1-cuda13.tar.gz"
ARCHIVE_SHA256 = "59a9233db34b75568acbcc5284e6cefe6fad5577ee644f85044971d62eeea353"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "requant-research/mathdx-26.06.1")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    archive = args.root / URL.rsplit("/", 1)[-1]
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=30) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if sha(temporary) != ARCHIVE_SHA256:
            raise ValueError("Downloaded MathDx archive does not match the pinned checksum")
        temporary.replace(archive)
    if sha(archive) != ARCHIVE_SHA256:
        raise ValueError("Cached MathDx archive changed")
    sdk = args.root / "nvidia-mathdx-26.06.1-cuda13/nvidia/mathdx/26.06"
    if not (sdk / "include/cufftdx.hpp").exists():
        with tarfile.open(archive, "r:gz") as package:
            package.extractall(args.root, filter="data")
    version = (sdk / "include/cufftdx/cufftdx_version.hpp").read_text(encoding="utf-8")
    number = int(re.search(r"#define CUFFTDX_VERSION (\d+)", version).group(1))
    headers = {p.relative_to(sdk).as_posix(): sha(p) for p in sorted((sdk / "include").rglob("*")) if p.is_file()}
    record = {"scope": "External, unmodified NVIDIA MathDx SDK; not vendored or relicensed by this repository.",
              "source_url": URL, "archive_sha256": ARCHIVE_SHA256, "archive_bytes": archive.stat().st_size,
              "cufftdx_version": number, "header_sha256": headers,
              "source_sha256": {"prepare_fft_dx.py": sha(ROOT / "prepare_fft_dx.py")}}
    (ROOT / "results/fft-dx-dependency.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Pinned cuFFTDx {number}: {len(headers)} headers; dependency manifest written")


if __name__ == "__main__":
    main()
