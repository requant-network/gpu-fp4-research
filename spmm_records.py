"""Read reproduced JSON or the byte-preserving gzip archives distributed in Git."""
import gzip
import json
from pathlib import Path


def load_record(path):
    path = Path(path)
    if not path.exists():
        path = path.with_name(path.name + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)
