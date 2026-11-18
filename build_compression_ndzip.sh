#!/usr/bin/env bash
# Build only our original adapter against the external, source-bound ndzip build.
set -euo pipefail
: "${NDZIP_ROOT:?Set NDZIP_ROOT to the pinned external ndzip checkout}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export NDZIP_ROOT
python3 - <<'PY'
from pathlib import Path
import hashlib,json,os
root=Path(os.environ['NDZIP_ROOT']).resolve()
record=json.loads(Path('results/compression-ndzip-cli-preflight/manifest.json').read_text())
assert record['upstream_commit']=='ff4e6702bf0abb86d4aeef8249bd30344dfeef75'
for name,digest in record['vendor_sources'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
for name,digest in record['binaries'].items():
    assert hashlib.sha256((root/'build-sm120-module-boost'/name).read_bytes()).hexdigest()==digest,name
print('NDZIP_PINNED_DEPENDENCY_BYTES_VERIFIED',len(record['vendor_sources']))
PY
mkdir -p build
g++ -std=c++17 -O3 -fPIC -shared -Wl,--no-undefined native/compression_ndzip_bridge.cpp \
    -I"$NDZIP_ROOT/include" -I"$CUDA_HOME/include" \
    -L"$NDZIP_ROOT/build-sm120-module-boost" -lndzip-cuda -lndzip \
    -L"$CUDA_HOME/lib64" -lcudart \
    -Wl,-rpath,"$NDZIP_ROOT/build-sm120-module-boost:$CUDA_HOME/lib64" \
    -o build/libcompression_ndzip.so
