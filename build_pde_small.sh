#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CUDA_DIR=${CUDA_DIR:-/usr/local/cuda-13.3}
mkdir -p build
"$CUDA_DIR/bin/nvcc" --version
"$CUDA_DIR/bin/nvcc" -O3 -std=c++17 -shared -Xcompiler=-fPIC -arch=sm_120 \
  native/pde_small_bridge.cu -L"$CUDA_DIR/lib64" -Xlinker=-rpath -Xlinker="$CUDA_DIR/lib64" \
  -lcublasLt -o build/libpde_small.so
sha256sum build/libpde_small.so
