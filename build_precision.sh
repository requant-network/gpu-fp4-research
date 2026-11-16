#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
cuda_root="${CUDA_ROOT:-/usr/local/cuda-13.3}"
"$cuda_root/bin/nvcc" -O3 -std=c++17 -arch=sm_120a -shared -Xcompiler=-fPIC \
  native/precision_bridge.cu -L"$cuda_root/lib64" -lcublasLt \
  -Xlinker -rpath -Xlinker "$cuda_root/lib64" -o build/libprecision.so
