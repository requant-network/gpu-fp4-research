#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cuda_root="${CUDA_ROOT:-/usr/local/cuda-13.3}"
mkdir -p build
"$cuda_root/bin/nvcc" -O3 -std=c++17 -arch=sm_120 -Xcompiler=-fPIC -shared native/fft_preflight.cu \
  -L"$cuda_root/lib64" -lcufft -Xlinker -rpath -Xlinker "$cuda_root/lib64" -o build/libfft-preflight.so
