#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cuda_root="${CUDA_ROOT:-/usr/local/cuda-13.3}"
mathdx_root="${MATHDX_ROOT:-$HOME/requant-research/mathdx-26.06.1/nvidia-mathdx-26.06.1-cuda13/nvidia/mathdx/26.06}"
mkdir -p build
"$cuda_root/bin/nvcc" -O3 -std=c++17 -arch=sm_120 -Xcompiler=-fPIC -shared native/fft_bridge.cu \
  -I"$mathdx_root/include" -L"$cuda_root/lib64" -lcufft -lcublasLt -lcusparse \
  -Xlinker -rpath -Xlinker "$cuda_root/lib64" -o build/libfft.so
