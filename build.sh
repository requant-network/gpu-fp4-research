#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cutlass_dir="${1:?Usage: bash build.sh /path/to/pinned/cutlass}"
expected=b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd
test "$(git -C "$cutlass_dir" rev-parse HEAD)" = "$expected"
cuda_root="${CUDA_ROOT:-/usr/local/cuda-13.3}"
mkdir -p build
"$cuda_root/bin/nvcc" -O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr \
  -shared -Xcompiler=-fPIC -I"$cutlass_dir/include" -I"$cutlass_dir/tools/util/include" \
  native/bridge.cu -L"$cuda_root/lib64" -lcublasLt \
  -Xlinker -rpath -Xlinker "$cuda_root/lib64" -o build/libstudy.so
