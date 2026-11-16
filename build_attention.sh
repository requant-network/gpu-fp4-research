#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cutlass_dir="${1:?Usage: bash build_attention.sh /path/to/pinned/cutlass}"
test "$(git -C "$cutlass_dir" rev-parse HEAD)" = b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd
test -z "$(git -C "$cutlass_dir" status --porcelain --untracked-files=no)"
cuda_root="${CUDA_ROOT:-/usr/local/cuda-13.3}"
mkdir -p build
"$cuda_root/bin/nvcc" -O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr \
  -shared -Xcompiler=-fPIC -I"$cutlass_dir/include" -I"$cutlass_dir/tools/util/include" \
  -Inative native/attention_bridge.cu -L"$cuda_root/lib64" -lcublasLt \
  -Xlinker -rpath -Xlinker "$cuda_root/lib64" -o build/libattention.so
