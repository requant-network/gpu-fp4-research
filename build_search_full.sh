#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cutlass_dir="${1:?Pass the pinned CUTLASS root}"
search_env="${2:?Pass the isolated Faiss/cuVS environment root}"
test "$(git -C "$cutlass_dir" rev-parse HEAD)" = b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd
test -z "$(git -C "$cutlass_dir" status --porcelain --untracked-files=no)"
cmp native/bridge.cu build/base-128.cuh
test -f build/search-select.o
cuda_root="${CUDA_ROOT:-/usr/local/cuda-13.3}"
site="$search_env/lib/python3.12/site-packages"
"$cuda_root/bin/nvcc" -O3 -std=c++20 -arch=sm_120a --expt-relaxed-constexpr \
  -shared -Xcompiler=-fPIC -I"$cutlass_dir/include" -I"$cutlass_dir/tools/util/include" \
  -Inative -Ibuild -DRQ_BASE_SOURCE='"base-128.cuh"' native/search_full_bridge.cu build/search-select.o \
  -L"$cuda_root/lib64" -lcublasLt -L"$site/libcuvs/lib64" -lcuvs -lcuvs_c \
  -Xlinker -rpath -Xlinker "$cuda_root/lib64" -Xlinker -rpath -Xlinker "$site/libcuvs/lib64" \
  -o build/libsearch-full.so
