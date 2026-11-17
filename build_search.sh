#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cutlass_dir="${1:?Usage: bash build_search.sh /path/to/pinned/cutlass /path/to/search-env}"
search_env="${2:?Pass the isolated Faiss/cuVS environment root}"
test "$(git -C "$cutlass_dir" rev-parse HEAD)" = b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd
test -z "$(git -C "$cutlass_dir" status --porcelain --untracked-files=no)"
cmp native/bridge.cu build/base-128.cuh
cuda_root="${CUDA_ROOT:-/usr/local/cuda-13.3}"
site="$search_env/lib/python3.12/site-packages"
mkdir -p build
mkdir -p build/search-include/dlpack
cp "$site/cupy/_core/include/cupy/_dlpack/dlpack.h" build/search-include/dlpack/dlpack.h
g++ -O3 -std=c++20 -fPIC -c native/search_select.cpp -o build/search-select.o \
  -Ibuild/search-include -I"$site/libcuvs/include" -I"$site/libraft/include" -I"$site/libraft/include/rapids" \
  -I"$site/librmm/include" -I"$site/rapids_logger/include" -I"$cuda_root/include" -I"$cuda_root/include/cccl"
"$cuda_root/bin/nvcc" -O3 -std=c++20 -arch=sm_120a --expt-relaxed-constexpr \
  -shared -Xcompiler=-fPIC -I"$cutlass_dir/include" -I"$cutlass_dir/tools/util/include" \
  -Inative -Ibuild -DRQ_BASE_SOURCE='"base-128.cuh"' native/search_bridge.cu build/search-select.o \
  -L"$cuda_root/lib64" -lcublasLt -L"$site/libcuvs/lib64" -lcuvs -lcuvs_c \
  -Xlinker -rpath -Xlinker "$cuda_root/lib64" -Xlinker -rpath -Xlinker "$site/libcuvs/lib64" \
  -o build/libsearch.so
