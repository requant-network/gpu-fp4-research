#!/usr/bin/env bash
set -euo pipefail
ceed_dir=${1:?Usage: build_pde_ceed.sh /path/to/pinned/libCEED}
expected=3e98d546622c05ba46cfce6d4ba2e7d0b9ad4b97
actual=$(git -C "$ceed_dir" rev-parse HEAD)
test "$actual" = "$expected"
test -f "$ceed_dir/lib/libceed.so"
mkdir -p build
${CXX:-c++} -O3 -std=c++17 -shared -fPIC -I"$ceed_dir/include" \
  native/pde_ceed_bridge.cpp -L"$ceed_dir/lib" -lceed \
  -Wl,-rpath,"$ceed_dir/lib" -o build/libpde_ceed.so
