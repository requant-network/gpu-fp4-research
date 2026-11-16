#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cutlass_dir="${1:?Usage: bash build_extended.sh /path/to/pinned/cutlass [128|256]}"
tile="${2:-128}"
test "$tile" = 128 || test "$tile" = 256
test "$(git -C "$cutlass_dir" rev-parse HEAD)" = b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd
test -z "$(git -C "$cutlass_dir" status --porcelain --untracked-files=no)"
cuda_root="${CUDA_ROOT:-/usr/local/cuda-13.3}"
mkdir -p build
python3 - "$tile" <<'PY'
from pathlib import Path
import sys
tile = int(sys.argv[1])
source = Path('native/bridge.cu').read_text()
needle = 'using ThreadBlockShape    = Shape<_128,_128,_128>;'
if source.count(needle) != 1:
    raise RuntimeError('Pinned kernel declaration changed')
source = source.replace(needle, f'using ThreadBlockShape    = Shape<_{tile},_128,_128>;')
Path(f'build/base-{tile}.cuh').write_text(source, encoding='utf-8')
PY
"$cuda_root/bin/nvcc" -O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr \
  -shared -Xcompiler=-fPIC -I"$cutlass_dir/include" -I"$cutlass_dir/tools/util/include" \
  -Inative -Ibuild -DRQ_BASE_SOURCE="\"base-$tile.cuh\"" native/optimized_bridge.cu \
  -L"$cuda_root/lib64" -lcublasLt -Xlinker -rpath -Xlinker "$cuda_root/lib64" \
  -o "build/libextended-$tile.so"
