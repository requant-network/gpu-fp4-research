#!/usr/bin/env bash
# Original CPU codec only; CUDA decoding is separately compiled by NVRTC.
set -euo pipefail
mkdir -p build
g++ -std=c++17 -O3 -fPIC -fopenmp -fno-strict-aliasing -shared -Wl,--no-undefined \
    native/xbp_cpu.cpp -o build/libxbp_cpu.so
