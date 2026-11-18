// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr unsigned words = 4096;
constexpr unsigned threads = 256;
constexpr unsigned width = 32;
constexpr unsigned max_batch = 8;

static_assert(std::uint64_t{65535} * 65535 * words < (std::uint64_t{1} << 44));

#if STUDY15_FP64
using Accumulator = double;

__device__ Accumulator add_product(Accumulator sum, std::uint16_t a, std::uint16_t b) {
    // Products, partial sums and the complete result are exact in binary64.
    return __fma_rn(static_cast<double>(a), static_cast<double>(b), sum);
}
#else
using Accumulator = unsigned long long;

__device__ Accumulator add_product(Accumulator sum, std::uint16_t a, std::uint16_t b) {
    return sum + std::uint32_t{a} * std::uint32_t{b};
}
#endif

__device__ Accumulator warp_sum(Accumulator value) {
    for (unsigned delta = width / 2; delta != 0; delta /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, delta, width);
    }
    return value;
}

__global__ void dot4096(const std::uint16_t* input, std::uint64_t* output) {
    auto a = input + blockIdx.x * (2 * words);
    auto b = a + words;
    Accumulator sum = 0;
    for (unsigned index = threadIdx.x; index < words; index += threads) {
        sum = add_product(sum, a[index], b[index]);
    }

    __shared__ Accumulator partial[threads / width];
    const unsigned lane = threadIdx.x % width;
    const unsigned warp = threadIdx.x / width;
    sum = warp_sum(sum);
    if (lane == 0) {
        partial[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        sum = warp_sum(lane < threads / width ? partial[lane] : Accumulator{0});
        if (lane == 0) {
            output[blockIdx.x] = static_cast<std::uint64_t>(sum);
        }
    }
}
}

extern "C" __attribute__((visibility("default")))
int study15_dot_launch(const void* input, void* output, unsigned count, void* stream) {
    if (!input || !output || !stream || count == 0 || count > max_batch) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    dot4096<<<count, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const std::uint16_t*>(input), static_cast<std::uint64_t*>(output));
    return static_cast<int>(cudaGetLastError());
}
