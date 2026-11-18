// Study 15 exact-output pilot: same kernel for both batching policies.
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr unsigned kWords = 4096;
constexpr unsigned kThreads = 256;
constexpr unsigned kMaxBatch = 8;
static_assert(std::uint64_t{65535} * 65535 * kWords < (std::uint64_t{1} << 44),
              "the complete unsigned dot product must fit in uint64");

__global__ void dot4096(const std::uint16_t* input, std::uint64_t* output) {
    const unsigned lane = threadIdx.x;
    const auto* a = input + blockIdx.x * (2 * kWords);
    const auto* b = a + kWords;
    std::uint64_t sum = 0;
    for (unsigned j = lane; j < kWords; j += kThreads) {
        sum += std::uint32_t{a[j]} * std::uint32_t{b[j]};
    }
    __shared__ std::uint64_t partial[kThreads];
    partial[lane] = sum;
    __syncthreads();
    for (unsigned stride = kThreads / 2; stride != 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) output[blockIdx.x] = partial[0];
}
}  // namespace

// Device pointers only. One ordered batch of count consecutive [A, B] records.
// The caller owns lifetimes, H2D/D2H, device identity, and completion/error checks.
// Returns a cudaError_t as int; zero is successful enqueue, not successful delivery.
#if defined(_WIN32)
#define STUDY15_EXPORT __declspec(dllexport)
#else
#define STUDY15_EXPORT __attribute__((visibility("default")))
#endif
extern "C" STUDY15_EXPORT int study15_dot_launch(const void* device_input, void* device_output,
                                               unsigned count, void* cuda_stream) {
    if (!device_input || !device_output || !cuda_stream || count < 1 || count > kMaxBatch)
        return static_cast<int>(cudaErrorInvalidValue);
    dot4096<<<count, kThreads, 0, reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
        static_cast<const std::uint16_t*>(device_input), static_cast<std::uint64_t*>(device_output));
    return static_cast<int>(cudaGetLastError());
}
