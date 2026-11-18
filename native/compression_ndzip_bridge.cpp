// Original C ABI adapter; ndzip remains an external pinned dependency.
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <ndzip/ndzip.hh>
#include <ndzip/offload.hh>
#include <ndzip/cuda.hh>

namespace {
thread_local std::string last_error;
struct Plan {
    virtual ~Plan() = default;
    virtual uint64_t bound() const = 0;
    virtual uint64_t encode(const void*, void*) = 0;
    virtual void decode(const void*, void*) = 0;
};

template<class T> struct TypedPlan final : Plan {
    ndzip::extent shape;
    std::unique_ptr<ndzip::offloader<T>> encoder;
    std::unique_ptr<ndzip::cuda_decompressor<T>> decoder;
    TypedPlan(int dimensions, const uint32_t* sizes, unsigned threads, cudaStream_t stream)
        : shape(dimensions), encoder(ndzip::make_cpu_offloader<T>(dimensions, threads)),
          decoder(ndzip::make_cuda_decompressor<T>(dimensions, stream)) {
        for (int i = 0; i < dimensions; ++i) shape[i] = sizes[i];
    }
    uint64_t bound() const override { return ndzip::compressed_length_bound<T>(shape); }
    uint64_t encode(const void* input, void* output) override {
        return encoder->compress(static_cast<const T*>(input), shape,
                                 static_cast<ndzip::compressed_type<T>*>(output));
    }
    void decode(const void* input, void* output) override {
        decoder->decompress(static_cast<const ndzip::compressed_type<T>*>(input),
                            static_cast<T*>(output), shape);
        const auto status = cudaPeekAtLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
};
}

extern "C" {
const char* ndz_error() { return last_error.c_str(); }
void* ndz_create(int bytes, int dimensions, const uint32_t* sizes, unsigned threads, void* stream) {
    try {
        last_error.clear();
        if (dimensions < 1 || dimensions > 3 || !sizes || !threads)
            throw std::runtime_error("Invalid ndzip geometry or CPU thread count");
        uint64_t elements = 1;
        for (int i = 0; i < dimensions; ++i) {
            if (!sizes[i]) throw std::runtime_error("Zero ndzip extent");
            elements *= sizes[i];
            if (elements > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error("ndzip uint32 element-count implementation limit");
        }
        if (bytes == 4) return new TypedPlan<float>(dimensions, sizes, threads, static_cast<cudaStream_t>(stream));
        if (bytes == 8) return new TypedPlan<double>(dimensions, sizes, threads, static_cast<cudaStream_t>(stream));
        throw std::runtime_error("Expected FP32 or FP64");
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
uint64_t ndz_bound(void* handle) {
    try {
        if (!handle) throw std::runtime_error("Missing ndzip handle");
        return static_cast<Plan*>(handle)->bound();
    } catch (const std::exception& e) { last_error = e.what(); return 0; }
}
int ndz_encode(void* handle, const void* input, void* output, uint64_t capacity, uint64_t* written) {
    try {
        last_error.clear();
        if (!handle || !input || !output || !written) throw std::runtime_error("Null ndzip encoding argument");
        auto* plan = static_cast<Plan*>(handle);
        if (capacity < plan->bound()) throw std::runtime_error("ndzip output buffer below declared bound");
        *written = plan->encode(input, output);
        if (*written > capacity) throw std::runtime_error("ndzip exceeded its compressed-length bound");
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return 1; }
}
int ndz_decode(void* handle, const void* input, void* output) {
    try {
        last_error.clear();
        if (!handle || !input || !output) throw std::runtime_error("Null ndzip decoding argument");
        static_cast<Plan*>(handle)->decode(input, output);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return 1; }
}
int ndz_free(void* handle) {
    try { delete static_cast<Plan*>(handle); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return 1; }
}
}
