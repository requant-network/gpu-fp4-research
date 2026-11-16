/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// Requant Network: FP32 output and original C ABI around the pinned CUTLASS configuration.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
using namespace cute;
// A matrix configuration
using         ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand
using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementD    = float;                            // Element type for D matrix operand
using         ElementC    = float;                            // Element type for C matrix operand
using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm120;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

// Kernel Perf config
using ThreadBlockShape    = Shape<_128,_128,_128>;                          // Threadblock's tile size
using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto                      // Epilogue schedule policy
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto                             // Kernel schedule policy. Auto defaults to cooperative kernel schedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,                                                   // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideC   = typename Gemm::GemmKernel::StrideC;
using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

//

// Original Requant Network experiment harness; the preceding kernel configuration
// is derived from NVIDIA CUTLASS example 79a under its retained BSD-3-Clause notice.
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <memory>
#include <vector>
#include <algorithm>

static thread_local std::string last_error;
static void check_cuda(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
static void check_blas(cublasStatus_t s) {
  if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLASLt status " + std::to_string(int(s)));
}
static void check_cutlass(cutlass::Status s) {
  if (s != cutlass::Status::kSuccess) throw std::runtime_error(cutlassGetStatusString(s));
}

template<class ScaleLayout>
__global__ void quantize_fp4(const float* input, uint8_t* packed,
                            cutlass::float_ue4m3_t* scales,
                            cutlass::float_ue4m3_t* logical_scales,
                            int rows, int k, ScaleLayout layout) {
  int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= rows * (k / 16)) return;
  int row = group / (k / 16), column = (group % (k / 16)) * 16;
  float maximum = 0;
  for (int j = 0; j < 16; ++j) maximum = fmaxf(maximum, fabsf(input[group * 16 + j]));
  float candidate = maximum == 0 ? 1.f : fminf(448.f, fmaxf(1.f / 512.f, maximum / 6.f));
  cutlass::float_ue4m3_t scale(candidate);
  logical_scales[group] = scale;
  scales[layout(make_coord(row, column, 0))] = scale;
  for (int j = 0; j < 8; ++j) {
    cutlass::Array<cutlass::float_e2m1_t, 2> pair;
    pair[0] = cutlass::float_e2m1_t(input[group * 16 + j * 2] / float(scale));
    pair[1] = cutlass::float_e2m1_t(input[group * 16 + j * 2 + 1] / float(scale));
    packed[group * 8 + j] = reinterpret_cast<const uint8_t*>(&pair)[0];
  }
}

__global__ void convert_input(const float* input, void* output, int count, int mode) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  if (mode == 1) reinterpret_cast<cutlass::float_e4m3_t*>(output)[i] = cutlass::float_e4m3_t(input[i]);
  else if (mode == 2) reinterpret_cast<__half*>(output)[i] = __float2half_rn(input[i]);
  else if (mode == 3) reinterpret_cast<__nv_bfloat16*>(output)[i] = __float2bfloat16_rn(input[i]);
  else reinterpret_cast<float*>(output)[i] = input[i];
}

struct StudyPlan {
  int mode = 0, m = 0, n = 0, k = 0;
  const float *input_a = nullptr, *input_b = nullptr;
  float* output = nullptr;
  void *a = nullptr, *b = nullptr, *workspace = nullptr;
  size_t a_bytes = 0, b_bytes = 0, workspace_bytes = 32 * 1024 * 1024;
  cutlass::float_ue4m3_t *sfa = nullptr, *sfb = nullptr, *flat_a = nullptr, *flat_b = nullptr;
  StrideA sa; StrideB sb; StrideC sc; StrideD sd; LayoutSFA la; LayoutSFB lb;
  Gemm gemm;
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t matrix_a = nullptr, matrix_b = nullptr, matrix_c = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> algorithms;

  ~StudyPlan() {
    if (matrix_a) cublasLtMatrixLayoutDestroy(matrix_a);
    if (matrix_b) cublasLtMatrixLayoutDestroy(matrix_b);
    if (matrix_c) cublasLtMatrixLayoutDestroy(matrix_c);
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (handle) cublasLtDestroy(handle);
    cudaFree(a); cudaFree(b); cudaFree(workspace);
    cudaFree(sfa); cudaFree(sfb); cudaFree(flat_a); cudaFree(flat_b);
  }

  Gemm::Arguments arguments() {
    return {cutlass::gemm::GemmUniversalMode::kGemm, {m,n,k,1},
      {static_cast<const ElementA::DataType*>(a), sa,
       static_cast<const ElementB::DataType*>(b), sb, sfa, la, sfb, lb},
      {{1.f,0.f}, output, sc, output, sd}};
  }

  void initialize(int kind, int rows, int columns, int depth,
                  const float* ia, const float* ib, float* result) {
    if (kind < 0 || kind > 4 || rows <= 0 || columns <= 0 || depth <= 0 ||
        rows > 8192 || columns > 8192 || depth > 8192 ||
        columns % 128 || depth % 128 || !ia || !ib || !result)
      throw std::runtime_error("Expected mode 0..4, dimensions 1..8192, N/K multiples of 128 and non-null device pointers");
    int device; cudaDeviceProp properties;
    check_cuda(cudaGetDevice(&device)); check_cuda(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 12 || properties.minor != 0)
      throw std::runtime_error("This bounded experiment requires SM120");
    mode = kind; m = rows; n = columns; k = depth; input_a = ia; input_b = ib; output = result;
    int bits = mode == 0 ? 4 : mode == 1 ? 8 : mode == 4 ? 32 : 16;
    a_bytes = size_t(m) * k * bits / 8; b_bytes = size_t(n) * k * bits / 8;
    check_cuda(cudaMalloc(&a, a_bytes)); check_cuda(cudaMalloc(&b, b_bytes));
    if (mode == 0) {
      using Scale = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
      sa = cutlass::make_cute_packed_stride(StrideA{}, {m,k,1});
      sb = cutlass::make_cute_packed_stride(StrideB{}, {n,k,1});
      sc = cutlass::make_cute_packed_stride(StrideC{}, {m,n,1});
      sd = cutlass::make_cute_packed_stride(StrideD{}, {m,n,1});
      la = Scale::tile_atom_to_shape_SFA(make_shape(m,n,k,1));
      lb = Scale::tile_atom_to_shape_SFB(make_shape(m,n,k,1));
      check_cuda(cudaMalloc(&sfa, size(filter_zeros(la))));
      check_cuda(cudaMalloc(&sfb, size(filter_zeros(lb))));
      check_cuda(cudaMalloc(&flat_a, size_t(m) * k / 16));
      check_cuda(cudaMalloc(&flat_b, size_t(n) * k / 16));
      check_cuda(cudaMemset(sfa, 0x38, size(filter_zeros(la))));
      check_cuda(cudaMemset(sfb, 0x38, size(filter_zeros(lb))));
      auto args = arguments(); check_cutlass(gemm.can_implement(args));
      workspace_bytes = Gemm::get_workspace_size(args);
    } else {
      check_blas(cublasLtCreate(&handle));
      check_blas(cublasLtMatmulDescCreate(&operation,
        mode == 4 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F, CUDA_R_32F));
      cublasOperation_t transpose = CUBLAS_OP_T, normal = CUBLAS_OP_N;
      check_blas(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
      check_blas(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &normal, sizeof(normal)));
      cudaDataType type = mode == 1 ? CUDA_R_8F_E4M3 : mode == 2 ? CUDA_R_16F : mode == 3 ? CUDA_R_16BF : CUDA_R_32F;
      check_blas(cublasLtMatrixLayoutCreate(&matrix_a, type, k, n, k));
      check_blas(cublasLtMatrixLayoutCreate(&matrix_b, type, k, m, k));
      check_blas(cublasLtMatrixLayoutCreate(&matrix_c, CUDA_R_32F, n, m, n));
      cublasLtMatmulPreference_t preference = nullptr;
      check_blas(cublasLtMatmulPreferenceCreate(&preference));
      auto status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &workspace_bytes, sizeof(workspace_bytes));
      if (status != CUBLAS_STATUS_SUCCESS) { cublasLtMatmulPreferenceDestroy(preference); check_blas(status); }
      algorithms.resize(8); int count = 0;
      status = cublasLtMatmulAlgoGetHeuristic(handle, operation, matrix_a, matrix_b, matrix_c, matrix_c,
                                             preference, 8, algorithms.data(), &count);
      cublasLtMatmulPreferenceDestroy(preference); check_blas(status);
      algorithms.resize(count);
      if (count == 0) throw std::runtime_error("No cuBLASLt heuristic candidate for this mode/shape");
    }
    if (workspace_bytes) check_cuda(cudaMalloc(&workspace, workspace_bytes));
  }

  void quantize(int operands, cudaStream_t stream) {
    if (operands < 1 || operands > 3) throw std::runtime_error("Operand mask must be 1, 2 or 3");
    if (operands & 1) {
      if (mode == 0) quantize_fp4<<<(m*(k/16)+127)/128,128,0,stream>>>(input_a, (uint8_t*)a, sfa, flat_a, m,k,la);
      else convert_input<<<(m*k+255)/256,256,0,stream>>>(input_a,a,m*k,mode);
    }
    if (operands & 2) {
      if (mode == 0) quantize_fp4<<<(n*(k/16)+127)/128,128,0,stream>>>(input_b, (uint8_t*)b, sfb, flat_b, n,k,lb);
      else convert_input<<<(n*k+255)/256,256,0,stream>>>(input_b,b,n*k,mode);
    }
    check_cuda(cudaGetLastError());
  }
  void run(int algorithm, cudaStream_t stream) {
    if (mode == 0) { if (algorithm != 0) throw std::runtime_error("Invalid FP4 algorithm");
      check_cutlass(gemm.run(arguments(), workspace, stream)); return; }
    if (algorithm < 0 || algorithm >= int(algorithms.size())) throw std::runtime_error("Invalid cuBLASLt candidate");
    const float alpha = 1, beta = 0;
    check_blas(cublasLtMatmul(handle, operation, &alpha, b, matrix_a, a, matrix_b,
      &beta, output, matrix_c, output, matrix_c, &algorithms[algorithm].algo,
      workspace, workspace_bytes, stream));
  }
};

extern "C" const char* rq_error() { return last_error.c_str(); }
extern "C" void* rq_create(int mode,int m,int n,int k,const float* a,const float* b,float* out) {
  try { auto plan = std::make_unique<StudyPlan>(); plan->initialize(mode,m,n,k,a,b,out); return plan.release(); }
  catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
extern "C" void rq_destroy(StudyPlan* p) { delete p; }
extern "C" int rq_candidates(StudyPlan* p) { return p ? (p->mode == 0 ? 1 : int(p->algorithms.size())) : 0; }
extern "C" size_t rq_cublas_version() { return cublasLtGetVersion(); }
extern "C" void* rq_buffer(StudyPlan* p,int index) {
  if (!p) return nullptr;
  switch (index) { case 0: return p->a; case 1: return p->b; case 2: return p->flat_a; case 3: return p->flat_b; default: return nullptr; }
}
extern "C" size_t rq_bytes(StudyPlan* p,int index) {
  if (!p) return 0;
  switch (index) { case 0: return p->a_bytes; case 1: return p->b_bytes;
    case 2: return p->mode == 0 ? size_t(p->m)*p->k/16 : 0;
    case 3: return p->mode == 0 ? size_t(p->n)*p->k/16 : 0; default: return 0; }
}
extern "C" int rq_quantize(StudyPlan* p,int mask,cudaStream_t stream) {
  try { if (!p) throw std::runtime_error("Null plan"); p->quantize(mask,stream); return 0; }
  catch (const std::exception& e) { last_error = e.what(); return 1; }
}
extern "C" int rq_run(StudyPlan* p,int algorithm,cudaStream_t stream) {
  try { if (!p) throw std::runtime_error("Null plan"); p->run(algorithm,stream); return 0; }
  catch (const std::exception& e) { last_error = e.what(); return 1; }
}
extern "C" int rq_algorithm_id(StudyPlan* p,int index) {
  if (!p || index < 0 || index >= rq_candidates(p)) return -1;
  if (p->mode == 0) return 0;
  int value = -1; size_t written;
  if (cublasLtMatmulAlgoConfigGetAttribute(&p->algorithms[index].algo, CUBLASLT_ALGO_CONFIG_ID,
                                         &value,sizeof(value),&written) != CUBLAS_STATUS_SUCCESS) return -1;
  return value;
}
