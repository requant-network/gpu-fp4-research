// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
#ifndef RQ_BASE_SOURCE
#define RQ_BASE_SOURCE "bridge.cu"
#endif
#include RQ_BASE_SOURCE

template<int ValuesPerLane, class ScaleLayout>
__global__ void cooperative_fp4(const float* input, const float* bias, uint8_t* packed,
                               cutlass::float_ue4m3_t* scales,
                               cutlass::float_ue4m3_t* logical_scales,
                               int rows, int k, ScaleLayout layout) {
  constexpr int Width = 16 / ValuesPerLane;
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int first = thread * ValuesPerLane;
  if (first >= rows * k) return;
  float values[ValuesPerLane], maximum = 0;
  #pragma unroll
  for (int j = 0; j < ValuesPerLane; ++j) {
    float value = input[first + j];
    values[j] = bias ? fmaxf(value + bias[(first + j) % k], 0.f) : value;
    maximum = fmaxf(maximum, fabsf(values[j]));
  }
  // Dimensions are multiples of 128 values, so active groups contain whole warps.
  #pragma unroll
  for (int offset = Width / 2; offset; offset /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset, Width));
  float candidate = maximum == 0 ? 1.f : fminf(448.f, fmaxf(1.f/512.f, maximum/6.f));
  cutlass::float_ue4m3_t scale(candidate);
  int group = first / 16;
  if ((thread % Width) == 0) {
    logical_scales[group] = scale;
    scales[layout(make_coord(group/(k/16), (group%(k/16))*16, 0))] = scale;
  }
  unsigned low = cutlass::float_e2m1_t(values[0] / float(scale)).raw();
  if constexpr (ValuesPerLane == 1) {
    unsigned high = __shfl_down_sync(0xffffffff, low, 1, Width);
    if (!(thread & 1)) packed[first/2] = uint8_t((low & 15) | ((high & 15) << 4));
  } else {
    unsigned high = cutlass::float_e2m1_t(values[1] / float(scale)).raw();
    packed[first/2] = uint8_t((low & 15) | ((high & 15) << 4));
  }
}

template<class ScaleLayout>
void launch_cooperative(const float* source, const float* bias, void* encoded,
                        cutlass::float_ue4m3_t* scales, cutlass::float_ue4m3_t* flat,
                        int rows,int k,ScaleLayout layout,int variant,cudaStream_t stream) {
  int threads = variant == 3 ? 256 : variant == 4 ? 512 : 128;
  if (variant == 1)
    cooperative_fp4<1><<<(rows*k+threads-1)/threads,threads,0,stream>>>(source,bias,(uint8_t*)encoded,scales,flat,rows,k,layout);
  else
    cooperative_fp4<2><<<(rows*k/2+threads-1)/threads,threads,0,stream>>>(source,bias,(uint8_t*)encoded,scales,flat,rows,k,layout);
}

extern "C" int rq_quantize_variant(StudyPlan* p,int mask,int variant,cudaStream_t stream) {
  try {
    if (!p || mask < 1 || mask > 3 || variant < 0 || variant > 4)
      throw std::runtime_error("Invalid plan, operand mask or encoder variant");
    if (p->mode != 0 || variant == 0) { p->quantize(mask,stream); return 0; }
    if (mask & 1) launch_cooperative(p->input_a,nullptr,p->a,p->sfa,p->flat_a,p->m,p->k,p->la,variant,stream);
    if (mask & 2) launch_cooperative(p->input_b,nullptr,p->b,p->sfb,p->flat_b,p->n,p->k,p->lb,variant,stream);
    check_cuda(cudaGetLastError()); return 0;
  } catch(const std::exception& e) { last_error=e.what(); return 1; }
}

__global__ void bias_relu(const float* source,const float* bias,float* output,int count,int k) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) output[i]=fmaxf(source[i]+bias[i%k],0.f);
}
extern "C" int rq_activate(const float* source,const float* bias,float* output,int count,int k,cudaStream_t stream) {
  try {
    if(!source || !bias || !output || count<=0 || k<=0 || count%k)
      throw std::runtime_error("Invalid activation operands");
    bias_relu<<<(count+255)/256,256,0,stream>>>(source,bias,output,count,k);
    check_cuda(cudaGetLastError()); return 0;
  } catch(const std::exception& e) { last_error=e.what(); return 1; }
}
extern "C" int rq_prepare_fused(StudyPlan* p,const float* source,const float* bias,int variant,cudaStream_t stream) {
  try {
    if(!p || p->mode!=0 || !source || !bias || variant<1 || variant>4)
      throw std::runtime_error("Fused preparation requires FP4, valid inputs and variant 1..4");
    launch_cooperative(source,bias,p->a,p->sfa,p->flat_a,p->m,p->k,p->la,variant,stream);
    check_cuda(cudaGetLastError()); return 0;
  } catch(const std::exception& e) { last_error=e.what(); return 1; }
}
