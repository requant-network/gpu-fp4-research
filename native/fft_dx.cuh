// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
// Original filtering wrapper around the external NVIDIA cuFFTDx API.
#include <cufftdx.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

struct FftDxInfo {
  const void* normal=nullptr;const void* diagnostic=nullptr;
  dim3 block;size_t shared=0;int transforms_per_block=1;int version=CUFFTDX_VERSION;
};

template<int N,class Forward,class Inverse,bool Diagnostic>
__global__ void fft_dx_filter(const float2* input,const float* window,const float2* response,
                              float2* output,float2* spectrum,float2* filtered,int batch){
  using Value=typename Forward::value_type;
  Value data[Forward::storage_size];
  extern __shared__ __align__(16) unsigned char workspace[];
  auto* shared=reinterpret_cast<Value*>(workspace);
  int transform=blockIdx.x*Forward::ffts_per_block+threadIdx.y;
  #pragma unroll
  for(unsigned i=0;i<Forward::elements_per_thread;++i){unsigned index=threadIdx.x+i*Forward::stride;
    float2 v={0,0};if(transform<batch && index<N)v=input[size_t(transform)*N+index];
    float w=index<N?window[index]:0.f;data[i].x=__fmul_rn(v.x,w);data[i].y=__fmul_rn(v.y,w);}
  Forward().execute(data,shared);
  #pragma unroll
  for(unsigned i=0;i<Forward::elements_per_thread;++i){unsigned index=threadIdx.x+i*Forward::stride;
    if(index<N){float2 value={data[i].x,data[i].y};
      if constexpr(Diagnostic)if(transform<batch)spectrum[size_t(transform)*N+index]=value;
      float2 h=response[index];
      data[i].x=__fsub_rn(__fmul_rn(value.x,h.x),__fmul_rn(value.y,h.y));
      data[i].y=__fadd_rn(__fmul_rn(value.x,h.y),__fmul_rn(value.y,h.x));
      if constexpr(Diagnostic)if(transform<batch)filtered[size_t(transform)*N+index]={data[i].x,data[i].y};}}
  __syncthreads();
  Inverse().execute(data,shared);
  #pragma unroll
  for(unsigned i=0;i<Inverse::elements_per_thread;++i){unsigned index=threadIdx.x+i*Inverse::stride;
    if(transform<batch && index<N)output[size_t(transform)*N+index]={data[i].x/N,data[i].y/N};}
}

template<int N,int Pack> static FftDxInfo fft_dx_factory(){
  using namespace cufftdx;
  constexpr unsigned EPT=N>=8192?(Pack==4?64:32):(N>=4096?(Pack==4?16:8):8);
  using Base=decltype(Block()+Size<N>()+Precision<float>()+Type<fft_type::c2c>()+ElementsPerThread<EPT>()+FFTsPerBlock<Pack>());
  using Desc=decltype(Base()+Direction<fft_direction::forward>());
  if constexpr(is_supported<Desc,1200>::value){
    using Forward=decltype(Desc()+SM<1200>());
    using Inverse=decltype(Base()+Direction<fft_direction::inverse>()+SM<1200>());
    if constexpr(!Forward::requires_workspace && !Inverse::requires_workspace){
      static_assert(Forward::storage_size==Inverse::storage_size,"Matching forward/inverse register layouts required");
      return {reinterpret_cast<const void*>(fft_dx_filter<N,Forward,Inverse,false>),
              reinterpret_cast<const void*>(fft_dx_filter<N,Forward,Inverse,true>),Forward::block_dim,
              std::max(Forward::shared_memory_size,Inverse::shared_memory_size),Pack,CUFFTDX_VERSION};
    }else throw std::runtime_error("This cuFFTDx wrapper does not implement a required Bluestein workspace");
  }else throw std::runtime_error("The selected cuFFTDx descriptor is not supported on SM120");
}

template<int Pack> static FftDxInfo fft_dx_choose(int n){
  switch(n){case 128:return fft_dx_factory<128,Pack>();case 512:return fft_dx_factory<512,Pack>();
    case 1000:return fft_dx_factory<1000,Pack>();case 1024:return fft_dx_factory<1024,Pack>();
    case 4096:return fft_dx_factory<4096,Pack>();case 16384:return fft_dx_factory<16384,Pack>();
    default:throw std::runtime_error("Length not instantiated for the cuFFTDx reference");}
}

static FftDxInfo fft_dx_info(int n,int pack){
  if(pack==1)return fft_dx_choose<1>(n);if(pack==4)return fft_dx_choose<4>(n);
  throw std::runtime_error("cuFFTDx reference supports one or four transforms per block");
}

static void fft_dx_launch(const FftDxInfo& info,int batch,cudaStream_t stream,const float2* input,
                          const float* window,const float2* response,float2* output,float2* spectrum,float2* filtered,bool diagnostic){
  void* args[]={&input,&window,&response,&output,&spectrum,&filtered,&batch};
  auto status=cudaLaunchKernel(diagnostic?info.diagnostic:info.normal,dim3((batch+info.transforms_per_block-1)/info.transforms_per_block),
                               info.block,args,info.shared,stream);
  if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
