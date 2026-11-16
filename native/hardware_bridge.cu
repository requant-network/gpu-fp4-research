// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
#include "application_bridge.cu"
#include <cuda_fp4.h>

__device__ __forceinline__ uint8_t pack_native(float a,float b,float scale) {
  return __nv_cvt_float2_to_fp4x2(make_float2(a/scale,b/scale),__NV_E2M1,cudaRoundNearest);
}

template<bool Activate,class ScaleLayout>
__global__ void native_scalar16(const float* source,const float* bias,uint8_t* output,
                               cutlass::float_ue4m3_t* scales,cutlass::float_ue4m3_t* flat,
                               int rows,int k,ScaleLayout layout) {
  int group=blockIdx.x*blockDim.x+threadIdx.x;
  if(group>=rows*(k/16))return;
  float values[16],maximum=0;
  #pragma unroll
  for(int j=0;j<4;++j){
    float4 block=reinterpret_cast<const float4*>(source)[group*4+j];
    float local[4]={block.x,block.y,block.z,block.w};
    #pragma unroll
    for(int t=0;t<4;++t){
      float value=local[t];
      if constexpr(Activate)value=fmaxf(value+bias[(group*16+j*4+t)%k],0.f);
      values[j*4+t]=value;
      maximum=fmaxf(maximum,fabsf(value));
    }
  }
  cutlass::float_ue4m3_t scale(maximum==0?1.f:fminf(448.f,fmaxf(1.f/512.f,maximum/6.f)));
  flat[group]=scale;
  scales[layout(make_coord(group/(k/16),(group%(k/16))*16,0))]=scale;
  uint2 packed=make_uint2(0,0);
  #pragma unroll
  for(int j=0;j<4;++j){
    packed.x|=uint32_t(pack_native(values[2*j],values[2*j+1],float(scale)))<<(8*j);
    packed.y|=uint32_t(pack_native(values[8+2*j],values[9+2*j],float(scale)))<<(8*j);
  }
  reinterpret_cast<uint2*>(output)[group]=packed;
}

template<int Values,bool Redux,bool Activate,class ScaleLayout>
__global__ void native_cooperative(const float* source,const float* bias,uint8_t* output,
                                  cutlass::float_ue4m3_t* scales,cutlass::float_ue4m3_t* flat,
                                  int rows,int k,ScaleLayout layout) {
  constexpr int Width=16/Values;
  int thread=blockIdx.x*blockDim.x+threadIdx.x,first=thread*Values;
  if(first>=rows*k)return;
  float values[Values],maximum=0;
  if constexpr(Values==4){
    float4 block=reinterpret_cast<const float4*>(source)[thread];
    values[0]=block.x;values[1]=block.y;values[2]=block.z;values[3]=block.w;
  }else{
    float2 block=reinterpret_cast<const float2*>(source)[thread];
    values[0]=block.x;values[1]=block.y;
  }
  #pragma unroll
  for(int j=0;j<Values;++j){
    if constexpr(Activate)values[j]=fmaxf(values[j]+bias[(first+j)%k],0.f);
    maximum=fmaxf(maximum,fabsf(values[j]));
  }
  if constexpr(Redux){
    unsigned mask=((1u<<Width)-1)<<((threadIdx.x%32)/Width*Width);
    maximum=__uint_as_float(__reduce_max_sync(mask,__float_as_uint(maximum)));
  }else{
    #pragma unroll
    for(int offset=Width/2;offset;offset/=2)
      maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,offset,Width));
  }
  cutlass::float_ue4m3_t scale(maximum==0?1.f:fminf(448.f,fmaxf(1.f/512.f,maximum/6.f)));
  int group=first/16;
  if(!(thread%Width)){
    flat[group]=scale;
    scales[layout(make_coord(group/(k/16),(group%(k/16))*16,0))]=scale;
  }
  uint16_t packed=pack_native(values[0],values[1],float(scale));
  if constexpr(Values==4){
    packed|=uint16_t(pack_native(values[2],values[3],float(scale)))<<8;
    reinterpret_cast<uint16_t*>(output)[thread]=packed;
  }else output[thread]=uint8_t(packed);
}

template<bool Activate,class ScaleLayout>
void launch_native(const float* input,const float* bias,void* output,cutlass::float_ue4m3_t* scales,
                   cutlass::float_ue4m3_t* flat,int rows,int k,ScaleLayout layout,int variant,cudaStream_t stream) {
  constexpr int threads=128;
  if(variant==10)native_scalar16<Activate><<<(rows*k/16+threads-1)/threads,threads,0,stream>>>(input,bias,(uint8_t*)output,scales,flat,rows,k,layout);
  else if(variant==11)native_cooperative<2,false,Activate><<<(rows*k/2+threads-1)/threads,threads,0,stream>>>(input,bias,(uint8_t*)output,scales,flat,rows,k,layout);
  else if(variant==12)native_cooperative<4,false,Activate><<<(rows*k/4+threads-1)/threads,threads,0,stream>>>(input,bias,(uint8_t*)output,scales,flat,rows,k,layout);
  else native_cooperative<4,true,Activate><<<(rows*k/4+threads-1)/threads,threads,0,stream>>>(input,bias,(uint8_t*)output,scales,flat,rows,k,layout);
}
extern "C" int rq_quantize_native(StudyPlan* p,int mask,int variant,cudaStream_t stream){
  try{
    if(!p || p->mode!=0 || mask<1 || mask>3 || variant<10 || variant>13)
      throw std::runtime_error("Native FP4 preparation requires valid plan, mask and variant10..13");
    if(((mask&1) && (reinterpret_cast<uintptr_t>(p->input_a)%16)) ||
       ((mask&2) && (reinterpret_cast<uintptr_t>(p->input_b)%16)))
      throw std::runtime_error("Native vector loads require 16-byte aligned FP32 operands");
    if(mask&1)launch_native<false>(p->input_a,nullptr,p->a,p->sfa,p->flat_a,p->m,p->k,p->la,variant,stream);
    if(mask&2)launch_native<false>(p->input_b,nullptr,p->b,p->sfb,p->flat_b,p->n,p->k,p->lb,variant,stream);
    check_cuda(cudaGetLastError());return 0;
  }catch(const std::exception& e){last_error=e.what();return 1;}
}
extern "C" int rq_fused_native(StudyPlan* p,const float* source,const float* bias,int variant,cudaStream_t stream){
  try{
    if(!p || p->mode!=0 || !source || !bias || variant<10 || variant>13)
      throw std::runtime_error("Invalid native fused preparation operands");
    if(reinterpret_cast<uintptr_t>(source)%16)
      throw std::runtime_error("Native fused source must be 16-byte aligned");
    launch_native<true>(source,bias,p->a,p->sfa,p->flat_a,p->m,p->k,p->la,variant,stream);
    check_cuda(cudaGetLastError());return 0;
  }catch(const std::exception& e){last_error=e.what();return 1;}
}

extern "C" int rq_native_resources(int variant,int threads,int* output){
  try{
    if(!output || threads<32 || threads>1024 || threads%32)
      throw std::runtime_error("Invalid resource-query arguments");
    const void* function=nullptr;
    if(variant==0)function=(const void*)quantize_fp4<LayoutSFA>;
    else if(variant==2)function=(const void*)cooperative_fp4<2,LayoutSFA>;
    else if(variant==10)function=(const void*)native_scalar16<false,LayoutSFA>;
    else if(variant==11)function=(const void*)native_cooperative<2,false,false,LayoutSFA>;
    else if(variant==12)function=(const void*)native_cooperative<4,false,false,LayoutSFA>;
    else if(variant==13)function=(const void*)native_cooperative<4,true,false,LayoutSFA>;
    else throw std::runtime_error("Unsupported resource-query variant");
    cudaFuncAttributes attributes;
    int blocks=0,device;
    cudaDeviceProp properties;
    check_cuda(cudaGetDevice(&device));check_cuda(cudaGetDeviceProperties(&properties,device));
    check_cuda(cudaFuncGetAttributes(&attributes,function));
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,function,threads,0));
    output[0]=attributes.numRegs;
    output[1]=int(attributes.sharedSizeBytes);
    output[2]=int(attributes.localSizeBytes);
    output[3]=attributes.maxThreadsPerBlock;
    output[4]=blocks;
    output[5]=properties.maxThreadsPerMultiProcessor;
    output[6]=properties.regsPerMultiprocessor;
    output[7]=properties.warpSize;
    output[8]=properties.multiProcessorCount;
    output[9]=properties.l2CacheSize;
    output[10]=int(properties.sharedMemPerMultiprocessor);
    output[11]=attributes.binaryVersion;
    return 0;
  }catch(const std::exception& e){last_error=e.what();return 1;}
}
