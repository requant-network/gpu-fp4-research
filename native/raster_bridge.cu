// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
#include "pipeline_bridge.cu"

extern "C" int rq_run_raster(StudyPlan* p,int raster,int swizzle,cudaStream_t stream){
  try{
    if(!p || p->mode!=0 || raster<0 || raster>2 ||
       !((raster==0 && swizzle==0) || swizzle==1 || swizzle==2 || swizzle==4 || swizzle==8))
      throw std::runtime_error("Invalid FP4 raster/swizzle configuration");
    auto args=p->arguments();
    using Order=decltype(args.scheduler.raster_order);
    args.scheduler.raster_order=raster==0?Order::Heuristic:raster==1?Order::AlongM:Order::AlongN;
    args.scheduler.max_swizzle_size=swizzle;
    if(Gemm::get_workspace_size(args)>p->workspace_bytes)throw std::runtime_error("Raster requires a larger workspace");
    check_cutlass(p->gemm.can_implement(args));check_cutlass(p->gemm.run(args,p->workspace,stream));return 0;
  }catch(const std::exception& e){last_error=e.what();return 1;}
}

__global__ void cache_pressure(float* scratch,size_t count){
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x){
    volatile float* address=scratch+i;
    *address=*address+1.f;
  }
}
extern "C" int rq_cache_pressure(float* scratch,size_t count,cudaStream_t stream){
  try{
    if(!scratch || !count || count>size_t(1)<<30)throw std::runtime_error("Invalid cache pressure buffer");
    cache_pressure<<<1024,256,0,stream>>>(scratch,count);check_cuda(cudaGetLastError());return 0;
  }catch(const std::exception& e){last_error=e.what();return 1;}
}
extern "C" int rq_record_external(cudaEvent_t event,cudaStream_t stream){
  try{check_cuda(cudaEventRecordWithFlags(event,stream,cudaEventRecordExternal));return 0;}
  catch(const std::exception& e){last_error=e.what();return 1;}
}

extern "C" int rq_fusion_resources(int variant,int fused,int threads,int* output){
  try{
    if(!output || (fused!=0 && fused!=1) || (threads!=128 && threads!=256 && threads!=512))
      throw std::runtime_error("Invalid fusion resource query");
    const void* function=nullptr;
    if(variant==2)function=(const void*)cooperative_fp4<2,LayoutSFA>;
    else if(variant==10)function=fused?(const void*)native_scalar16<true,LayoutSFA>:(const void*)native_scalar16<false,LayoutSFA>;
    else if(variant==11)function=fused?(const void*)native_cooperative<2,false,true,LayoutSFA>:(const void*)native_cooperative<2,false,false,LayoutSFA>;
    else if(variant==12)function=fused?(const void*)native_cooperative<4,false,true,LayoutSFA>:(const void*)native_cooperative<4,false,false,LayoutSFA>;
    else if(variant==13)function=fused?(const void*)native_cooperative<4,true,true,LayoutSFA>:(const void*)native_cooperative<4,true,false,LayoutSFA>;
    else throw std::runtime_error("Unsupported fusion encoder");
    cudaFuncAttributes attributes;int blocks=0,device;cudaDeviceProp properties;
    check_cuda(cudaGetDevice(&device));check_cuda(cudaGetDeviceProperties(&properties,device));
    check_cuda(cudaFuncGetAttributes(&attributes,function));
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,function,threads,0));
    output[0]=attributes.numRegs;output[1]=attributes.sharedSizeBytes;output[2]=attributes.localSizeBytes;
    output[3]=blocks;output[4]=properties.maxThreadsPerMultiProcessor;return 0;
  }catch(const std::exception& e){last_error=e.what();return 1;}
}
