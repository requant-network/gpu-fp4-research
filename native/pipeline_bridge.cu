// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
#include "hardware_bridge.cu"

extern "C" int rq_pipeline_resources(int* output){
  try{
    if(!output)throw std::runtime_error("Null pipeline resource output");
    const void* function=(const void*)cutlass::device_kernel<GemmKernel>;
    constexpr int shared=sizeof(typename GemmKernel::SharedStorage);
    constexpr int threads=GemmKernel::MaxThreadsPerBlock;
    check_cuda(cudaFuncSetAttribute(function,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    cudaFuncAttributes attributes;
    int blocks=0;
    check_cuda(cudaFuncGetAttributes(&attributes,function));
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,function,threads,shared));
    output[0]=GemmKernel::CollectiveMainloop::DispatchPolicy::Stages;
    output[1]=threads;output[2]=shared;output[3]=attributes.numRegs;
    output[4]=attributes.sharedSizeBytes;output[5]=attributes.localSizeBytes;output[6]=blocks;
    return 0;
  }catch(const std::exception& e){last_error=e.what();return 1;}
}
