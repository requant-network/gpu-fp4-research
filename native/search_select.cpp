// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
// Original wrapper around the external cuVS batched selection API.
#include <cuvs/core/c_api.h>
#include <cuvs/selection/select_k.hpp>
#include <memory>
#include <string>

static thread_local std::string select_error;
extern "C" const char* sq_select_error(){return select_error.c_str();}
extern "C" uintptr_t sq_select_create(cudaStream_t stream){
  cuvsResources_t handle=0;
  try{
    if(cuvsResourcesCreate(&handle)!=CUVS_SUCCESS || cuvsStreamSet(handle,stream)!=CUVS_SUCCESS ||
       cuvsResourcesSetWorkspacePool(handle,64ull<<20)!=CUVS_SUCCESS)
      throw std::runtime_error(cuvsGetLastErrorText()?cuvsGetLastErrorText():"cuVS selection resource setup failed");
    return handle;
  }catch(const std::exception& e){if(handle)cuvsResourcesDestroy(handle);select_error=e.what();return 0;}
}
extern "C" void sq_select_delete(uintptr_t handle){if(handle)cuvsResourcesDestroy(handle);}
extern "C" int sq_select(uintptr_t handle,const float* input,const int64_t* payload,float* values,int64_t* ids,
                           int batch,int columns,int count){
  try{
    if(!handle || !input || !values || !ids || batch<1 || columns<count || count<1)
      throw std::runtime_error("Invalid batched selection buffers/extents");
    auto in=raft::make_device_matrix_view<const float,int64_t>(input,batch,columns);
    auto out=raft::make_device_matrix_view<float,int64_t>(values,batch,count);
    auto indices=raft::make_device_matrix_view<int64_t,int64_t>(ids,batch,count);
    std::optional<raft::device_matrix_view<const int64_t,int64_t,raft::row_major>> source;
    if(payload)source=raft::make_device_matrix_view<const int64_t,int64_t>(payload,batch,columns);
    cuvs::selection::select_k(*reinterpret_cast<raft::resources*>(handle),in,source,out,indices,true,true);
    return 0;
  }catch(const std::exception& e){select_error=e.what();return -1;}
}
