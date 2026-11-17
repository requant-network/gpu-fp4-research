// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
// Add global-ID mapping without changing the source bound to the kernel probe.
#include "search_bridge.cu"

__global__ void search_global_ids(int size,int rows,const int64_t* local,const int64_t* mapping,int64_t* global){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size)return;int64_t id=local[i];global[i]=id>=0 && id<rows?mapping[id]:-1;
}
struct FullSearch{
  SearchPlan* base=nullptr;int64_t *mapping=nullptr,*global_ids=nullptr;
  cudaGraph_t graph=nullptr;cudaGraphExec_t execution=nullptr;size_t extra=0;int reps=1;
  ~FullSearch(){if(base)cudaStreamSynchronize(base->stream);if(execution)cudaGraphExecDestroy(execution);if(graph)cudaGraphDestroy(graph);
    if(mapping){cudaFree(mapping);cudaFree(global_ids);}}
  void initialize(SearchPlan* p,const int64_t* host_mapping){
    if(!p)throw std::runtime_error("A live native search plan is required");base=p;global_ids=p->output_ids;
    if(host_mapping){
      for(int i=0;i<p->n;++i)if(host_mapping[i]<-1)throw std::runtime_error("Invalid global-ID map");
      check_cuda(cudaMalloc(&mapping,size_t(p->n)*8));global_ids=nullptr;check_cuda(cudaMalloc(&global_ids,size_t(p->batch)*p->k*8));
      extra=(size_t(p->n)+size_t(p->batch)*p->k)*8;
      check_cuda(cudaMemcpyAsync(mapping,host_mapping,size_t(p->n)*8,cudaMemcpyHostToDevice,p->stream));check_cuda(cudaStreamSynchronize(p->stream));}
  }
  void operation(){base->run();if(mapping){int size=base->batch*base->k;search_global_ids<<<(size+255)/256,256,0,base->stream>>>(size,base->n,base->output_ids,mapping,global_ids);check_cuda(cudaGetLastError());}}
  void capture(int count){if(count<1 || count>32 || graph || !base->ready)throw std::runtime_error("Invalid full search capture");
    check_cuda(cudaStreamBeginCapture(base->stream,cudaStreamCaptureModeThreadLocal));
    try{for(int i=0;i<count;++i)operation();check_cuda(cudaStreamEndCapture(base->stream,&graph));}
    catch(...){cudaGraph_t invalid=nullptr;cudaStreamEndCapture(base->stream,&invalid);if(invalid)cudaGraphDestroy(invalid);throw;}
    check_cuda(cudaGraphInstantiate(&execution,graph,0));reps=count;}
  void time(bool captured){if(captured && !execution)throw std::runtime_error("Capture full search first");auto start=std::chrono::steady_clock::now();float ms=0;
    check_cuda(cudaEventRecord(base->begin,base->stream));if(captured)check_cuda(cudaGraphLaunch(execution,base->stream));else operation();
    check_cuda(cudaEventRecord(base->end,base->stream));check_cuda(cudaEventSynchronize(base->end));check_cuda(cudaEventElapsedTime(&ms,base->begin,base->end));
    int divisor=captured?reps:1;base->last_ms=ms/divisor;base->last_wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/divisor;base->ready=true;}
};
static thread_local std::string full_json;
extern "C" {
void* sf_wrap(void* base,const int64_t* mapping){try{auto p=std::make_unique<FullSearch>();p->initialize(static_cast<SearchPlan*>(base),mapping);return p.release();}catch(const std::exception& e){search_error=e.what();return nullptr;}}
void sf_delete(void* p){delete static_cast<FullSearch*>(p);}
int sf_run(void* p){try{if(!p)throw std::runtime_error("Null full search wrapper");auto& s=*static_cast<FullSearch*>(p);s.operation();check_cuda(cudaStreamSynchronize(s.base->stream));return 0;}catch(const std::exception& e){search_error=e.what();return -1;}}
int sf_time(void* p,int captured){try{if(!p)throw std::runtime_error("Null full search wrapper");static_cast<FullSearch*>(p)->time(captured!=0);return 0;}catch(const std::exception& e){search_error=e.what();return -1;}}
int sf_capture(void* p,int repetitions){try{if(!p)throw std::runtime_error("Null full search wrapper");static_cast<FullSearch*>(p)->capture(repetitions);return 0;}catch(const std::exception& e){search_error=e.what();return -1;}}
int sf_gather(void* p,int which,size_t count,size_t width,const uint64_t* offsets,void* host){try{
  if(!p || !host || !offsets || !count || (width!=1 && width!=2 && width!=4 && width!=8))throw std::runtime_error("Invalid sampled search snapshot");
  auto& s=*static_cast<FullSearch*>(p)->base;size_t bytes=s.bytes(which);
  if((which==6 || which==8 || (which>=9 && which<=14)) && !s.ready)throw std::runtime_error("Run search before sampling computed buffers");
  void* pointers[]={s.database,s.queries,s.gemm.a,s.gemm.b,s.gemm.flat_a,s.gemm.flat_b,s.dot,s.database_norm,s.query_norm,s.distances,s.candidates,s.candidate_distances,s.refined_distances,s.output_ids,s.output_distances,s.allowed};
  for(size_t i=0;i<count;++i){if(offsets[i]>bytes || width>bytes-offsets[i])throw std::runtime_error("Sample exceeds search buffer");}
  for(size_t i=0;i<count;++i)check_cuda(cudaMemcpyAsync(static_cast<char*>(host)+i*width,static_cast<const char*>(pointers[which])+offsets[i],width,cudaMemcpyDeviceToHost,s.stream));
  check_cuda(cudaStreamSynchronize(s.stream));return 0;
  }catch(const std::exception& e){search_error=e.what();return -1;}}
int sf_output(void* p,float* distances,int64_t* ids){try{if(!p || !distances || !ids)throw std::runtime_error("Null full search output buffer");auto& s=*static_cast<FullSearch*>(p);
  if(!s.base->ready)throw std::runtime_error("Run search before reading outputs");size_t size=size_t(s.base->batch)*s.base->k;
  check_cuda(cudaMemcpyAsync(distances,s.base->output_distances,size*4,cudaMemcpyDeviceToHost,s.base->stream));check_cuda(cudaMemcpyAsync(ids,s.global_ids,size*8,cudaMemcpyDeviceToHost,s.base->stream));check_cuda(cudaStreamSynchronize(s.base->stream));return 0;
  }catch(const std::exception& e){search_error=e.what();return -1;}}
const char* sf_metadata(void* p){try{if(!p)throw std::runtime_error("Null full search wrapper");auto& s=*static_cast<FullSearch*>(p);std::ostringstream out;
  out<<"{\"native\":"<<s.base->describe()<<",\"global_id_mapping\":"<<(s.mapping?"true":"false")<<",\"mapping_owned_bytes\":"<<s.extra<<",\"graph_repetitions\":"<<s.reps<<"}";
  full_json=out.str();return full_json.c_str();}catch(const std::exception& e){search_error=e.what();return nullptr;}}
}
