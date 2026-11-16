// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

static void timer_check(cudaError_t status){if(status)throw std::runtime_error(cudaGetErrorString(status));}
struct TimerControl{
  cudaStream_t stream=nullptr;cudaEvent_t begin=nullptr,end=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t execution=nullptr;
  ~TimerControl(){if(stream)cudaStreamSynchronize(stream);if(execution)cudaGraphExecDestroy(execution);if(graph)cudaGraphDestroy(graph);
    if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);if(stream)cudaStreamDestroy(stream);}
};
extern "C" const char* fft_timer_control(int repetitions){
  static thread_local std::string result;
  try{
    if(repetitions!=1 && repetitions!=8)throw std::runtime_error("Expected one or eight graph repetitions");
    timer_check(cudaSetDevice(0));TimerControl control;
    timer_check(cudaStreamCreateWithFlags(&control.stream,cudaStreamNonBlocking));
    timer_check(cudaEventCreate(&control.begin));timer_check(cudaEventCreate(&control.end));timer_check(cudaGraphCreate(&control.graph,0));
    cudaGraphNode_t previous=nullptr;
    for(int i=0;i<repetitions;++i){cudaGraphNode_t current;
      timer_check(cudaGraphAddEmptyNode(&current,control.graph,previous?&previous:nullptr,previous?1:0));previous=current;}
    timer_check(cudaGraphInstantiate(&control.execution,control.graph,0));
    for(int i=0;i<5;++i){timer_check(cudaGraphLaunch(control.execution,control.stream));timer_check(cudaStreamSynchronize(control.stream));}
    std::ostringstream out;out<<std::setprecision(17)<<"{\"repetitions\":"<<repetitions<<",\"observations\":[";
    for(int i=0;i<30;++i){auto start=std::chrono::steady_clock::now();float elapsed=0;
      timer_check(cudaEventRecord(control.begin,control.stream));timer_check(cudaGraphLaunch(control.execution,control.stream));
      timer_check(cudaEventRecord(control.end,control.stream));timer_check(cudaEventSynchronize(control.end));
      timer_check(cudaEventElapsedTime(&elapsed,control.begin,control.end));
      double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      if(i)out<<",";out<<"{\"cuda_ms\":"<<elapsed/repetitions<<",\"wall_ms\":"<<wall/repetitions<<"}";}
    out<<"]}";result=out.str();
  }catch(const std::exception& error){result=std::string("{\"error\":\"")+error.what()+"\"}";}
  return result.c_str();
}
