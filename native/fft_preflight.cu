// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
// cuFFT capability probe; not a complete filtering or Tensor Core FFT study.
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

static thread_local std::string error_message,json_text;
static void ck(cudaError_t s){if(s)throw std::runtime_error(cudaGetErrorString(s));}
static void fk(cufftResult s,const char* where){if(s)throw std::runtime_error(std::string(where)+": cuFFT status "+std::to_string(int(s)));}
struct Probe {
  int n,batch,precision,version=0;size_t bytes=0,workspace_bytes=0;
  void *input=nullptr,*working=nullptr,*spectrum=nullptr,*inverse=nullptr,*workspace=nullptr;
  cufftHandle plan=0;cudaStream_t stream=nullptr;cudaEvent_t begin=nullptr,end=nullptr;
  float elapsed_ms=0;bool forward_ready=false,inverse_ready=false;
  ~Probe(){if(stream)cudaStreamSynchronize(stream);if(plan)cufftDestroy(plan);
    for(void* p:{input,working,spectrum,inverse,workspace})cudaFree(p);
    if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);if(stream)cudaStreamDestroy(stream);}
  void setup(int length,int count,int p,const void* host){
    if(length<1 || length>65536 || count<1 || count>64 || p<0 || p>3 || !host)throw std::runtime_error("Invalid FFT probe arguments");
    n=length;batch=count;precision=p;bytes=size_t(n)*batch*(p==1?16:(p==0?8:4));
    ck(cudaSetDevice(0));cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));if(prop.major!=12 || prop.minor!=0)throw std::runtime_error("SM120 GPU required");
    ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
    fk(cufftGetVersion(&version),"version");fk(cufftCreate(&plan),"create");fk(cufftSetAutoAllocation(plan,0),"manual workspace");
    auto type=p==0?CUDA_C_32F:(p==1?CUDA_C_64F:(p==2?CUDA_C_16F:CUDA_C_16BF));
    long long dimensions=n;
    fk(cufftXtMakePlanMany(plan,1,&dimensions,&dimensions,1,n,type,&dimensions,1,n,type,batch,&workspace_bytes,type),"makePlan");
    fk(cufftSetStream(plan,stream),"stream");
    if(workspace_bytes){ck(cudaMalloc(&workspace,workspace_bytes));fk(cufftSetWorkArea(plan,workspace),"workspace");}
    for(void** p:{&input,&working,&spectrum,&inverse})ck(cudaMalloc(p,bytes));
    ck(cudaMemcpyAsync(input,host,bytes,cudaMemcpyHostToDevice,stream));ck(cudaStreamSynchronize(stream));
  }
  void run(int direction){if(direction!=CUFFT_FORWARD && direction!=CUFFT_INVERSE)throw std::runtime_error("Invalid FFT direction");
    if(direction==CUFFT_INVERSE && !forward_ready)throw std::runtime_error("Forward FFT required first");
    ck(cudaMemcpyAsync(working,direction==CUFFT_FORWARD?input:spectrum,bytes,cudaMemcpyDeviceToDevice,stream));
    ck(cudaEventRecord(begin,stream));fk(cufftXtExec(plan,working,direction==CUFFT_FORWARD?spectrum:inverse,direction),"execute");
    ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));ck(cudaEventElapsedTime(&elapsed_ms,begin,end));
    if(direction==CUFFT_FORWARD){forward_ready=true;inverse_ready=false;}else inverse_ready=true;
  }
  void output(int which,void* host){if(which<0 || which>2 || !host || (which==1 && !forward_ready) || (which==2 && !inverse_ready))throw std::runtime_error("Invalid/unready FFT output");
    ck(cudaMemcpyAsync(host,which==0?input:(which==1?spectrum:inverse),bytes,cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));}
};
extern "C" {
const char* ffp_error(){return error_message.c_str();}
void* ffp_new(int n,int batch,int precision,const void* host){try{auto p=std::make_unique<Probe>();p->setup(n,batch,precision,host);return p.release();}catch(const std::exception& e){error_message=e.what();return nullptr;}}
void ffp_delete(void* p){delete static_cast<Probe*>(p);}
int ffp_run(void* p,int direction){try{if(!p)throw std::runtime_error("Null FFT probe");static_cast<Probe*>(p)->run(direction);return 0;}catch(const std::exception& e){error_message=e.what();return -1;}}
int ffp_output(void* p,int which,void* host){try{if(!p)throw std::runtime_error("Null FFT probe");static_cast<Probe*>(p)->output(which,host);return 0;}catch(const std::exception& e){error_message=e.what();return -1;}}
const char* ffp_metadata(void* p){try{if(!p)throw std::runtime_error("Null FFT probe");auto& q=*static_cast<Probe*>(p);std::ostringstream out;
  out<<std::setprecision(17)<<"{\"cuda_ms\":"<<q.elapsed_ms<<",\"workspace_bytes\":"<<q.workspace_bytes<<",\"owned_device_bytes\":"<<4*q.bytes+q.workspace_bytes<<",\"cufft_version\":"<<q.version<<"}";
  json_text=out.str();return json_text.c_str();}catch(const std::exception& e){error_message=e.what();return nullptr;}}
}
