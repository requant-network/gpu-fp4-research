// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static thread_local std::string hp_last_error;
static void hp_cuda(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
static void hp_blas(cublasStatus_t status){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLASLt status "+std::to_string(int(status)));}

__global__ void hp_stats(const float* input,int* exponent,int* required,int rows,int k){
  __shared__ unsigned maxima[256];__shared__ int minima[256];
  int row=blockIdx.x;unsigned maximum=0;int minimum=INT_MAX;
  for(int j=threadIdx.x;j<k;j+=256){
    float x=input[row*k+j];
    unsigned raw=__float_as_uint(x)&0x7fffffff,field=raw>>23,mantissa=(raw&0x7fffff)|(field?0x800000:0);
    maximum=max(maximum,raw);
    if(mantissa)minimum=min(minimum,(field?int(field)-150:-149)+__ffs(mantissa)-1);
  }
  maxima[threadIdx.x]=maximum;minima[threadIdx.x]=minimum;__syncthreads();
  for(int step=128;step;step/=2){if(threadIdx.x<step){maxima[threadIdx.x]=max(maxima[threadIdx.x],maxima[threadIdx.x+step]);minima[threadIdx.x]=min(minima[threadIdx.x],minima[threadIdx.x+step]);}__syncthreads();}
  if(!threadIdx.x){int field=maxima[0]>>23;int e=field?field-126:maxima[0]?31-__clz(maxima[0])-148:0;exponent[row]=e;required[row]=minima[0]==INT_MAX?0:(e-minima[0]+6)/7;}
}
__global__ void hp_required(const int* a,const int* b,int m,int n,int* result){
  __shared__ int values[256];int value=0;
  for(int i=threadIdx.x;i<m;i+=256)value=max(value,a[i]);
  for(int i=threadIdx.x;i<n;i+=256)value=max(value,b[i]);
  values[threadIdx.x]=value;__syncthreads();
  for(int step=128;step;step/=2){if(threadIdx.x<step)values[threadIdx.x]=max(values[threadIdx.x],values[threadIdx.x+step]);__syncthreads();}
  if(!threadIdx.x)*result=max(values[0],1);
}
__global__ void hp_slices(const float* input,const int* exponent,int8_t* output,int count,int k,int components){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  double residual=ldexp(double(input[i]),-exponent[i/k]);
  for(int slice=0;slice<components;++slice){residual*=128.;int value=int(residual);output[size_t(slice)*count+i]=int8_t(value);residual-=value;}
}
template<class T> __global__ void hp_convert(const float* input,T* output,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]=T(input[i]);}
__global__ void hp_promote(const float* input,double* output,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]=double(input[i]);}
__global__ void hp_diagonal(const int32_t* partial,int64_t* diagonal,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)diagonal[i]+=int64_t(partial[i]);}
__global__ void hp_recombine(const int64_t* diagonal,const int* ea,const int* eb,double* output,int m,int n,int sum){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m*n)output[i]+=ldexp(double(diagonal[i]),ea[i/n]+eb[i%n]-7*(sum+2));
}

struct HpGemm{
  cublasLtHandle_t handle=nullptr;cublasLtMatmulDesc_t operation=nullptr;
  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;cublasLtEmulationDesc_t emulation=nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> algorithms;int arithmetic=0;
  ~HpGemm(){if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);if(operation)cublasLtMatmulDescDestroy(operation);if(emulation)cublasLtEmulationDescDestroy(emulation);if(handle)cublasLtDestroy(handle);}
  void init(int m,int n,int k,int kind,size_t workspace,int* retained,int bits){
    arithmetic=kind==4?4:(kind==0 || kind==3)?0:1;
    hp_blas(cublasLtCreate(&handle));
    cublasComputeType_t compute=kind==0?CUBLAS_COMPUTE_32F_PEDANTIC:kind==1?CUBLAS_COMPUTE_64F_PEDANTIC:
      kind==2?CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT:kind==3?CUBLAS_COMPUTE_32F_EMULATED_16BFX9:CUBLAS_COMPUTE_32I;
    cudaDataType scalar=arithmetic==4?CUDA_R_32I:arithmetic==0?CUDA_R_32F:CUDA_R_64F;
    cudaDataType input=arithmetic==4?CUDA_R_8I:scalar;
    hp_blas(cublasLtMatmulDescCreate(&operation,compute,scalar));
    cublasOperation_t transpose=CUBLAS_OP_T,normal=CUBLAS_OP_N;
    hp_blas(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
    hp_blas(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSB,&normal,sizeof(normal)));
    if(kind==2 || kind==3){
      hp_blas(cublasLtEmulationDescCreate(&emulation));int eager=kind==2 && bits==-1?CUBLAS_EMULATION_STRATEGY_PERFORMANT:CUBLAS_EMULATION_STRATEGY_EAGER;
      hp_blas(cublasLtEmulationDescSetAttribute(emulation,CUBLASLT_EMULATION_DESC_STRATEGY,&eager,sizeof(eager)));
      int special=CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE;
      hp_blas(cublasLtEmulationDescSetAttribute(emulation,CUBLASLT_EMULATION_DESC_SPECIAL_VALUES_SUPPORT,&special,sizeof(special)));
      if(kind==2){
        int control=bits>0?CUDA_EMULATION_MANTISSA_CONTROL_FIXED:CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
        hp_blas(cublasLtEmulationDescSetAttribute(emulation,CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL,&control,sizeof(control)));
        if(bits>0)hp_blas(cublasLtEmulationDescSetAttribute(emulation,CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT,&bits,sizeof(bits)));
        hp_blas(cublasLtEmulationDescSetAttribute(emulation,CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_COUNT_POINTER,&retained,sizeof(retained)));
      }
      hp_blas(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR,&emulation,sizeof(emulation)));
    }
    hp_blas(cublasLtMatrixLayoutCreate(&a,input,k,n,k));hp_blas(cublasLtMatrixLayoutCreate(&b,input,k,m,k));hp_blas(cublasLtMatrixLayoutCreate(&c,scalar,n,m,n));
    cublasLtMatmulPreference_t preference=nullptr;hp_blas(cublasLtMatmulPreferenceCreate(&preference));
    auto status=cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace));
    if(status!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(preference);hp_blas(status);}
    algorithms.resize(8);int count=0;status=cublasLtMatmulAlgoGetHeuristic(handle,operation,a,b,c,c,preference,8,algorithms.data(),&count);
    cublasLtMatmulPreferenceDestroy(preference);hp_blas(status);algorithms.resize(count);
    if(!count)throw std::runtime_error("No cuBLASLt heuristic for requested precision");
  }
  void run(const void* x,const void* y,void* output,void* workspace,size_t bytes,int algorithm,cudaStream_t stream){
    if(algorithm<0 || algorithm>=int(algorithms.size()))throw std::runtime_error("Invalid precision algorithm");
    int32_t ai=1,bi=0;float af=1,bf=0;double ad=1,bd=0;
    const void* alpha=arithmetic==4?(void*)&ai:arithmetic==0?(void*)&af:(void*)&ad;
    const void* beta=arithmetic==4?(void*)&bi:arithmetic==0?(void*)&bf:(void*)&bd;
    hp_blas(cublasLtMatmul(handle,operation,alpha,y,a,x,b,beta,output,c,output,c,&algorithms[algorithm].algo,workspace,bytes,stream));
  }
};

struct HpPlan{
  int mode=0,parameter=0,selected=0,m=0,n=0,k=0;const float *input_a=nullptr,*input_b=nullptr;double* output=nullptr;
  void *base_a=nullptr,*base_b=nullptr,*workspace=nullptr;int8_t *slices_a=nullptr,*slices_b=nullptr;
  int *ea=nullptr,*eb=nullptr,*ra=nullptr,*rb=nullptr,*required=nullptr,*retained=nullptr;
  int32_t* partial=nullptr;int64_t* diagonal=nullptr;float* single=nullptr;
  size_t workspace_bytes=512ull*1024*1024;std::unique_ptr<HpGemm> primary,fallback;
  ~HpPlan(){primary.reset();fallback.reset();cudaFree(base_a);cudaFree(base_b);cudaFree(workspace);cudaFree(slices_a);cudaFree(slices_b);cudaFree(ea);cudaFree(eb);cudaFree(ra);cudaFree(rb);cudaFree(required);cudaFree(retained);cudaFree(partial);cudaFree(diagonal);cudaFree(single);}
  void init(int kind,int param,int rows,int columns,int depth,const float* x,const float* y,double* result){
    if(kind<0 || kind>5 || rows<=0 || rows>2048 || rows%16 || columns<=0 || columns>2048 || columns%32 || depth<=0 || depth>8192 || depth%32 || !x || !y || !result ||
       ((kind==4 || kind==5)?(param<1 || param>8):(kind==2?(param< -1 || param>79):param!=0)))
      throw std::runtime_error("Invalid precision mode, parameter, aligned dimensions or operands");
    mode=kind;parameter=param;selected=kind>=4?param:0;m=rows;n=columns;k=depth;input_a=x;input_b=y;output=result;
    hp_cuda(cudaMalloc(&workspace,workspace_bytes));hp_cuda(cudaMalloc(&retained,sizeof(int)));hp_cuda(cudaMemset(retained,0xff,sizeof(int)));
    hp_cuda(cudaMalloc(&base_a,size_t(m)*k*sizeof(double)));hp_cuda(cudaMalloc(&base_b,size_t(n)*k*sizeof(double)));
    hp_cuda(cudaMalloc(&single,size_t(m)*n*sizeof(float)));
    primary=std::make_unique<HpGemm>();primary->init(m,n,k,kind>=4?4:kind,workspace_bytes,retained,kind==2?param:0);
    if(kind>=4){
      hp_cuda(cudaMalloc(&slices_a,size_t(m)*k*param));hp_cuda(cudaMalloc(&slices_b,size_t(n)*k*param));
      hp_cuda(cudaMalloc(&ea,m*sizeof(int)));hp_cuda(cudaMalloc(&eb,n*sizeof(int)));hp_cuda(cudaMalloc(&ra,m*sizeof(int)));hp_cuda(cudaMalloc(&rb,n*sizeof(int)));hp_cuda(cudaMalloc(&required,sizeof(int)));
      hp_cuda(cudaMalloc(&partial,size_t(m)*n*sizeof(int32_t)));hp_cuda(cudaMalloc(&diagonal,size_t(m)*n*sizeof(int64_t)));
      if(kind==5){fallback=std::make_unique<HpGemm>();fallback->init(m,n,k,1,workspace_bytes,retained,0);}
    }
  }
  void analyze(int mask,cudaStream_t stream){
    if(mask<1 || mask>3)throw std::runtime_error("Precision operand mask must be1..3");
    if(mode>=4){
      if(mask&1)hp_stats<<<m,256,0,stream>>>(input_a,ea,ra,m,k);
      if(mask&2)hp_stats<<<n,256,0,stream>>>(input_b,eb,rb,n,k);
      hp_required<<<1,256,0,stream>>>(ra,rb,m,n,required);hp_cuda(cudaGetLastError());
    }
  }
  void encode(int mask,cudaStream_t stream){
    if(mask<1 || mask>3)throw std::runtime_error("Precision operand mask must be1..3");
    if(mode>=4 && selected){
      if(mask&1)hp_slices<<<(m*k+255)/256,256,0,stream>>>(input_a,ea,slices_a,m*k,k,selected);
      if(mask&2)hp_slices<<<(n*k+255)/256,256,0,stream>>>(input_b,eb,slices_b,n*k,k,selected);
    }else if(mode==0 || mode==3){
      if(mask&1)hp_convert<<<(m*k+255)/256,256,0,stream>>>(input_a,(float*)base_a,m*k);
      if(mask&2)hp_convert<<<(n*k+255)/256,256,0,stream>>>(input_b,(float*)base_b,n*k);
    }else{
      if(mask&1)hp_convert<<<(m*k+255)/256,256,0,stream>>>(input_a,(double*)base_a,m*k);
      if(mask&2)hp_convert<<<(n*k+255)/256,256,0,stream>>>(input_b,(double*)base_b,n*k);
    }
    hp_cuda(cudaGetLastError());
  }
  void run(int algorithm,int alternative,cudaStream_t stream){
    if(mode>=4 && selected){
      hp_cuda(cudaMemsetAsync(output,0,size_t(m)*n*sizeof(double),stream));
      for(int sum=2*selected-2;sum>=0;--sum){
        hp_cuda(cudaMemsetAsync(diagonal,0,size_t(m)*n*sizeof(int64_t),stream));
        for(int i=0;i<selected;++i){int j=sum-i;if(j<0 || j>=selected)continue;
          primary->run(slices_a+size_t(i)*m*k,slices_b+size_t(j)*n*k,partial,workspace,workspace_bytes,algorithm,stream);
          hp_diagonal<<<(m*n+255)/256,256,0,stream>>>(partial,diagonal,m*n);
        }
        hp_recombine<<<(m*n+255)/256,256,0,stream>>>(diagonal,ea,eb,output,m,n,sum);
      }
    }else if(mode==5)fallback->run(base_a,base_b,output,workspace,workspace_bytes,alternative,stream);
    else if(mode==0 || mode==3){primary->run(base_a,base_b,single,workspace,workspace_bytes,algorithm,stream);hp_promote<<<(m*n+255)/256,256,0,stream>>>(single,output,m*n);}
    else primary->run(base_a,base_b,output,workspace,workspace_bytes,algorithm,stream);
    hp_cuda(cudaGetLastError());
  }
};

extern "C" const char* hp_error(){return hp_last_error.c_str();}
extern "C" size_t rq_cublas_version(){return cublasLtGetVersion();}
extern "C" void* hp_create(int mode,int parameter,int m,int n,int k,const float* a,const float* b,double* output){try{auto p=std::make_unique<HpPlan>();p->init(mode,parameter,m,n,k,a,b,output);return p.release();}catch(const std::exception& e){hp_last_error=e.what();return nullptr;}}
extern "C" void hp_destroy(HpPlan* p){delete p;}
extern "C" int hp_analyze(HpPlan* p,int mask,cudaStream_t s){try{if(!p)throw std::runtime_error("Null precision plan");p->analyze(mask,s);return 0;}catch(const std::exception& e){hp_last_error=e.what();return 1;}}
extern "C" int hp_encode(HpPlan* p,int mask,cudaStream_t s){try{if(!p)throw std::runtime_error("Null precision plan");p->encode(mask,s);return 0;}catch(const std::exception& e){hp_last_error=e.what();return 1;}}
extern "C" int hp_select(HpPlan* p,cudaStream_t s){try{if(!p || p->mode!=5)throw std::runtime_error("Selection requires adaptive plan");int count;hp_cuda(cudaMemcpyAsync(&count,p->required,sizeof(int),cudaMemcpyDeviceToHost,s));hp_cuda(cudaStreamSynchronize(s));p->selected=count<=p->parameter?count:0;return p->selected;}catch(const std::exception& e){hp_last_error=e.what();return -1;}}
extern "C" int hp_run(HpPlan* p,int algorithm,int alternative,cudaStream_t s){try{if(!p)throw std::runtime_error("Null precision plan");p->run(algorithm,alternative,s);return 0;}catch(const std::exception& e){hp_last_error=e.what();return 1;}}
extern "C" int hp_candidates(HpPlan* p,int which){return !p || which<0 || which>1?0:which?(p->fallback?int(p->fallback->algorithms.size()):0):int(p->primary->algorithms.size());}
extern "C" int hp_tune_prepare(HpPlan* p,cudaStream_t s){try{
  if(!p)throw std::runtime_error("Null precision plan");
  if(p->mode>=4){
    hp_slices<<<(p->m*p->k+255)/256,256,0,s>>>(p->input_a,p->ea,p->slices_a,p->m*p->k,p->k,1);
    hp_slices<<<(p->n*p->k+255)/256,256,0,s>>>(p->input_b,p->eb,p->slices_b,p->n*p->k,p->k,1);
    hp_convert<<<(p->m*p->k+255)/256,256,0,s>>>(p->input_a,(double*)p->base_a,p->m*p->k);
    hp_convert<<<(p->n*p->k+255)/256,256,0,s>>>(p->input_b,(double*)p->base_b,p->n*p->k);
  }
  hp_cuda(cudaGetLastError());return 0;
}catch(const std::exception& e){hp_last_error=e.what();return 1;}}
extern "C" int hp_tune_gemm(HpPlan* p,int which,int algorithm,cudaStream_t s){try{
  if(!p || which<0 || which>1 || (which && !p->fallback))throw std::runtime_error("Invalid precision tuning plan");
  HpGemm* g=which?p->fallback.get():p->primary.get();
  const void* a=!which && p->mode>=4?(void*)p->slices_a:p->base_a;
  const void* b=!which && p->mode>=4?(void*)p->slices_b:p->base_b;
  void* output=g->arithmetic==4?(void*)p->partial:g->arithmetic==0?(void*)p->single:(void*)p->output;
  g->run(a,b,output,p->workspace,p->workspace_bytes,algorithm,s);return 0;
}catch(const std::exception& e){hp_last_error=e.what();return 1;}}
extern "C" int hp_algorithm_info(HpPlan* p,int which,int candidate,uint64_t* flags,int* emulated,int* id){try{
  if(!p || !flags || !emulated || !id || which<0 || which>1)throw std::runtime_error("Invalid precision algorithm query");
  auto* g=which?p->fallback.get():p->primary.get();if(!g || candidate<0 || candidate>=int(g->algorithms.size()))throw std::runtime_error("Invalid precision candidate");size_t written;
  auto* algorithm=&g->algorithms[candidate].algo;
  hp_blas(cublasLtMatmulAlgoCapGetAttribute(algorithm,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,flags,sizeof(*flags),&written));
  hp_blas(cublasLtMatmulAlgoCapGetAttribute(algorithm,CUBLASLT_ALGO_CAP_FLOATING_POINT_EMULATION_SUPPORT,emulated,sizeof(*emulated),&written));
  hp_blas(cublasLtMatmulAlgoConfigGetAttribute(algorithm,CUBLASLT_ALGO_CONFIG_ID,id,sizeof(*id),&written));return 0;
}catch(const std::exception& e){hp_last_error=e.what();return 1;}}
extern "C" void* hp_buffer(HpPlan* p,int index){if(!p)return nullptr;switch(index){case 0:return p->mode>=4 && p->selected?(void*)p->slices_a:p->base_a;case 1:return p->mode>=4 && p->selected?(void*)p->slices_b:p->base_b;case 2:return p->ea;case 3:return p->eb;case 4:return p->ra;case 5:return p->rb;case 6:return p->required;case 7:return p->retained;case 8:return p->partial;default:return nullptr;}}
extern "C" int hp_emulation_config(HpPlan* p,int* output){try{
  if(!p || !output || !p->primary->emulation)throw std::runtime_error("Emulation descriptor unavailable");
  cublasLtEmulationDescAttributes_t names[5]={CUBLASLT_EMULATION_DESC_STRATEGY,CUBLASLT_EMULATION_DESC_SPECIAL_VALUES_SUPPORT,
    CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL,CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT,CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_OFFSET};
  for(int i=0;i<5;++i){size_t written;hp_blas(cublasLtEmulationDescGetAttribute(p->primary->emulation,names[i],output+i,sizeof(int),&written));}
  return 0;
}catch(const std::exception& e){hp_last_error=e.what();return 1;}}
