// Original ungrouped short-q cuBLASLt control; no original bridge size guard.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static thread_local std::string error_text;
static void cuda_ok(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
static void blas_ok(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS status "+std::to_string(int(s)));}
__global__ void pde_small_cast(const float* input,void* output,int count,int mode){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  if(mode==1)((__nv_fp8_e4m3*)output)[i]=__nv_fp8_e4m3(input[i]);
  else if(mode==2)((__half*)output)[i]=__float2half_rn(input[i]);
  else if(mode==3)((__nv_bfloat16*)output)[i]=__float2bfloat16_rn(input[i]);
  else ((float*)output)[i]=input[i];
}
struct Small {
  int mode=0,m=0,n=0;const float *input_a=nullptr,*input_b=nullptr;float* output=nullptr;
  void *a=nullptr,*b=nullptr,*workspace=nullptr;size_t a_bytes=0,b_bytes=0,workspace_bytes=32*1024*1024;
  cublasLtHandle_t handle=nullptr;cublasLtMatmulDesc_t operation=nullptr;
  cublasLtMatrixLayout_t ma=nullptr,mb=nullptr,mc=nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> algorithms;
  ~Small(){if(ma)cublasLtMatrixLayoutDestroy(ma);if(mb)cublasLtMatrixLayoutDestroy(mb);if(mc)cublasLtMatrixLayoutDestroy(mc);
    if(operation)cublasLtMatmulDescDestroy(operation);if(handle)cublasLtDestroy(handle);cudaFree(a);cudaFree(b);cudaFree(workspace);}
  void init(int kind,int rows,int columns,const float* ia,const float* ib,float* result){
    if(kind<1||kind>4||rows<1||rows>16||columns<1||!ia||!ib||!result)throw std::runtime_error("Invalid bounded short-q geometry");
    mode=kind;m=rows;n=columns;input_a=ia;input_b=ib;output=result;
    int bytes=mode==1?1:mode==4?4:2;a_bytes=size_t(m)*m*bytes;b_bytes=size_t(n)*m*bytes;
    cuda_ok(cudaMalloc(&a,a_bytes));cuda_ok(cudaMalloc(&b,b_bytes));cuda_ok(cudaMalloc(&workspace,workspace_bytes));
    blas_ok(cublasLtCreate(&handle));blas_ok(cublasLtMatmulDescCreate(&operation,mode==4?CUBLAS_COMPUTE_32F_PEDANTIC:CUBLAS_COMPUTE_32F,CUDA_R_32F));
    cublasOperation_t trans=CUBLAS_OP_T,normal=CUBLAS_OP_N;
    blas_ok(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
    blas_ok(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSB,&normal,sizeof(normal)));
    cudaDataType type=mode==1?CUDA_R_8F_E4M3:mode==2?CUDA_R_16F:mode==3?CUDA_R_16BF:CUDA_R_32F;
    blas_ok(cublasLtMatrixLayoutCreate(&ma,type,m,n,m));blas_ok(cublasLtMatrixLayoutCreate(&mb,type,m,m,m));
    blas_ok(cublasLtMatrixLayoutCreate(&mc,CUDA_R_32F,n,m,n));
    cublasLtMatmulPreference_t preference=nullptr;blas_ok(cublasLtMatmulPreferenceCreate(&preference));
    auto status=cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes));
    int count=0;algorithms.resize(16);
    if(status==CUBLAS_STATUS_SUCCESS)status=cublasLtMatmulAlgoGetHeuristic(handle,operation,ma,mb,mc,mc,preference,16,algorithms.data(),&count);
    cublasLtMatmulPreferenceDestroy(preference);blas_ok(status);algorithms.resize(count);
    if(!count)throw std::runtime_error("No short-q cuBLASLt heuristic candidate");
  }
};
extern "C" const char* pde_small_error(){return error_text.c_str();}
extern "C" void* pde_small_create(int mode,int m,int n,const float*a,const float*b,float*out){
  try{auto p=std::make_unique<Small>();p->init(mode,m,n,a,b,out);return p.release();}catch(const std::exception&e){error_text=e.what();return nullptr;}}
extern "C" void pde_small_destroy(void*p){delete static_cast<Small*>(p);}
extern "C" int pde_small_prepare(void* opaque,int mask,void* stream){try{
  auto*p=static_cast<Small*>(opaque);if(mask<1||mask>3)throw std::runtime_error("Invalid operand mask");
  if(mask&1)pde_small_cast<<<(p->m*p->m+255)/256,256,0,(cudaStream_t)stream>>>(p->input_a,p->a,p->m*p->m,p->mode);
  if(mask&2)pde_small_cast<<<(p->n*p->m+255)/256,256,0,(cudaStream_t)stream>>>(p->input_b,p->b,p->n*p->m,p->mode);
  cuda_ok(cudaGetLastError());return 0;}catch(const std::exception&e){error_text=e.what();return 1;}}
extern "C" int pde_small_run(void*opaque,int candidate,void*stream){try{auto*p=static_cast<Small*>(opaque);float alpha=1,beta=0;
  auto&a=p->algorithms.at(candidate).algo;
  blas_ok(cublasLtMatmul(p->handle,p->operation,&alpha,p->b,p->ma,p->a,p->mb,&beta,p->output,p->mc,p->output,p->mc,&a,p->workspace,p->workspace_bytes,(cudaStream_t)stream));return 0;
  }catch(const std::exception&e){error_text=e.what();return 1;}}
extern "C" int pde_small_candidates(void*p){return static_cast<Small*>(p)->algorithms.size();}
extern "C" uint64_t pde_small_flags(void*opaque,int candidate){auto*p=static_cast<Small*>(opaque);uint64_t value=0;size_t written=0;
  auto status=cublasLtMatmulAlgoCapGetAttribute(&p->algorithms.at(candidate).algo,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,&value,sizeof(value),&written);
  return status==CUBLAS_STATUS_SUCCESS?value:0;}
extern "C" int pde_small_id(void*opaque,int candidate){auto*p=static_cast<Small*>(opaque);int value=-1;size_t written=0;
  auto status=cublasLtMatmulAlgoConfigGetAttribute(&p->algorithms.at(candidate).algo,CUBLASLT_ALGO_CONFIG_ID,&value,sizeof(value),&written);
  return status==CUBLAS_STATUS_SUCCESS?value:-1;}
extern "C" size_t pde_small_bytes(void*opaque,int index){auto*p=static_cast<Small*>(opaque);return index==0?p->a_bytes:index==1?p->b_bytes:0;}
extern "C" void* pde_small_buffer(void*opaque,int index){auto*p=static_cast<Small*>(opaque);return index==0?p->a:index==1?p->b:nullptr;}
extern "C" size_t pde_small_owned_bytes(void*opaque){auto*p=static_cast<Small*>(opaque);return p->a_bytes+p->b_bytes+p->workspace_bytes;}
extern "C" size_t pde_small_cublas_version(){return cublasLtGetVersion();}
