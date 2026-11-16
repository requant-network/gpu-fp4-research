// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
#include "optimized_bridge.cu"

struct LibraryFP4 {
  StudyPlan* source=nullptr;
  cublasLtHandle_t handle=nullptr;
  cublasLtMatmulDesc_t operation=nullptr;
  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
  void* workspace=nullptr;
  size_t workspace_bytes=32*1024*1024;
  std::vector<cublasLtMatmulHeuristicResult_t> candidates;
  ~LibraryFP4() {
    if(a)cublasLtMatrixLayoutDestroy(a);
    if(b)cublasLtMatrixLayoutDestroy(b);
    if(c)cublasLtMatrixLayoutDestroy(c);
    if(operation)cublasLtMatmulDescDestroy(operation);
    if(handle)cublasLtDestroy(handle);
    cudaFree(workspace);
  }
  void initialize(StudyPlan* p) {
    if(!p || p->mode!=0) throw std::runtime_error("Library FP4 requires a live FP4 source plan");
    source=p;
    check_blas(cublasLtCreate(&handle));
    check_blas(cublasLtMatmulDescCreate(&operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    cublasOperation_t transpose=CUBLAS_OP_T,normal=CUBLAS_OP_N;
    check_blas(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
    check_blas(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSB,&normal,sizeof(normal)));
    cublasLtMatmulMatrixScale_t scaling=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    check_blas(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&scaling,sizeof(scaling)));
    check_blas(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&scaling,sizeof(scaling)));
    const void* ascale=p->sfb;
    const void* bscale=p->sfa;
    check_blas(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&ascale,sizeof(ascale)));
    check_blas(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&bscale,sizeof(bscale)));
    check_blas(cublasLtMatrixLayoutCreate(&a,CUDA_R_4F_E2M1,p->k,p->n,p->k));
    check_blas(cublasLtMatrixLayoutCreate(&b,CUDA_R_4F_E2M1,p->k,p->m,p->k));
    check_blas(cublasLtMatrixLayoutCreate(&c,CUDA_R_32F,p->n,p->m,p->n));
    cublasLtMatmulPreference_t preference=nullptr;
    check_blas(cublasLtMatmulPreferenceCreate(&preference));
    auto status=cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &workspace_bytes,sizeof(workspace_bytes));
    if(status!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(preference);check_blas(status);}
    candidates.resize(8);
    int count=0;
    status=cublasLtMatmulAlgoGetHeuristic(handle,operation,a,b,c,c,preference,8,candidates.data(),&count);
    cublasLtMatmulPreferenceDestroy(preference);
    check_blas(status);
    candidates.resize(count);
    if(!count)throw std::runtime_error("No block-scaled FP4 cuBLASLt candidate for this shape/output type");
    check_cuda(cudaMalloc(&workspace,workspace_bytes));
  }
  void run(int index,cudaStream_t stream) {
    if(index<0 || index>=int(candidates.size()))throw std::runtime_error("Invalid library FP4 candidate");
    const float one=1,zero=0;
    check_blas(cublasLtMatmul(handle,operation,&one,source->b,a,source->a,b,
      &zero,source->output,c,source->output,c,&candidates[index].algo,workspace,workspace_bytes,stream));
  }
};

extern "C" void* rq_library_fp4_create(StudyPlan* source) {
  try {auto result=std::make_unique<LibraryFP4>();result->initialize(source);return result.release();}
  catch(const std::exception& e){last_error=e.what();return nullptr;}
}
extern "C" void rq_library_fp4_destroy(LibraryFP4* p){delete p;}
extern "C" int rq_library_fp4_count(LibraryFP4* p){return p?int(p->candidates.size()):0;}
extern "C" int rq_library_fp4_run(LibraryFP4* p,int candidate,cudaStream_t stream) {
  try {if(!p)throw std::runtime_error("Null library FP4 plan");p->run(candidate,stream);return 0;}
  catch(const std::exception& e){last_error=e.what();return 1;}
}

__global__ void classify(const float* scores,const float* bias,int* labels,int rows,int width,int classes) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=rows)return;
  int label=0;
  float maximum=scores[row*width]+bias[0];
  for(int column=1;column<classes;++column){
    float value=scores[row*width+column]+bias[column];
    if(value>maximum){maximum=value;label=column;}
  }
  labels[row]=label;
}
extern "C" int rq_classify(const float* scores,const float* bias,int* labels,int rows,int width,int classes,cudaStream_t stream) {
  try {
    if(!scores || !bias || !labels || rows<=0 || width<=0 || classes<=0 || classes>width)
      throw std::runtime_error("Invalid classification dimensions or pointers");
    classify<<<(rows+127)/128,128,0,stream>>>(scores,bias,labels,rows,width,classes);
    check_cuda(cudaGetLastError());return 0;
  }catch(const std::exception& e){last_error=e.what();return 1;}
}
