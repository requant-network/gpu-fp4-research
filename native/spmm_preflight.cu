// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
// Capability/reference probe only; not the hybrid SpMM experiment.
#include <cuda_runtime.h>
#include <cusparse.h>
#include <chrono>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

static thread_local std::string error_text, meta_text;
static void cuda_check(cudaError_t value) {if(value)throw std::runtime_error(cudaGetErrorString(value));}
static void sparse_check(cusparseStatus_t value) {if(value)throw std::runtime_error("cuSPARSE status "+std::to_string(value));}
struct Probe {
  int m,k,n,nnz,precision,algorithm,row_major,preprocessed,version=0;
  cudaDataType type;size_t input_bytes,workspace_bytes=0,owned_bytes=0;
  cudaStream_t stream=nullptr;cudaEvent_t begin=nullptr,end=nullptr;
  cusparseHandle_t handle=nullptr;cusparseSpMatDescr_t a=nullptr;
  cusparseDnMatDescr_t b=nullptr,c=nullptr;
  int *rowptr=nullptr,*columns=nullptr;void *values=nullptr,*right=nullptr,*workspace=nullptr;
  float* output=nullptr;float elapsed_ms=0;double setup_ms=0;
  ~Probe() {
    if(stream)cudaStreamSynchronize(stream);
    if(a)cusparseDestroySpMat(a);if(b)cusparseDestroyDnMat(b);if(c)cusparseDestroyDnMat(c);
    for(void* p:{static_cast<void*>(rowptr),static_cast<void*>(columns),values,right,workspace,static_cast<void*>(output)})cudaFree(p);
    if(handle)cusparseDestroy(handle);if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);if(stream)cudaStreamDestroy(stream);
  }
  void initialize(int rows,int inner,int width,int count,int p,int algo,int layout,int preprocess,
                  const int* rp,const int* ci,const void* av,const void* bv) {
    if(rows<1 || rows>4096 || inner<1 || inner>4096 || width<1 || width>1024 || count<1 || count>int64_t(rows)*inner || p<0 || p>2 || (layout!=0 && layout!=1) || (preprocess!=0 && preprocess!=1) || !rp || !ci || !av || !bv)
      throw std::runtime_error("Invalid SpMM probe arguments");
    if(rp[0]!=0 || rp[rows]!=count)throw std::runtime_error("Invalid CSR offsets");
    for(int row=0;row<rows;++row) {
      if(rp[row]<0 || rp[row]>rp[row+1] || rp[row+1]>count)throw std::runtime_error("Invalid CSR row extent");
      for(int index=rp[row];index<rp[row+1];++index)
        if(ci[index]<0 || ci[index]>=inner || (index>rp[row] && ci[index]<=ci[index-1]))throw std::runtime_error("Expected sorted unique CSR columns");
    }
    m=rows;k=inner;n=width;nnz=count;precision=p;algorithm=algo;row_major=layout;preprocessed=preprocess;
    type=p==0?CUDA_R_32F:(p==1?CUDA_R_16F:CUDA_R_16BF);input_bytes=p==0?4:2;
    cuda_check(cudaSetDevice(0));cudaDeviceProp properties;cuda_check(cudaGetDeviceProperties(&properties,0));
    if(properties.major!=12 || properties.minor!=0)throw std::runtime_error("SM120 GPU required");
    cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cuda_check(cudaEventCreate(&begin));cuda_check(cudaEventCreate(&end));
    sparse_check(cusparseCreate(&handle));sparse_check(cusparseSetStream(handle,stream));sparse_check(cusparseGetVersion(handle,&version));
    cuda_check(cudaMalloc(&rowptr,size_t(m+1)*4));cuda_check(cudaMalloc(&columns,size_t(nnz)*4));
    cuda_check(cudaMalloc(&values,size_t(nnz)*input_bytes));cuda_check(cudaMalloc(&right,size_t(k)*n*input_bytes));
    cuda_check(cudaMalloc(&output,size_t(m)*n*4));
    cuda_check(cudaMemcpyAsync(rowptr,rp,size_t(m+1)*4,cudaMemcpyHostToDevice,stream));
    cuda_check(cudaMemcpyAsync(columns,ci,size_t(nnz)*4,cudaMemcpyHostToDevice,stream));
    cuda_check(cudaMemcpyAsync(values,av,size_t(nnz)*input_bytes,cudaMemcpyHostToDevice,stream));
    cuda_check(cudaMemcpyAsync(right,bv,size_t(k)*n*input_bytes,cudaMemcpyHostToDevice,stream));
    auto order=row_major?CUSPARSE_ORDER_ROW:CUSPARSE_ORDER_COL;
    sparse_check(cusparseCreateCsr(&a,m,k,nnz,rowptr,columns,values,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,type));
    sparse_check(cusparseCreateDnMat(&b,k,n,row_major?n:k,right,type,order));
    sparse_check(cusparseCreateDnMat(&c,m,n,row_major?n:m,output,CUDA_R_32F,order));
    float one=1,zero=0;
    sparse_check(cusparseSpMM_bufferSize(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,a,b,&zero,c,CUDA_R_32F,static_cast<cusparseSpMMAlg_t>(algorithm),&workspace_bytes));
    if(workspace_bytes)cuda_check(cudaMalloc(&workspace,workspace_bytes));
    cuda_check(cudaStreamSynchronize(stream));
    auto started=std::chrono::steady_clock::now();
    if(preprocessed)sparse_check(cusparseSpMM_preprocess(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,a,b,&zero,c,CUDA_R_32F,static_cast<cusparseSpMMAlg_t>(algorithm),workspace));
    cuda_check(cudaStreamSynchronize(stream));
    setup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    owned_bytes=size_t(m+1+nnz)*4+size_t(nnz+k*n)*input_bytes+size_t(m)*n*4+workspace_bytes;
  }
  void run() {
    float one=1,zero=0;
    cuda_check(cudaEventRecord(begin,stream));
    sparse_check(cusparseSpMM(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,a,b,&zero,c,CUDA_R_32F,static_cast<cusparseSpMMAlg_t>(algorithm),workspace));
    cuda_check(cudaEventRecord(end,stream));cuda_check(cudaEventSynchronize(end));cuda_check(cudaEventElapsedTime(&elapsed_ms,begin,end));
  }
};
extern "C" {
const char* mmp_error(){return error_text.c_str();}
void* mmp_new(int m,int k,int n,int nnz,int precision,int algorithm,int row_major,int preprocess,
              const int* rowptr,const int* columns,const void* values,const void* right) {
  try {auto plan=std::make_unique<Probe>();plan->initialize(m,k,n,nnz,precision,algorithm,row_major,preprocess,rowptr,columns,values,right);return plan.release();}
  catch(const std::exception& e){error_text=e.what();return nullptr;}
}
void mmp_delete(void* p){delete static_cast<Probe*>(p);}
int mmp_run(void* p){try{if(!p)throw std::runtime_error("Null SpMM probe");static_cast<Probe*>(p)->run();return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
int mmp_output(void* p,float* host){try{if(!p || !host)throw std::runtime_error("Null SpMM output");auto& q=*static_cast<Probe*>(p);cuda_check(cudaMemcpyAsync(host,q.output,size_t(q.m)*q.n*4,cudaMemcpyDeviceToHost,q.stream));cuda_check(cudaStreamSynchronize(q.stream));return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
int mmp_update_b(void* p,const void* host){try{if(!p || !host)throw std::runtime_error("Null SpMM B");auto& q=*static_cast<Probe*>(p);cuda_check(cudaMemcpyAsync(q.right,host,size_t(q.k)*q.n*q.input_bytes,cudaMemcpyHostToDevice,q.stream));cuda_check(cudaStreamSynchronize(q.stream));return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
const char* mmp_metadata(void* p){try{if(!p)throw std::runtime_error("Null SpMM probe");auto& q=*static_cast<Probe*>(p);std::ostringstream out;
  out<<std::setprecision(17)<<"{\"cuda_ms\":"<<q.elapsed_ms<<",\"preprocess_wall_ms\":"<<q.setup_ms<<",\"workspace_bytes\":"<<q.workspace_bytes
     <<",\"owned_device_bytes\":"<<q.owned_bytes<<",\"cusparse_version\":"<<q.version<<"}";meta_text=out.str();return meta_text.c_str();}
  catch(const std::exception& e){error_text=e.what();return nullptr;}}
}
