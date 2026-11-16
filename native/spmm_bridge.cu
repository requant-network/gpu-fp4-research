// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cusparse.h>
#include <cublasLt.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static thread_local std::string last_error, last_json;
static void ck(cudaError_t s){if(s)throw std::runtime_error(cudaGetErrorString(s));}
static void sk(cusparseStatus_t s,const char* where){if(s)throw std::runtime_error(std::string(where)+": cuSPARSE status "+std::to_string(s));}
static void bk(cublasStatus_t s,const char* where){if(s)throw std::runtime_error(std::string(where)+": cuBLASLt status "+std::to_string(s));}
template<class T> __device__ T cast(float v){return T(v);}
template<> __device__ __half cast(float v){return __float2half_rn(v);}
template<> __device__ __nv_bfloat16 cast(float v){return __float2bfloat16_rn(v);}
template<class T> __global__ void encode_a(size_t count,int kind,int m,int k,int block,const float* src,T* dst){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  size_t source=i;
  if(kind==1){size_t offset=i%(block*block);source=(i/(block*block))*block*block+(offset%block)*block+offset/block;}
  dst[i]=cast<T>(src[source]);
}
template<class T> __global__ void encode_b(size_t count,int kind,int k,int n,int block,bool column_major,
                                         const int* block_columns,const float* src,T* dst){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  int row,column;
  if(kind==1){size_t matrix=i/(size_t(block)*n),offset=i%(size_t(block)*n);row=block_columns[matrix]*block+offset%block;column=offset/block;}
  else if(column_major){row=i%k;column=i/k;}
  else {row=i/n;column=i%n;}
  dst[i]=cast<T>(row<k?src[size_t(row)*n+column]:0.f);
}
template<class T> __global__ void gather_b_tiles(int n,int block,const int* columns,const float* source,T* out){
  __shared__ float tile[32][33];
  int source_block=columns[blockIdx.y],tile_row=blockIdx.z*32,tile_column=blockIdx.x*32;
  for(int j=0;j<32;j+=8){int row=threadIdx.y+j,column=threadIdx.x;
    tile[row][column]=(tile_row+row<block && tile_column+column<n)?source[size_t(source_block*block+tile_row+row)*n+tile_column+column]:0.f;}
  __syncthreads();
  for(int j=0;j<32;j+=8){int row=threadIdx.x,column=threadIdx.y+j;
    if(tile_row+row<block && tile_column+column<n)
      out[size_t(blockIdx.y)*block*n+tile_row+row+size_t(tile_column+column)*block]=cast<T>(tile[row][column]);}
}
__global__ void finish(int m,int mp,int n,int kind,int block,bool column_major,const int* permutation,
                        const int* brp,const float* base,const float* tiles,float* out){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(m)*n)return;
  int row=i/n,column=i%n;float value=base[column_major?row+size_t(column)*mp:i];
  if(kind==1)for(int index=brp[row/block];index<brp[row/block+1];++index)
    value+=tiles[size_t(index)*block*n+row%block+size_t(column)*block];
  out[size_t(permutation[row])*n+column]=value;
}

struct Lt {
  cublasLtHandle_t handle=nullptr;cublasLtMatmulDesc_t operation=nullptr;
  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;cublasLtMatmulAlgo_t selected{};
  void* workspace=nullptr;size_t workspace_bytes=64ull<<20;
  uint64_t flags=0;int chosen=-1;std::vector<uint64_t> candidate_flags;std::vector<int> states;std::vector<double> times;
  ~Lt(){if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);
    if(operation)cublasLtMatmulDescDestroy(operation);if(handle)cublasLtDestroy(handle);}
  void run(const void* av,const void* bv,float* cv,cudaStream_t stream){float one=1,zero=0;
    bk(cublasLtMatmul(handle,operation,&one,av,a,bv,b,&zero,cv,c,cv,c,&selected,workspace,workspace_bytes,stream),"matmul");}
  void setup(int m,int n,int k,int batch,int precision,const void* av,const void* bv,float* cv,
             void* work,cudaStream_t stream,cudaEvent_t begin,cudaEvent_t end,bool row_major){
    workspace=work;bk(cublasLtCreate(&handle),"create");
    bk(cublasLtMatmulDescCreate(&operation,precision==0?CUBLAS_COMPUTE_32F_PEDANTIC:CUBLAS_COMPUTE_32F,CUDA_R_32F),"desc");
    auto type=precision==0?CUDA_R_32F:(precision==1?CUDA_R_16F:CUDA_R_16BF);
    bk(cublasLtMatrixLayoutCreate(&a,type,m,k,row_major?k:m),"layout A");bk(cublasLtMatrixLayoutCreate(&b,type,k,n,row_major?n:k),"layout B");
    bk(cublasLtMatrixLayoutCreate(&c,CUDA_R_32F,m,n,row_major?n:m),"layout C");
    int64_t strides[]={int64_t(m)*k,int64_t(k)*n,int64_t(m)*n};
    cublasLtMatrixLayout_t layouts[]={a,b,c};
    for(int i=0;i<3;++i){auto order=row_major?CUBLASLT_ORDER_ROW:CUBLASLT_ORDER_COL;
      bk(cublasLtMatrixLayoutSetAttribute(layouts[i],CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order)),"layout order");
      bk(cublasLtMatrixLayoutSetAttribute(layouts[i],CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch)),"batch");
      bk(cublasLtMatrixLayoutSetAttribute(layouts[i],CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&strides[i],sizeof(int64_t)),"stride");}
    cublasLtMatmulPreference_t preference=nullptr;bk(cublasLtMatmulPreferenceCreate(&preference),"preference");
    auto status=cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes));
    if(status){cublasLtMatmulPreferenceDestroy(preference);bk(status,"workspace preference");}
    cublasLtMatmulHeuristicResult_t candidates[8];int count=0;
    status=cublasLtMatmulAlgoGetHeuristic(handle,operation,a,b,c,c,preference,8,candidates,&count);
    cublasLtMatmulPreferenceDestroy(preference);bk(status,"heuristic");double best=1e100;
    for(int i=0;i<count;++i){uint64_t numerical=0;size_t written=0;
      auto cap=cublasLtMatmulAlgoCapGetAttribute(&candidates[i].algo,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,&numerical,sizeof(numerical),&written);
      candidate_flags.push_back(numerical);states.push_back(int(candidates[i].state));times.push_back(-1);
      if(cap || candidates[i].state || (precision!=0 && !(numerical&CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA)))continue;
      selected=candidates[i].algo;double measured=0;bool usable=true;
      for(int repeat=0;repeat<3;++repeat){float one=1,zero=0,ms=0;ck(cudaEventRecord(begin,stream));
        auto result=cublasLtMatmul(handle,operation,&one,av,a,bv,b,&zero,cv,c,cv,c,&selected,workspace,workspace_bytes,stream);
        if(result){states.back()=int(result);usable=false;break;}
        ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));ck(cudaEventElapsedTime(&ms,begin,end));measured+=ms;}
      if(usable){times.back()=measured/3;if(times.back()<best){best=times.back();chosen=i;flags=numerical;}}
    }
    if(chosen<0)throw std::runtime_error("No usable cuBLASLt candidate with required numerical implementation flags");
    selected=candidates[chosen].algo;
  }
};

struct Plan {
  int m,k,n,mp,kp,kind,precision,block,algorithm,layout,preprocess,nnz,blocks,ellcols;
  size_t owned=0,acount=0,bcount=0,element=4,sparse_workspace_bytes=0;
  float *a_source=nullptr,*block_source=nullptr,*b_source=nullptr,*c_base=nullptr,*c_tiles=nullptr,*output=nullptr;
  void *a_encoded=nullptr,*b_encoded=nullptr,*sparse_workspace=nullptr;
  int *rp=nullptr,*ci=nullptr,*brp=nullptr,*bci=nullptr,*permutation=nullptr;
  cudaStream_t stream=nullptr;cudaEvent_t begin=nullptr,end=nullptr;
  cudaGraph_t graphs[3]={};cudaGraphExec_t executions[3]={};int repetitions=0;
  cusparseHandle_t sparse=nullptr;cusparseSpMatDescr_t sparse_a=nullptr;cusparseDnMatDescr_t sparse_b=nullptr,sparse_c=nullptr;
  std::unique_ptr<Lt> lt;std::vector<void*> allocations;
  int sparse_version=0;double preprocess_ms=0,last_ms=0,last_wall_ms=0;
  bool direct_output=false;
  ~Plan(){if(stream)cudaStreamSynchronize(stream);for(auto graph:executions)if(graph)cudaGraphExecDestroy(graph);for(auto graph:graphs)if(graph)cudaGraphDestroy(graph);
    lt.reset();if(sparse_a)cusparseDestroySpMat(sparse_a);if(sparse_b)cusparseDestroyDnMat(sparse_b);if(sparse_c)cusparseDestroyDnMat(sparse_c);if(sparse)cusparseDestroy(sparse);
    for(void* pointer:allocations)cudaFree(pointer);if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);if(stream)cudaStreamDestroy(stream);}
  template<class T> T* allocate(size_t count){T* pointer=nullptr;size_t bytes=std::max(size_t(1),count)*sizeof(T);ck(cudaMalloc(&pointer,bytes));allocations.push_back(pointer);owned+=bytes;return pointer;}
  template<class T> T* upload(const T* host,size_t count){auto pointer=allocate<T>(count);if(count)ck(cudaMemcpyAsync(pointer,host,count*sizeof(T),cudaMemcpyHostToDevice,stream));return pointer;}
  static void validate_rows(int rows,int columns,int count,const int* offsets,const int* indices){
    if(!offsets || (count && !indices) || offsets[0]!=0 || offsets[rows]!=count)throw std::runtime_error("Invalid compressed row extents");
    for(int row=0;row<rows;++row){if(offsets[row]<0 || offsets[row]>offsets[row+1] || offsets[row+1]>count)throw std::runtime_error("Invalid compressed row offsets");
      for(int i=offsets[row];i<offsets[row+1];++i)if(indices[i]<0 || indices[i]>=columns || (i>offsets[row] && indices[i]<=indices[i-1]))throw std::runtime_error("Expected sorted unique in-range compressed columns");}
  }
  void initialize(int rows,int inner,int width,int padded_rows,int padded_inner,int format,int p,int bs,int algo,int order,int prep,int count,int nb,int ell,
                  const int* hrp,const int* hci,const float* ha,const int* hbrp,const int* hbci,const float* hblocks,const int* hperm,const float* hb){
    if(rows<1 || rows>4096 || inner<1 || inner>4096 || width<1 || width>1024 || padded_rows<rows || padded_rows>4160 || padded_inner<inner || padded_inner>4160 || format<0 || format>4 || p<0 || p>2 || bs<1 || bs>64 || count<0 || nb<0 || ell<0 || !hperm || !hb || (order!=0 && order!=1) || (prep!=0 && prep!=1))throw std::runtime_error("Invalid SpMM dimensions/configuration");
    m=rows;k=inner;n=width;mp=padded_rows;kp=padded_inner;kind=format;precision=p;block=bs;algorithm=algo;layout=order;preprocess=prep;nnz=count;blocks=nb;ellcols=ell;
    if((kind!=0 && (block!=16 && block!=32 && block!=64)) || mp%block || kp%block || (kind==1 && p==0))throw std::runtime_error("Invalid padded block configuration");
    bool identity=true;std::vector<bool> seen(m,false);for(int i=0;i<m;++i){if(hperm[i]<0 || hperm[i]>=m || seen[hperm[i]])throw std::runtime_error("Invalid row permutation");seen[hperm[i]]=true;identity=identity && hperm[i]==i;}
    direct_output=identity && layout==0 && (kind!=1 || blocks==0);
    if(kind==0 || kind==1)validate_rows(mp,kp,nnz,hrp,hci);
    if(kind==1)validate_rows(mp/block,kp/block,blocks,hbrp,hbci);
    if(kind==2)validate_rows(mp/block,kp/block,blocks,hrp,hci);
    if(kind==3){if(ellcols%block || ellcols>kp || !hci)throw std::runtime_error("Invalid Blocked-ELL extent");
      for(int i=0;i<(mp/block)*(ellcols/block);++i)if(hci[i]<-1 || hci[i]>=kp/block)throw std::runtime_error("Invalid Blocked-ELL column");}
    size_t source_count=(kind==0 || kind==1)?nnz:(kind==2?size_t(blocks)*block*block:(kind==3?size_t(mp)*ellcols:size_t(mp)*kp));
    if(source_count && !ha)throw std::runtime_error("Missing A values");
    for(size_t i=0;i<source_count;++i)if(!std::isfinite(ha[i]))throw std::runtime_error("Nonfinite A value");
    if(kind==1 && blocks && !hblocks)throw std::runtime_error("Missing dense block values");
    for(size_t i=0;i<(kind==1?size_t(blocks)*block*block:0);++i)if(!std::isfinite(hblocks[i]))throw std::runtime_error("Nonfinite dense block value");
    ck(cudaSetDevice(0));cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));if(prop.major!=12 || prop.minor!=0)throw std::runtime_error("SM120 GPU required");
    ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
    a_source=upload(ha,source_count);permutation=upload(hperm,m);b_source=allocate<float>(size_t(kp)*n);
    ck(cudaMemsetAsync(b_source,0,size_t(kp)*n*4,stream));load_b(hb);
    c_base=allocate<float>(size_t(mp)*n);output=direct_output?c_base:allocate<float>(size_t(m)*n);
    if(kind==0 || kind==1){rp=upload(hrp,mp+1);ci=upload(hci,nnz);}
    if(kind==2){rp=upload(hrp,mp/block+1);ci=upload(hci,blocks);}
    if(kind==3)ci=upload(hci,size_t(mp/block)*(ellcols/block));
    if(kind==1){brp=upload(hbrp,mp/block+1);bci=upload(hbci,blocks);block_source=upload(hblocks,size_t(blocks)*block*block);c_tiles=allocate<float>(size_t(blocks)*block*n);}
    element=p==0?4:2;acount=kind==1?size_t(blocks)*block*block:source_count;
    bcount=kind==1?size_t(blocks)*block*n:size_t(kp)*n;
    if(p==0)a_encoded=a_source;else a_encoded=allocate<unsigned char>(acount*element);
    if(p==0 && layout==0)b_encoded=b_source;else b_encoded=allocate<unsigned char>(bcount*element);
    prepare_a();prepare_b();ck(cudaStreamSynchronize(stream));
    if(kind!=4 && !(kind==0 && !nnz) && !(kind==1 && !nnz)){
      sk(cusparseCreate(&sparse),"create");sk(cusparseSetStream(sparse,stream),"stream");sk(cusparseGetVersion(sparse,&sparse_version),"version");
      auto type=p==0?CUDA_R_32F:(p==1?CUDA_R_16F:CUDA_R_16BF);auto sparse_type=kind==1?CUDA_R_32F:type;
      if(kind==0 || kind==1)sk(cusparseCreateCsr(&sparse_a,mp,kp,nnz,rp,ci,kind==1?static_cast<void*>(a_source):a_encoded,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,sparse_type),"CSR");
      if(kind==2)sk(cusparseCreateBsr(&sparse_a,mp/block,kp/block,blocks,block,block,rp,ci,a_encoded,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,type,CUSPARSE_ORDER_ROW),"BSR");
      if(kind==3)sk(cusparseCreateBlockedEll(&sparse_a,mp,kp,block,ellcols,ci,a_encoded,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,type),"BlockedELL");
      bool col=kind!=1 && layout==1;auto dense_order=col?CUSPARSE_ORDER_COL:CUSPARSE_ORDER_ROW;
      sk(cusparseCreateDnMat(&sparse_b,kp,n,col?kp:n,kind==1?static_cast<void*>(b_source):b_encoded,sparse_type,dense_order),"B");
      sk(cusparseCreateDnMat(&sparse_c,mp,n,col?mp:n,c_base,CUDA_R_32F,dense_order),"C");
      float one=1,zero=0;auto algorithm_value=static_cast<cusparseSpMMAlg_t>(algorithm);
      sk(cusparseSpMM_bufferSize(sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,sparse_a,sparse_b,&zero,sparse_c,CUDA_R_32F,algorithm_value,&sparse_workspace_bytes),"bufferSize");
      sparse_workspace=allocate<unsigned char>(sparse_workspace_bytes);
      auto time=std::chrono::steady_clock::now();
      if(preprocess)sk(cusparseSpMM_preprocess(sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,sparse_a,sparse_b,&zero,sparse_c,CUDA_R_32F,algorithm_value,sparse_workspace),"preprocess");
      ck(cudaStreamSynchronize(stream));preprocess_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-time).count();
    }
    if(kind==4 || (kind==1 && blocks)){
      lt=std::make_unique<Lt>();void* work=allocate<unsigned char>(64ull<<20);
      lt->setup(kind==1?block:mp,n,kind==1?block:kp,kind==1?blocks:1,p,a_encoded,b_encoded,kind==1?c_tiles:c_base,work,stream,begin,end,kind==4);
    }
    ck(cudaStreamSynchronize(stream));
  }
  void load_b(const float* host){if(!host)throw std::runtime_error("Missing B");ck(cudaMemcpyAsync(b_source,host,size_t(k)*n*4,cudaMemcpyHostToDevice,stream));ck(cudaStreamSynchronize(stream));}
  void prepare_a(){if(!acount || precision==0)return;const float* source=kind==1?block_source:a_source;
    int grid=(acount+255)/256;
    if(precision==0)encode_a<<<grid,256,0,stream>>>(acount,kind,mp,kp,block,source,static_cast<float*>(a_encoded));
    if(precision==1)encode_a<<<grid,256,0,stream>>>(acount,kind,mp,kp,block,source,static_cast<__half*>(a_encoded));
    if(precision==2)encode_a<<<grid,256,0,stream>>>(acount,kind,mp,kp,block,source,static_cast<__nv_bfloat16*>(a_encoded));ck(cudaGetLastError());}
  void prepare_b(){if(!bcount || (precision==0 && layout==0))return;
    if(kind==1){dim3 grid((n+31)/32,blocks,(block+31)/32),threads(32,8);
      if(precision==1)gather_b_tiles<<<grid,threads,0,stream>>>(n,block,bci,b_source,static_cast<__half*>(b_encoded));
      if(precision==2)gather_b_tiles<<<grid,threads,0,stream>>>(n,block,bci,b_source,static_cast<__nv_bfloat16*>(b_encoded));
      ck(cudaGetLastError());return;}
    int grid=(bcount+255)/256;bool col=layout==1;
    if(precision==0)encode_b<<<grid,256,0,stream>>>(bcount,kind,kp,n,block,col,bci,b_source,static_cast<float*>(b_encoded));
    if(precision==1)encode_b<<<grid,256,0,stream>>>(bcount,kind,kp,n,block,col,bci,b_source,static_cast<__half*>(b_encoded));
    if(precision==2)encode_b<<<grid,256,0,stream>>>(bcount,kind,kp,n,block,col,bci,b_source,static_cast<__nv_bfloat16*>(b_encoded));ck(cudaGetLastError());}
  void operation(int boundary){if(boundary<0 || boundary>2)throw std::runtime_error("Invalid timing boundary");if(boundary==2)prepare_a();if(boundary>=1)prepare_b();
    if(kind!=4){if(sparse){float one=1,zero=0;sk(cusparseSpMM(sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,sparse_a,sparse_b,&zero,sparse_c,CUDA_R_32F,static_cast<cusparseSpMMAlg_t>(algorithm),sparse_workspace),"SpMM");}
      else ck(cudaMemsetAsync(c_base,0,size_t(mp)*n*4,stream));}
    if(lt)lt->run(a_encoded,b_encoded,kind==1?c_tiles:c_base,stream);
    bool col=kind!=1 && layout==1;
    if(!direct_output){finish<<<(size_t(m)*n+255)/256,256,0,stream>>>(m,mp,n,kind,block,col,permutation,brp,c_base,c_tiles,output);ck(cudaGetLastError());}}
  void capture(int repeats){if(repeats<1 || repeats>64 || repetitions)throw std::runtime_error("Invalid/repeated graph construction");
    for(int boundary=0;boundary<3;++boundary){ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
      try{for(int i=0;i<repeats;++i)operation(boundary);ck(cudaStreamEndCapture(stream,&graphs[boundary]));}
      catch(...){cudaGraph_t invalid=nullptr;cudaStreamEndCapture(stream,&invalid);if(invalid)cudaGraphDestroy(invalid);throw;}
      ck(cudaGraphInstantiate(&executions[boundary],graphs[boundary],0));}repetitions=repeats;}
  void time(int boundary){if(boundary<0 || boundary>2 || !executions[boundary])throw std::runtime_error("Capture first");
    auto started=std::chrono::steady_clock::now();float ms=0;ck(cudaEventRecord(begin,stream));ck(cudaGraphLaunch(executions[boundary],stream));ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));ck(cudaEventElapsedTime(&ms,begin,end));
    last_ms=ms/repetitions;last_wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()/repetitions;}
  size_t buffer_size(int which){if(which==0)return acount*element;if(which==1)return bcount*element;if(which==2)return (kind==1?size_t(nnz):(kind==0?size_t(nnz):(kind==2?size_t(blocks)*block*block:(kind==3?size_t(mp)*ellcols:size_t(mp)*kp))))*4;if(which==3)return size_t(kp)*n*4;throw std::runtime_error("Invalid buffer selection");}
  void snapshot(int which,void* host){size_t bytes=buffer_size(which);void* pointer=which==0?a_encoded:(which==1?b_encoded:(which==2?static_cast<void*>(a_source):static_cast<void*>(b_source)));
    if(bytes && !host)throw std::runtime_error("Missing snapshot buffer");if(bytes)ck(cudaMemcpyAsync(host,pointer,bytes,cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));}
  std::string describe(){std::ostringstream out;out<<std::setprecision(17)<<"{\"cuda_ms\":"<<last_ms<<",\"graph_wall_ms\":"<<last_wall_ms<<",\"owned_device_bytes\":"<<owned
    <<",\"cusparse_version\":"<<sparse_version<<",\"preprocess_wall_ms\":"<<preprocess_ms<<",\"sparse_workspace_bytes\":"<<sparse_workspace_bytes<<",\"graph_repetitions\":"<<repetitions
    <<",\"packing\":\"shared_transpose32\",\"direct_output\":"<<(direct_output?"true":"false")
    <<",\"lt_selected_index\":"<<(lt?lt->chosen:-1)<<",\"lt_numerical_flags\":"<<(lt?lt->flags:0)<<",\"lt_candidates\":[";
    if(lt)for(size_t i=0;i<lt->states.size();++i){if(i)out<<",";out<<"{\"state\":"<<lt->states[i]<<",\"flags\":"<<lt->candidate_flags[i]<<",\"mean_ms\":"<<lt->times[i]<<"}";}
    out<<"],\"buffer_bytes\":[";for(int i=0;i<4;++i){if(i)out<<",";out<<buffer_size(i);}out<<"]}";return out.str();}
};

extern "C" {
const char* mm_error(){return last_error.c_str();}
const char* mm_environment(){try{ck(cudaSetDevice(0));cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));
  int runtime=0,driver=0,version=0;ck(cudaRuntimeGetVersion(&runtime));ck(cudaDriverGetVersion(&driver));
  cusparseHandle_t handle=nullptr;sk(cusparseCreate(&handle),"environment");auto status=cusparseGetVersion(handle,&version);cusparseDestroy(handle);sk(status,"version");
  std::ostringstream out;out<<"{\"gpu_name\":\""<<prop.name<<"\",\"compute_capability\":["<<prop.major<<","<<prop.minor<<"],\"multiprocessors\":"<<prop.multiProcessorCount
    <<",\"runtime_api_version\":"<<runtime<<",\"driver_api_version\":"<<driver<<",\"cusparse_version\":"<<version<<",\"cublaslt_version\":"<<cublasLtGetVersion()<<"}";
  last_json=out.str();return last_json.c_str();}catch(const std::exception& e){last_error=e.what();return nullptr;}}
void* mm_new(int m,int k,int n,int mp,int kp,int kind,int precision,int block,int algorithm,int layout,int preprocess,int nnz,int blocks,int ellcols,
              const int* rp,const int* ci,const float* av,const int* brp,const int* bci,const float* bv,const int* permutation,const float* right){
  try{auto plan=std::make_unique<Plan>();plan->initialize(m,k,n,mp,kp,kind,precision,block,algorithm,layout,preprocess,nnz,blocks,ellcols,rp,ci,av,brp,bci,bv,permutation,right);return plan.release();}
  catch(const std::exception& e){last_error=e.what();return nullptr;}}
void mm_delete(void* p){delete static_cast<Plan*>(p);}
int mm_load_b(void* p,const float* host){try{if(!p)throw std::runtime_error("Null plan");static_cast<Plan*>(p)->load_b(host);return 0;}catch(const std::exception& e){last_error=e.what();return -1;}}
int mm_run(void* p,int boundary){try{if(!p)throw std::runtime_error("Null plan");auto& plan=*static_cast<Plan*>(p);plan.operation(boundary);ck(cudaStreamSynchronize(plan.stream));return 0;}catch(const std::exception& e){last_error=e.what();return -1;}}
int mm_capture(void* p,int repeats){try{if(!p)throw std::runtime_error("Null plan");static_cast<Plan*>(p)->capture(repeats);return 0;}catch(const std::exception& e){last_error=e.what();return -1;}}
int mm_time(void* p,int boundary){try{if(!p)throw std::runtime_error("Null plan");static_cast<Plan*>(p)->time(boundary);return 0;}catch(const std::exception& e){last_error=e.what();return -1;}}
int mm_output(void* p,float* host){try{if(!p || !host)throw std::runtime_error("Null output/plan");auto& q=*static_cast<Plan*>(p);ck(cudaMemcpyAsync(host,q.output,size_t(q.m)*q.n*4,cudaMemcpyDeviceToHost,q.stream));ck(cudaStreamSynchronize(q.stream));return 0;}catch(const std::exception& e){last_error=e.what();return -1;}}
int mm_snapshot(void* p,int which,void* host){try{if(!p)throw std::runtime_error("Null plan");static_cast<Plan*>(p)->snapshot(which,host);return 0;}catch(const std::exception& e){last_error=e.what();return -1;}}
const char* mm_metadata(void* p){try{if(!p)throw std::runtime_error("Null plan");last_json=static_cast<Plan*>(p)->describe();return last_json.c_str();}catch(const std::exception& e){last_error=e.what();return nullptr;}}
}
