// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
// Reuse the frozen native NVFP4 packer and CUTLASS configuration without editing them.
#include "hardware_bridge.cu"
#include <chrono>
#include <iomanip>
#include <limits>
#include <sstream>
#include <math_constants.h>

extern "C" uintptr_t sq_select_create(cudaStream_t);
extern "C" void sq_select_delete(uintptr_t);
extern "C" const char* sq_select_error();
extern "C" int sq_select(uintptr_t,const float*,const int64_t*,float*,int64_t*,int,int,int);
static thread_local std::string search_error,search_json;

struct SearchGemm:StudyPlan{
  std::vector<uint64_t> flags;
  std::vector<int> candidate_states;
  std::vector<double> candidate_ms;
  std::unique_ptr<LibraryFP4> library_fp4;
  int selected=-1;
  ~SearchGemm(){if(mode==4){a=nullptr;b=nullptr;}}
  void init(int kind,int rows,int columns,int depth,const float* query,const float* database,float* result,cudaStream_t stream){
    if(kind<0 || kind>5 || rows<1 || rows>128 || columns<128 || columns>1048576 || columns%128 ||
       depth<128 || depth>256 || depth%128 || !query || !database || !result)throw std::runtime_error("Invalid bounded search GEMM configuration");
    mode=kind==5?0:kind;m=rows;n=columns;k=depth;input_a=query;input_b=database;output=result;
    int bits=mode==0?4:(mode==1?8:(mode==4?32:16));a_bytes=size_t(m)*k*bits/8;b_bytes=size_t(n)*k*bits/8;
    if(mode==4){a=const_cast<float*>(query);b=const_cast<float*>(database);}
    else{check_cuda(cudaMalloc(&a,a_bytes));check_cuda(cudaMalloc(&b,b_bytes));}
    if(mode==0){
      using Scale=typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
      sa=cutlass::make_cute_packed_stride(StrideA{},{m,k,1});sb=cutlass::make_cute_packed_stride(StrideB{},{n,k,1});
      sc=cutlass::make_cute_packed_stride(StrideC{},{m,n,1});sd=cutlass::make_cute_packed_stride(StrideD{},{m,n,1});
      la=Scale::tile_atom_to_shape_SFA(make_shape(m,n,k,1));lb=Scale::tile_atom_to_shape_SFB(make_shape(m,n,k,1));
      check_cuda(cudaMalloc(&sfa,size(filter_zeros(la))));check_cuda(cudaMalloc(&sfb,size(filter_zeros(lb))));
      check_cuda(cudaMalloc(&flat_a,size_t(m)*k/16));check_cuda(cudaMalloc(&flat_b,size_t(n)*k/16));
      check_cuda(cudaMemsetAsync(sfa,0x38,size(filter_zeros(la)),stream));check_cuda(cudaMemsetAsync(sfb,0x38,size(filter_zeros(lb)),stream));
      check_cutlass(gemm.can_implement(arguments()));workspace_bytes=Gemm::get_workspace_size(arguments());
    }else{
      check_blas(cublasLtCreate(&handle));check_blas(cublasLtMatmulDescCreate(&operation,mode==4?CUBLAS_COMPUTE_32F_PEDANTIC:CUBLAS_COMPUTE_32F,CUDA_R_32F));
      cublasOperation_t transpose=CUBLAS_OP_T,normal=CUBLAS_OP_N;
      check_blas(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
      check_blas(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSB,&normal,sizeof(normal)));
      cudaDataType type=mode==1?CUDA_R_8F_E4M3:(mode==2?CUDA_R_16F:(mode==3?CUDA_R_16BF:CUDA_R_32F));
      check_blas(cublasLtMatrixLayoutCreate(&matrix_a,type,k,n,k));check_blas(cublasLtMatrixLayoutCreate(&matrix_b,type,k,m,k));
      check_blas(cublasLtMatrixLayoutCreate(&matrix_c,CUDA_R_32F,n,m,n));
      cublasLtMatmulPreference_t preference=nullptr;check_blas(cublasLtMatmulPreferenceCreate(&preference));
      auto status=cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes));
      if(status){cublasLtMatmulPreferenceDestroy(preference);check_blas(status);}
      algorithms.resize(8);int count=0;status=cublasLtMatmulAlgoGetHeuristic(handle,operation,matrix_a,matrix_b,matrix_c,matrix_c,preference,8,algorithms.data(),&count);
      cublasLtMatmulPreferenceDestroy(preference);check_blas(status);algorithms.resize(count);
      if(!count)throw std::runtime_error("No usable search GEMM heuristic");
    }
    if(workspace_bytes)check_cuda(cudaMalloc(&workspace,workspace_bytes));
    if(kind==5){library_fp4=std::make_unique<LibraryFP4>();library_fp4->initialize(this);}
  }
  void prepare(int mask,cudaStream_t stream){
    if(mode==4)return;
    if(mode){quantize(mask,stream);return;}
    if(mask&1)launch_native<false>(input_a,nullptr,a,sfa,flat_a,m,k,la,12,stream);
    if(mask&2)launch_native<false>(input_b,nullptr,b,sfb,flat_b,n,k,lb,12,stream);check_cuda(cudaGetLastError());
  }
  void tune(cudaStream_t stream,cudaEvent_t begin,cudaEvent_t end){
    if(mode==0 && !library_fp4){selected=0;return;}double best=1e100;
    if(library_fp4){
      for(int i=0;i<int(library_fp4->candidates.size());++i){uint64_t implementation=0;size_t written=0;
        auto cap=cublasLtMatmulAlgoCapGetAttribute(&library_fp4->candidates[i].algo,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,&implementation,sizeof(implementation),&written);
        flags.push_back(implementation);candidate_states.push_back(int(library_fp4->candidates[i].state));candidate_ms.push_back(-1);
        if(cap || library_fp4->candidates[i].state!=CUBLAS_STATUS_SUCCESS)continue;
        bool valid=true;double total=0;
        for(int j=0;j<3;++j){float ms=0;check_cuda(cudaEventRecord(begin,stream));try{library_fp4->run(i,stream);}catch(const std::exception&){valid=false;candidate_states.back()=-1;break;}
          check_cuda(cudaEventRecord(end,stream));check_cuda(cudaEventSynchronize(end));check_cuda(cudaEventElapsedTime(&ms,begin,end));total+=ms;}
        if(valid){candidate_ms.back()=total/3;if(total<best){best=total;selected=i;}}}
      if(selected<0)throw std::runtime_error("No executable library FP4 search GEMM");return;
    }
    for(int i=0;i<int(algorithms.size());++i){uint64_t implementation=0;size_t written=0;
      auto cap=cublasLtMatmulAlgoCapGetAttribute(&algorithms[i].algo,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,&implementation,sizeof(implementation),&written);flags.push_back(implementation);
      candidate_states.push_back(int(algorithms[i].state));candidate_ms.push_back(-1);
      if(cap || algorithms[i].state!=CUBLAS_STATUS_SUCCESS || ((mode==2 || mode==3) && !(implementation&CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA)))continue;
      bool valid=true;double total=0;
      for(int j=0;j<3;++j){float one=1,zero=0,ms=0;check_cuda(cudaEventRecord(begin,stream));
        auto status=cublasLtMatmul(handle,operation,&one,b,matrix_a,a,matrix_b,&zero,output,matrix_c,output,matrix_c,&algorithms[i].algo,workspace,workspace_bytes,stream);
        if(status){valid=false;candidate_states.back()=int(status);break;}check_cuda(cudaEventRecord(end,stream));check_cuda(cudaEventSynchronize(end));check_cuda(cudaEventElapsedTime(&ms,begin,end));total+=ms;}
      if(valid){candidate_ms.back()=total/3;if(total<best){best=total;selected=i;}}}
    if(selected<0)throw std::runtime_error("No executable search GEMM with required arithmetic implementation");
  }
  void multiply(cudaStream_t stream){if(library_fp4)library_fp4->run(selected,stream);else run(selected,stream);}
  size_t bytes(){return (mode==4?0:a_bytes+b_bytes)+workspace_bytes+(library_fp4?library_fp4->workspace_bytes:0)+(mode==0?size(filter_zeros(la))+size(filter_zeros(lb))+size_t(m+n)*k/16:0);}
};

static __device__ float search_value(const void* encoded,const cutlass::float_ue4m3_t* scales,size_t index,int mode){
  if(mode==0){uint8_t nibble=(static_cast<const uint8_t*>(encoded)[index/2]>>((index&1)*4))&15;
    float value=nibble%8==0?0:(nibble%8==1?.5f:(nibble%8==2?1.f:(nibble%8==3?1.5f:(nibble%8==4?2.f:(nibble%8==5?3.f:(nibble%8==6?4.f:6.f))))));
    return (nibble&8?-value:value)*float(scales[index/16]);}
  if(mode==1)return float(static_cast<const cutlass::float_e4m3_t*>(encoded)[index]);
  if(mode==2)return __half2float(static_cast<const __half*>(encoded)[index]);
  if(mode==3)return __bfloat162float(static_cast<const __nv_bfloat16*>(encoded)[index]);
  return static_cast<const float*>(encoded)[index];
}
__global__ void search_norms(int rows,int depth,const float* original,const void* encoded,const cutlass::float_ue4m3_t* scales,int mode,bool reconstructed,float* norms){
  int row=blockIdx.x*4+threadIdx.y;if(row>=rows)return;float sum=0;
  for(int j=threadIdx.x;j<depth;j+=32){size_t i=size_t(row)*depth+j;float v=reconstructed?search_value(encoded,scales,i,mode):original[i];sum=__fadd_rn(sum,__fmul_rn(v,v));}
  for(int offset=16;offset;offset/=2)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,offset));if(!threadIdx.x)norms[row]=sum;
}
__global__ void search_distances(size_t count,int columns,const float* dot,const float* query_norm,const float* database_norm,const uint8_t* allowed,float* distance){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;int id=i%columns;
  distance[i]=allowed[id]?__fsub_rn(__fadd_rn(query_norm[i/columns],database_norm[id]),__fmul_rn(2.f,dot[i])):CUDART_INF_F;
}
__global__ void search_refine(int batch,int count,int depth,int database_rows,const float* database,const float* queries,const int64_t* candidates,const uint8_t* allowed,float* output){
  int index=blockIdx.x*4+threadIdx.y;if(index>=batch*count)return;int64_t id=candidates[index];float total=0;
  if(id>=0 && id<database_rows && allowed[id]){
    for(int j=threadIdx.x;j<depth;j+=32){float d=__fsub_rn(database[size_t(id)*depth+j],queries[size_t(index/count)*depth+j]);total=__fadd_rn(total,__fmul_rn(d,d));}
    for(int offset=16;offset;offset/=2)total=__fadd_rn(total,__shfl_down_sync(0xffffffff,total,offset));
  }else total=CUDART_INF_F;
  if(!threadIdx.x)output[index]=total;
}
__global__ void search_sanitize(int count,int rows,const uint8_t* allowed,float* distances,int64_t* ids){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;int64_t id=ids[i];
  if(id<0 || id>=rows || !allowed[id] || !isfinite(distances[i])){ids[i]=-1;distances[i]=CUDART_INF_F;}
}

struct SearchPlan{
  int n=0,d=0,batch=0,mode=0,count=10,k=10,norm_mode=0;bool refinement=true,ready=false;
  cudaStream_t stream=nullptr;cudaEvent_t begin=nullptr,end=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t execution=nullptr;
  uintptr_t selection=0;SearchGemm gemm;std::vector<void*> allocations;size_t owned=0;int repetitions=1;
  float *database=nullptr,*queries=nullptr,*dot=nullptr,*distances=nullptr,*database_norm=nullptr,*query_norm=nullptr;
  float *candidate_distances=nullptr,*refined_distances=nullptr,*output_distances=nullptr;int64_t *candidates=nullptr,*output_ids=nullptr;
  uint8_t* allowed=nullptr;double last_ms=0,last_wall_ms=0;
  ~SearchPlan(){if(stream)cudaStreamSynchronize(stream);if(execution)cudaGraphExecDestroy(execution);if(graph)cudaGraphDestroy(graph);
    sq_select_delete(selection);for(void* p:allocations)cudaFree(p);if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);if(stream)cudaStreamDestroy(stream);}
  template<class T>T* allocate(size_t count){T* p=nullptr;size_t bytes=count*sizeof(T);check_cuda(cudaMalloc(&p,bytes));allocations.push_back(p);owned+=bytes;return p;}
  void init(int rows,int depth,int b,int precision,int candidate_count,int output_count,int norms,bool rerank,const float* x,const float* q,const uint8_t* mask){
    if(rows<128 || rows>1048576 || rows%128 || depth<128 || depth>256 || depth%128 || b<1 || b>128 || precision<0 || precision>5 ||
       output_count<1 || output_count>100 || candidate_count<output_count || candidate_count>512 || candidate_count>rows || norms<0 || norms>1 ||
       (!rerank && candidate_count!=output_count) || !x || !q || !mask)throw std::runtime_error("Invalid complete search plan arguments");
    check_cuda(cudaSetDevice(0));cudaDeviceProp prop;check_cuda(cudaGetDeviceProperties(&prop,0));if(prop.major!=12 || prop.minor!=0)throw std::runtime_error("SM120 GPU required");
    n=rows;d=depth;batch=b;mode=precision==5?0:precision;count=candidate_count;k=output_count;norm_mode=norms;refinement=rerank;
    check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check_cuda(cudaEventCreate(&begin));check_cuda(cudaEventCreate(&end));
    database=allocate<float>(size_t(n)*d);queries=allocate<float>(size_t(batch)*d);allowed=allocate<uint8_t>(n);
    dot=allocate<float>(size_t(batch)*n);distances=allocate<float>(size_t(batch)*n);database_norm=allocate<float>(n);query_norm=allocate<float>(batch);
    candidates=allocate<int64_t>(size_t(batch)*count);candidate_distances=allocate<float>(size_t(batch)*count);refined_distances=allocate<float>(size_t(batch)*count);
    output_ids=allocate<int64_t>(size_t(batch)*k);output_distances=allocate<float>(size_t(batch)*k);
    check_cuda(cudaMemcpyAsync(database,x,size_t(n)*d*4,cudaMemcpyHostToDevice,stream));check_cuda(cudaMemcpyAsync(queries,q,size_t(batch)*d*4,cudaMemcpyHostToDevice,stream));
    check_cuda(cudaMemcpyAsync(allowed,mask,n,cudaMemcpyHostToDevice,stream));
    gemm.init(precision,batch,n,d,queries,database,dot,stream);gemm.prepare(3,stream);
    search_norms<<<(n+3)/4,dim3(32,4),0,stream>>>(n,d,database,gemm.b,gemm.flat_b,mode,norm_mode,database_norm);check_cuda(cudaGetLastError());
    gemm.tune(stream,begin,end);selection=sq_select_create(stream);if(!selection)throw std::runtime_error(sq_select_error());
    check_cuda(cudaStreamSynchronize(stream));
  }
  void load(const float* q){if(!q)throw std::runtime_error("Missing query buffer");check_cuda(cudaMemcpyAsync(queries,q,size_t(batch)*d*4,cudaMemcpyHostToDevice,stream));check_cuda(cudaStreamSynchronize(stream));ready=false;}
  void run(){gemm.prepare(1,stream);search_norms<<<(batch+3)/4,dim3(32,4),0,stream>>>(batch,d,queries,gemm.a,gemm.flat_a,mode,norm_mode,query_norm);
    gemm.multiply(stream);search_distances<<<(size_t(batch)*n+255)/256,256,0,stream>>>(size_t(batch)*n,n,dot,query_norm,database_norm,allowed,distances);check_cuda(cudaGetLastError());
    if(sq_select(selection,distances,nullptr,candidate_distances,candidates,batch,n,count))throw std::runtime_error(sq_select_error());
    search_sanitize<<<(batch*count+255)/256,256,0,stream>>>(batch*count,n,allowed,candidate_distances,candidates);check_cuda(cudaGetLastError());
    if(refinement){search_refine<<<(batch*count+3)/4,dim3(32,4),0,stream>>>(batch,count,d,n,database,queries,candidates,allowed,refined_distances);check_cuda(cudaGetLastError());
      if(sq_select(selection,refined_distances,candidates,output_distances,output_ids,batch,count,k))throw std::runtime_error(sq_select_error());
    }else{check_cuda(cudaMemcpyAsync(output_ids,candidates,size_t(batch)*k*8,cudaMemcpyDeviceToDevice,stream));check_cuda(cudaMemcpyAsync(output_distances,candidate_distances,size_t(batch)*k*4,cudaMemcpyDeviceToDevice,stream));}
    search_sanitize<<<(batch*k+255)/256,256,0,stream>>>(batch*k,n,allowed,output_distances,output_ids);check_cuda(cudaGetLastError());ready=true;}
  void capture(int reps){if(reps<1 || reps>32 || graph || !ready)throw std::runtime_error("Invalid/repeated search capture or missing initial run");
    check_cuda(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));try{for(int i=0;i<reps;++i)run();check_cuda(cudaStreamEndCapture(stream,&graph));}
    catch(...){cudaGraph_t invalid=nullptr;cudaStreamEndCapture(stream,&invalid);if(invalid)cudaGraphDestroy(invalid);throw;}
    check_cuda(cudaGraphInstantiate(&execution,graph,0));repetitions=reps;}
  void time(bool captured){if(captured && !execution)throw std::runtime_error("Capture search first");auto start=std::chrono::steady_clock::now();float ms=0;
    check_cuda(cudaEventRecord(begin,stream));if(captured)check_cuda(cudaGraphLaunch(execution,stream));else run();check_cuda(cudaEventRecord(end,stream));check_cuda(cudaEventSynchronize(end));
    check_cuda(cudaEventElapsedTime(&ms,begin,end));int divisor=captured?repetitions:1;last_ms=ms/divisor;last_wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/divisor;ready=true;}
  size_t bytes(int which){switch(which){case 0:return size_t(n)*d*4;case 1:return size_t(batch)*d*4;case 2:return gemm.a_bytes;case 3:return gemm.b_bytes;
    case 4:return mode==0?size_t(batch)*d/16:0;case 5:return mode==0?size_t(n)*d/16:0;case 6:return size_t(batch)*n*4;case 7:return n*4;case 8:return batch*4;
    case 9:return size_t(batch)*n*4;case 10:return size_t(batch)*count*8;case 11:return size_t(batch)*count*4;case 12:return refinement?size_t(batch)*count*4:0;case 13:return size_t(batch)*k*8;case 14:return size_t(batch)*k*4;case 15:return n;default:throw std::runtime_error("Invalid search snapshot selector");}}
  void snapshot(int which,void* host){size_t size=bytes(which);void* pointers[]={database,queries,gemm.a,gemm.b,gemm.flat_a,gemm.flat_b,dot,database_norm,query_norm,distances,candidates,candidate_distances,refined_distances,output_ids,output_distances,allowed};
    if((which==6 || which==8 || (which>=9 && which<=14)) && !ready)throw std::runtime_error("Run the search before reading its computed buffers");
    if(size && !host)throw std::runtime_error("Null search snapshot output");if(size)check_cuda(cudaMemcpyAsync(host,pointers[which],size,cudaMemcpyDeviceToHost,stream));check_cuda(cudaStreamSynchronize(stream));}
  std::string describe(){std::ostringstream out;out<<std::setprecision(17)<<"{\"cuda_ms\":"<<last_ms<<",\"wall_ms\":"<<last_wall_ms<<",\"owned_device_bytes\":"<<owned+gemm.bytes()
    <<",\"lt_workspace_bytes\":"<<gemm.workspace_bytes+(gemm.library_fp4?gemm.library_fp4->workspace_bytes:0)<<",\"gemm_backend\":\""<<(mode==0?(gemm.library_fp4?"cublaslt-fp4":"cutlass-sm120-fp4"):"cublaslt")
    <<"\",\"fp32_input_alias\":"<<(mode==4?"true":"false")<<",\"selection_initial_workspace_pool_bytes\":67108864,\"graph_repetitions\":"<<repetitions<<",\"lt_selected_index\":"<<gemm.selected<<",\"lt_candidate_numerical_flags\":[";
    for(size_t i=0;i<gemm.flags.size();++i){if(i)out<<",";out<<gemm.flags[i];}out<<"],\"lt_candidate_states\":[";
    for(size_t i=0;i<gemm.candidate_states.size();++i){if(i)out<<",";out<<gemm.candidate_states[i];}out<<"],\"lt_candidate_ms\":[";
    for(size_t i=0;i<gemm.candidate_ms.size();++i){if(i)out<<",";out<<gemm.candidate_ms[i];}
    out<<"],\"buffer_bytes\":[";for(int i=0;i<16;++i){if(i)out<<",";out<<bytes(i);}out<<"]}";return out.str();}
};

extern "C" {
const char* sq_error(){return search_error.c_str();}
void* sq_new(int n,int d,int batch,int precision,int count,int k,int norm_mode,int refine,const float* x,const float* q,const uint8_t* allowed){
  try{auto p=std::make_unique<SearchPlan>();p->init(n,d,batch,precision,count,k,norm_mode,refine!=0,x,q,allowed);return p.release();}catch(const std::exception& e){search_error=e.what();return nullptr;}}
void sq_delete(void* p){delete static_cast<SearchPlan*>(p);}
int sq_load(void* p,const float* q){try{if(!p)throw std::runtime_error("Null search plan");static_cast<SearchPlan*>(p)->load(q);return 0;}catch(const std::exception& e){search_error=e.what();return -1;}}
int sq_run(void* p){try{if(!p)throw std::runtime_error("Null search plan");auto& s=*static_cast<SearchPlan*>(p);s.run();check_cuda(cudaStreamSynchronize(s.stream));return 0;}catch(const std::exception& e){search_error=e.what();return -1;}}
int sq_time(void* p,int graph){try{if(!p)throw std::runtime_error("Null search plan");static_cast<SearchPlan*>(p)->time(graph!=0);return 0;}catch(const std::exception& e){search_error=e.what();return -1;}}
int sq_capture(void* p,int repetitions){try{if(!p)throw std::runtime_error("Null search plan");static_cast<SearchPlan*>(p)->capture(repetitions);return 0;}catch(const std::exception& e){search_error=e.what();return -1;}}
int sq_snapshot(void* p,int which,void* output){try{if(!p)throw std::runtime_error("Null search plan");static_cast<SearchPlan*>(p)->snapshot(which,output);return 0;}catch(const std::exception& e){search_error=e.what();return -1;}}
const char* sq_metadata(void* p){try{if(!p)throw std::runtime_error("Null search plan");search_json=static_cast<SearchPlan*>(p)->describe();return search_json.c_str();}catch(const std::exception& e){search_error=e.what();return nullptr;}}
}
