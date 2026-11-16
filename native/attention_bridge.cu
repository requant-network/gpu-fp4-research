// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
#include "optimized_bridge.cu"
#include <math_constants.h>

template<class ScaleLayout>
__global__ void attention_fp4(const float* input,uint8_t* packed,cutlass::float_ue4m3_t* physical,
    cutlass::float_ue4m3_t* logical,int rows,int k,int batches,ScaleLayout layout){
  int lane=blockIdx.x*blockDim.x+threadIdx.x,first=lane*2;if(first>=batches*rows*k)return;
  float x=input[first],y=input[first+1],maximum=fmaxf(fabsf(x),fabsf(y));
  for(int step=4;step;step/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,step,8));
  cutlass::float_ue4m3_t scale(maximum==0?1.f:fminf(448.f,fmaxf(1.f/512.f,maximum/6.f)));
  int group=first/16,within=group%(rows*k/16),batch=group/(rows*k/16);
  if(!(lane%8)){logical[group]=scale;physical[layout(make_coord(within/(k/16),(within%(k/16))*16,batch))]=scale;}
  packed[first/2]=uint8_t((cutlass::float_e2m1_t(x/float(scale)).raw()&15)|((cutlass::float_e2m1_t(y/float(scale)).raw()&15)<<4));
}

struct AttentionGemm:StudyPlan{
  int batches=1;
  Gemm::Arguments batched_arguments(){return {cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k,batches},
    {static_cast<const ElementA::DataType*>(a),sa,static_cast<const ElementB::DataType*>(b),sb,sfa,la,sfb,lb},
    {{1.f,0.f},output,sc,output,sd}};}
  template<class T>void resize(T*& pointer,size_t bytes){check_cuda(cudaFree(pointer));pointer=nullptr;if(bytes)check_cuda(cudaMalloc(&pointer,bytes));}
  void init(int kind,int batch,int rows,int columns,int depth,const float* x,const float* y,float* result){
    initialize(kind,rows,columns,depth,x,y,result);batches=batch;
    if(batch==1)return;
    a_bytes*=batch;b_bytes*=batch;resize(a,a_bytes);resize(b,b_bytes);
    if(mode==0){
      using Scale=typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
      sa=cutlass::make_cute_packed_stride(StrideA{},{m,k,batch});sb=cutlass::make_cute_packed_stride(StrideB{},{n,k,batch});
      sc=cutlass::make_cute_packed_stride(StrideC{},{m,n,batch});sd=cutlass::make_cute_packed_stride(StrideD{},{m,n,batch});
      la=Scale::tile_atom_to_shape_SFA(make_shape(m,n,k,batch));lb=Scale::tile_atom_to_shape_SFB(make_shape(m,n,k,batch));
      resize(sfa,size(filter_zeros(la)));resize(sfb,size(filter_zeros(lb)));resize(flat_a,size_t(batch)*m*k/16);resize(flat_b,size_t(batch)*n*k/16);
      check_cuda(cudaMemset(sfa,0x38,size(filter_zeros(la))));check_cuda(cudaMemset(sfb,0x38,size(filter_zeros(lb))));
      auto args=batched_arguments();check_cutlass(gemm.can_implement(args));workspace_bytes=Gemm::get_workspace_size(args);resize(workspace,workspace_bytes);
    }else{
      int64_t offsets[3]={int64_t(n)*k,int64_t(m)*k,int64_t(m)*n};cublasLtMatrixLayout_t layouts[3]={matrix_a,matrix_b,matrix_c};
      for(int index=0;index<3;++index){
        check_blas(cublasLtMatrixLayoutSetAttribute(layouts[index],CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch)));
        check_blas(cublasLtMatrixLayoutSetAttribute(layouts[index],CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&offsets[index],sizeof(offsets[index])));
      }
      cublasLtMatmulPreference_t preference=nullptr;check_blas(cublasLtMatmulPreferenceCreate(&preference));
      auto status=cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes));
      if(status!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(preference);check_blas(status);}
      algorithms.resize(8);int count=0;status=cublasLtMatmulAlgoGetHeuristic(handle,operation,matrix_a,matrix_b,matrix_c,matrix_c,preference,8,algorithms.data(),&count);
      cublasLtMatmulPreferenceDestroy(preference);check_blas(status);algorithms.resize(count);if(!count)throw std::runtime_error("No batched cuBLASLt attention candidate");
    }
  }
  void encode(int mask,cudaStream_t stream){
    if(mask<1 || mask>3)throw std::runtime_error("Invalid attention operand mask");
    if(mode==0){
      if(mask&1)attention_fp4<<<(batches*m*k/2+127)/128,128,0,stream>>>(input_a,(uint8_t*)a,sfa,flat_a,m,k,batches,la);
      if(mask&2)attention_fp4<<<(batches*n*k/2+127)/128,128,0,stream>>>(input_b,(uint8_t*)b,sfb,flat_b,n,k,batches,lb);
    }else{
      if(mask&1)convert_input<<<(batches*m*k+255)/256,256,0,stream>>>(input_a,a,batches*m*k,mode);
      if(mask&2)convert_input<<<(batches*n*k+255)/256,256,0,stream>>>(input_b,b,batches*n*k,mode);
    }
    check_cuda(cudaGetLastError());
  }
  void multiply(int algorithm,cudaStream_t stream){
    if(mode==0){if(algorithm)throw std::runtime_error("Invalid batched FP4 algorithm");check_cutlass(gemm.run(batched_arguments(),workspace,stream));}
    else run(algorithm,stream);
  }
  size_t bytes(){return a_bytes+b_bytes+workspace_bytes+(mode==0?size(filter_zeros(la))+size(filter_zeros(lb))+size_t(batches)*(m+n)*k/16:0);}
};

__global__ void attention_pad(const float* q,const float* k,const float* v,float* qp,float* kp,float* vt,int heads,int length,int depth,int sp,int dp){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=heads*sp*dp)return;
  int column=i%dp,row=(i/dp)%sp,head=i/(sp*dp);int source=(head*length+row)*depth+column;
  bool valid=row<length && column<depth;qp[i]=valid?q[source]:0.f;kp[i]=valid?k[source]:0.f;
  vt[(head*dp+column)*sp+row]=valid?v[source]:0.f;
}

__device__ bool attention_allowed(int row,int column,int length,int mask){return row<length && column<length && (mask!=1 || column<=row) && (mask!=2 || (row!=0 && column<length-length/4));}
__global__ void attention_softmax(const float* scores,float* probabilities,int length,int depth,int sp,int mask){
  int row=blockIdx.x%sp,lane=threadIdx.x;const float* source=scores+size_t(blockIdx.x)*sp;float* result=probabilities+size_t(blockIdx.x)*sp;
  __shared__ float sums[256];float maximum=-CUDART_INF_F,scale=rsqrtf(float(depth));
  for(int column=lane;column<sp;column+=256)if(attention_allowed(row,column,length,mask))maximum=fmaxf(maximum,source[column]*scale);
  sums[lane]=maximum;__syncthreads();for(int step=128;step;step/=2){if(lane<step)sums[lane]=fmaxf(sums[lane],sums[lane+step]);__syncthreads();}
  maximum=sums[0];__syncthreads();float total=0;
  for(int column=lane;column<sp;column+=256){float value=attention_allowed(row,column,length,mask)?expf(source[column]*scale-maximum):0.f;result[column]=value;total+=value;}
  sums[lane]=total;__syncthreads();for(int step=128;step;step/=2){if(lane<step)sums[lane]+=sums[lane+step];__syncthreads();}
  total=sums[0];for(int column=lane;column<sp;column+=256)result[column]=total>0?result[column]/total:0.f;
}
__global__ void attention_crop(const float* source,float* output,int heads,int length,int depth,int sp,int dp){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<heads*length*depth)output[i]=source[(i/(length*depth)*sp+i/depth%length)*dp+i%depth];
}

struct AttentionPlan{
  int heads=0,length=0,depth=0,sp=0,dp=0,mask=0;const float *q=nullptr,*k=nullptr,*v=nullptr;float* output=nullptr;
  float *qp=nullptr,*kp=nullptr,*vt=nullptr,*scores=nullptr,*probabilities=nullptr,*padded_output=nullptr;
  AttentionGemm qk,pv;
  ~AttentionPlan(){cudaFree(qp);cudaFree(kp);cudaFree(vt);cudaFree(scores);cudaFree(probabilities);cudaFree(padded_output);}
  void init(int first,int second,int batch,int count,int l,int d,int masking,const float* query,const float* key,const float* value,float* result){
    if(first<0 || first>4 || second<0 || second>4 || batch<1 || count<1 || batch>64 || count>64 || batch*count>64 || l<1 || l>2048 || d<1 || d>256 || masking<0 || masking>2 || !query || !key || !value || !result)
      throw std::runtime_error("Invalid bounded attention configuration");
    heads=batch*count;length=l;depth=d;sp=(l+127)/128*128;dp=(d+127)/128*128;mask=masking;q=query;k=key;v=value;output=result;
    if(size_t(heads)*sp*sp>(size_t(1)<<27))throw std::runtime_error("Attention materialization exceeds bounded buffer limit");
    size_t linear=size_t(heads)*sp*dp*sizeof(float),square=size_t(heads)*sp*sp*sizeof(float);
    check_cuda(cudaMalloc(&qp,linear));check_cuda(cudaMalloc(&kp,linear));check_cuda(cudaMalloc(&vt,linear));check_cuda(cudaMalloc(&padded_output,linear));
    check_cuda(cudaMalloc(&scores,square));check_cuda(cudaMalloc(&probabilities,square));
    qk.init(first,heads,sp,sp,dp,qp,kp,scores);pv.init(second,heads,sp,dp,sp,probabilities,vt,padded_output);
  }
  void prepare(cudaStream_t stream){attention_pad<<<(heads*sp*dp+255)/256,256,0,stream>>>(q,k,v,qp,kp,vt,heads,length,depth,sp,dp);qk.encode(3,stream);pv.encode(2,stream);check_cuda(cudaGetLastError());}
  void compute(int first,int second,cudaStream_t stream){
    qk.multiply(first,stream);attention_softmax<<<heads*sp,256,0,stream>>>(scores,probabilities,length,depth,sp,mask);pv.encode(1,stream);pv.multiply(second,stream);
    attention_crop<<<(heads*length*depth+255)/256,256,0,stream>>>(padded_output,output,heads,length,depth,sp,dp);check_cuda(cudaGetLastError());
  }
  size_t bytes(){return (size_t(heads)*sp*dp*4+size_t(heads)*sp*sp*2)*sizeof(float)+qk.bytes()+pv.bytes();}
};

extern "C" void* att_create(int a,int b,int batch,int heads,int length,int depth,int mask,const float* q,const float* k,const float* v,float* out){try{auto p=std::make_unique<AttentionPlan>();p->init(a,b,batch,heads,length,depth,mask,q,k,v,out);return p.release();}catch(const std::exception& e){last_error=e.what();return nullptr;}}
extern "C" void att_destroy(AttentionPlan* p){delete p;}
extern "C" int att_prepare(AttentionPlan* p,cudaStream_t stream){try{if(!p)throw std::runtime_error("Null attention plan");p->prepare(stream);return 0;}catch(const std::exception& e){last_error=e.what();return 1;}}
extern "C" int att_run(AttentionPlan* p,int a,int b,cudaStream_t stream){try{if(!p)throw std::runtime_error("Null attention plan");p->compute(a,b,stream);return 0;}catch(const std::exception& e){last_error=e.what();return 1;}}
extern "C" int att_stage(AttentionPlan* p,int stage,int algorithm,cudaStream_t stream){try{if(!p || stage<0 || stage>1)throw std::runtime_error("Invalid attention stage");(stage?p->pv:p->qk).multiply(algorithm,stream);return 0;}catch(const std::exception& e){last_error=e.what();return 1;}}
extern "C" StudyPlan* att_stage_plan(AttentionPlan* p,int stage){return !p || stage<0 || stage>1?nullptr:stage?&p->pv:&p->qk;}
extern "C" void* att_buffer(AttentionPlan* p,int index){if(!p)return nullptr;switch(index){case 0:return p->qp;case 1:return p->kp;case 2:return p->vt;case 3:return p->scores;case 4:return p->probabilities;case 5:return p->padded_output;default:return nullptr;}}
extern "C" size_t att_encoded_bytes(AttentionPlan* p,int stage,int index){if(!p || stage<0 || stage>1 || index<0 || index>3)return 0;auto& g=stage?p->pv:p->qk;return index<2?(index?g.b_bytes:g.a_bytes):g.mode==0?size_t(g.batches)*(index==2?g.m:g.n)*g.k/16:0;}
extern "C" size_t att_owned_bytes(AttentionPlan* p){return p?p->bytes():0;}
