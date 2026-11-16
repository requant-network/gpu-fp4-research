// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
#include "hardware_bridge.cu"

__device__ float decoded_fp4(const uint8_t* packed,const cutlass::float_ue4m3_t* scales,int i){
  constexpr float levels[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
  int code=(packed[i/2]>>((i%2)*4))&15;
  return (code&8?-levels[code&7]:levels[code&7])*float(scales[i/16]);
}

__global__ void column_maximum(const float* input,float* maxima,int rows,int k){
  __shared__ float values[256];
  int column=blockIdx.x;
  float maximum=0;
  for(int row=threadIdx.x;row<rows;row+=blockDim.x)maximum=fmaxf(maximum,fabsf(input[row*k+column]));
  values[threadIdx.x]=maximum;__syncthreads();
  for(int step=128;step;step/=2){if(threadIdx.x<step)values[threadIdx.x]=fmaxf(values[threadIdx.x],values[threadIdx.x+step]);__syncthreads();}
  if(!threadIdx.x)maxima[column]=values[0];
}

__global__ void select_columns(const float* a,const float* b,int* selected,int k,int padded){
  extern __shared__ unsigned storage[];
  float* scores=reinterpret_cast<float*>(storage);
  int* indices=reinterpret_cast<int*>(storage+padded);
  for(int i=threadIdx.x;i<padded;i+=blockDim.x){scores[i]=i<k?a[i]*b[i]:-1.f;indices[i]=i;}
  __syncthreads();
  for(int length=2;length<=padded;length*=2){
    for(int step=length/2;step;step/=2){
      for(int i=threadIdx.x;i<padded;i+=blockDim.x){
        int j=i^step;
        if(j>i){
          bool before=scores[i]>scores[j] || (scores[i]==scores[j] && indices[i]<indices[j]);
          bool descending=!(i&length);
          if(before!=descending){float value=scores[i];scores[i]=scores[j];scores[j]=value;int index=indices[i];indices[i]=indices[j];indices[j]=index;}
        }
      }
      __syncthreads();
    }
  }
  for(int i=threadIdx.x;i<k;i+=blockDim.x)selected[i]=indices[i];
}

__global__ void reduce_maximum(float* maxima,int k){
  __shared__ float values[256];
  float maximum=0;
  for(int i=threadIdx.x;i<k;i+=blockDim.x)maximum=fmaxf(maximum,maxima[i]);
  values[threadIdx.x]=maximum;__syncthreads();
  for(int step=128;step;step/=2){if(threadIdx.x<step)values[threadIdx.x]=fmaxf(values[threadIdx.x],values[threadIdx.x+step]);__syncthreads();}
  if(!threadIdx.x)maxima[0]=values[0];
}

__global__ void gather_columns(const float* input,const uint8_t* packed,const cutlass::float_ue4m3_t* scales,
                               const int* selected,__half* original,__half* quantized,int rows,int k,int budget){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=rows*budget)return;
  int source=(i/budget)*k+selected[i%budget];
  original[i]=__float2half_rn(input[source]);quantized[i]=__float2half_rn(decoded_fp4(packed,scales,source));
}

__global__ void residual_factors(const float* input,const uint8_t* packed,const cutlass::float_ue4m3_t* scales,
                                const float* maxima,__half* first,__half* second,uint8_t* mask,
                                int rows,int k,float fraction,bool operand_b,bool full){
  int group=blockIdx.x*blockDim.x+threadIdx.x;
  if(group>=rows*k/16)return;
  float global=0,block=0;
  if(!full){
    global=maxima[0];
    for(int j=0;j<16;++j)block=fmaxf(block,fabsf(input[group*16+j]));
  }
  bool selected=full || block>fraction*global;
  mask[group]=selected;
  for(int j=0;j<16;++j){
    int i=group*16+j;
    float q=decoded_fp4(packed,scales,i),residual=selected?input[i]-q:0.f;
    if(operand_b){first[i]=__float2half_rn(selected?input[i]:q);second[i]=__float2half_rn(residual);}
    else{first[i]=__float2half_rn(residual);second[i]=__float2half_rn(q);}
  }
}

__global__ void combine_correction(float* output,const float* first,const float* second,int count,bool subtract){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count)output[i]=(output[i]+first[i])+(subtract?-second[i]:second[i]);
}

struct CorrectionPlan{
  StudyPlan* base=nullptr;
  std::unique_ptr<StudyPlan> first,second;
  float *temporary1=nullptr,*temporary2=nullptr,*max_a=nullptr,*max_b=nullptr;
  int* selected=nullptr;
  uint8_t *mask_a=nullptr,*mask_b=nullptr;
  int method=0,budget=0,padded=0;
  float fraction=0;
  ~CorrectionPlan(){first.reset();second.reset();cudaFree(temporary1);cudaFree(temporary2);cudaFree(max_a);cudaFree(max_b);cudaFree(selected);cudaFree(mask_a);cudaFree(mask_b);}
  void initialize(StudyPlan* p,int kind,int columns,float threshold){
    if(!p || p->mode!=0 || kind<1 || kind>3 || p->k>2048 || columns<128 || columns>p->k || columns%128 ||
       !std::isfinite(threshold) || threshold<=0 || threshold>1)
      throw std::runtime_error("Correction requires FP4 plan, K<=2048, method1..3, aligned budget and fraction in (0,1]");
    base=p;method=kind;budget=kind==1?columns:p->k;fraction=threshold;padded=1;while(padded<p->k)padded*=2;
    check_cuda(cudaMalloc(&temporary1,size_t(p->m)*p->n*sizeof(float)));
    check_cuda(cudaMalloc(&temporary2,size_t(p->m)*p->n*sizeof(float)));
    check_cuda(cudaMalloc(&max_a,p->k*sizeof(float)));check_cuda(cudaMalloc(&max_b,p->k*sizeof(float)));
    check_cuda(cudaMalloc(&selected,p->k*sizeof(int)));
    check_cuda(cudaMalloc(&mask_a,size_t(p->m)*p->k/16));check_cuda(cudaMalloc(&mask_b,size_t(p->n)*p->k/16));
    first=std::make_unique<StudyPlan>();second=std::make_unique<StudyPlan>();
    first->initialize(2,p->m,p->n,budget,p->input_a,p->input_b,temporary1);
    second->initialize(2,p->m,p->n,budget,p->input_a,p->input_b,temporary2);
  }
  void prepare(int mask,cudaStream_t stream){
    if(mask<1 || mask>3)throw std::runtime_error("Correction operand mask must be 1..3");
    if(rq_quantize_native(base,mask,12,stream))throw std::runtime_error(last_error);
    if(method!=3){
      if(mask&1)column_maximum<<<base->k,256,0,stream>>>(base->input_a,max_a,base->m,base->k);
      if(mask&2)column_maximum<<<base->k,256,0,stream>>>(base->input_b,max_b,base->n,base->k);
    }
    if(method==1){
      select_columns<<<1,256,size_t(padded)*8,stream>>>(max_a,max_b,selected,base->k,padded);
      gather_columns<<<(base->m*budget+255)/256,256,0,stream>>>(base->input_a,(uint8_t*)base->a,base->flat_a,selected,(__half*)first->a,(__half*)second->a,base->m,base->k,budget);
      gather_columns<<<(base->n*budget+255)/256,256,0,stream>>>(base->input_b,(uint8_t*)base->b,base->flat_b,selected,(__half*)first->b,(__half*)second->b,base->n,base->k,budget);
    }else{
      if(method==2){
        if(mask&1)reduce_maximum<<<1,256,0,stream>>>(max_a,base->k);
        if(mask&2)reduce_maximum<<<1,256,0,stream>>>(max_b,base->k);
      }
      if(mask&1)residual_factors<<<(base->m*base->k/16+127)/128,128,0,stream>>>(base->input_a,(uint8_t*)base->a,base->flat_a,max_a,(__half*)first->a,(__half*)second->a,mask_a,base->m,base->k,fraction,false,method==3);
      if(mask&2)residual_factors<<<(base->n*base->k/16+127)/128,128,0,stream>>>(base->input_b,(uint8_t*)base->b,base->flat_b,max_b,(__half*)first->b,(__half*)second->b,mask_b,base->n,base->k,fraction,true,method==3);
    }
    check_cuda(cudaGetLastError());
  }
  void run(int a,int b,cudaStream_t stream){
    base->run(0,stream);first->run(a,stream);second->run(b,stream);
    combine_correction<<<(base->m*base->n+255)/256,256,0,stream>>>(base->output,temporary1,temporary2,base->m*base->n,method==1);
    check_cuda(cudaGetLastError());
  }
};
extern "C" void* rq_correction_create(StudyPlan* base,int method,int budget,float fraction){
  try{auto p=std::make_unique<CorrectionPlan>();p->initialize(base,method,budget,fraction);return p.release();}
  catch(const std::exception& e){last_error=e.what();return nullptr;}
}
extern "C" void rq_correction_destroy(CorrectionPlan* p){delete p;}
extern "C" int rq_correction_prepare(CorrectionPlan* p,int mask,cudaStream_t stream){
  try{if(!p)throw std::runtime_error("Null correction plan");p->prepare(mask,stream);return 0;}
  catch(const std::exception& e){last_error=e.what();return 1;}
}
extern "C" int rq_correction_run(CorrectionPlan* p,int a,int b,cudaStream_t stream){
  try{if(!p)throw std::runtime_error("Null correction plan");p->run(a,b,stream);return 0;}
  catch(const std::exception& e){last_error=e.what();return 1;}
}
extern "C" int rq_correction_candidates(CorrectionPlan* p,int which){
  return p && which>=0 && which<2 ? int((which?p->second:p->first)->algorithms.size()):0;
}
extern "C" int rq_correction_gemm(CorrectionPlan* p,int which,int candidate,cudaStream_t stream){
  try{if(!p || which<0 || which>1)throw std::runtime_error("Invalid correction GEMM");(which?p->second:p->first)->run(candidate,stream);return 0;}
  catch(const std::exception& e){last_error=e.what();return 1;}
}
extern "C" void* rq_correction_buffer(CorrectionPlan* p,int index){
  if(!p)return nullptr;
  switch(index){case 0:return p->first->a;case 1:return p->first->b;case 2:return p->second->a;case 3:return p->second->b;case 4:return p->selected;case 5:return p->mask_a;case 6:return p->mask_b;default:return nullptr;}
}
