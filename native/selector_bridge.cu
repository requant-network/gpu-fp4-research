// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
#include "optimized_bridge.cu"

__device__ double selector_value(const void* values,const cutlass::float_ue4m3_t* scales,int index,int mode){
  if(mode==0){
    unsigned raw=static_cast<const uint8_t*>(values)[index/2];raw=(raw>>(4*(index%2)))&15;
    const double levels[8]={0,.5,1,1.5,2,3,4,6};
    return (raw&8?-1.:1.)*levels[raw&7]*double(float(scales[index/16]));
  }
  if(mode==1)return double(float(static_cast<const cutlass::float_e4m3_t*>(values)[index]));
  if(mode==2)return double(__half2float(static_cast<const __half*>(values)[index]));
  if(mode==3)return double(__bfloat162float(static_cast<const __nv_bfloat16*>(values)[index]));
  return double(static_cast<const float*>(values)[index]);
}

__global__ void selector_samples(const float* x,const float* y,const void* a,const void* b,
    const cutlass::float_ue4m3_t* sa,const cutlass::float_ue4m3_t* sb,const int* indices,
    double* result,int k,int mode){
  int sample=blockIdx.x,row=indices[2*sample],column=indices[2*sample+1],lane=threadIdx.x;
  double original=0,quantized=0,absolute=0;__shared__ double sums[3][128];
  for(int q=lane;q<k;q+=128){
    double aa=selector_value(a,sa,row*k+q,mode),bb=selector_value(b,sb,column*k+q,mode);
    original+=double(x[row*k+q])*double(y[column*k+q]);quantized+=aa*bb;absolute+=fabs(aa*bb);
  }
  sums[0][lane]=original;sums[1][lane]=quantized;sums[2][lane]=absolute;__syncthreads();
  for(int step=64;step;step/=2){if(lane<step)for(int j=0;j<3;++j)sums[j][lane]+=sums[j][lane+step];__syncthreads();}
  if(!lane)for(int j=0;j<3;++j)result[3*sample+j]=sums[j][0];
}

__global__ void selector_promote(const float* source,double* result,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)result[i]=double(source[i]);}
extern "C" int rq_selector_probe(StudyPlan* p,const int* indices,int count,double* result,cudaStream_t stream){try{
  if(!p || !indices || !result || count<1 || count>1024)throw std::runtime_error("Invalid selector probe");
  selector_samples<<<count,128,0,stream>>>(p->input_a,p->input_b,p->a,p->b,p->flat_a,p->flat_b,indices,result,p->k,p->mode);
  check_cuda(cudaGetLastError());return 0;
}catch(const std::exception& e){last_error=e.what();return 1;}}
extern "C" int rq_selector_promote(StudyPlan* p,double* result,cudaStream_t stream){try{
  if(!p || !result)throw std::runtime_error("Invalid selector output");
  selector_promote<<<(p->m*p->n+255)/256,256,0,stream>>>(p->output,result,p->m*p->n);check_cuda(cudaGetLastError());return 0;
}catch(const std::exception& e){last_error=e.what();return 1;}}
