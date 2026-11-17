// Original changing-input grouping and post-weighting for affine GLL contractions.
extern "C" __global__ void pde_tensor_pack(const double* x,const int* indices,
                                           const unsigned char* boundary,float* right,
                                           int elements,int q,int columns) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=columns*128)return;
  const int column=index/128, feature=index%128;
  const int group=feature/q, reduced=feature%q;
  const int vector=group*columns+column;
  const int per_axis=elements*q*q;
  if(vector>=3*per_axis){right[index]=0.f;return;}
  const int axis=vector/per_axis, within=vector%per_axis;
  const int element=within/(q*q), a=(within/q)%q, b=within%q;
  int local;
  if(axis==0)local=(reduced*q+a)*q+b;
  else if(axis==1)local=(a*q+reduced)*q+b;
  else local=(a*q+b)*q+reduced;
  const int global=indices[element*q*q*q+local];
  right[index]=boundary[global]?0.f:float(x[global]);
}

extern "C" __global__ void pde_tensor_unweight(const float* product,const double* weights,
                                               double* local,int elements,int q,
                                               int columns,double half_h) {
  const int t=blockIdx.x*blockDim.x+threadIdx.x;
  const int volume=q*q*q;
  if(t>=elements*volume)return;
  const int element=t/volume,node=t%volume;
  const int i=node/(q*q),j=(node/q)%q,k=node%q;
  const int per_axis=elements*q*q;
  const int vectors[3]={element*q*q+j*q+k,
                        per_axis+element*q*q+i*q+k,
                        2*per_axis+element*q*q+i*q+j};
  const int output_index[3]={i,j,k};
  double values[3];
  #pragma unroll
  for(int axis=0;axis<3;++axis){
    const int group=vectors[axis]/columns,column=vectors[axis]%columns;
    values[axis]=double(product[(group*q+output_index[axis])*columns+column]);
  }
  local[t]=half_h*(values[0]*weights[j]*weights[k]
                  +values[1]*weights[i]*weights[k]
                  +values[2]*weights[i]*weights[j]);
}
