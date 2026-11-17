// Original direct short-q packing: one physical directional vector per row.
extern "C" __global__ void pde_small_pack(const double*x,const int*indices,const unsigned char*boundary,
                                          float*right,int elements,int q,int pad){
  int index=blockIdx.x*blockDim.x+threadIdx.x;
  int per_axis=elements*q*q,total=3*per_axis;
  if(index>=total*pad)return;
  int vector=index/pad,reduced=index%pad;
  if(reduced>=q){right[index]=0.f;return;}
  int axis=vector/per_axis,within=vector%per_axis,element=within/(q*q),a=(within/q)%q,b=within%q;
  int local=axis==0?(reduced*q+a)*q+b:axis==1?(a*q+reduced)*q+b:(a*q+b)*q+reduced;
  int global=indices[element*q*q*q+local];right[index]=boundary[global]?0.f:float(x[global]);
}
extern "C" __global__ void pde_small_unweight(const float*product,const double*w,double*local,
                                              int elements,int q,double half_h){
  int t=blockIdx.x*blockDim.x+threadIdx.x,volume=q*q*q;if(t>=elements*volume)return;
  int e=t/volume,node=t%volume,i=node/(q*q),j=(node/q)%q,k=node%q;
  int per_axis=elements*q*q,columns=3*per_axis;
  double a=product[i*columns+e*q*q+j*q+k];
  double b=product[j*columns+per_axis+e*q*q+i*q+k];
  double c=product[k*columns+2*per_axis+e*q*q+i*q+j];
  local[t]=half_h*(a*w[j]*w[k]+b*w[i]*w[k]+c*w[i]*w[j]);
}
