// Original matrix-free affine GLL operator kernels, not an assembled sparse matrix.
// Element-local contractions plus deterministic global contribution gathering.

template <typename T, int Q>
__global__ void pde_local(const double* __restrict__ x,
                          const int* __restrict__ indices,
                          const unsigned char* __restrict__ boundary,
                          const double* __restrict__ stiffness,
                          const double* __restrict__ weights,
                          T* __restrict__ element_output, double half_h) {
  extern __shared__ unsigned char storage[];
  T* values = reinterpret_cast<T*>(storage);
  constexpr int volume = Q*Q*Q;
  const int element = blockIdx.x;
  for (int t=threadIdx.x; t<volume; t+=blockDim.x) {
    const int global = indices[element*volume+t];
    values[t] = boundary[global] ? T(0) : T(x[global]);
  }
  __syncthreads();
  for (int t=threadIdx.x; t<volume; t+=blockDim.x) {
    const int i=t/(Q*Q), j=(t/Q)%Q, k=t%Q;
    T along_i=0, along_j=0, along_k=0;
    #pragma unroll
    for (int l=0; l<Q; ++l) {
      along_i += T(stiffness[i*Q+l])*values[(l*Q+j)*Q+k];
      along_j += T(stiffness[j*Q+l])*values[(i*Q+l)*Q+k];
      along_k += T(stiffness[k*Q+l])*values[(i*Q+j)*Q+l];
    }
    const T wi=T(weights[i]), wj=T(weights[j]), wk=T(weights[k]);
    element_output[element*volume+t] = T(half_h)*
      (along_i*wj*wk + along_j*wi*wk + along_k*wi*wj);
  }
}

template <typename T>
__global__ void pde_assemble(const T* __restrict__ element_output,
                             const double* __restrict__ x,
                             const double* __restrict__ mass,
                             const unsigned char* __restrict__ boundary,
                             double* __restrict__ output,
                             int n, int points, int elements, int q, double reaction) {
  const int global = blockIdx.x*blockDim.x+threadIdx.x;
  if (global>=n) return;
  // P A P + boundary identity: neither boundary columns nor rows leak into A_ii.
  if (boundary[global]) { output[global]=x[global]; return; }
  const int p=q-1, volume=q*q*q;
  const int coordinates[3]={global/(points*points),(global/points)%points,global%points};
  int first[3], count[3], local[3];
  #pragma unroll
  for (int axis=0; axis<3; ++axis) {
    const int e=coordinates[axis]/p, r=coordinates[axis]%p;
    first[axis]=e-(r==0);
    count[axis]=(r==0) ? 2 : 1;
    local[axis]=r;
  }
  T sum=0;
  for (int a=0; a<count[0]; ++a) {
    const int ei=first[0]+a, i=local[0] ? local[0] : (a ? 0 : p);
    for (int b=0; b<count[1]; ++b) {
      const int ej=first[1]+b, j=local[1] ? local[1] : (b ? 0 : p);
      for (int c=0; c<count[2]; ++c) {
        const int ek=first[2]+c, k=local[2] ? local[2] : (c ? 0 : p);
        const int element=(ei*elements+ej)*elements+ek;
        sum += element_output[element*volume+(i*q+j)*q+k];
      }
    }
  }
  sum += T(reaction)*T(mass[global])*T(x[global]);
  output[global]=double(sum);
}

extern "C" __global__ void pde_mask_boundary(const double* x,
                                             const unsigned char* boundary,
                                             double* masked, int n) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) masked[i]=boundary[i] ? 0. : x[i];
}

extern "C" __global__ void pde_restore_boundary(const double* x,
                                                const unsigned char* boundary,
                                                double* output, int n) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n && boundary[i]) output[i]=x[i];
}
