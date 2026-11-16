// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static thread_local std::string error_message, metadata;
static void ck(cudaError_t code) { if(code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
static void sk(cusolverStatus_t code) { if(code) throw std::runtime_error("cuSOLVER setup status " + std::to_string(code)); }
static void bk(cublasStatus_t code) { if(code) throw std::runtime_error("cuBLAS status " + std::to_string(code)); }
template<class T> static void alloc(T*& pointer, size_t elements) { ck(cudaMalloc(&pointer, elements*sizeof(T))); }

__global__ void diagonal_scale(int n, const double* a, double* scale, int* invalid) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) { double diagonal=a[i+size_t(i)*n];
    if(!(diagonal>0) || !isfinite(diagonal)) { atomicExch(invalid,1); scale[i]=1; }
    else scale[i]=1./sqrt(diagonal);
  }
}
__global__ void scale_inputs(int n, int rhs, const double* a, const double* b,
                             const double* scale, double* aw, double* bw) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<size_t(n)*n) aw[i]=a[i]*scale[i%n]*scale[i/n];
  if(i<size_t(n)*rhs) bw[i]=b[i]*scale[i%n];
}
__global__ void unscale_output(int n, int rhs, const double* scale, double* x) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<size_t(n)*rhs) x[i]*=scale[i%n];
}
__global__ void residual_norms(int n, const double* r, const double* b, double* result) {
  __shared__ double rs[256], bs[256];
  int column=blockIdx.x, t=threadIdx.x;
  double rsum=0, bsum=0;
  for(int i=t;i<n;i+=blockDim.x) {
    double rv=r[i+size_t(column)*n], bv=b[i+size_t(column)*n];
    rsum+=isfinite(rv)?rv*rv:CUDART_INF; bsum+=bv*bv;
  }
  rs[t]=rsum;bs[t]=bsum;__syncthreads();
  for(int stride=128;stride;stride/=2) { if(t<stride) {rs[t]+=rs[t+stride];bs[t]+=bs[t+stride];} __syncthreads(); }
  if(!t) result[column]=(bs[0]>0 && isfinite(bs[0]))?sqrt(rs[0]/bs[0]):CUDART_INF;
}

struct IRS {
  cusolverDnIRSParams_t params=nullptr;
  cusolverDnIRSInfos_t infos=nullptr;
  void* workspace=nullptr;
  size_t bytes=0;
  int precision, refinement, maximum;
  ~IRS() { cudaFree(workspace);if(infos)cusolverDnIRSInfosDestroy(infos);if(params)cusolverDnIRSParamsDestroy(params); }
  void setup(cusolverDnHandle_t handle,int n,int rhs,int p,int refine,bool fallback,int limit) {
    precision=p;refinement=refine;maximum=limit;
    sk(cusolverDnIRSParamsCreate(&params));sk(cusolverDnIRSInfosCreate(&infos));
    sk(cusolverDnIRSParamsSetSolverPrecisions(params,CUSOLVER_R_64F,static_cast<cusolverPrecType_t>(p)));
    sk(cusolverDnIRSParamsSetRefinementSolver(params,static_cast<cusolverIRSRefinement_t>(refine)));
    sk(cusolverDnIRSParamsSetMaxIters(params,limit));sk(cusolverDnIRSParamsSetMaxItersInner(params,limit));
    sk(fallback?cusolverDnIRSParamsEnableFallback(params):cusolverDnIRSParamsDisableFallback(params));
    sk(cusolverDnIRSXgesv_bufferSize(handle,params,n,rhs,&bytes));
    if(bytes) ck(cudaMalloc(&workspace,bytes));
  }
};
struct Stage { int precision,status,info,iterations,outer=-9999;double residual;bool accepted; };
struct Plan {
  int n,rhs,mode;bool equilibrate,loaded_a=false,loaded_b=false,factor_ready=false;
  cudaStream_t stream=nullptr;cudaEvent_t start=nullptr,end=nullptr;
  cusolverDnHandle_t solver=nullptr;cublasHandle_t blas=nullptr;
  double *a=nullptr,*b=nullptr,*aw=nullptr,*bw=nullptr,*x=nullptr,*r=nullptr,*scale=nullptr,*norms=nullptr,*cholwork=nullptr;
  int* info=nullptr;int cholsize=0;size_t owned=0;
  std::vector<std::unique_ptr<IRS>> irs;
  std::vector<Stage> stages;
  float gpu_ms=0;double wall_ms=0;
  ~Plan() {
    if(stream)cudaStreamSynchronize(stream);
    irs.clear();
    for(double* p:{a,b,aw,bw,x,r,scale,norms,cholwork})cudaFree(p);
    cudaFree(info);if(blas)cublasDestroy(blas);if(solver)cusolverDnDestroy(solver);
    if(start)cudaEventDestroy(start);if(end)cudaEventDestroy(end);if(stream)cudaStreamDestroy(stream);
  }
  void initialize(int dimension,int columns,int kind,int precision,int refine,bool fallback,int maximum,bool eq) {
    if(dimension<1 || dimension>4096 || (columns!=1 && columns!=4) || kind<0 || kind>3 || maximum<1 || maximum>100)
      throw std::runtime_error("Invalid dimensions, mode or iteration limit");
    if(refine==1104 && columns!=1)throw std::runtime_error("GMRES requires one RHS in this protocol");
    if(eq && kind!=0 && kind!=2)throw std::runtime_error("Equilibration only defined for IRS/adaptive modes");
    n=dimension;rhs=columns;mode=kind;equilibrate=eq;
    ck(cudaSetDevice(0));cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));
    if(prop.major!=12 || prop.minor!=0)throw std::runtime_error("SM120 GPU required");
    ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
    sk(cusolverDnCreate(&solver));sk(cusolverDnSetStream(solver,stream));
    bk(cublasCreate(&blas));bk(cublasSetStream(blas,stream));bk(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH));
    alloc(a,size_t(n)*n);alloc(aw,size_t(n)*n);
    alloc(b,size_t(n)*rhs);alloc(bw,size_t(n)*rhs);alloc(x,size_t(n)*rhs);alloc(r,size_t(n)*rhs);
    alloc(scale,n);alloc(norms,rhs);alloc(info,1);
    sk(cusolverDnDpotrf_bufferSize(solver,CUBLAS_FILL_MODE_LOWER,n,aw,n,&cholsize));
    alloc(cholwork,cholsize);
    owned=(2*size_t(n)*n+4*size_t(n)*rhs+n+rhs+cholsize)*sizeof(double)+sizeof(int);
    if(mode==0 || mode==2) {
      int count=mode==2?2:1;
      for(int i=0;i<count;++i) {
        auto item=std::make_unique<IRS>();
        item->setup(solver,n,rhs,mode==2?(i==0?1205:1204):precision,
                    mode==2?1102:refine,mode==2?false:fallback,mode==2?(i==0?4:12):maximum);
        owned+=item->bytes;irs.push_back(std::move(item));
      }
    }
    ck(cudaStreamSynchronize(stream));
  }
  void load(const double* host_a,const double* host_b) {
    if(host_a) {ck(cudaMemcpyAsync(a,host_a,size_t(n)*n*sizeof(double),cudaMemcpyHostToDevice,stream));loaded_a=true;factor_ready=false;}
    if(host_b) {ck(cudaMemcpyAsync(b,host_b,size_t(n)*rhs*sizeof(double),cudaMemcpyHostToDevice,stream));loaded_b=true;}
    ck(cudaStreamSynchronize(stream));
  }
  void restore(bool use_scale) {
    ck(cudaMemsetAsync(x,0,size_t(n)*rhs*sizeof(double),stream));
    if(use_scale) {
      ck(cudaMemsetAsync(info,0,sizeof(int),stream));
      diagonal_scale<<<(n+255)/256,256,0,stream>>>(n,a,scale,info);
      int invalid=0;ck(cudaMemcpyAsync(&invalid,info,sizeof(int),cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
      if(invalid)throw std::runtime_error("Jacobi equilibration requires a finite positive diagonal");
      scale_inputs<<<(size_t(n)*n+255)/256,256,0,stream>>>(n,rhs,a,b,scale,aw,bw);
    } else {
      ck(cudaMemcpyAsync(aw,a,size_t(n)*n*sizeof(double),cudaMemcpyDeviceToDevice,stream));
      ck(cudaMemcpyAsync(bw,b,size_t(n)*rhs*sizeof(double),cudaMemcpyDeviceToDevice,stream));
    }
    ck(cudaMemsetAsync(info,0xff,sizeof(int),stream));ck(cudaGetLastError());
  }
  double residual() {
    ck(cudaMemcpyAsync(r,b,size_t(n)*rhs*sizeof(double),cudaMemcpyDeviceToDevice,stream));
    double one=1,minus=-1;
    bk(cublasDgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,n,rhs,n,&one,a,n,x,n,&minus,r,n));
    residual_norms<<<rhs,256,0,stream>>>(n,r,b,norms);ck(cudaGetLastError());
    std::vector<double> values(rhs);
    ck(cudaMemcpyAsync(values.data(),norms,rhs*sizeof(double),cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
    double result=0;
    for(double value:values) {if(!std::isfinite(value))return INFINITY;result=std::max(result,value);}
    return result;
  }
  Stage cholesky(bool reuse) {
    int status=0,device_info=-9999;
    if(!reuse) {
      restore(false);
      status=int(cusolverDnDpotrf(solver,CUBLAS_FILL_MODE_LOWER,n,aw,n,cholwork,cholsize,info));
      ck(cudaMemcpyAsync(&device_info,info,sizeof(int),cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
      factor_ready=status==0 && device_info==0;
    } else if(!factor_ready)throw std::runtime_error("Factor reuse requires successful sp_factor after latest A load");
    if(factor_ready) {
      ck(cudaMemcpyAsync(x,b,size_t(n)*rhs*sizeof(double),cudaMemcpyDeviceToDevice,stream));
      status=int(cusolverDnDpotrs(solver,CUBLAS_FILL_MODE_LOWER,n,rhs,aw,n,x,n,info));
      ck(cudaMemcpyAsync(&device_info,info,sizeof(int),cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
    } else ck(cudaMemsetAsync(x,0,size_t(n)*rhs*sizeof(double),stream));
    double value=residual();
    return {1203,status,device_info,0,-9999,value,status==0 && device_info==0 && value<=1e-12};
  }
  void timer_start() {ck(cudaStreamSynchronize(stream));ck(cudaEventRecord(start,stream));}
  void timer_end(std::chrono::steady_clock::time_point begin) {
    ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));ck(cudaEventElapsedTime(&gpu_ms,start,end));
    wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
  }
  void factor() {
    if(mode!=3 || !loaded_a || !loaded_b)throw std::runtime_error("Factor setup requires reuse mode and loaded A/B");
    stages.clear();auto begin=std::chrono::steady_clock::now();timer_start();restore(false);
    int status=int(cusolverDnDpotrf(solver,CUBLAS_FILL_MODE_LOWER,n,aw,n,cholwork,cholsize,info)),device_info=-9999;
    ck(cudaMemcpyAsync(&device_info,info,sizeof(int),cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
    factor_ready=status==0 && device_info==0;
    stages.push_back({1203,status,device_info,0,-9999,INFINITY,false});timer_end(begin);
  }
  void run() {
    if(!loaded_a || !loaded_b)throw std::runtime_error("Load A and B before solving");
    stages.clear();auto begin=std::chrono::steady_clock::now();timer_start();
    if(mode==1 || mode==3)stages.push_back(cholesky(mode==3));
    else {
      for(auto& item:irs) {
        restore(equilibrate);int iterations=-9999,device_info=-9999;
        // Infos describes one solve; reusing it produced ALLOC_FAILED on call two.
        sk(cusolverDnIRSInfosDestroy(item->infos));item->infos=nullptr;
        sk(cusolverDnIRSInfosCreate(&item->infos));
        if(item->refinement!=1101)sk(cusolverDnIRSInfosRequestResidual(item->infos));
        int status=int(cusolverDnIRSXgesv(solver,item->params,item->infos,n,rhs,aw,n,bw,n,x,n,
                                         item->workspace,item->bytes,&iterations,info));
        if(equilibrate)unscale_output<<<(size_t(n)*rhs+255)/256,256,0,stream>>>(n,rhs,scale,x);
        ck(cudaGetLastError());ck(cudaMemcpyAsync(&device_info,info,sizeof(int),cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
        double value=residual();
        bool accepted=status==0 && device_info==0 && value<=1e-12;
        stages.push_back({item->precision,status,device_info,iterations,-9999,value,accepted});
        if(mode==0 || accepted)break;
      }
      if(mode==2 && !stages.back().accepted)stages.push_back(cholesky(false));
    }
    timer_end(begin);
  }
  std::string describe() {
    std::ostringstream out;out<<std::setprecision(17)<<"{\"cuda_ms\":"<<gpu_ms<<",\"wall_ms\":"<<wall_ms
      <<",\"owned_device_bytes\":"<<owned<<",\"cholesky_workspace_bytes\":"<<size_t(cholsize)*8
      <<",\"factor_ready\":"<<(factor_ready?"true":"false")<<",\"stages\":[";
    for(size_t i=0;i<stages.size();++i) {
      auto& stage=stages[i];if(i)out<<",";
      out<<"{\"requested_precision\":"<<stage.precision<<",\"status\":"<<stage.status<<",\"device_info\":"<<stage.info
         <<",\"iterations\":"<<stage.iterations<<",\"gate_pass\":"<<(stage.accepted?"true":"false")<<",\"max_relative_residual\":";
      if(std::isfinite(stage.residual))out<<stage.residual;else out<<"null";
      out<<",\"outer_iterations\":null,\"history\":null,\"history_status\":null";
      // Only valid successful nonnegative iteration histories have a defined prefix.
      if(i<irs.size() && irs[i]->refinement!=1101 && stage.status==0 && stage.info==0 && stage.iterations>0) {
        auto& item=irs[i];int outer=-9999,maximum=-9999;void* pointer=nullptr;
        int s1=cusolverDnIRSInfosGetOuterNiters(item->infos,&outer);
        int s2=cusolverDnIRSInfosGetMaxIters(item->infos,&maximum);
        int s3=cusolverDnIRSInfosGetResidualHistory(item->infos,&pointer);
        out<<",\"history_api_status\":["<<s1<<","<<s2<<","<<s3<<"],\"reported_outer_iterations\":"<<outer
           <<",\"reported_max_iterations\":"<<maximum;
        if(s1==0 && s2==0 && s3==0 && pointer && outer>=0 && outer<=maximum && maximum<=100) {
          std::vector<double> values(size_t(maximum+1)*2);
          ck(cudaMemcpy(values.data(),pointer,values.size()*sizeof(double),cudaMemcpyDefault));
          out<<",\"valid_history\":[";
          for(int j=0;j<=outer;++j) {
            if(j)out<<",";out<<"[";
            if(std::isfinite(values[j]))out<<values[j];else out<<"null";out<<",";
            if(std::isfinite(values[maximum+1+j]))out<<values[maximum+1+j];else out<<"null";out<<"]";
          }
          out<<"]";
        }
      }
      out<<"}";
    }
    out<<"]}";return out.str();
  }
};

extern "C" {
const char* sp_error() { return error_message.c_str(); }
const char* sp_environment() {
  try {ck(cudaSetDevice(0));cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));
    int driver=0,runtime=0,major=0,minor=0,patch=0,blas_version=0;
    ck(cudaDriverGetVersion(&driver));ck(cudaRuntimeGetVersion(&runtime));
    sk(cusolverGetProperty(MAJOR_VERSION,&major));sk(cusolverGetProperty(MINOR_VERSION,&minor));sk(cusolverGetProperty(PATCH_LEVEL,&patch));
    cublasHandle_t handle=nullptr;bk(cublasCreate(&handle));bk(cublasGetVersion(handle,&blas_version));bk(cublasDestroy(handle));
    std::ostringstream out;out<<"{\"gpu_name\":\""<<prop.name<<"\",\"compute_capability\":["<<prop.major<<","<<prop.minor
      <<"],\"multiprocessors\":"<<prop.multiProcessorCount<<",\"driver_api_version\":"<<driver
      <<",\"runtime_api_version\":"<<runtime<<",\"cusolver_version\":["<<major<<","<<minor<<","<<patch
      <<"],\"cublas_version\":"<<blas_version<<"}";metadata=out.str();return metadata.c_str();}
  catch(const std::exception& e){error_message=e.what();return nullptr;}
}
void* sp_new(int n,int rhs,int mode,int precision,int refinement,int fallback,int maximum,int equilibrate) {
  try {auto plan=std::make_unique<Plan>();plan->initialize(n,rhs,mode,precision,refinement,fallback,maximum,equilibrate);return plan.release();}
  catch(const std::exception& e){error_message=e.what();return nullptr;}
}
void sp_delete(void* p) {delete static_cast<Plan*>(p);}
int sp_load(void* p,const double* a,const double* b) {
  try {if(!p)throw std::runtime_error("Null plan");static_cast<Plan*>(p)->load(a,b);return 0;}
  catch(const std::exception& e){error_message=e.what();return -1;}
}
int sp_run(void* p) {
  try {if(!p)throw std::runtime_error("Null plan");static_cast<Plan*>(p)->run();return 0;}
  catch(const std::exception& e){error_message=e.what();return -1;}
}
int sp_factor(void* p) {
  try {if(!p)throw std::runtime_error("Null plan");static_cast<Plan*>(p)->factor();return 0;}
  catch(const std::exception& e){error_message=e.what();return -1;}
}
int sp_output(void* p,double* host) {
  try {if(!p || !host)throw std::runtime_error("Null output/plan");auto& q=*static_cast<Plan*>(p);
    ck(cudaMemcpyAsync(host,q.x,size_t(q.n)*q.rhs*sizeof(double),cudaMemcpyDeviceToHost,q.stream));ck(cudaStreamSynchronize(q.stream));return 0;}
  catch(const std::exception& e){error_message=e.what();return -1;}
}
int sp_prepared(void* p,double* host_a,double* host_b,double* host_scale) {
  try {if(!p || !host_a || !host_b || !host_scale)throw std::runtime_error("Null preparation output/plan");
    auto& q=*static_cast<Plan*>(p);if(!q.loaded_a || !q.loaded_b)throw std::runtime_error("Load A/B first");
    q.restore(q.equilibrate);q.factor_ready=false;
    ck(cudaMemcpyAsync(host_a,q.aw,size_t(q.n)*q.n*sizeof(double),cudaMemcpyDeviceToHost,q.stream));
    ck(cudaMemcpyAsync(host_b,q.bw,size_t(q.n)*q.rhs*sizeof(double),cudaMemcpyDeviceToHost,q.stream));
    if(q.equilibrate)ck(cudaMemcpyAsync(host_scale,q.scale,q.n*sizeof(double),cudaMemcpyDeviceToHost,q.stream));
    else std::fill(host_scale,host_scale+q.n,1.);
    ck(cudaStreamSynchronize(q.stream));return 0;}
  catch(const std::exception& e){error_message=e.what();return -1;}
}
const char* sp_metadata(void* p) {
  try {if(!p)throw std::runtime_error("Null plan");metadata=static_cast<Plan*>(p)->describe();return metadata.c_str();}
  catch(const std::exception& e){error_message=e.what();return nullptr;}
}
}
