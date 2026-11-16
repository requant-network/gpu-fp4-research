// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

void cuda_check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
void solver_check(cusolverStatus_t status){if(status!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("cuSOLVER setup status "+std::to_string(int(status)));}
struct SolverProbe{
  cusolverDnHandle_t handle=nullptr;cusolverDnIRSParams_t parameters=nullptr;cusolverDnIRSInfos_t information=nullptr;
  double *a=nullptr,*b=nullptr,*x=nullptr;int* info=nullptr;void* workspace=nullptr;size_t workspace_bytes=0;
  ~SolverProbe(){cudaFree(a);cudaFree(b);cudaFree(x);cudaFree(info);cudaFree(workspace);if(information)cusolverDnIRSInfosDestroy(information);if(parameters)cusolverDnIRSParamsDestroy(parameters);if(handle)cusolverDnDestroy(handle);}
  void initialize(int n,int precision,int refinement,bool fallback){
    solver_check(cusolverDnCreate(&handle));solver_check(cusolverDnIRSParamsCreate(&parameters));solver_check(cusolverDnIRSInfosCreate(&information));
    solver_check(cusolverDnIRSParamsSetSolverPrecisions(parameters,CUSOLVER_R_64F,static_cast<cusolverPrecType_t>(precision)));
    solver_check(cusolverDnIRSParamsSetRefinementSolver(parameters,static_cast<cusolverIRSRefinement_t>(refinement)));
    solver_check(cusolverDnIRSParamsSetMaxIters(parameters,20));solver_check(cusolverDnIRSParamsSetMaxItersInner(parameters,20));
    solver_check(fallback?cusolverDnIRSParamsEnableFallback(parameters):cusolverDnIRSParamsDisableFallback(parameters));
    solver_check(cusolverDnIRSXgesv_bufferSize(handle,parameters,n,1,&workspace_bytes));
    cuda_check(cudaMalloc(&a,size_t(n)*n*sizeof(double)));cuda_check(cudaMalloc(&b,n*sizeof(double)));cuda_check(cudaMalloc(&x,n*sizeof(double)));cuda_check(cudaMalloc(&info,sizeof(int)));
    if(workspace_bytes)cuda_check(cudaMalloc(&workspace,workspace_bytes));
  }
};

int main(){try{
  cuda_check(cudaSetDevice(0));cudaDeviceProp properties;cuda_check(cudaGetDeviceProperties(&properties,0));if(properties.major!=12 || properties.minor!=0)throw std::runtime_error("SM120 GPU required");
  std::cout<<std::setprecision(17)<<"{\"n\":128,\"nrhs\":1,\"gpu_name\":\""<<properties.name<<"\",\"compute_capability\":[12,0],\"cases\":[";bool first=true;int n=128;
  for(int profile=0;profile<2;++profile)for(int precision:{1203,1204,1205,1206,1207})for(int refinement:{1101,1102,1104})for(bool fallback:{false,true}){
    if(!first)std::cout<<",";first=false;std::cout<<"{\"profile\":"<<profile<<",\"lowest_precision\":"<<precision<<",\"refinement\":"<<refinement<<",\"fallback_enabled\":"<<(fallback?"true":"false");
    try{
      SolverProbe probe;probe.initialize(n,precision,refinement,fallback);std::vector<double> a(n*n),b(n),x(n),truth(n);
      for(int j=0;j<n;++j)for(int i=0;i<n;++i)a[i+j*n]=1./1024.+(i==j?(profile?std::ldexp(1.,-(i/4)):1.):0.);
      for(int i=0;i<n;++i){truth[i]=i%2?-1.:1.;b[i]=(profile?std::ldexp(1.,-(i/4)):1.)*truth[i];}
      cuda_check(cudaMemcpy(probe.a,a.data(),a.size()*sizeof(double),cudaMemcpyHostToDevice));cuda_check(cudaMemcpy(probe.b,b.data(),b.size()*sizeof(double),cudaMemcpyHostToDevice));cuda_check(cudaMemset(probe.x,0,n*sizeof(double)));cuda_check(cudaMemset(probe.info,0xff,sizeof(int)));cuda_check(cudaDeviceSynchronize());
      int iterations=-9999,info=-9999;auto start=std::chrono::steady_clock::now();auto status=cusolverDnIRSXgesv(probe.handle,probe.parameters,probe.information,n,1,probe.a,n,probe.b,n,probe.x,n,probe.workspace,probe.workspace_bytes,&iterations,probe.info);cuda_check(cudaDeviceSynchronize());
      double milliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      cuda_check(cudaMemcpy(x.data(),probe.x,n*sizeof(double),cudaMemcpyDeviceToHost));cuda_check(cudaMemcpy(&info,probe.info,sizeof(int),cudaMemcpyDeviceToHost));
      double residual=0,error=0,denominator=0;bool finite=true;
      for(int i=0;i<n;++i){double value=-b[i];for(int j=0;j<n;++j)value+=a[i+j*n]*x[j];residual+=value*value;error+=(x[i]-truth[i])*(x[i]-truth[i]);denominator+=b[i]*b[i];finite=finite&&std::isfinite(x[i]);}
      std::cout<<",\"status\":"<<int(status)<<",\"device_info\":"<<info<<",\"iterations\":"<<iterations<<",\"workspace_bytes\":"<<probe.workspace_bytes<<",\"single_solve_wall_ms\":"<<milliseconds<<",\"finite\":"<<(finite?"true":"false")<<",\"relative_residual\":";
      if(finite && std::isfinite(residual))std::cout<<std::sqrt(residual/denominator);else std::cout<<"null";
      std::cout<<",\"forward_error\":";if(finite && std::isfinite(error))std::cout<<std::sqrt(error/n);else std::cout<<"null";
      std::cout<<",\"solution\":[";for(int i=0;i<n;++i){if(i)std::cout<<",";if(std::isfinite(x[i]))std::cout<<x[i];else std::cout<<"null";}std::cout<<"]";
    }catch(const std::exception& error){std::cout<<",\"setup_or_execution_error\":\""<<error.what()<<"\"";}
    std::cout<<"}";
  }
  std::cout<<"],\"complete\":true}\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}}
