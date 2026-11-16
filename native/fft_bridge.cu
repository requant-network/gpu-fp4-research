// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Requant Network
// Reuse the already validated cuBLASLt setup/tuning helper without modifying it.
#include "spmm_bridge.cu"
#include "fft_dx.cuh"
#include <cufftXt.h>
#include <map>

static thread_local std::string fft_error_text,fft_json_text;
static void fft_check(cufftResult s,const char* where){if(s)throw std::runtime_error(std::string(where)+": cuFFT status "+std::to_string(int(s)));}
template<class T> struct FftPair{T x,y;};
template<class T> __global__ void fft_prepare_input(size_t count,int n,const float2* input,const float* window,FftPair<T>* out){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  float2 v=input[i];float w=window[i%n];out[i]={cast<T>(__fmul_rn(v.x,w)),cast<T>(__fmul_rn(v.y,w))};
}
template<> __global__ void fft_prepare_input<double>(size_t count,int n,const float2* input,const float* window,FftPair<double>* out){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count){float2 v=input[i];double w=window[i%n];out[i]={double(v.x)*w,double(v.y)*w};}
}
template<class T> __global__ void fft_filter_values(size_t count,int n,const FftPair<T>* input,const float2* response,FftPair<T>* output){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  float x=float(input[i].x),y=float(input[i].y);float2 h=response[i%n];
  output[i]={cast<T>(__fsub_rn(__fmul_rn(x,h.x),__fmul_rn(y,h.y))),cast<T>(__fadd_rn(__fmul_rn(x,h.y),__fmul_rn(y,h.x)))};
}
template<> __global__ void fft_filter_values<double>(size_t count,int n,const FftPair<double>* input,const float2* response,FftPair<double>* output){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count){double x=input[i].x,y=input[i].y;float2 h=response[i%n];output[i]={x*double(h.x)-y*double(h.y),x*double(h.y)+y*double(h.x)};}
}
template<class T> __global__ void fft_normalize(size_t count,int n,const FftPair<T>* input,float2* output){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)output[i]={float(input[i].x)/n,float(input[i].y)/n};
}
template<> __global__ void fft_normalize<double>(size_t count,int n,const FftPair<double>* input,float2* output){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)output[i]={float(input[i].x/n),float(input[i].y/n)};
}
__global__ void fft_reverse(size_t count,int n,const int* permutation,const float2* input,const float* window,bool apply_window,float2* output){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;int index=permutation[i%n];
  float2 value=input[(i/n)*n+index];if(apply_window){float w=window[index];value={__fmul_rn(value.x,w),__fmul_rn(value.y,w)};}output[i]=value;
}
template<class T> __global__ void fft_encode_basis(size_t count,const float* input,T* output){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)output[i]=cast<T>(input[i]);
}
static __device__ float2 fft_rotate(float2 v,float2 w,bool inverse){
  if(inverse)w.y=-w.y;
  return {__fsub_rn(__fmul_rn(v.x,w.x),__fmul_rn(v.y,w.y)),__fadd_rn(__fmul_rn(v.x,w.y),__fmul_rn(v.y,w.x))};
}
template<class T> __global__ void fft_pack_direct(int n,int batch,int radix,int tile,int columns,int padded,int previous,
                                                const float2* input,const float2* phase,bool inverse,T* out){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(tile)*padded)return;
  int row=i%tile,column=i/tile;float value=0;
  if(column<columns && row<2*radix){int t=row%radix,j=column%previous,g=(column/(previous))%(n/(radix*previous)),b=column/(n/radix);
    float2 v=input[size_t(b)*n+g*radix*previous+j+t*previous];v=fft_rotate(v,phase[t*previous+j],inverse);value=row<radix?v.x:v.y;}
  out[i]=cast<T>(value);
}
template<class T> __global__ void fft_pack_tiled(int n,int batch,int radix,int tile,int columns,int padded,int previous,
                                               const float2* input,const float2* phase,bool inverse,T* out){
  __shared__ float real[16][33],imag[16][33];
  int column=blockIdx.x*32+threadIdx.x;
  for(int t=threadIdx.y;t<16;t+=8){float2 v={0,0};
    if(t<radix && column<columns){int j=column%previous,g=(column/previous)%(n/(radix*previous)),b=column/(n/radix);
      v=fft_rotate(input[size_t(b)*n+g*radix*previous+j+t*previous],phase[t*previous+j],inverse);}
    real[t][threadIdx.x]=v.x;imag[t][threadIdx.x]=v.y;}
  __syncthreads();
  int row=threadIdx.x;
  for(int offset=threadIdx.y;offset<32;offset+=8){int c=blockIdx.x*32+offset;
    if(row<tile && c<padded){float value=0;if(c<columns && row<radix)value=real[row][offset];
      if(c<columns && row>=radix && row<2*radix)value=imag[row-radix][offset];out[size_t(c)*tile+row]=cast<T>(value);}}
}
__global__ void fft_unpack(size_t count,int n,int radix,int previous,int tile,const float* product,float2* out){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  int index=i%n,g=index/(radix*previous),q=(index%(radix*previous))/previous,j=index%previous;
  size_t column=(i/n)*(n/radix)+g*previous+j;out[i]={product[column*tile+q]/radix,product[column*tile+radix+q]/radix};
}
__global__ void fft_rescale(size_t count,int n,const float2* input,float2* output){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count){float2 v=input[i];output[i]={v.x*n,v.y*n};}
}

struct TensorFftGeometry{int radix,previous,tile,columns,padded,phase_offset,basis_offset;};
struct FftPlan {
  int n,batch,backend,precision,maximum,dx_pack,phase_count,basis_count;
  size_t total,owned=0,element=8,workspace_bytes=0,max_pack=0;
  float2 *input=nullptr,*output=nullptr,*response=nullptr,*phases=nullptr,*sequence[2]={},*filtered=nullptr,*current=nullptr,*previous_sequence=nullptr;
  float *window=nullptr,*basis=nullptr,*product=nullptr;int* permutation=nullptr;
  void *encoded_basis=nullptr,*packed=nullptr,*diag_spectrum=nullptr,*diag_filtered=nullptr,*cu_input=nullptr,*cu_spectrum=nullptr,*cu_output=nullptr,*workspace=nullptr;
  std::vector<void*> allocations;std::vector<TensorFftGeometry> stages;std::map<int,std::unique_ptr<Lt>> gemms;
  cudaStream_t stream=nullptr;cudaEvent_t begin=nullptr,end=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t execution=nullptr;int repetitions=0;
  cufftHandle cufft=0;int cufft_version=0;FftDxInfo dx;
  int dx_registers=0,dx_diagnostic_registers=0,dx_blocks_per_sm=0,last_stage=-1;bool last_inverse=false,diagnostics_ready=false;
  double last_ms=0,last_wall_ms=0;
  ~FftPlan(){if(stream)cudaStreamSynchronize(stream);if(execution)cudaGraphExecDestroy(execution);if(graph)cudaGraphDestroy(graph);
    gemms.clear();if(cufft)cufftDestroy(cufft);for(void* p:allocations)cudaFree(p);
    if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);if(stream)cudaStreamDestroy(stream);}
  template<class T> T* allocate(size_t count){T* p=nullptr;size_t bytes=std::max(size_t(1),count)*sizeof(T);ck(cudaMalloc(&p,bytes));owned+=bytes;allocations.push_back(p);return p;}
  template<class T> T* upload(const T* host,size_t count){auto p=allocate<T>(count);if(count)ck(cudaMemcpyAsync(p,host,count*sizeof(T),cudaMemcpyHostToDevice,stream));return p;}
  void initialize(int length,int transforms,int mode,int p,int radix,int pack,int phase_size,int basis_size,
                  const float2* source,const float* win,const float2* h,const int* perm,const float2* twiddle,const float* weights){
    if(length<2 || length>65536 || transforms<1 || transforms>64 || mode<0 || mode>2 || p<0 || p>(mode==0?3:2) || !source || !win || !h)throw std::runtime_error("Invalid FFT pipeline arguments");
    n=length;batch=transforms;backend=mode;precision=p;maximum=radix;dx_pack=pack;phase_count=phase_size;basis_count=basis_size;total=size_t(n)*batch;
    if(mode==1 && ((n&(n-1)) || (radix!=8 && radix!=16)))throw std::runtime_error("Tensor matrix FFT is implemented only for power-of-two lengths with radix 8 or 16");
    if(mode==2 && p!=0)throw std::runtime_error("The fused cuFFTDx reference uses FP32");
    for(int i=0;i<n;++i)if(!std::isfinite(win[i]) || !std::isfinite(h[i].x) || !std::isfinite(h[i].y))throw std::runtime_error("Nonfinite window/filter parameter");
    ck(cudaSetDevice(0));cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));if(prop.major!=12 || prop.minor!=0)throw std::runtime_error("SM120 GPU required");
    ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
    fft_check(cufftGetVersion(&cufft_version),"version");
    input=upload(source,total);window=upload(win,n);response=upload(h,n);output=allocate<float2>(total);
    if(mode==0){element=p==1?16:(p==0?8:4);auto type=p==0?CUDA_C_32F:(p==1?CUDA_C_64F:(p==2?CUDA_C_16F:CUDA_C_16BF));
      fft_check(cufftCreate(&cufft),"create");fft_check(cufftSetAutoAllocation(cufft,0),"manual workspace");long long dimensions=n;
      fft_check(cufftXtMakePlanMany(cufft,1,&dimensions,&dimensions,1,n,type,&dimensions,1,n,type,batch,&workspace_bytes,type),"makePlan");
      if(workspace_bytes){workspace=allocate<unsigned char>(workspace_bytes);fft_check(cufftSetWorkArea(cufft,workspace),"workspace");}
      fft_check(cufftSetStream(cufft,stream),"stream");cu_input=allocate<unsigned char>(total*element);cu_spectrum=allocate<unsigned char>(total*element);cu_output=allocate<unsigned char>(total*element);
    }else if(mode==2){dx=fft_dx_info(n,dx_pack);cudaFuncAttributes a{},b{};
      ck(cudaFuncSetAttribute(dx.normal,cudaFuncAttributeMaxDynamicSharedMemorySize,int(dx.shared)));ck(cudaFuncSetAttribute(dx.diagnostic,cudaFuncAttributeMaxDynamicSharedMemorySize,int(dx.shared)));
      ck(cudaFuncGetAttributes(&a,dx.normal));ck(cudaFuncGetAttributes(&b,dx.diagnostic));dx_registers=a.numRegs;dx_diagnostic_registers=b.numRegs;
      ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&dx_blocks_per_sm,dx.normal,dx.block.x*dx.block.y*dx.block.z,dx.shared));
      int diagnostic_blocks=0;ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&diagnostic_blocks,dx.diagnostic,dx.block.x*dx.block.y*dx.block.z,dx.shared));
      if(dx_blocks_per_sm==0 || diagnostic_blocks==0)throw std::runtime_error("cuFFTDx descriptor exceeds per-block resource limits");
    }else{
      if(!perm || !twiddle || !weights || basis_count!=3584)throw std::runtime_error("Missing Tensor FFT coefficient arrays");
      std::vector<bool> seen(n,false);for(int i=0;i<n;++i){if(perm[i]<0 || perm[i]>=n || seen[perm[i]])throw std::runtime_error("Invalid mixed-radix permutation");seen[perm[i]]=true;}
      int previous=1,remaining=n,offset=0;
      while(remaining>1){int r=std::min(maximum,remaining),tile=std::max(16,2*r),columns=int(total/r),padded=(columns+15)/16*16;
        int basis_offset=r==2?0:(r==4?256:(r==8?512:768));stages.push_back({r,previous,tile,columns,padded,offset,basis_offset});
        max_pack=std::max(max_pack,size_t(tile)*padded);previous*=r;remaining/=r;offset+=previous;}
      if(offset!=phase_count)throw std::runtime_error("Twiddle table extent does not match the factorization");
      permutation=upload(perm,n);phases=upload(twiddle,phase_count);basis=upload(weights,basis_count);
      sequence[0]=allocate<float2>(total);sequence[1]=allocate<float2>(total);filtered=allocate<float2>(total);
      size_t scalar=precision==0?4:2;packed=allocate<unsigned char>(max_pack*scalar);product=allocate<float>(max_pack);
      ck(cudaMemsetAsync(packed,0,max_pack*scalar,stream));ck(cudaMemsetAsync(product,0,max_pack*4,stream));
      if(precision==0)encoded_basis=basis;else{encoded_basis=allocate<unsigned char>(basis_count*scalar);
        if(precision==1)fft_encode_basis<<<(basis_count+255)/256,256,0,stream>>>(basis_count,basis,static_cast<__half*>(encoded_basis));
        else fft_encode_basis<<<(basis_count+255)/256,256,0,stream>>>(basis_count,basis,static_cast<__nv_bfloat16*>(encoded_basis));ck(cudaGetLastError());}
      workspace_bytes=64ull<<20;workspace=allocate<unsigned char>(workspace_bytes);ck(cudaStreamSynchronize(stream));
      for(const auto& stage:stages)if(!gemms.count(stage.radix)){auto lt=std::make_unique<Lt>();
        auto* weight=static_cast<unsigned char*>(encoded_basis)+stage.basis_offset*scalar;
        lt->setup(stage.tile,stage.padded,stage.tile,1,precision,weight,packed,product,workspace,stream,begin,end,false);gemms.emplace(stage.radix,std::move(lt));}
    }
    diag_spectrum=allocate<unsigned char>(total*element);diag_filtered=allocate<unsigned char>(total*element);ck(cudaStreamSynchronize(stream));
  }
  void load(const float2* host){if(!host)throw std::runtime_error("Missing input");ck(cudaMemcpyAsync(input,host,total*8,cudaMemcpyHostToDevice,stream));ck(cudaStreamSynchronize(stream));diagnostics_ready=false;}
  template<class T> void cufft_pipeline(bool debug){int grid=(total+255)/256;
    fft_prepare_input<<<grid,256,0,stream>>>(total,n,input,window,static_cast<FftPair<T>*>(cu_input));ck(cudaGetLastError());
    fft_check(cufftXtExec(cufft,cu_input,cu_spectrum,CUFFT_FORWARD),"forward");
    if(debug)ck(cudaMemcpyAsync(diag_spectrum,cu_spectrum,total*element,cudaMemcpyDeviceToDevice,stream));
    fft_filter_values<<<grid,256,0,stream>>>(total,n,static_cast<FftPair<T>*>(cu_spectrum),response,static_cast<FftPair<T>*>(cu_input));ck(cudaGetLastError());
    if(debug)ck(cudaMemcpyAsync(diag_filtered,cu_input,total*element,cudaMemcpyDeviceToDevice,stream));
    fft_check(cufftXtExec(cufft,cu_input,cu_output,CUFFT_INVERSE),"inverse");
    fft_normalize<<<grid,256,0,stream>>>(total,n,static_cast<FftPair<T>*>(cu_output),output);ck(cudaGetLastError());
  }
  void tc_begin(bool inverse){fft_reverse<<<(total+255)/256,256,0,stream>>>(total,n,permutation,inverse?filtered:input,window,!inverse,sequence[0]);ck(cudaGetLastError());current=sequence[0];last_stage=-1;last_inverse=inverse;}
  template<class T> void tc_pack(const TensorFftGeometry& stage,bool inverse){
    if(stage.previous==1)fft_pack_direct<<<(size_t(stage.tile)*stage.padded+255)/256,256,0,stream>>>(n,batch,stage.radix,stage.tile,stage.columns,stage.padded,stage.previous,current,phases+stage.phase_offset,inverse,static_cast<T*>(packed));
    else fft_pack_tiled<<<(stage.padded+31)/32,dim3(32,8),0,stream>>>(n,batch,stage.radix,stage.tile,stage.columns,stage.padded,stage.previous,current,phases+stage.phase_offset,inverse,static_cast<T*>(packed));ck(cudaGetLastError());
  }
  void tc_step(int index,bool inverse){if(index<0 || index>=int(stages.size()) || index!=last_stage+1 || inverse!=last_inverse)throw std::runtime_error("Invalid Tensor FFT stage sequence");
    const auto& stage=stages[index];previous_sequence=current;
    if(precision==0)tc_pack<float>(stage,inverse);if(precision==1)tc_pack<__half>(stage,inverse);if(precision==2)tc_pack<__nv_bfloat16>(stage,inverse);
    size_t scalar=precision==0?4:2;auto* weight=static_cast<unsigned char*>(encoded_basis)+(stage.basis_offset+(inverse?1792:0))*scalar;
    gemms[stage.radix]->run(weight,packed,product,stream);float2* next=current==sequence[0]?sequence[1]:sequence[0];
    fft_unpack<<<(total+255)/256,256,0,stream>>>(total,n,stage.radix,stage.previous,stage.tile,product,next);ck(cudaGetLastError());current=next;last_stage=index;
  }
  void tensor_pipeline(bool debug){tc_begin(false);for(int i=0;i<int(stages.size());++i)tc_step(i,false);
    if(debug)ck(cudaMemcpyAsync(diag_spectrum,current,total*8,cudaMemcpyDeviceToDevice,stream));
    fft_filter_values<<<(total+255)/256,256,0,stream>>>(total,n,reinterpret_cast<FftPair<float>*>(current),response,reinterpret_cast<FftPair<float>*>(filtered));ck(cudaGetLastError());
    if(debug)ck(cudaMemcpyAsync(diag_filtered,filtered,total*8,cudaMemcpyDeviceToDevice,stream));
    tc_begin(true);for(int i=0;i<int(stages.size());++i)tc_step(i,true);fft_rescale<<<(total+255)/256,256,0,stream>>>(total,n,current,output);ck(cudaGetLastError());
  }
  void operation(bool debug){
    if(backend==0){if(precision==0)cufft_pipeline<float>(debug);if(precision==1)cufft_pipeline<double>(debug);if(precision==2)cufft_pipeline<__half>(debug);if(precision==3)cufft_pipeline<__nv_bfloat16>(debug);}
    if(backend==1)tensor_pipeline(debug);
    if(backend==2)fft_dx_launch(dx,batch,stream,input,window,response,output,static_cast<float2*>(diag_spectrum),static_cast<float2*>(diag_filtered),debug);
    if(debug)diagnostics_ready=true;
  }
  void capture(int repeats){if(repeats<1 || repeats>64 || graph)throw std::runtime_error("Invalid/repeated graph construction");
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));try{for(int i=0;i<repeats;++i)operation(false);ck(cudaStreamEndCapture(stream,&graph));}
    catch(...){cudaGraph_t invalid=nullptr;cudaStreamEndCapture(stream,&invalid);if(invalid)cudaGraphDestroy(invalid);throw;}
    ck(cudaGraphInstantiate(&execution,graph,0));repetitions=repeats;
  }
  void time(bool captured){if(captured && !execution)throw std::runtime_error("Capture first");auto started=std::chrono::steady_clock::now();float ms=0;
    ck(cudaEventRecord(begin,stream));if(captured)ck(cudaGraphLaunch(execution,stream));else operation(false);
    ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));ck(cudaEventElapsedTime(&ms,begin,end));int divisor=captured?repetitions:1;
    last_ms=ms/divisor;last_wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()/divisor;
  }
  size_t bytes(int which){
    switch(which){case 0:case 1:return total*element;case 2:return total*8;case 3:return n*4;case 4:return n*8;
      case 5:return backend==1?n*4:0;case 6:return backend==1?phase_count*8:0;case 7:return backend==1?basis_count*4:0;
      case 8:return backend==1?basis_count*(precision==0?4:2):0;case 9:case 12:return backend==1?total*8:0;
      case 10:case 11:if(backend!=1 || last_stage<0)return 0;return size_t(stages[last_stage].tile)*stages[last_stage].padded*(which==11 || precision==0?4:2);
      default:throw std::runtime_error("Invalid FFT snapshot selection");}
  }
  void snapshot(int which,void* host){size_t count=bytes(which);if((which==0 || which==1) && !diagnostics_ready)throw std::runtime_error("Run a diagnostic pipeline before reading spectra");
    void* pointers[]={diag_spectrum,diag_filtered,input,window,response,permutation,phases,basis,encoded_basis,previous_sequence,packed,product,current};
    void* p=pointers[which];
    if(count && (!p || !host))throw std::runtime_error("Missing FFT snapshot buffer");if(count)ck(cudaMemcpyAsync(host,p,count,cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
  }
  std::string describe(){std::ostringstream out;out<<std::setprecision(17)<<"{\"cuda_ms\":"<<last_ms<<",\"wall_ms\":"<<last_wall_ms<<",\"owned_device_bytes\":"<<owned
    <<",\"workspace_bytes\":"<<workspace_bytes<<",\"cufft_version\":"<<cufft_version<<",\"cufftdx_version\":"<<(backend==2?dx.version:0)<<",\"spectrum_element_bytes\":"<<element
    <<",\"spectrum_scale\":"<<(backend==1?n:1)<<",\"graph_repetitions\":"<<repetitions<<",\"dx_block\":["<<dx.block.x<<","<<dx.block.y<<","<<dx.block.z<<"]"
    <<",\"dx_shared_bytes\":"<<(backend==2?dx.shared:0)<<",\"dx_registers\":"<<dx_registers<<",\"dx_diagnostic_registers\":"<<dx_diagnostic_registers<<",\"dx_theoretical_blocks_per_sm\":"<<dx_blocks_per_sm
    <<",\"last_stage\":"<<last_stage<<",\"last_inverse\":"<<(last_inverse?"true":"false")<<",\"stages\":[";
    for(size_t i=0;i<stages.size();++i){if(i)out<<",";const auto& s=stages[i];out<<"{\"radix\":"<<s.radix<<",\"previous\":"<<s.previous<<",\"tile\":"<<s.tile<<",\"columns\":"<<s.columns<<",\"padded\":"<<s.padded<<",\"phase_offset\":"<<s.phase_offset<<",\"basis_offset\":"<<s.basis_offset<<",\"lt_numerical_flags\":"<<gemms[s.radix]->flags<<",\"lt_selected_index\":"<<gemms[s.radix]->chosen<<"}";}
    out<<"],\"buffer_bytes\":[";for(int i=0;i<13;++i){if(i)out<<",";out<<bytes(i);}out<<"]}";return out.str();
  }
};

extern "C" {
const char* ff_error(){return fft_error_text.c_str();}
void* ff_new(int n,int batch,int backend,int precision,int radix,int dx_pack,int phase_count,int basis_count,
             const float2* input,const float* window,const float2* response,const int* perm,const float2* phases,const float* basis){
  try{auto p=std::make_unique<FftPlan>();p->initialize(n,batch,backend,precision,radix,dx_pack,phase_count,basis_count,input,window,response,perm,phases,basis);return p.release();}
  catch(const std::exception& e){fft_error_text=e.what();return nullptr;}}
void ff_delete(void* p){delete static_cast<FftPlan*>(p);}
int ff_load(void* p,const float2* input){try{if(!p)throw std::runtime_error("Null FFT plan");static_cast<FftPlan*>(p)->load(input);return 0;}catch(const std::exception& e){fft_error_text=e.what();return -1;}}
int ff_run(void* p,int diagnostic){try{if(!p)throw std::runtime_error("Null FFT plan");auto& q=*static_cast<FftPlan*>(p);q.operation(diagnostic!=0);ck(cudaStreamSynchronize(q.stream));return 0;}catch(const std::exception& e){fft_error_text=e.what();return -1;}}
int ff_capture(void* p,int repetitions){try{if(!p)throw std::runtime_error("Null FFT plan");static_cast<FftPlan*>(p)->capture(repetitions);return 0;}catch(const std::exception& e){fft_error_text=e.what();return -1;}}
int ff_time(void* p,int captured){try{if(!p)throw std::runtime_error("Null FFT plan");static_cast<FftPlan*>(p)->time(captured!=0);return 0;}catch(const std::exception& e){fft_error_text=e.what();return -1;}}
int ff_output(void* p,float2* host){try{if(!p || !host)throw std::runtime_error("Null FFT output/plan");auto& q=*static_cast<FftPlan*>(p);ck(cudaMemcpyAsync(host,q.output,q.total*8,cudaMemcpyDeviceToHost,q.stream));ck(cudaStreamSynchronize(q.stream));return 0;}catch(const std::exception& e){fft_error_text=e.what();return -1;}}
int ff_snapshot(void* p,int which,void* host){try{if(!p)throw std::runtime_error("Null FFT plan");static_cast<FftPlan*>(p)->snapshot(which,host);return 0;}catch(const std::exception& e){fft_error_text=e.what();return -1;}}
int ff_tc_begin(void* p,int inverse){try{if(!p || static_cast<FftPlan*>(p)->backend!=1)throw std::runtime_error("Tensor FFT plan required");auto& q=*static_cast<FftPlan*>(p);q.tc_begin(inverse!=0);ck(cudaStreamSynchronize(q.stream));return 0;}catch(const std::exception& e){fft_error_text=e.what();return -1;}}
int ff_tc_step(void* p,int stage,int inverse){try{if(!p || static_cast<FftPlan*>(p)->backend!=1)throw std::runtime_error("Tensor FFT plan required");auto& q=*static_cast<FftPlan*>(p);q.tc_step(stage,inverse!=0);ck(cudaStreamSynchronize(q.stream));return 0;}catch(const std::exception& e){fft_error_text=e.what();return -1;}}
const char* ff_metadata(void* p){try{if(!p)throw std::runtime_error("Null FFT plan");fft_json_text=static_cast<FftPlan*>(p)->describe();return fft_json_text.c_str();}catch(const std::exception& e){fft_error_text=e.what();return nullptr;}}
}
