// SPDX-License-Identifier: MIT
// Original PCA metadata helpers; included CUTLASS-derived code retains its notice.
#include "hardware_bridge.cu"
#include <cstdint>
extern "C" size_t rq_pca_owned_bytes(StudyPlan* p) {
  if(!p)return 0;
  size_t total=p->a_bytes+p->b_bytes+p->workspace_bytes;
  if(p->mode==0)total+=size(filter_zeros(p->la))+size(filter_zeros(p->lb))+size_t(p->m+p->n)*p->k/16;
  return total;
}
extern "C" std::uint64_t rq_pca_flags(StudyPlan* p,int index) {
  if(!p || p->mode==0 || index<0 || index>=int(p->algorithms.size()))return 0;
  std::uint64_t flags=0;size_t written=0;
  auto status=cublasLtMatmulAlgoCapGetAttribute(&p->algorithms[index].algo,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,&flags,sizeof(flags),&written);
  return status==CUBLAS_STATUS_SUCCESS?flags:0;
}
