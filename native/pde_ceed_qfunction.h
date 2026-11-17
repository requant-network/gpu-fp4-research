// Original affine diffusion/reaction weak-form QFunction using the libCEED API.
#include <ceed/types.h>

CEED_QFUNCTION(PdeApply)(void *ctx, CeedInt Q,
                         const CeedScalar *const *in,
                         CeedScalar *const *out) {
  const CeedScalar *u=in[0], *du=in[1], *qdata=in[2];
  CeedScalar *v=out[0], *dv=out[1];
  for (CeedInt i=0; i<Q; ++i) {
    v[i]=qdata[3*Q+i]*u[i];
    for (CeedInt direction=0; direction<3; ++direction)
      dv[direction*Q+i]=qdata[direction*Q+i]*du[direction*Q+i];
  }
  return 0;
}
