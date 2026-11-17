// Thin original C ABI over pinned libCEED; no vendor sources/SDK headers bundled.
#include <ceed.h>
#include <stdexcept>
#include <string>
#include "pde_ceed_qfunction.h"

static_assert(sizeof(CeedScalar)==sizeof(double),"Reference requires FP64 libCEED");
static thread_local std::string error_message;
struct PdeCeed {
  Ceed ceed=nullptr;
  CeedElemRestriction restriction=nullptr, qrestriction=nullptr;
  CeedBasis basis=nullptr;
  CeedVector qdata=nullptr, input=nullptr, output=nullptr;
  CeedQFunction qfunction=nullptr;
  CeedOperator op=nullptr;
  std::string resource;
  ~PdeCeed() {
    CeedOperatorDestroy(&op);
    CeedQFunctionDestroy(&qfunction);
    CeedVectorDestroy(&input);
    CeedVectorDestroy(&output);
    CeedVectorDestroy(&qdata);
    CeedBasisDestroy(&basis);
    CeedElemRestrictionDestroy(&qrestriction);
    CeedElemRestrictionDestroy(&restriction);
    CeedDestroy(&ceed);
  }
  void checked(int code,const char* stage) {
    if (code) {
      const char* message=nullptr;
      if (ceed) CeedGetErrorMessage(ceed,&message);
      throw std::runtime_error(std::string(stage)+": "+(message ? message : std::to_string(code)));
    }
  }
};
extern "C" const char* rq_pde_ceed_error() { return error_message.c_str(); }
extern "C" void* rq_pde_ceed_create(const char* resource,int elements,int q,int n,
                                     const int* offsets,const double* interpolation,
                                     const double* derivative,const double* nodes,
                                     const double* weights,const double* qdata,
                                     const char* qfunction_source) {
  auto* p=new PdeCeed;
  try {
    p->checked(CeedInit(resource,&p->ceed),"CeedInit");
    p->checked(CeedSetErrorHandler(p->ceed,CeedErrorStore),"CeedSetErrorHandler");
    const char* actual=nullptr;
    p->checked(CeedGetResource(p->ceed,&actual),"CeedGetResource");
    p->resource=actual;
    const CeedInt volume=q*q*q;
    p->checked(CeedElemRestrictionCreate(p->ceed,elements,volume,1,1,n,CEED_MEM_HOST,CEED_COPY_VALUES,offsets,&p->restriction),"Restriction");
    const CeedInt strides[3]={1,volume,4*volume};
    p->checked(CeedElemRestrictionCreateStrided(p->ceed,elements,volume,4,CeedSize(elements)*4*volume,strides,&p->qrestriction),"Qdata restriction");
    p->checked(CeedBasisCreateTensorH1(p->ceed,3,1,q,q,interpolation,derivative,nodes,weights,&p->basis),"Canonical basis");
    p->checked(CeedVectorCreate(p->ceed,CeedSize(elements)*4*volume,&p->qdata),"Qdata vector");
    p->checked(CeedVectorSetArray(p->qdata,CEED_MEM_HOST,CEED_COPY_VALUES,const_cast<double*>(qdata)),"Qdata copy");
    p->checked(CeedQFunctionCreateInterior(p->ceed,1,PdeApply,qfunction_source,&p->qfunction),"QFunction");
    p->checked(CeedQFunctionAddInput(p->qfunction,"u",1,CEED_EVAL_INTERP),"Input u");
    p->checked(CeedQFunctionAddInput(p->qfunction,"du",3,CEED_EVAL_GRAD),"Input du");
    p->checked(CeedQFunctionAddInput(p->qfunction,"qdata",4,CEED_EVAL_NONE),"Input qdata");
    p->checked(CeedQFunctionAddOutput(p->qfunction,"v",1,CEED_EVAL_INTERP),"Output v");
    p->checked(CeedQFunctionAddOutput(p->qfunction,"dv",3,CEED_EVAL_GRAD),"Output dv");
    p->checked(CeedOperatorCreate(p->ceed,p->qfunction,CEED_QFUNCTION_NONE,CEED_QFUNCTION_NONE,&p->op),"Operator");
    for (const char* name : {"u","du","v","dv"})
      p->checked(CeedOperatorSetField(p->op,name,p->restriction,p->basis,CEED_VECTOR_ACTIVE),name);
    p->checked(CeedOperatorSetField(p->op,"qdata",p->qrestriction,CEED_BASIS_NONE,p->qdata),"Qdata field");
    p->checked(CeedVectorCreate(p->ceed,n,&p->input),"Input vector");
    p->checked(CeedVectorCreate(p->ceed,n,&p->output),"Output vector");
    return p;
  } catch (const std::exception& error) {
    error_message=error.what(); delete p; return nullptr;
  }
}
extern "C" int rq_pde_ceed_apply(void* handle,double* input,double* output) {
  auto* p=static_cast<PdeCeed*>(handle);
  try {
    // External CuPy writes must invalidate the vector's previous value state.
    p->checked(CeedVectorSetArray(p->input,CEED_MEM_DEVICE,CEED_USE_POINTER,input),"Bind changed input");
    p->checked(CeedVectorSetArray(p->output,CEED_MEM_DEVICE,CEED_USE_POINTER,output),"Bind output");
    p->checked(CeedOperatorApply(p->op,p->input,p->output,CEED_REQUEST_IMMEDIATE),"Operator apply");
    return 0;
  } catch (const std::exception& error) { error_message=error.what(); return 1; }
}
extern "C" const char* rq_pde_ceed_resource(void* handle) {
  return static_cast<PdeCeed*>(handle)->resource.c_str();
}
extern "C" void rq_pde_ceed_destroy(void* handle) { delete static_cast<PdeCeed*>(handle); }
