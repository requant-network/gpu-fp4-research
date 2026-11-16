"""Independent CPU contracts for column, block and residual FP4 correction."""
import numpy as np

from reference import digest,quantize,relative_error


def factors(a,b,method,budget=128,fraction=.25):
    if method not in (1,2,3) or a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1]:
        raise ValueError("Invalid correction operands/method")
    k=a.shape[1]
    if k%128 or k>2048 or budget<128 or budget>k or budget%128 or not 0<fraction<=1:
        raise ValueError("Invalid correction budget/fraction")
    if any(x.dtype!=np.float32 or not np.isfinite(x).all() for x in (a,b)):
        raise ValueError("Correction inputs must be finite FP32")
    qa,qb=(quantize(x,"fp4")[2] for x in (a,b))
    metadata={}
    if method==1:
        score=np.max(np.abs(a),axis=0)*np.max(np.abs(b),axis=0)
        selected=np.lexsort((np.arange(k),-score)).astype(np.int32)[:budget]
        arrays=[a[:,selected],b[:,selected],qa[:,selected],qb[:,selected]]
        metadata["selected_columns"]=selected
    else:
        masks=[]
        for x in (a,b):
            maxima=np.max(np.abs(x.reshape(-1,16)),axis=1)
            masks.append(np.ones(maxima.shape,np.uint8) if method==3 else
                         (maxima>np.float32(fraction)*np.max(np.abs(x))).astype(np.uint8))
        ra=np.where(np.repeat(masks[0],16).reshape(a.shape),a-qa,np.float32(0))
        rb=np.where(np.repeat(masks[1],16).reshape(b.shape),b-qb,np.float32(0))
        corrected_b=np.where(np.repeat(masks[1],16).reshape(b.shape),b,qb)
        arrays=[ra,corrected_b,qa,rb]
        metadata.update(mask_a=masks[0],mask_b=masks[1])
    arrays=[np.ascontiguousarray(x,dtype=np.float16) for x in arrays]
    if not all(np.isfinite(x).all() for x in arrays):raise ValueError("Correction FP16 factors overflow")
    return arrays,metadata,(qa,qb)


def oracle(a,b,method,budget=128,fraction=.25):
    arrays,metadata,(qa,qb)=factors(a,b,method,budget,fraction)
    x,y,z,w=(v.astype(np.float64) for v in arrays)
    qx,qy=qa.astype(np.float64),qb.astype(np.float64)
    base=qx@qy.T
    first,second=x@y.T,z@w.T
    expected=base+first+(-second if method==1 else second)
    gamma=(a.shape[1]+4)*2**-24/(1-(a.shape[1]+4)*2**-24)
    magnitude=np.abs(qx)@np.abs(qy).T+np.abs(x)@np.abs(y).T+np.abs(z)@np.abs(w).T
    bound=8*gamma*magnitude+np.finfo(np.float32).tiny
    return {"factors":arrays,"metadata":metadata,"quantized":(qa,qb),"expected":expected,"bound":bound,
            "original":a.astype(np.float64)@b.astype(np.float64).T,
            "factor_sha256":[digest(v) for v in arrays],"metadata_sha256":{key:digest(v) for key,v in metadata.items()}}


def check_output(actual,reference):
    ratio=float(np.max(np.abs(actual.astype(np.float64)-reference["expected"])/reference["bound"]))
    if not np.isfinite(actual).all() or ratio>1:raise ValueError(f"Correction arithmetic failed: {ratio}")
    return {"arithmetic_max_bound_ratio":ratio,"relative_frobenius_error":relative_error(actual,reference["original"]),
            "output_sha256":digest(actual),"checked_output_entries":int(actual.size)}
