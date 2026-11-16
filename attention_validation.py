"""CPU-only arithmetic contracts for the materialized native attention pipeline."""
import numpy as np

from attention_reference import allowed,attention,encoding,padded,probabilities
from reference import digest,relative_error


def product(actual,a,b):
    a,b=a.astype(np.float64),b.astype(np.float64);expected=a@b.swapaxes(-1,-2);k=a.shape[-1];gamma=k*2**-24/(1-k*2**-24)
    bound=8*gamma*(np.abs(a)@np.abs(b).swapaxes(-1,-2))+1e-6
    ratio=float(np.max(np.abs(actual.astype(np.float64)-expected)/bound))
    if not np.isfinite(actual).all() or ratio>1:raise ValueError(f"Attention GEMM arithmetic failed: {ratio}")
    return ratio


def validate_arrays(q,k,v,modes,kind,scores,p,full_output,output,reference_output=None):
    qp,kp,vt=padded(q,k,v);length,depth=q.shape[-2:];sp,dp=qp.shape[-2:];hashes={};decoded=[]
    for name,x,mode in (("q",qp,modes[0]),("k",kp,modes[0]),("p",p,modes[1]),("v",vt,modes[1])):
        raw,scale,value=encoding(x,mode);hashes[name]=[digest(raw),digest(scale)];decoded.append(value)
    first=product(scores,*decoded[:2]);target_p=probabilities(scores.astype(np.float64),depth,allowed(length,kind)) if scores.shape[-1]==length else None
    if target_p is None:
        permitted=np.zeros((sp,sp),bool);permitted[:length,:length]=allowed(length,kind);target_p=probabilities(scores.astype(np.float64),depth,permitted)
    difference=float(np.max(np.abs(p-target_p)))
    if not np.isfinite(p).all() or np.any(np.abs(p-target_p)>1e-6+3e-6*target_p):raise ValueError(f"Attention softmax arithmetic failed: {difference}")
    if np.any(p[:,length:]) or np.any(p[:,:,length:]):raise ValueError("Padded attention probabilities must be zero")
    if kind=="padding" and np.any(p[:,0]):raise ValueError("Fully masked attention rows must be zero")
    second=product(full_output,decoded[2],decoded[3]);np.testing.assert_array_equal(output.reshape(-1,length,depth),full_output[:,:length,:depth])
    target=attention(q,k,v,kind) if reference_output is None else reference_output
    return {"encoded_sha256":hashes,"qk_max_bound_ratio":first,"pv_max_bound_ratio":second,"softmax_max_absolute_error":difference,
            "relative_frobenius_error":relative_error(output,target),"output_sha256":digest(output),"scores_sha256":digest(scores),"probabilities_sha256":digest(p),
            "padded_output_sha256":digest(full_output),"checked_output_entries":int(scores.size+p.size+full_output.size+output.size)}
