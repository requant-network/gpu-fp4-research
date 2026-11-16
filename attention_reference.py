"""Independent padded encodings, masked attention and classifier arithmetic."""
import numpy as np

from reference import digest,inputs,quantize,relative_error

SHAPES=((1,1,128,64),(2,4,512,64),(1,4,1024,128),(1,2,256,32),(1,2,193,48),(1,1,768,96))
MASKS=("none","causal","padding")
PROFILES=("normal","peaky")
COMBINATIONS=tuple((a,b) for a in ("fp4","fp8","fp16") for b in ("fp4","fp8","fp16"))+(("bf16","bf16"),("fp32","fp32"))


def arrays(seed,shape,profile="normal"):
    if len(shape)!=4 or min(shape)<1 or profile not in PROFILES:raise ValueError("Positive BHLD shape and known attention profile required")
    batch,heads,length,depth=shape;rows=batch*heads*length
    q,k=inputs(seed,(rows,rows,depth),"normal");v,_=inputs(seed+1,(rows,rows,depth),"normal")
    if profile=="peaky":q*=np.float32(4);k*=np.float32(4)
    return tuple(np.ascontiguousarray(x.reshape(shape)) for x in (q,k,v))


def allowed(length,kind,padded=None):
    if kind not in MASKS or length<1 or padded is not None and padded<length:raise ValueError("Invalid attention mask")
    size=length if padded is None else padded;result=np.zeros((size,size),bool);result[:length,:length]=True
    if kind=="causal":result&=np.tril(np.ones_like(result))
    if kind=="padding":result[:,length-length//4:]=False;result[0]=False
    return result


def probabilities(scores,depth,mask):
    scores=np.asarray(scores,np.float64)/np.sqrt(depth);scores=np.where(mask,scores,-np.inf)
    maximum=np.max(scores,axis=-1,keepdims=True);maximum=np.where(np.isfinite(maximum),maximum,0.)
    values=np.exp(scores-maximum);total=values.sum(axis=-1,keepdims=True)
    return values/np.where(total>0,total,1.)


def attention(q,k,v,kind="none"):
    q,k,v=(x.astype(np.float64) for x in (q,k,v));p=probabilities(q@k.swapaxes(-1,-2),q.shape[-1],allowed(q.shape[-2],kind))
    return p@v


def padded(q,k,v):
    if q.shape!=k.shape or q.shape!=v.shape or q.ndim!=4 or any(x.dtype!=np.float32 or not np.isfinite(x).all() for x in (q,k,v)):raise ValueError("Matching finite FP32 BHLD operands required")
    batch,heads,length,depth=q.shape;sp=(length+127)//128*128;dp=(depth+127)//128*128
    values=[]
    for source in (q,k,v):
        out=np.zeros((batch*heads,sp,dp),np.float32);out[:,:length,:depth]=source.reshape(batch*heads,length,depth);values.append(out)
    return values[0],values[1],np.ascontiguousarray(values[2].swapaxes(-1,-2))


def encoding(x,mode):
    flat=x.reshape(-1,x.shape[-1]);raw,scales,decoded=quantize(flat,mode)
    return raw,scales,decoded.reshape(x.shape)


def product_check(actual,a,b):
    a,b=a.astype(np.float64),b.astype(np.float64);expected=a@b.swapaxes(-1,-2);k=a.shape[-1];gamma=k*2**-24/(1-k*2**-24)
    bound=8*gamma*(np.abs(a)@np.abs(b).swapaxes(-1,-2))+1e-6
    if actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError("Invalid attention product")
    ratio=float(np.max(np.abs(actual.astype(np.float64)-expected)/bound))
    if ratio>1:raise ValueError(f"Attention GEMM arithmetic failed: {ratio}")
    return ratio


def network(x,weights,attend=None):
    def linear(a,name):return a@weights[name+".weight"].astype(np.float64).T+weights[name+".bias"].astype(np.float64)
    def norm(a,name):
        mean=a.mean(axis=-1,keepdims=True);variance=((a-mean)**2).mean(axis=-1,keepdims=True)
        return (a-mean)/np.sqrt(variance+1e-5)*weights[name+".weight"]+weights[name+".bias"]
    embedded=linear(x[:,:64,None].astype(np.float64),"embed")+weights["position"]
    joined=linear(norm(embedded,"norm1"),"qkv");batch,length,_=joined.shape
    q,k,v=(np.ascontiguousarray(joined.reshape(batch,length,3,2,64)[:,:,i].transpose(0,2,1,3)) for i in range(3))
    output=attention(q,k,v) if attend is None else attend(q,k,v)
    residual=embedded+linear(output.transpose(0,2,1,3).reshape(batch,length,128),"projection")
    final=residual+linear(np.maximum(linear(norm(residual,"norm2"),"ff1"),0),"ff2")
    return linear(norm(final,"norm3").mean(axis=1),"classifier")
