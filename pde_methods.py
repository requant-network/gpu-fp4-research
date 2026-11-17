"""Declared physical-operator configurations and common charged solver calls."""
import cupy as cp
from pde_cuda import PdeCUDA
from pde_ceed import PdeCEED
from pde_tensor import PdeTensor
from pde_small import PdeSmall


def configurations():
    result=[dict(family='cuda',precision=p,threads=t) for p in ('fp32','fp64') for t in (128,256)]
    result.extend(dict(family='ceed',precision='fp64',backend=b,threads=256) for b in ('ref','shared','gen'))
    result.extend(dict(family='tensor',precision=p,threads=256) for p in ('fp4','fp8','fp16','bf16','fp32'))
    result.extend(dict(family='small',precision=p,threads=256) for p in ('fp8','fp16','bf16','fp32'))
    return result


def create(mesh,configuration,reaction):
    family=configuration['family']
    if family=='cuda':return PdeCUDA(mesh,configuration['precision'],configuration['threads'],reaction)
    if family=='ceed':return PdeCEED(mesh,configuration['backend'],reaction)
    if family=='tensor':return PdeTensor(mesh,configuration['precision'],configuration['threads'],reaction)
    if family=='small':return PdeSmall(mesh,configuration['precision'],configuration['threads'],reaction)
    raise ValueError('Unknown declared physical operator')


class DeviceOperator:
    """Common full-operation call: D2D staging is included, never a free alias."""
    def __init__(self,plan):self.plan,self.calls=plan,0

    def __call__(self,values):
        if values.shape!=self.plan.data.shape or values.dtype!=cp.float64:raise ValueError('Full FP64 common solver vector required')
        with self.plan.stream:cp.copyto(self.plan.data,values)
        self.plan.run()
        self.calls+=1
        return self.plan.output
