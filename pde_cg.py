"""Explicit host-controlled FP64-vector Jacobi PCG, not a best-in-class solver.

The fixed operator may be nonlinear/inexact. Its recursive residual status is
internal only; the caller must charge/check the original-system final residual.
"""
import math
import numpy as np


def jacobi_diagonal(mesh,reaction=0.):
    if not np.isfinite(reaction) or reaction<0:raise ValueError('Nonnegative reaction required')
    w,s=mesh.weights,np.diag(mesh.stiffness)
    local=mesh.h*.5*(s[:,None,None]*w[None,:,None]*w[None,None,:]+
                     w[:,None,None]*s[None,:,None]*w[None,None,:]+
                     w[:,None,None]*w[None,:,None]*s[None,None,:])
    result=np.bincount(mesh.indices.ravel(),weights=np.tile(local.ravel(),mesh.elements**3),minlength=mesh.size)
    result+=reaction*mesh.mass
    result[mesh.boundary]=1
    if not np.isfinite(result).all() or not np.all(result>0):raise ValueError('Invalid physical preconditioner')
    return result


def _dot(xp,left,right):return float(xp.dot(left,right).item())


def solve(apply,rhs,diagonal,*,xp,tolerance=1e-8,max_iterations=512,x0=None):
    if rhs.ndim!=1 or diagonal.shape!=rhs.shape or not 0<tolerance<1 or type(max_iterations) is not int or max_iterations<1:raise ValueError('Bounded PCG arguments required')
    if not bool(xp.isfinite(rhs).all()) or not bool(xp.isfinite(diagonal).all()) or not bool((diagonal>0).all()):raise ValueError('Finite data and positive preconditioner required')
    x=xp.zeros_like(rhs,dtype=xp.float64) if x0 is None else xp.asarray(x0,dtype=xp.float64).copy()
    if x.shape!=rhs.shape or not bool(xp.isfinite(x).all()):raise ValueError('Finite full starting vector required')
    calls=0
    if x0 is None:r=rhs.copy()
    else:r=rhs-apply(x);calls+=1
    normb=math.sqrt(_dot(xp,rhs,rhs));denominator=normb if normb>0 else 1.
    relative=math.sqrt(_dot(xp,r,r))/denominator
    trace=[dict(iteration=0,recursive_relative=relative)]
    if relative<=tolerance:return dict(x=x,status='internal-converged',iterations=0,operator_calls=calls,trace=trace)
    z=r/diagonal;p=z.copy();rz=_dot(xp,r,z)
    status='iteration-limit';completed=0
    for iteration in range(1,max_iterations+1):
        product=apply(p);calls+=1
        curvature=_dot(xp,p,product)
        if not math.isfinite(curvature) or curvature<=0 or not math.isfinite(rz) or rz<=0:
            status='nonpositive-curvature' if math.isfinite(curvature) and curvature<=0 else 'invalid-reduction'
            trace.append(dict(iteration=iteration,status=status,curvature=curvature if math.isfinite(curvature) else None));break
        alpha=rz/curvature
        x+=alpha*p;r-=alpha*product
        relative=math.sqrt(_dot(xp,r,r))/denominator
        trace.append(dict(iteration=iteration,recursive_relative=relative if math.isfinite(relative) else None,alpha=alpha,curvature=curvature))
        completed=iteration
        if not math.isfinite(relative):status='nonfinite-recursive-residual';break
        if relative<=tolerance:status='internal-converged';break
        z=r/diagonal;next_rz=_dot(xp,r,z)
        p=z+(next_rz/rz)*p;rz=next_rz
    return dict(x=x,status=status,iterations=completed,operator_calls=calls,trace=trace)


def solve_reliable(apply,original_apply,rhs,diagonal,*,xp,tolerance=1e-8,max_iterations=512,x0=None,low_budget=32,segment_iterations=16):
    """Bounded low-operator defect corrections, true residual, then FP64 fallback.

    Each candidate correction is accepted only after reducing the original norm.
    No low recursive stopping test certifies the physical target. All callbacks
    and rejected work are part of the caller's complete solve timing boundary.
    """
    if type(low_budget) is not int or low_budget<1 or type(segment_iterations) is not int or segment_iterations<1:raise ValueError('Positive low-work bounds required')
    if not 0<tolerance<1 or type(max_iterations) is not int or max_iterations<1:raise ValueError('Bounded solve controls required')
    x=xp.zeros_like(rhs,dtype=xp.float64) if x0 is None else xp.asarray(x0,dtype=xp.float64).copy()
    normb=math.sqrt(_dot(xp,rhs,rhs));denominator=normb if normb>0 else 1.
    high_calls=0
    def high(value):
        nonlocal high_calls
        result=original_apply(value);high_calls+=1;return result
    residual=rhs-high(x)
    relative=math.sqrt(_dot(xp,residual,residual))/denominator
    trace=[];low_calls=0;iterations=0;fallback=False;reason=None
    while relative>tolerance and low_calls<low_budget:
        part=solve(apply,residual,diagonal,xp=xp,tolerance=.1,max_iterations=min(segment_iterations,low_budget-low_calls))
        low_calls+=part['operator_calls'];iterations+=part['iterations']
        accepted=False;new_relative=None
        if bool(xp.isfinite(part['x']).all()) and part['status'] not in ('nonpositive-curvature','invalid-reduction','nonfinite-recursive-residual'):
            proposal=x+part['x']
            candidate=rhs-high(proposal)
            new_relative=math.sqrt(_dot(xp,candidate,candidate))/denominator
            accepted=math.isfinite(new_relative) and new_relative<=.75*relative
            if accepted:x=proposal;residual=candidate
        trace.append(dict(phase='low-correction',low_operator_calls=part['operator_calls'],inner_status=part['status'],inner_trace=part['trace'],
                          original_before=relative,original_candidate=new_relative if new_relative is not None and math.isfinite(new_relative) else None,accepted=accepted))
        if not accepted:reason='Rejected original-residual progress or invalid low correction';break
        relative=new_relative
    if relative>tolerance:
        fallback=True
        if reason is None:reason='Declared low-operator work budget exhausted'
        correction=solve(high,residual,diagonal,xp=xp,tolerance=min(.5,max(np.finfo(np.float64).eps,tolerance*.5/relative)),max_iterations=max_iterations)
        x+=correction['x'];iterations+=correction['iterations']
        residual=rhs-high(x);relative=math.sqrt(_dot(xp,residual,residual))/denominator
        trace.append(dict(phase='fp64-fallback',reason=reason,inner_status=correction['status'],inner_trace=correction['trace'],original_relative=relative if math.isfinite(relative) else None))
    return dict(x=x,status='original-converged' if relative<=tolerance else 'original-target-failed',iterations=iterations,
                operator_calls=low_calls+high_calls,low_operator_calls=low_calls,high_operator_calls=high_calls,
                fallback=fallback,original_relative=relative if math.isfinite(relative) else None,trace=trace)
