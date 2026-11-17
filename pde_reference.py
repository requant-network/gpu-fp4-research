"""Independent FP64 nodal spectral-element reference for a bounded 3D PDE study.

This file defines physical discretization, not the GPU's packed Tensor Core
arithmetic. Collocated GLL quadrature, homogeneous Dirichlet projection and all
manufactured-solution/discretization errors remain explicit.
"""
import numpy as np
from numpy.polynomial.legendre import Legendre
from pde_coefficients import CHOICES,canonical_gll


def gll(q):
    """Gauss-Lobatto-Legendre nodes, quadrature weights and nodal derivative."""
    if type(q) is not int or not 2<=q<=32:raise ValueError('Bounded GLL node count must be 2..32')
    degree=q-1
    polynomial=Legendre.basis(degree)
    nodes=np.concatenate(([-1.],np.sort(polynomial.deriv().roots()),[1.])).astype(np.float64)
    values=polynomial(nodes)
    weights=2./(degree*(degree+1)*values**2)
    derivative=np.zeros((q,q),np.float64)
    for i in range(q):
        for j in range(q):
            if i!=j:derivative[i,j]=values[i]/(values[j]*(nodes[i]-nodes[j]))
        derivative[i,i]=-derivative[i].sum()
    return nodes,weights,derivative


class Mesh:
    """Conforming tensor-product GLL grid on the unit cube, affine equal cells."""
    def __init__(self,elements,q):
        if type(elements) is not int or not 1<=elements<=128:raise ValueError('Bounded positive element count required')
        self.elements,self.q=elements,q
        if type(q) is not int:raise ValueError('Integer GLL node count required')
        if q in CHOICES:
            self.nodes,self.weights,self.derivative,canonical_stiffness=canonical_gll(q)
        else:
            self.nodes,self.weights,self.derivative=gll(q)
            canonical_stiffness=None
        self.h=1./elements
        self.points_per_axis=elements*(q-1)+1
        self.size=self.points_per_axis**3
        if self.size>8_000_000:raise ValueError('Declared CPU mesh bound exceeded')
        if canonical_stiffness is not None:self.stiffness=canonical_stiffness
        else:
            self.stiffness=self.derivative.T@(self.weights[:,None]*self.derivative)
            self.stiffness=(self.stiffness+self.stiffness.T)*.5
        axis=np.concatenate([e*self.h+(self.nodes[:-1]+1)*self.h*.5 for e in range(elements)]+[np.array([1.])])
        self.coordinates=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
        self.boundary=np.any((self.coordinates==0)|(self.coordinates==1),axis=1)
        mapping=[]
        for ei in range(elements):
            for ej in range(elements):
                for ek in range(elements):
                    i,j,k=np.meshgrid(ei*(q-1)+np.arange(q),ej*(q-1)+np.arange(q),ek*(q-1)+np.arange(q),indexing='ij')
                    mapping.append(((i*self.points_per_axis+j)*self.points_per_axis+k).astype(np.int32))
        self.indices=np.stack(mapping)
        local_mass=(self.h*.5)**3*self.weights[:,None,None]*self.weights[None,:,None]*self.weights[None,None,:]
        self.mass=np.bincount(self.indices.ravel(),weights=np.tile(local_mass.ravel(),elements**3),minlength=self.size)
        for value in (self.nodes,self.weights,self.derivative,self.stiffness,self.coordinates,self.boundary,self.indices,self.mass):value.setflags(write=False)


def apply_operator(mesh,values,reaction=0.):
    """FP64 sumfactorization/scatter; boundary columns projected, rows identity."""
    values=np.asarray(values)
    if values.shape!=(mesh.size,) or values.dtype not in (np.float32,np.float64) or not np.isfinite(values).all():
        raise ValueError('Finite full nodal floating-point vector required')
    if not np.isfinite(reaction) or reaction<0:raise ValueError('Nonnegative finite reaction required')
    masked=values.astype(np.float64,copy=True)
    masked[mesh.boundary]=0
    local=masked[mesh.indices]
    weight,stiffness=mesh.weights,mesh.stiffness
    x=np.einsum('il,eljk->eijk',stiffness,local,optimize=True)*weight[None,None,:,None]*weight[None,None,None,:]
    y=np.einsum('jl,eilk->eijk',stiffness,local,optimize=True)*weight[None,:,None,None]*weight[None,None,None,:]
    z=np.einsum('kl,eijl->eijk',stiffness,local,optimize=True)*weight[None,:,None,None]*weight[None,None,:,None]
    result=np.bincount(mesh.indices.ravel(),weights=((mesh.h*.5)*(x+y+z)).ravel(),minlength=mesh.size)
    result+=float(reaction)*mesh.mass*masked
    result[mesh.boundary]=values[mesh.boundary]
    return result


def dense_operator(mesh,reaction=0.):
    """Independent tiny-control assembly via Kronecker products, never large runs."""
    if mesh.size>4096:raise ValueError('Dense control deliberately limited to4096 nodes')
    if not np.isfinite(reaction) or reaction<0:raise ValueError('Nonnegative finite reaction required')
    w,s=np.diag(mesh.weights),mesh.stiffness
    local=(mesh.h*.5)*(np.kron(s,np.kron(w,w))+np.kron(w,np.kron(s,w))+np.kron(w,np.kron(w,s)))
    local+=float(reaction)*(mesh.h*.5)**3*np.kron(w,np.kron(w,w))
    matrix=np.zeros((mesh.size,mesh.size),np.float64)
    for indices in mesh.indices.reshape(-1,mesh.q**3):matrix[np.ix_(indices,indices)]+=local
    matrix[mesh.boundary,:]=0
    matrix[:,mesh.boundary]=0
    selected=np.flatnonzero(mesh.boundary)
    matrix[selected,selected]=1
    return matrix


def manufactured(mesh,cohort,reaction=0.,time=0.)->dict:
    """Smooth zero-boundary sine data with analytic continuum Laplacian/heat law."""
    if cohort not in ('a','b'):raise ValueError('Unknown independent physical input cohort')
    if not np.isfinite(reaction) or reaction<0 or not np.isfinite(time) or time<0:raise ValueError('Nonnegative finite reaction/time required')
    modes=((1.,(1,1,1)),) if cohort=='a' else ((.75,(1,2,3)),(-.125,(2,1,2)))
    exact=np.zeros(mesh.size,np.float64)
    forcing=np.zeros_like(exact)
    eigenvalues=[]
    for amplitude,frequency in modes:
        wave=np.asarray(frequency,dtype=np.float64)
        eigenvalue=float(np.pi**2*np.dot(wave,wave))
        mode=amplitude*np.exp(-eigenvalue*time)*np.prod(np.sin(np.pi*mesh.coordinates*wave[None,:]),axis=1)
        exact+=mode
        forcing+=(eigenvalue+reaction)*mode
        eigenvalues.append(eigenvalue)
    exact[mesh.boundary]=0
    forcing[mesh.boundary]=0
    rhs=mesh.mass*forcing
    rhs[mesh.boundary]=0
    return dict(exact=exact,forcing=forcing,rhs=rhs,laplacian_eigenvalues=eigenvalues,modes=modes,reaction=float(reaction),time=float(time))


def implicit_euler_control(mesh,cohort,steps,final_time)->dict:
    """Small independent dense implicit-Euler control, not the GPU solver path."""
    if type(steps) is not int or not 1<=steps<=256 or not np.isfinite(final_time) or final_time<=0:
        raise ValueError('Bounded positive steps/final time required')
    dt=float(final_time)/steps
    matrix=dense_operator(mesh,reaction=1./dt)
    state=manufactured(mesh,cohort)['exact'].copy()
    history=[]
    for step in range(steps):
        rhs=mesh.mass*state/dt
        rhs[mesh.boundary]=0
        state=np.linalg.solve(matrix,rhs)
        history.append(dict(step=step+1,state=state.copy(),rhs=rhs))
    return dict(state=state,dt=dt,final_time=float(final_time),history=history)
