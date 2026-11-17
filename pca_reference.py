"""CPU-only stable covariance and independently merged streaming controls."""
import numpy as np


def validate(data,chunk):
    if data.dtype!=np.float32 or data.ndim!=2 or len(data)<2 or data.shape[1]<1 or not np.isfinite(data).all():
        raise ValueError('Expected at least two finite FP32 observations')
    if not isinstance(chunk,int) or chunk<1:
        raise ValueError('A positive integer chunk size is required')


def stable_reference(data,chunk=2048):
    """Two-pass FP64 centered covariance, normalized by N-1, full eigh."""
    validate(data,chunk)
    mean=np.mean(data,axis=0,dtype=np.float64)
    moment=np.zeros((data.shape[1],data.shape[1]),np.float64)
    for begin in range(0,len(data),chunk):
        centered=data[begin:begin+chunk].astype(np.float64)-mean
        moment+=centered.T@centered
    covariance=moment/(len(data)-1)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    return dict(mean=mean,covariance=covariance,eigenvalues=eigenvalues,eigenvectors=eigenvectors)


def streaming_reference(data,chunk=2048):
    """Independent chunk-centered moments plus Chan-style count/mean merge."""
    validate(data,chunk)
    mean=np.zeros(data.shape[1],np.float64)
    moment=np.zeros((data.shape[1],data.shape[1]),np.float64)
    count=0
    for begin in range(0,len(data),chunk):
        block=data[begin:begin+chunk].astype(np.float64)
        block_mean=block.mean(axis=0)
        centered=block-block_mean
        block_moment=centered.T@centered
        delta=block_mean-mean
        total=count+len(block)
        moment+=block_moment+np.outer(delta,delta)*(count*len(block)/total)
        mean+=delta*(len(block)/total)
        count=total
    return mean,moment/(count-1)


def quality(mean,covariance,eigenvalues,eigenvectors,reference,rank=16):
    """Separate covariance, weak spectrum, eigensolver and subspace errors."""
    target=reference['covariance']
    d=len(target)
    if rank<1 or rank>=d or covariance.shape!=target.shape or mean.shape!=(d,) or eigenvalues.shape!=(d,) or eigenvectors.shape!=(d,d):
        raise ValueError('Invalid PCA output geometry or target rank')
    arrays=(mean,covariance,eigenvalues,eigenvectors)
    if not all(np.isfinite(a).all() for a in arrays):
        return dict(finite=False)
    mean,covariance,eigenvalues,eigenvectors=(a.astype(np.float64) for a in arrays)
    tiny=np.finfo(np.float64).tiny
    scale=max(float(np.linalg.norm(target)),tiny)
    target_values=reference['eigenvalues']
    top_scale=max(float(np.max(abs(target_values))),tiny)
    # A declared absolute floor handles zero/repeated null eigenvalues without
    # letting a large leading component hide meaningful weak-eigenvalue loss.
    weak_floor=top_scale*1e-12
    spectral=np.abs(eigenvalues-target_values)/np.maximum(abs(target_values),weak_floor)
    u=eigenvectors[:,-rank:]
    v=reference['eigenvectors'][:,-rank:]
    projection=float(np.linalg.norm(u@u.T-v@v.T)/np.sqrt(2*rank))
    residual=covariance@eigenvectors-eigenvectors*eigenvalues[None,:]
    return dict(finite=True,covariance_relative_error=float(np.linalg.norm(covariance-target)/scale),
                maximum_scaled_eigenvalue_error=float(np.max(spectral)),weak_eigenvalue_absolute_floor=weak_floor,
                eigenvalues_relative_error=float(np.linalg.norm(eigenvalues-target_values)/max(float(np.linalg.norm(target_values)),tiny)),
                mean_standard_deviation_relative_error=float(np.linalg.norm(mean-reference['mean'])/max(float(np.sqrt(max(np.trace(target),0.))),tiny)),
                subspace_projection_error=projection,target_relative_eigengap=float((target_values[-rank]-target_values[-rank-1])/top_scale),
                eigensolver_relative_residual=float(np.linalg.norm(residual)/max(float(np.linalg.norm(covariance)),tiny)),
                eigenvector_orthogonality_error=float(np.linalg.norm(eigenvectors.T@eigenvectors-np.eye(d))/np.sqrt(d)),
                covariance_symmetry_error=float(np.linalg.norm(covariance-covariance.T)/max(float(np.linalg.norm(covariance)),tiny)),
                negative_eigenvalues=int(np.sum(eigenvalues < -top_scale*1e-12)))


def quality_leading(mean,covariance,eigenvalues,leading,reference):
    """Complete covariance/spectrum, and explicitly leading-only vector checks.

    Retaining the leading subspace avoids storing a redundant full eigenbasis
    for every timing observation. It does not certify unretained eigenvectors.
    """
    target=reference['covariance']
    d=len(target)
    if leading.ndim!=2 or leading.shape[0]!=d or not 1<=leading.shape[1]<d or covariance.shape!=(d,d) or mean.shape!=(d,) or eigenvalues.shape!=(d,):
        raise ValueError('Invalid retained PCA output geometry')
    if not all(np.isfinite(a).all() for a in (mean,covariance,eigenvalues,leading)):
        return dict(finite=False)
    mean,covariance,eigenvalues,u=(a.astype(np.float64) for a in (mean,covariance,eigenvalues,leading))
    rank=u.shape[1]
    tiny=np.finfo(np.float64).tiny
    spectrum=reference['eigenvalues']
    top_scale=max(float(np.max(abs(spectrum))),tiny)
    floor=top_scale*1e-12
    v=reference['eigenvectors'][:,-rank:]
    residual=covariance@u-u*eigenvalues[None,-rank:]
    return dict(finite=True,covariance_relative_error=float(np.linalg.norm(covariance-target)/max(float(np.linalg.norm(target)),tiny)),
                maximum_scaled_eigenvalue_error=float(np.max(abs(eigenvalues-spectrum)/np.maximum(abs(spectrum),floor))),
                weak_eigenvalue_absolute_floor=floor,
                eigenvalues_relative_error=float(np.linalg.norm(eigenvalues-spectrum)/max(float(np.linalg.norm(spectrum)),tiny)),
                mean_standard_deviation_relative_error=float(np.linalg.norm(mean-reference['mean'])/max(float(np.sqrt(max(np.trace(target),0.))),tiny)),
                subspace_projection_error=float(np.linalg.norm(u@u.T-v@v.T)/np.sqrt(2*rank)),
                target_relative_eigengap=float((spectrum[-rank]-spectrum[-rank-1])/top_scale),
                leading_eigensolver_relative_residual=float(np.linalg.norm(residual)/max(float(np.linalg.norm(covariance)),tiny)),
                leading_orthogonality_error=float(np.linalg.norm(u.T@u-np.eye(rank))/np.sqrt(rank)),
                covariance_symmetry_error=float(np.linalg.norm(covariance-covariance.T)/max(float(np.linalg.norm(covariance)),tiny)),
                negative_eigenvalues=int(np.sum(eigenvalues < -top_scale*1e-12)))
