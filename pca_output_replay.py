"""CPU-only PCA truth/final-output replay, independent of the GPU producer."""
import numpy as np

from pca_final_replay import replay_case_outputs


def replay_truth(data,target):
    """Recompute all original-data covariance entries with interleaved reductions.

    Retained full reference eigenvectors are certified against their independently
    reconstructed covariance. A near-degenerate basis is not compared elementwise
    with a different LAPACK implementation's arbitrary basis/sign choices.
    """
    n,d=data.shape
    assert data.dtype==np.float32 and n>=2 and np.isfinite(data).all()
    assert set(target)=={'mean','covariance','eigenvalues','eigenvectors'}
    for field,shape in (('mean',(d,)),('covariance',(d,d)),('eigenvalues',(d,)),('eigenvectors',(d,d))):
        assert target[field].shape==shape and target[field].dtype==np.float64
        assert np.isfinite(target[field]).all()
    values=data.astype(np.float64)
    mean=(values[::2].sum(axis=0)+values[1::2].sum(axis=0))/n
    epsilon=2.**-53
    gamma=n*epsilon/(1-n*epsilon)
    mean_bound=8*gamma*np.mean(abs(values),axis=0)+np.finfo(np.float64).tiny
    assert np.all(abs(target['mean']-mean)<=mean_bound),'Original-data truth mean mismatch'
    centered=values-mean
    even,odd=centered[::2].copy(),centered[1::2].copy()
    covariance=(even.T@even+odd.T@odd)/(n-1)
    absolute=(abs(even).T@abs(even)+abs(odd).T@abs(odd))/(n-1)
    average_absolute=np.mean(abs(centered),axis=0)
    centering_bound=mean_bound+4*epsilon*np.mean(abs(values)+abs(mean),axis=0)
    bound=16*gamma*absolute+(n/(n-1))*(np.outer(centering_bound,average_absolute)+np.outer(average_absolute,centering_bound)+np.outer(centering_bound,centering_bound))
    assert np.all(abs(target['covariance']-covariance)<=bound+np.finfo(np.float64).tiny),'Original-data truth covariance mismatch'
    cov=target['covariance']
    scale=max(float(np.linalg.norm(cov)),np.finfo(np.float64).tiny)
    assert np.linalg.norm(cov-cov.T)<=64*d*epsilon*scale
    eigenvalues,eigenvectors=target['eigenvalues'],target['eigenvectors']
    assert np.all(np.diff(eigenvalues)>=0)
    independently_solved=np.linalg.eigvalsh((cov+cov.T)*.5)
    assert np.max(abs(eigenvalues-independently_solved))<=64*d*epsilon*scale,'Truth spectrum does not solve its covariance'
    residual=cov@eigenvectors-eigenvectors*eigenvalues[None,:]
    assert np.linalg.norm(residual)<=64*d*epsilon*scale,'Truth full eigenbasis residual is invalid'
    assert np.linalg.norm(eigenvectors.T@eigenvectors-np.eye(d))<=64*d*epsilon
    return dict(truth_mean_entries=d,truth_covariance_entries=d*d,truth_spectrum_entries=d,truth_eigenvector_entries=d*d)
