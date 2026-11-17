"""Executed matrix-free Poisson and implicit heat solver preflight.

Fixed low-precision recursive convergence is NOT physical convergence. Every
retained final vector is checked against the original independent CPU operator.
"""
import hashlib
import json
from pathlib import Path
import time
import cupy as cp
import numpy as np
from pde_reference import Mesh,apply_operator,manufactured
from pde_cg import jacobi_diagonal,solve,solve_reliable
from pde_methods import configurations,create,DeviceOperator
from reference import digest

ROOT=Path(__file__).resolve().parent


def main():
    cp.cuda.Device(0).use()
    assert cp.cuda.runtime.getDeviceProperties(0)['name'].decode()=='NVIDIA GeForce RTX 5060 Ti'
    mesh=Mesh(2,8)
    rows=[];arrays={}
    for task,reaction in (('poisson',0.),('heat',100.)):
        original_plan=create(mesh,dict(family='cuda',precision='fp64',threads=256),reaction)
        original_apply=DeviceOperator(original_plan)
        diagonal=cp.asarray(jacobi_diagonal(mesh,reaction))
        try:
            for config in configurations():
                plan=create(mesh,config,reaction)
                operator=DeviceOperator(plan)
                try:
                    plan.load(manufactured(mesh,'a')['exact'])
                    if hasattr(plan,'calibrate'):plan.calibrate()
                    for cohort in ('a','b'):
                        physical=manufactured(mesh,cohort)
                        exact=physical['exact'] if task=='poisson' else manufactured(mesh,cohort,time=.01)['exact']
                        host_rhs=physical['rhs'] if task=='poisson' else mesh.mass*physical['exact']*reaction
                        host_rhs[mesh.boundary]=0;host_rhs.setflags(write=False)
                        identity=digest(host_rhs);rhs=cp.asarray(host_rhs)
                        for policy in ('fixed','reliable'):
                            before=(operator.calls,original_apply.calls)
                            started=time.perf_counter()
                            if policy=='fixed':result=solve(operator,rhs,diagonal,xp=cp,tolerance=1e-8,max_iterations=512)
                            else:result=solve_reliable(operator,original_apply,rhs,diagonal,xp=cp,tolerance=1e-8,max_iterations=512)
                            solution=cp.asnumpy(result.pop('x'))
                            cp.cuda.runtime.deviceSynchronize()
                            elapsed=(time.perf_counter()-started)*1000
                            assert np.isfinite(solution).all(),'Nonfinite solver state must be investigated, not claimed converged'
                            residual=host_rhs-apply_operator(mesh,solution,reaction)
                            original_relative=float(np.linalg.norm(residual)/np.linalg.norm(host_rhs))
                            mass_error=float(np.sqrt(np.sum(mesh.mass*(solution-exact)**2)/np.sum(mesh.mass*exact**2)))
                            if policy=='reliable':assert original_relative<=1.01e-8,('Reliable solver missed original target',config,cohort,original_relative)
                            if config['family'] in ('ceed',) or config.get('precision')=='fp64':
                                assert original_relative<=1.01e-8,('High-precision reference missed original target',config,cohort,original_relative)
                            assert digest(host_rhs)==identity and digest(cp.asnumpy(rhs))==identity
                            key='x'+str(len(rows));arrays[key]=solution
                            rows.append(dict(task=task,reaction=reaction,cohort=cohort,configuration=config,policy=policy,array=key,
                                             input_sha256=identity,output_sha256=digest(solution),cpu_original_relative=original_relative,
                                             continuum_mass_relative=mass_error,wall_ms_including_output_copy=elapsed,
                                             actual_operator_calls=operator.calls-before[0],actual_original_calls=original_apply.calls-before[1],controller=result))
                finally:plan.close()
        finally:original_plan.close()
    np.savez_compressed(ROOT/'results/pde-solver-preflight.npz',**arrays)
    sources=('check_pde_solver.py','pde_methods.py','pde_cg.py','pde_reference.py','pde_coefficients.py','prepare_pde_coefficients.py','reference.py',
             'pde_cuda.py','native/pde_kernels.cu','pde_ceed.py','native/pde_ceed_bridge.cpp','native/pde_ceed_qfunction.h','build_pde_ceed.sh',
             'pde_tensor.py','native/pde_tensor_kernels.cu','pde_packing.py','pca_native.py','native/pca_bridge.cu','native/hardware_bridge.cu',
             'native/application_bridge.cu','native/optimized_bridge.cu','native/bridge.cu','build_pca.sh','pde_small.py','native/pde_small_bridge.cu',
             'native/pde_small_kernels.cu','build_pde_small.sh')
    result=dict(scope='One E2/q8 Poisson and one-step heat complete-solver preflight, not full spatial/temporal/latency study',cases=rows,
                source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources},
                data_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('fixtures/pde-gll.json','fixtures/pde-gll.npz','results/pde-solver-preflight.npz')})
    (ROOT/'results/pde-solver-preflight.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('PDE_ORIGINAL_SYSTEM_SOLVER_PREFLIGHT_PASS',len(rows),flush=True)


if __name__=='__main__':main()
