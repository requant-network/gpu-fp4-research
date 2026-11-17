"""A priori bounded physical campaign; unavailable shapes stay in the matrix."""


def campaign_cases():
    return [dict(cohort=c,elements=e,q=q,heat=e<=2) for c in ('a','b') for e in (1,2,4,8) for q in (4,8,16)]


def configuration_specs():
    result=[dict(family='cuda',precision=p,threads=t) for p in ('fp32','fp64') for t in (128,256)]
    result.extend(dict(family='ceed',precision='fp64',backend=b,threads=256) for b in ('ref','shared','gen'))
    result.extend(dict(family='tensor',precision=p,threads=256) for p in ('fp4','fp8','fp16','bf16','fp32'))
    result.extend(dict(family='small',precision=p,threads=256) for p in ('fp8','fp16','bf16','fp32'))
    return result


def protocol(smoke=False):
    return dict(phase='smoke' if smoke else 'full',targets=[1e-3] if smoke else [1e-3,1e-8],
                resident_observations=4 if smoke else 16,host_observations=4 if smoke else 16,
                poisson_repetitions_per_route=1 if smoke else 2,heat_repetitions_per_route=1,
                policies=['fixed','reliable'],heat_steps=[2] if smoke else [4,8],heat_final_time=.01,
                max_iterations=1024,low_budget=32,segment_iterations=16,
                heat_scope='E1/2 at q4/8/16 only; operator/Poisson include all declared geometries',
                operator_inputs='Independent cohort-specific smooth sine field and sine plus changing dyadic rough perturbation with nonzero boundary',
                resident_scope='Already resident source, charged D2D staging plus complete operation and context acknowledgment',
                host_scope='H2D input plus complete operation plus D2H output',
                solver_scope='Common host-controlled FP64-vector Jacobi PCG, charged staging/reductions/true checks/fallback; not a best-of-all solver claim',
                observation_integrity='Full changing-original host/device fingerprints outside named service boundaries; these diagnostics alter inter-request spacing',
                retention='Every timed final output losslessly retained; first resident output of each variant additionally has complete GEMM-product/encoding witnesses',
                reuse=[1,100,10000],exclusive_device=False,instruction_trace=False,physical_peak_memory=False)
