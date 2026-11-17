"""GPU regression: filtered CAGRA cleanup must exit cleanly after query changes.

This is a correctness/lifetime check, not comparative performance evidence.
Run on the declared device with CUDA_VISIBLE_DEVICES=0.
"""
import argparse
from pathlib import Path
import subprocess
import sys


def worker():
    import numpy as np
    from search_baselines import Baseline
    from search_reference import evaluate,reference_search
    rng=np.random.Generator(np.random.PCG64(20261130))
    data=rng.integers(-2048,2049,(8192,128)).astype(np.float32)/4096
    queries=rng.integers(-2048,2049,(4,128)).astype(np.float32)/4096
    for iteration in range(4):
        allowed=np.arange(len(data))%10==iteration
        plan=Baseline(data,queries,allowed,dict(family='cagra',itopk=256))
        try:
            plan.run()
            first=plan.result()[1].copy()
            for query in (queries,np.ascontiguousarray(-queries)):
                plan.load(query)
                plan.run()
                distances,ids=plan.result()
                gold,truth=reference_search(data,query,10,allowed)
                metrics,exact=evaluate(data,query,ids,gold,truth,allowed)
                assert metrics['duplicate_ids']==metrics['invalid_or_filtered_ids']==0
                np.testing.assert_allclose(distances,exact,rtol=3e-6,atol=1e-5)
            assert not np.array_equal(first,ids)
            try:
                plan.capture(1)
            except Exception:
                # Filter population counting is not capturable in this pinned API.
                pass
            plan.load(queries)
            plan.run()
            np.testing.assert_array_equal(plan.result()[1],first)
        finally:
            plan.close()
        print('ROUND_PASS',iteration,flush=True)
    print('LIFETIME_WORKER_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.worker:
        worker()
        return
    if args.repeats<1:
        raise ValueError('At least one subprocess is required')
    for iteration in range(args.repeats):
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],capture_output=True,text=True,timeout=120)
        assert result.returncode==0,(iteration,result.returncode,result.stdout,result.stderr)
        assert result.stdout.count('ROUND_PASS')==4 and 'LIFETIME_WORKER_COMPLETE' in result.stdout,result.stdout
        forbidden=('context is destroyed','cudaFreeAsync(device_payload','Segmentation fault','SIGSEGV')
        assert not any(message in result.stdout+result.stderr for message in forbidden),(iteration,result.stdout,result.stderr)
        print('CLEAN_PROCESS_EXIT_PASS',iteration,flush=True)
    print('Filtered CAGRA lifetime and changed-query regression PASS',flush=True)


if __name__=='__main__':
    main()
