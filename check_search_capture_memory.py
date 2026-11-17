"""GPU regression for Faiss scratch lifetime at full B128 search geometries.

A capture rejection or complete-output mismatch makes a graph ineligible; an
illegal address, process abort or an unusable uncaptured route fails the check.
This is not comparative performance evidence.
"""
import argparse
from pathlib import Path
import subprocess
import sys


def worker(args):
    import numpy as np
    from search_baselines import Baseline
    from search_cases import allow_mask,inputs
    from search_memory import faiss_reservation_bytes
    data,queries,_=inputs(args.dataset,'a',4,128)
    allowed=allow_mask(len(data),100,'a')
    plan=Baseline(data,queries[0],allowed,dict(family='faiss',precision=args.precision,use_cuvs=bool(args.cuvs)))
    try:
        info=plan.faiss_resources.getMemoryInfo()
        actual=info[0]['TemporaryMemoryBuffer'][1]
        assert actual>=faiss_reservation_bytes(len(data),128)
        plan.run()
        captured=False
        try:
            plan.capture(1)
            captured=True
        except Exception as error:
            print('CAPTURE_UNAVAILABLE',type(error).__name__,flush=True)
        matching=True
        for variant in range(4):
            plan.load(queries[variant].copy())
            if captured:
                plan.time(True)
                graph_distances,graph_ids=plan.result()
            plan.run()
            distances,ids=plan.result()
            assert ids.shape==distances.shape==(128,10)
            if captured:
                matching&=np.array_equal(graph_ids,ids) and np.array_equal(graph_distances,distances,equal_nan=True)
            print('CURRENT_QUERY_RUN_PASS',variant,flush=True)
        print('GRAPH_ELIGIBLE',captured and matching,'ACTUAL_TEMPORARY_BYTES',actual,flush=True)
    finally:plan.close()
    print('FAISS_WORKER_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--dataset',choices=('shifted','uniform-million'),default='shifted')
    parser.add_argument('--precision',choices=('fp32','fp16'),default='fp32')
    parser.add_argument('--cuvs',type=int,choices=(0,1),default=0)
    args=parser.parse_args()
    if args.worker:
        worker(args)
        return
    for name in ('shifted','uniform-million'):
        for precision in ('fp32','fp16'):
            for backend in (0,1):
                command=[sys.executable,str(Path(__file__).resolve()),'--worker','--dataset',name,'--precision',precision,'--cuvs',str(backend)]
                result=subprocess.run(command,capture_output=True,text=True,timeout=180)
                assert result.returncode==0,(command,result.returncode,result.stdout,result.stderr)
                assert result.stdout.count('CURRENT_QUERY_RUN_PASS')==4 and 'FAISS_WORKER_COMPLETE' in result.stdout,result.stdout
                forbidden=('cudaErrorIllegalAddress','an illegal memory access','context is destroyed','Segmentation fault','SIGSEGV')
                assert not any(s in result.stdout+result.stderr for s in forbidden),(result.stdout,result.stderr)
                details=next(line for line in result.stdout.splitlines() if line.startswith('GRAPH_ELIGIBLE'))
                print('FAISS_CAPTURE_SCRATCH_PROCESS_PASS',name,precision,backend,details,flush=True)
    print('Faiss B128 scratch/capture/changed-query regression PASS',flush=True)


if __name__=='__main__':main()
