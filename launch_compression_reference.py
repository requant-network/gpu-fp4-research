"""External process envelope for frozen compression producers and post-unwind failure capture."""


def measure_child(command,log,environment,cwd):
    import subprocess,time
    with log.open('wb') as output:
        started=time.perf_counter_ns()
        child=subprocess.run(command,env=environment,cwd=cwd,stdout=output,stderr=subprocess.STDOUT,check=False)
        elapsed=time.perf_counter_ns()-started
    return {'exit_code':child.returncode,'process_wall_ns':elapsed}


def failure_state(error):
    operation=None;boundary=None;cursor=error.__traceback__
    while cursor is not None:
        values=cursor.tb_frame.f_locals
        if {'cached','specs','frame_records','profile','dtype','seed','slots'}.issubset(values):
            operation=dict(values)
        if {'self','pending','table'}.issubset(values) and getattr(values['self'],'pending',None) is not None:
            boundary={'slot':values['self'],'pending':dict(values['pending']),'returned':values['table']}
        cursor=cursor.tb_next
    if operation is None or boundary is None:return None
    return {**boundary,'operation':operation}


def capture_failure(error,directory,*,to_host):
    import hashlib,json
    from pathlib import Path
    import numpy as np
    from compression_integrity import digest_chunks
    from compression_pair_cases import input_words
    state=failure_state(error)
    if state is None:return None
    context=state['operation'];slot=state['slot'];pending=state['pending'];frame=pending['frame']
    spec=context['specs'][frame];dtype=np.dtype(context['dtype']);wire=pending['wire_bytes']
    for other in context['slots']:
        if hasattr(other,'stream'):other.stream.synchronize()
    cached=context['cached'][frame].copy()
    provider=context.get('input_provider',input_words)
    regenerated=provider(context['profile'],dtype,spec['offset_words'],spec['bytes']//dtype.itemsize,context['seed']).reshape(spec['shape'])
    restored=to_host(slot.decoded[:spec['bytes']//dtype.itemsize]).copy().reshape(spec['shape'])
    returned=state['returned'].copy();expected=context['expected'][frame].copy()
    device_digests=to_host(slot.hash_gpu[:pending['hashes']]).copy()
    if slot.route['codec']=='plain':
        host_packet=cached.view(np.uint8).reshape(-1);device_packet=restored.view(np.uint8).reshape(-1)
    else:
        host_packet=(cached.view(np.uint8).reshape(-1) if slot.route['policy']=='precompressed' else slot.work[:wire].copy())
        device_packet=to_host(slot.encoded[:wire]).copy()
    word='<u4' if dtype.itemsize==4 else '<u8'
    previous=regenerated.view(word).reshape(-1);current=restored.view(word).reshape(-1)
    different=np.flatnonzero(previous!=current)
    arrays={'cached':cached,'regenerated':regenerated,'restored':restored,'expected':expected,
            'returned':returned,'device_digests':device_digests,'host_packet':host_packet,'device_packet':device_packet}
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);path=directory/'arrays.npz'
    np.savez_compressed(path,**arrays)
    result={'schema':'compression-post-unwind-failure-v1','frame':frame,'spec':spec,'route':slot.route,
            'profile':context['profile'],'dtype':dtype.str,'seed':context['seed'],
            'exception_type':type(error).__name__,'arrays_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'prepared_cache_sha256':context['frame_records'][frame]['cache_sha256'],
            'cache_matches_prepared_guard':hashlib.sha256(cached).hexdigest()==context['frame_records'][frame]['cache_sha256'],
            'expected_matches_regeneration':np.array_equal(expected,digest_chunks(regenerated,4096)),
            'restored_matches_regeneration':np.array_equal(previous,current),
            'restored_different_words':int(different.size),
            'gpu_digest_matches_cpu_restored':np.array_equal(device_digests,digest_chunks(restored,4096)),
            'returned_digest_matches_device':np.array_equal(returned,device_digests),
            'host_packet_matches_device':host_packet.tobytes()==device_packet.tobytes(),
            'first_differences':[{'word':int(i),'expected':int(previous[i]),'restored':int(current[i])} for i in different[:16]],
            'scope':'Observed arrays after frozen producer finally-block; not a physical-cause certificate'}
    (directory/'manifest.json').write_text(json.dumps(result,indent=2));return result


def main():
    import argparse,hashlib,json,os,runpy,sys,time
    from pathlib import Path
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ndzip-reference',choices=('ordered-copy','stock-serial'),required=True)
    parser.add_argument('--suite',choices=('smoke','paired','over-vram','above-probe'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--_child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    output=args.output.resolve()
    assert output.parent==(root/'results').resolve(),'Use an immediate project results collection'
    launch=output.with_name(output.name+'-launch')
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def clean(text):return text.replace(str(root),'<PROJECT_ROOT>').replace(str(Path.home()),'<HOME>')
    arguments=['--ndzip-reference',args.ndzip_reference,'--suite',args.suite,'--output',str(output.relative_to(root))]
    if args._child:
        sys.argv=[str(root/'check_compression_reference_pair.py'),*arguments]
        try:runpy.run_path(str(root/'check_compression_reference_pair.py'),run_name='__main__')
        except Exception as error:
            child_error={'exception_type':type(error).__name__,'message':clean(str(error)),
                         'capture_manifest_sha256':None,'capture_error':None}
            try:
                import cupy as cp
                capture=capture_failure(error,launch/'failure',to_host=cp.asnumpy)
                if capture is not None:child_error['capture_manifest_sha256']=sha(launch/'failure/manifest.json')
            except Exception as capture_error:
                child_error['capture_error']=clean(type(capture_error).__name__+': '+str(capture_error))
            (launch/'child-error.json').write_text(json.dumps(child_error,indent=2))
            raise
        return 0
    envelope_started=time.perf_counter_ns()
    assert not output.exists(),'Do not overwrite a previous collection'
    launch.mkdir(parents=True,exist_ok=False)
    from check_compression_reference_pair import SOURCES,prerequisites
    before={name:sha(root/name) for name in (*SOURCES,'launch_compression_reference.py')}
    accepted=json.loads((root/'results/compression-reference-ordered-smoke/manifest.json').read_text())
    assert {name:before[name] for name in SOURCES}==accepted['sources'],'Frozen producer changed'
    dependencies=prerequisites()
    private=root/'build/compression-launch-raw';private.mkdir(parents=True,exist_ok=True)
    raw=private/(output.name+'.log');assert not raw.exists()
    environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    record={'schema':'compression-reference-launch-v1','collection':str(output.relative_to(root)),
            'ndzip_reference':args.ndzip_reference,'suite':args.suite,'sources':before,'dependencies':dependencies,
            'python_binary_sha256':sha(sys.executable),'python_version':sys.version,
            'explicit_environment':{key:environment[key] for key in ('CUDA_VISIBLE_DEVICES','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
            'driver_arguments':arguments,'sealed':False}
    (launch/'manifest.json').write_text(json.dumps(record,indent=2))
    command=[sys.executable,str(root/'launch_compression_reference.py'),'--_child',*arguments]
    measured=measure_child(command,raw,environment,root)
    after={name:sha(root/name) for name in before};dependencies_after=prerequisites()
    record.update(measured)
    record['source_closure_valid']=before==after and dependencies==dependencies_after
    text=raw.read_text(errors='replace');public=launch/'child.log';public.write_text(clean(text))
    record['raw_private_log_sha256']=sha(raw);record['public_log_sha256']=sha(public)
    record['unhandled_error_marker']=any(marker in text for marker in ('Traceback (most recent call last):','Exception ignored in:','CUDA_ERROR_'))
    child_manifest=output/'manifest.json';record['collection_manifest_sha256']=sha(child_manifest) if child_manifest.exists() else None
    child=json.loads(child_manifest.read_text()) if child_manifest.exists() else {}
    record['collection_complete']=child.get('complete') is True
    if child:
        assert child['sources']=={name:before[name] for name in SOURCES}
        assert child['dependencies']==dependencies
    named=0
    for case in child.get('cases',[]):
        named+=sum(case[key] for key in ('setup_ns','prepare_ns','first_use_ns','control_replay_ns','post_guard_ns','allocator_cleanup_ns'))
        named+=sum(trial['host_ns'] for trial in case['trials'])
    record['summed_named_case_ns']=named;record['other_child_wall_ns']=measured['process_wall_ns']-named
    assert record['other_child_wall_ns']>=0,'Named costs overlap or exceed whole child wall'
    record['child_accepted']=measured['exit_code']==0 and record['collection_complete'] and record['source_closure_valid'] and not record['unhandled_error_marker']
    record['launcher_exit_code']=0 if record['child_accepted'] else (measured['exit_code'] or 65)
    record['parent_envelope_ns']=time.perf_counter_ns()-envelope_started
    record['scope']='Whole fresh child process includes Python startup/imports, frozen driver, source checks, data/archive I/O and teardown; parent audit separately scoped; failure capture included only for failures'
    record['sealed']=True;(launch/'manifest.json').write_text(json.dumps(record,indent=2))
    print('COMPRESSION_REFERENCE_LAUNCH',json.dumps({key:record[key] for key in ('child_accepted','exit_code','process_wall_ns','summed_named_case_ns','other_child_wall_ns')}),flush=True)
    return record['launcher_exit_code']


if __name__=='__main__':raise SystemExit(main())
