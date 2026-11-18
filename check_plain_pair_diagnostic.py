"""Capture all plain-transfer failure boundaries without changing frozen measured producers."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from check_compression_reference_pair import SOURCES as BASE_SOURCES,prerequisites,ROOT,digest
from compression_reference_engine import Slot,pinned,loaded_libraries
from compression_pair_cases import frame_specs,input_words
from compression_pair_protocol import WitnessStore
from compression_integrity import DeviceIntegrity,digest_chunks
from compression_plain_census import census

SOURCES=(*BASE_SOURCES,'compression_plain_census.py','check_plain_pair_diagnostic.py')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--frames',type=int,default=320)
    parser.add_argument('--depth',type=int,choices=(1,2),default=2);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert 2<=args.frames<=320
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    total=args.frames*64*1024**2
    assert total*1.15+64*1024**2*20<available*.65,'Insufficient conservative host-memory margin'
    import cupy as cp
    cp.cuda.Device(0).use();args.output.mkdir(parents=True,exist_ok=False)
    props=cp.cuda.runtime.getDeviceProperties(0)
    device={'name':props['name'].decode() if isinstance(props['name'],bytes) else props['name'],
            'total_memory_bytes':int(props['totalGlobalMem']),'major':int(props['major']),'minor':int(props['minor']),
            'runtime_api':cp.cuda.runtime.runtimeGetVersion(),'driver_api':cp.cuda.runtime.driverGetVersion()}
    specs=frame_specs('<f4',total,64*1024**2)
    route={'codec':'plain','policy':'online','cpu_threads':1,'queue_depth':args.depth}
    source_map={name:digest(ROOT/name) for name in SOURCES};dependencies=prerequisites()
    integrity=DeviceIntegrity(4096);slots=[Slot(route,specs,'<f4',integrity) for _ in range(args.depth)]
    events=[[cp.cuda.Event() for _ in range(4)] for _ in specs];origin=cp.cuda.Event()
    cached=[];expected=[];guards=[];captured=[];phase='prepare'
    record={'schema':'plain-pair-failure-diagnostic-v1','device':device,'route':route,
            'frames_requested':args.frames,'total_bytes':total,'sources':source_map,'dependencies':dependencies,
            'frame_specs':specs,'mismatch':None,'observations':[],'complete':False,
            'scope':'Boundary census with full failure arrays; no throughput or physical-cause certificate'}
    manifest=args.output/'manifest.json';manifest.write_text(json.dumps(record,indent=2))
    store=WitnessStore(args.output)
    def finish(slot):
        if slot.pending is None:return True
        pending=slot.pending.copy();frame=pending['frame'];spec=specs[frame]
        try:
            done=slot.finish(expected)
            if phase=='service':captured.append(done)
        except AssertionError:
            for other in slots:other.stream.synchronize()
            original=cached[frame].copy()
            regenerated=input_words('smooth','<f4',spec['offset_words'],spec['bytes']//4,71).reshape(spec['shape'])
            restored=cp.asnumpy(slot.decoded[:spec['bytes']//4]).reshape(spec['shape'])
            host=slot.hash_host[:pending['hashes']].copy();gpu=cp.asnumpy(slot.hash_gpu[:pending['hashes']])
            path=args.output/'failure.npz'
            np.savez_compressed(path,original=original,regenerated=regenerated,restored=restored,
                                expected=expected[frame],returned=host,gpu=gpu)
            record['mismatch']={'frame':frame,'phase':phase,'failure_sha256':digest(path),
                'census':census(original,regenerated,restored,expected[frame],host,gpu,guards[frame])}
            return False
        return True
    try:
        for spec in specs:
            data=input_words('smooth','<f4',spec['offset_words'],spec['bytes']//4,71).reshape(spec['shape'])
            source=pinned(data.shape,data.dtype);np.copyto(source,data);source.flags.writeable=False
            cached.append(source);expected.append(digest_chunks(data,4096));guards.append(hashlib.sha256(source).hexdigest())
        intervals=sorted((a.ctypes.data,a.ctypes.data+a.nbytes) for a in cached)
        assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
        for slot in slots:
            lo=slot.hash_host.ctypes.data;hi=lo+slot.hash_host.nbytes
            assert all(hi<=a or lo>=b for a,b in intervals),'Returned hash buffer aliases prepared source'
        phase='warm'
        for slot in slots:
            slot.enqueue(0,specs[0],cached[0],None,[cp.cuda.Event() for _ in range(4)])
            if not finish(slot):break
        if record['mismatch'] is None:
            phase='service';origin.record();origin.synchronize()
            for slot in slots:slot.stream.wait_event(origin)
            for frame,spec in enumerate(specs):
                slot=slots[frame%len(slots)]
                if not finish(slot):break
                slot.enqueue(frame,spec,cached[frame],None,events[frame])
            if record['mismatch'] is None:
                for slot in slots:
                    if not finish(slot):break
        for pending,table in sorted(captured,key=lambda pair:pair[0]['frame']):
            record['observations'].append({'frame':pending['frame'],'witness':store.put(table)})
        record['all_host_sources_unchanged']=all(hashlib.sha256(a).hexdigest()==guard for a,guard in zip(cached,guards))
        record['source_guards']=guards
        record['complete']=record['mismatch'] is None and [o['frame'] for o in record['observations']]==list(range(args.frames))
        assert source_map=={name:digest(ROOT/name) for name in SOURCES}
        assert dependencies==prerequisites()
        record['loaded_libraries']=loaded_libraries();manifest.write_text(json.dumps(record,indent=2))
        print('PLAIN_FAILURE_DIAGNOSTIC',json.dumps({'complete':record['complete'],'observations':len(record['observations']),
                                                   'mismatch':record['mismatch'],'sources_unchanged':record['all_host_sources_unchanged']}),flush=True)
    finally:
        for slot in slots:slot.close()
    raise SystemExit(0 if record['complete'] else 1)
