"""Shared measured engine for plain pinned, ndzip and original XBP1 movement."""
import ctypes as C
import hashlib
import os
from pathlib import Path
import time

import numpy as np
from compression_pair_cases import frame_specs, input_words
from compression_integrity import DeviceIntegrity, digest_chunks
from compression_native import NdzipPlan
from xbp_native import XbpNative


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pinned(shape, dtype):
    import cupy as cp
    dtype = np.dtype(dtype)
    count = int(np.prod(shape, dtype=np.int64))
    owner = cp.cuda.PinnedMemoryPointer(cp.cuda.PinnedMemory(count*dtype.itemsize), 0)
    return np.frombuffer(owner, dtype).reshape(shape)


class Slot:
    def __init__(self, route, specs, dtype, integrity):
        import cupy as cp
        self.route, self.dtype, self.integrity = route, np.dtype(dtype), integrity
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.codec = route['codec']; self.pending = None; self.plans = {}
        self.xbp = XbpNative() if self.codec == 'xbp' else None
        shapes = sorted({tuple(s['shape']) for s in specs})
        with self.stream:
            if self.codec == 'ndzip':
                self.plans = {shape: NdzipPlan(shape, self.dtype, route['cpu_threads'], self.stream) for shape in shapes}
                self.capacity = max(p.bound_words*self.dtype.itemsize for p in self.plans.values())
            elif self.codec == 'xbp':
                self.capacity = max(self.xbp.bound(int(np.prod(shape)), self.dtype) for shape in shapes)
            else:
                self.capacity = 0
            largest = max(s['bytes'] for s in specs)
            self.decoded = cp.empty(largest//self.dtype.itemsize, self.dtype)
            self.encoded = cp.empty(self.capacity, cp.uint8) if self.capacity else None
            self.work = pinned((self.capacity,), np.uint8) if self.capacity else None
            self.hash_gpu = cp.empty((max(1,(largest+4095)//4096),32), cp.uint8)
            self.hash_host = pinned(self.hash_gpu.shape, np.uint8)
            self.status_gpu = cp.empty(1,cp.uint32) if self.xbp else None
            self.status_host = pinned((1,),np.uint32) if self.xbp else None

    def encode(self, original):
        if self.codec == 'ndzip':
            plan = self.plans[tuple(original.shape)]; words = C.c_uint64()
            plan._check(plan.lib.ndz_encode(plan.handle, original.ctypes.data, self.work.ctypes.data,
                                            self.capacity//self.dtype.itemsize, C.byref(words)))
            assert 0 < words.value <= plan.bound_words
            return self.work[:words.value*self.dtype.itemsize], None
        packet = self.xbp.encode(original,threads=self.route['cpu_threads'],out=self.work)
        return packet, self.xbp.validate(packet)

    def enqueue(self, frame, spec, cached, metadata, events):
        import cupy as cp
        begin = time.perf_counter_ns()
        if self.codec != 'plain' and self.route['policy'] == 'online':
            host, metadata = self.encode(cached)
            encode_ns = time.perf_counter_ns()-begin
        else:
            host = cached; encode_ns = 0
        count = spec['bytes']//self.dtype.itemsize
        view = self.decoded[:count].reshape(spec['shape'])
        hashes = max(1,(spec['bytes']+4095)//4096)
        with self.stream:
            a,b,c,d = events
            a.record(self.stream)
            destination = self.decoded if self.codec == 'plain' else self.encoded
            cp.cuda.runtime.memcpyAsync(destination.data.ptr,host.ctypes.data,host.nbytes,
                                        cp.cuda.runtime.memcpyHostToDevice,self.stream.ptr)
            if self.codec == 'ndzip':
                plan=self.plans[tuple(spec['shape'])]
                plan.decode(self.encoded.view(plan.word_dtype)[:host.nbytes//self.dtype.itemsize],view)
            elif self.codec == 'xbp':
                self.xbp.decode(self.encoded[:host.nbytes],view,metadata,self.status_gpu)
            b.record(self.stream)
            self.integrity.hash_into(view,self.hash_gpu[:hashes])
            c.record(self.stream)
            cp.cuda.runtime.memcpyAsync(self.hash_host.ctypes.data,self.hash_gpu.data.ptr,hashes*32,
                                        cp.cuda.runtime.memcpyDeviceToHost,self.stream.ptr)
            if self.xbp:
                cp.cuda.runtime.memcpyAsync(self.status_host.ctypes.data,self.status_gpu.data.ptr,4,
                                            cp.cuda.runtime.memcpyDeviceToHost,self.stream.ptr)
            d.record(self.stream)
        self.pending={'frame':frame,'wire_bytes':host.nbytes,'cpu_encode_validate_ns':encode_ns,
                      'hashes':hashes,'shape':spec['shape'],'events':events}

    def finish(self, expected):
        if self.pending is None:
            return None
        pending=self.pending
        if self.codec == 'ndzip':
            self.plans[tuple(pending['shape'])].synchronize()
        else:
            self.stream.synchronize()
        error=int(self.status_host[0]) if self.xbp else 0
        assert error == 0, ('XBP GPU format/range guard',error)
        table=self.hash_host[:pending['hashes']].copy()
        assert np.array_equal(table,expected[pending['frame']]), 'Paired pipeline output differs from original input'
        self.pending=None
        return pending,table

    def close(self):
        self.stream.synchronize()
        for plan in self.plans.values():
            plan.close()

    def memory(self):
        arrays=(self.decoded,self.encoded,self.hash_gpu,self.status_gpu)
        hosts=(self.work,self.hash_host,self.status_host)
        return {'requested_device_bytes':sum(a.nbytes for a in arrays if a is not None),
                'requested_pinned_slot_bytes':sum(a.nbytes for a in hosts if a is not None),
                'candidate_bytes':self.capacity,
                'opaque_library_cpu_workspace_measured':False}


def packet_control(directory, packet):
    packet=np.asarray(packet).view(np.uint8).reshape(-1)
    result={'bytes':packet.nbytes,'sha256':sha(packet),'retained_file':None}
    if packet.nbytes <= 8*1024**2:
        relative=f'packets/{result["sha256"]}.npy'; path=Path(directory)/relative
        path.parent.mkdir(exist_ok=True)
        assert not path.is_symlink()
        if path.exists():
            assert np.array_equal(np.load(path,allow_pickle=False),packet)
        else:
            np.save(path,packet,allow_pickle=False)
        result['retained_file']={'file':relative,'sha256':sha(path.read_bytes())}
    return result


def run_route(directory, store, cohort, dtype, profile, route, config, physical_vram):
    import cupy as cp
    specs=frame_specs(dtype,config['total_bytes'],config['frame_bytes'])
    max_bytes=max(s['bytes'] for s in specs)
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:')))*1024
    assert config['total_bytes']*1.15+max_bytes*20 < available*.65, 'Insufficient conservative host-memory margin'
    before=time.perf_counter_ns(); integrity=DeviceIntegrity(4096)
    slots=[Slot(route,specs,dtype,integrity) for _ in range(route['queue_depth'])]
    events=[[[cp.cuda.Event() for _ in range(4)] for _ in specs] for _ in range(config['repeats'])]
    origins=[cp.cuda.Event() for _ in range(config['repeats'])]
    setup_ns=time.perf_counter_ns()-before
    cached=[]; metadata=[]; expected=[]; frame_records=[]
    prepare_start=time.perf_counter_ns(); input_ns=encode_ns=cache_ns=0
    seed=71 if cohort=='a' else 171
    try:
        for spec in specs:
            start=time.perf_counter_ns()
            data=input_words(profile,dtype,spec['offset_words'],spec['bytes']//np.dtype(dtype).itemsize,seed).reshape(spec['shape'])
            table=digest_chunks(data,4096); input_hash=sha(data)
            input_ns+=time.perf_counter_ns()-start
            packet=data; meta=None
            if route['codec']!='plain' and route['policy']=='precompressed':
                start=time.perf_counter_ns(); packet,meta=slots[0].encode(data)
                encode_ns+=time.perf_counter_ns()-start
            start=time.perf_counter_ns(); source=pinned(packet.shape,packet.dtype);np.copyto(source,packet);source.flags.writeable=False
            cache_ns+=time.perf_counter_ns()-start
            cached.append(source);metadata.append(meta);expected.append(table)
            frame_records.append({**spec,'input_sha256':input_hash,'cache_bytes':source.nbytes,
                                  'cache_sha256':sha(source),'expected_digest_sha256':sha(table)})
            del data,packet,table,source
        prepare_ns=time.perf_counter_ns()-prepare_start
        intervals=sorted((a.ctypes.data,a.ctypes.data+a.nbytes) for a in cached)
        assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])), 'Prepared host frames alias each other'
        for slot in slots:
            if slot.work is not None:
                low,high=slot.work.ctypes.data,slot.work.ctypes.data+slot.work.nbytes
                assert all(high<=a or low>=b for a,b in intervals), 'Encoding buffer aliases prepared source'
        relative_intervals=[{'begin':a-intervals[0][0],'bytes':b-a} for a,b in intervals]
        first=time.perf_counter_ns()
        warm_events=[[cp.cuda.Event() for _ in range(4)] for _ in slots]
        for slot,ev in zip(slots,warm_events):
            slot.enqueue(0,specs[0],cached[0],metadata[0],ev);slot.finish(expected)
        first_use_ns=time.perf_counter_ns()-first
        trials=[]
        for repeat in range(config['repeats']):
            origin=origins[repeat];origin.record();origin.synchronize()
            for slot in slots:slot.stream.wait_event(origin)
            captured=[];begin=time.perf_counter_ns()
            for frame,spec in enumerate(specs):
                slot=slots[frame%len(slots)]
                done=slot.finish(expected)
                if done is not None:captured.append(done)
                slot.enqueue(frame,spec,cached[frame],metadata[frame],events[repeat][frame])
            for slot in slots:
                done=slot.finish(expected)
                if done is not None:captured.append(done)
            host_ns=time.perf_counter_ns()-begin
            observations=[]
            for pending,table in sorted(captured,key=lambda x:x[0]['frame']):
                a,b,c,d=pending['events']
                observations.append({'frame':pending['frame'],'wire_bytes':pending['wire_bytes'],
                                     'cpu_encode_validate_ns':pending['cpu_encode_validate_ns'],
                                     'device_error_bits':0,'witness':store.put(table),
                                     'start_ms':cp.cuda.get_elapsed_time(origin,a),
                                     'ingest_end_ms':cp.cuda.get_elapsed_time(origin,b),
                                     'hash_end_ms':cp.cuda.get_elapsed_time(origin,c),
                                     'result_end_ms':cp.cuda.get_elapsed_time(origin,d)})
            assert len(observations)==len(specs)
            trials.append({'repeat':repeat,'host_ns':host_ns,'observations':observations,
                           'wire_bytes':sum(o['wire_bytes'] for o in observations)})
        start=time.perf_counter_ns();controls=[]
        for frame,spec in enumerate(specs):
            if route['codec']=='plain':control={'bytes':spec['bytes'],'sha256':frame_records[frame]['input_sha256'],'retained_file':None}
            else:
                packet,_=(slots[0].encode(cached[frame]) if route['policy']=='online' else (cached[frame],metadata[frame]))
                control=packet_control(directory,packet)
            assert all(t['observations'][frame]['wire_bytes']==control['bytes'] for t in trials)
            controls.append(control)
        control_replay_ns=time.perf_counter_ns()-start
        start=time.perf_counter_ns()
        assert all(sha(source)==desc['cache_sha256'] for source,desc in zip(cached,frame_records)),'Prepared host input changed'
        post_guard_ns=time.perf_counter_ns()-start
        return {'cohort':cohort,'dtype':np.dtype(dtype).str,'profile':profile,'seed':seed,'route':route,
                'total_bytes':config['total_bytes'],'frame_bytes':config['frame_bytes'],'frames':frame_records,
                'logical_source_exceeds_vram':config['total_bytes']>physical_vram,
                'physical_vram_bytes':physical_vram,'nonoverlapping_host_intervals':relative_intervals,
                'setup_ns':setup_ns,'prepare_ns':prepare_ns,'prepare_input_reference_ns':input_ns,
                'prepare_encoding_ns':encode_ns,'prepare_cache_ns':cache_ns,
                'prepare_other_ns':prepare_ns-input_ns-encode_ns-cache_ns,'first_use_ns':first_use_ns,
                'control_replay_ns':control_replay_ns,'post_guard_ns':post_guard_ns,
                'memory':{'slots':[s.memory() for s in slots],
                          'requested_pinned_source_bytes':sum(a.nbytes for a in cached),
                          'expected_digest_bytes':sum(a.nbytes for a in expected),
                          'physical_peak_measured':False},
                'encoded_controls':controls,'encoded_control_scope':'online controls are untimed deterministic reencoding; precompressed controls are actual cached input',
                'trials':trials}
    finally:
        for slot in slots:slot.close()


def loaded_libraries():
    """Actual mapped objects, with paths redacted; no causal runtime-loader attribution."""
    records={}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split()
        if len(fields)<6 or not fields[-1].startswith('/') or '.so' not in fields[-1]:continue
        path=Path(fields[-1])
        if not path.is_file():continue
        key=str(path.resolve())
        if key in records:continue
        h=hashlib.sha256()
        with path.open('rb') as stream:
            while block:=stream.read(1024*1024):h.update(block)
        records[key]={'basename':path.name,'sha256':h.hexdigest(),'bytes':path.stat().st_size}
    return sorted(records.values(),key=lambda r:(r['basename'],r['sha256']))
