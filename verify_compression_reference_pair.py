"""Independent streaming CPU replay of every timed three-way content fingerprint."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from check_compression_reference_pair import SOURCES,REFERENCE_FILES
from compression_reference_epoch import reference_config
from compression_pair_cases import suite_spec,frame_specs,input_words
from compression_pair_models import xbp_size,ndzip_bound_bytes
from xbp_reference import encode_reference,decode_reference

ROOT=Path(__file__).resolve().parent
HEX=re.compile(r'[0-9a-f]{64}\Z')


def digest(data):return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):h.update(block)
    return h.hexdigest()


def safe_file(root,relative):
    path=root/relative
    assert not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'Unsafe artifact path'
    return path


def original_digests(original):
    raw=original.view(np.uint8).reshape(-1)
    return np.asarray([np.frombuffer(hashlib.sha256(raw[i:i+4096]).digest(),np.uint8)
                       for i in range(0,raw.size,4096)],np.uint8)


def verify_collection(directory):
    root=Path(directory);manifest=root/'manifest.json';record=json.loads(manifest.read_text())
    assert record['schema']=='compression-pair-reference-suite-v2' and record['complete'] is True
    config=json.loads(json.dumps(reference_config(record['suite'],record['ndzip_reference'])))
    assert record['config']==config,'Fixed suite declaration differs'
    assert set(record['sources'])==set(SOURCES)
    for source,expected in record['sources'].items():assert file_digest(ROOT/source)==expected,source
    dependencies=record['dependencies'];expected_proofs={
        f'results/{name}/manifest.json' for name in ('compression-native-preflight','xbp-native-preflight',
                                                  'compression-integrity-preflight','compression-ndzip-cli-preflight')}
    assert set(dependencies['proofs'])==expected_proofs
    for source,expected in dependencies['proofs'].items():assert file_digest(ROOT/source)==expected,source
    vendor=json.loads((ROOT/'results/compression-ndzip-cli-preflight/manifest.json').read_text())
    assert dependencies['vendor_commit']==vendor['upstream_commit']=='ff4e6702bf0abb86d4aeef8249bd30344dfeef75'
    assert dependencies['vendor_sources']==vendor['vendor_sources'] and dependencies['vendor_binaries']==vendor['binaries']
    assert record['device']['name']=='NVIDIA GeForce RTX 5060 Ti'
    assert record['device']['major']==12 and record['device']['minor']==0
    assert 15*1024**3<record['device']['total_memory_bytes']<20*1024**3
    mapped=record['loaded_libraries'];assert mapped
    for library in mapped:
        assert '/' not in library['basename'] and '\\' not in library['basename']
        assert HEX.fullmatch(library['sha256']) and library['bytes']>0
    mapped_hashes={x['sha256'] for x in mapped}
    ordered=dependencies['ordered_reference']
    assert set(ordered['proofs'])==set(REFERENCE_FILES)
    for name,expected in ordered['proofs'].items():assert file_digest(ROOT/name)==expected,name
    build=json.loads((ROOT/'results/ndzip-ordered-build-evidence/manifest.json').read_text())
    assert ordered['build']==build and build['complete'] is True
    assert build['reference']=='ndzip-ff4e6702+ordered-copy'
    native=json.loads((ROOT/'results/ndzip-ordered-native/manifest.json').read_text())
    replay=json.loads((ROOT/'results/ndzip-ordered-native-replay.json').read_text())
    assert native['complete'] is True and native['build']==build
    assert native['build_manifest_sha256']==file_digest(ROOT/'results/ndzip-ordered-build-evidence/manifest.json')
    assert replay['manifest_sha256']==file_digest(ROOT/'results/ndzip-ordered-native/manifest.json')
    for name,expected in native['sources'].items():assert file_digest(ROOT/name)==expected,name
    for name,expected in replay['verifiers'].items():assert file_digest(ROOT/name)==expected,name
    own=json.loads((ROOT/'results/xbp-native-preflight/manifest.json').read_text())
    assert own['binary_sha256'] in mapped_hashes,'Original-codec binary absent from mapped object receipt'
    if record['ndzip_reference']=='ordered-copy':
        active={Path(name).name:value for name,value in build['binaries'].items() if name.endswith('.so')}
        forbidden='libcompression_ndzip.so'
    else:
        assert record['ndzip_reference']=='stock-serial'
        active={name:value for name,value in vendor['binaries'].items() if name.endswith('.so')}
        active['libcompression_ndzip.so']=json.loads((ROOT/'results/compression-native-preflight/manifest.json').read_text())['binary_sha256']
        forbidden='libcompression_ndzip_ordered.so'
    assert not any(obj['basename']==forbidden for obj in mapped),'Mixed reference objects'
    for name,expected in active.items():
        assert [obj['sha256'] for obj in mapped if obj['basename']==name]==[expected],'Wrong/mixed active reference object'

    declared=[]
    for cohort,dtype,profile in config['datasets']:
        routes=list(config['routes'])
        if cohort=='b':routes.reverse()
        declared.extend((cohort,dtype,profile,r) for r in routes)
    assert len(record['cases'])==len(declared),'Missing declared route cases'
    totals={'route_cases':len(declared),'timed_frames':0,'timed_restored_input_bytes':0,
            'observed_digest_bytes':0,'wire_bytes':0,'retained_encoded_control_bindings':0,
            'unretained_encoded_control_bindings':0,'unique_original_bytes_replayed':0}
    grouped={}
    for index,(case,identity) in enumerate(zip(record['cases'],declared)):
        cohort,dtype,profile,route=identity
        assert case['ndzip_reference']==record['ndzip_reference']
        assert case['index']==index and (case['cohort'],case['dtype'],case['profile'],case['route'])==identity
        assert case['seed']==(71 if cohort=='a' else 171)
        assert case['total_bytes']==config['total_bytes'] and case['frame_bytes']==config['frame_bytes']
        assert case['physical_vram_bytes']==record['device']['total_memory_bytes']
        assert case['logical_source_exceeds_vram']==(config['total_bytes']>case['physical_vram_bytes'])
        if record['suite'] in ('over-vram','above-probe'):assert case['logical_source_exceeds_vram'] is True
        specs=frame_specs(dtype,config['total_bytes'],config['frame_bytes'])
        assert len(case['frames'])==len(specs) and len(case['encoded_controls'])==len(specs),'Missing prepared frames'
        for desc,spec in zip(case['frames'],specs):
            for key,value in spec.items():assert desc[key]==value,'Invalid original frame geometry'
        timings=('setup_ns','prepare_ns','prepare_input_reference_ns','prepare_encoding_ns','prepare_cache_ns',
                 'prepare_other_ns','first_use_ns','control_replay_ns','post_guard_ns','allocator_cleanup_ns')
        assert all(isinstance(case[k],int) and case[k]>=0 for k in timings)
        assert case['prepare_ns']==sum(case[k] for k in ('prepare_input_reference_ns','prepare_encoding_ns','prepare_cache_ns','prepare_other_ns'))
        assert (case['prepare_encoding_ns']>0)==(route['policy']=='precompressed')
        largest=max(s['bytes'] for s in specs);itemsize=np.dtype(dtype).itemsize
        hash_bytes=((largest+4095)//4096)*32
        if route['codec']=='plain':candidate=0;status=0
        elif route['codec']=='xbp':candidate=32+16*((largest//itemsize+255)//256)+8*((largest+7)//8);status=4
        else:candidate=max(ndzip_bound_bytes(s['shape'],dtype) for s in specs);status=0
        expected_slot={'candidate_bytes':candidate,'requested_device_bytes':largest+candidate+hash_bytes+status,
                       'requested_pinned_slot_bytes':candidate+hash_bytes+status,
                       'opaque_library_cpu_workspace_measured':False}
        memory=case['memory'];assert memory['slots']==[expected_slot for _ in range(route['queue_depth'])],'Named slot allocation geometry differs'
        cache_lengths=[f['bytes'] if route['policy']=='online' else c['bytes'] for f,c in zip(specs,case['encoded_controls'])]
        assert [f['cache_bytes'] for f in case['frames']]==cache_lengths
        assert memory['requested_pinned_source_bytes']==sum(cache_lengths)
        assert memory['expected_digest_bytes']==sum(((f['bytes']+4095)//4096)*32 for f in specs)
        assert memory['physical_peak_measured'] is False
        intervals=case['nonoverlapping_host_intervals'];assert len(intervals)==len(specs)
        assert intervals[0]['begin']==0 and sorted(i['bytes'] for i in intervals)==sorted(cache_lengths)
        assert all(a['begin']+a['bytes']<=b['begin'] for a,b in zip(intervals,intervals[1:])), 'Host source interval alias'
        assert len(case['trials'])==config['repeats'],'Missing declared repeats'
        for repeat,trial in enumerate(case['trials']):
            assert trial['repeat']==repeat and isinstance(trial['host_ns'],int) and trial['host_ns']>0
            observations=trial['observations']
            assert [o['frame'] for o in observations]==list(range(len(specs))),'Missing timed frames'
            assert trial['wire_bytes']==sum(o['wire_bytes'] for o in observations)
            assert sum(o['cpu_encode_validate_ns'] for o in observations)<=trial['host_ns']
            for frame,(obs,spec,control) in enumerate(zip(observations,specs,case['encoded_controls'])):
                assert obs['wire_bytes']==control['bytes'] and obs['device_error_bits']==0
                assert (obs['cpu_encode_validate_ns']>0)==(route['policy']=='online' and route['codec']!='plain')
                values=[obs[k] for k in ('start_ms','ingest_end_ms','hash_end_ms','result_end_ms')]
                assert all(math.isfinite(x) and x>=0 for x in values) and values==sorted(values)
                if frame>=route['queue_depth']:
                    assert observations[frame-route['queue_depth']]['result_end_ms']<=obs['start_ms'],'Same slot reused before completion'
                totals['timed_frames']+=1;totals['timed_restored_input_bytes']+=spec['bytes']
                totals['observed_digest_bytes']+=((spec['bytes']+4095)//4096)*32
                totals['wire_bytes']+=obs['wire_bytes']
        grouped.setdefault((cohort,dtype,profile),[]).append(case)
    checked_witnesses=set();checked_packets=set()
    for (cohort,dtype,profile),cases in grouped.items():
        specs=frame_specs(dtype,config['total_bytes'],config['frame_bytes']);seed=71 if cohort=='a' else 171
        for frame,spec in enumerate(specs):
            original=input_words(profile,dtype,spec['offset_words'],spec['bytes']//np.dtype(dtype).itemsize,seed).reshape(spec['shape'])
            input_hash=digest(original);expected=original_digests(original);expected_hash=digest(expected)
            own_size=xbp_size(original);canonical=None
            totals['unique_original_bytes_replayed']+=spec['bytes']
            for case in cases:
                desc=case['frames'][frame];control=case['encoded_controls'][frame];codec=case['route']['codec']
                assert desc['input_sha256']==input_hash and desc['expected_digest_sha256']==expected_hash,'Original input identity differs'
                assert desc['cache_sha256']==(control['sha256'] if case['route']['policy']=='precompressed' else input_hash)
                if codec=='plain':
                    assert control=={'bytes':spec['bytes'],'sha256':input_hash,'retained_file':None}
                else:
                    assert HEX.fullmatch(control['sha256']) and control['bytes']>0
                    if codec=='xbp':assert control['bytes']==own_size,'Original-codec encoded length differs from independent word-width model'
                    else:assert control['bytes']<=ndzip_bound_bytes(spec['shape'],dtype)
                    if control['bytes']<=8*1024**2:
                        retained=control['retained_file'];assert retained is not None
                        assert retained['file']==f'packets/{control["sha256"]}.npy'
                        key=(retained['file'],retained['sha256'],control['bytes'],codec,input_hash)
                        if key not in checked_packets:
                            path=safe_file(root,retained['file']);assert file_digest(path)==retained['sha256']
                            packet=np.load(path,allow_pickle=False)
                            assert packet.dtype==np.uint8 and packet.ndim==1 and packet.nbytes==control['bytes']
                            assert digest(packet)==control['sha256']
                            if codec=='xbp':
                                if canonical is None:canonical=encode_reference(original)
                                assert packet.tobytes()==canonical,'Original-codec packet differs from independent original input'
                                restored=decode_reference(packet)
                                assert restored.view(np.uint8).tobytes()==original.view(np.uint8).tobytes()
                            checked_packets.add(key)
                        totals['retained_encoded_control_bindings']+=1
                    else:
                        assert control['retained_file'] is None
                        totals['unretained_encoded_control_bindings']+=1
                for trial in case['trials']:
                    witness=trial['observations'][frame]['witness']
                    assert witness['shape']==list(expected.shape) and witness['array_bytes']==expected.nbytes
                    assert HEX.fullmatch(witness['array_sha256'])
                    assert witness['file']==f'witnesses/{witness["array_sha256"]}.npy'
                    key=(witness['file'],witness['sha256'],witness['array_sha256'],expected_hash)
                    if key not in checked_witnesses:
                        path=safe_file(root,witness['file']);assert file_digest(path)==witness['sha256']
                        table=np.load(path,allow_pickle=False)
                        assert table.dtype==np.uint8 and list(table.shape)==witness['shape']
                        assert table.nbytes==witness['array_bytes'] and digest(table)==witness['array_sha256']
                        assert np.array_equal(table,expected),'Timed output fingerprints differ from independently regenerated original bytes'
                        checked_witnesses.add(key)
            del original,expected,canonical
    return {'schema':'compression-pair-reference-replay-v2','suite':record['suite'],'ndzip_reference':record['ndzip_reference'],'manifest_sha256':file_digest(manifest),
            'collection':root.name,**totals,'unique_digest_files':len({key[0] for key in checked_witnesses}),'unique_encoded_control_files':len({key[0] for key in checked_packets}),
            'verifiers':{name:file_digest(ROOT/name) for name in ('verify_compression_reference_pair.py','compression_reference_epoch.py','compression_pair_models.py','compression_pair_cases.py','xbp_reference.py')},
            'scope':'Every timed block fingerprint; exact named requested buffers, not physical peaks/opaque CPU workspaces; online packet controls are untimed reencoding, not a retained timed instruction trace; teardown not separately metered'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--summary',type=Path,required=True);args=parser.parse_args()
    receipt=verify_collection(args.directory);args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
