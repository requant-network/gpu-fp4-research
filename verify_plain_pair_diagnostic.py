"""Independent observed-byte replay for the plain-transfer boundary diagnostic."""
import hashlib
import numpy as np


def standard_table(array):
    raw=memoryview(array).cast('B')
    return np.array([list(hashlib.sha256(raw[i:i+4096]).digest()) for i in range(0,len(raw),4096)],dtype=np.uint8)


def replay_arrays(arrays,guard):
    assert set(arrays)=={'original','regenerated','restored','expected','returned','gpu'}
    original=arrays['original'];reference=arrays['regenerated'];restored=arrays['restored']
    assert original.dtype==reference.dtype==restored.dtype==np.dtype('<f4')
    assert original.shape==reference.shape==restored.shape
    expected=standard_table(reference);actual=standard_table(restored)
    for name in ('expected','returned','gpu'):
        assert arrays[name].dtype==np.uint8 and arrays[name].shape==expected.shape
    different=np.nonzero(restored.view('<u4').reshape(-1)!=reference.view('<u4').reshape(-1))[0]
    old=reference.view('<u4').reshape(-1);new=restored.view('<u4').reshape(-1)
    return {'host_source_matches_prepared_guard':hashlib.sha256(original).hexdigest()==guard,
            'host_source_matches_regeneration':original.tobytes()==reference.tobytes(),
            'expected_matches_regeneration':arrays['expected'].tobytes()==expected.tobytes(),
            'restored_matches_regeneration':restored.tobytes()==reference.tobytes(),
            'restored_matches_current_host':restored.tobytes()==original.tobytes(),
            'restored_different_words':int(len(different)),
            'gpu_digest_matches_cpu_restored':arrays['gpu'].tobytes()==actual.tobytes(),
            'returned_host_digest_matches_gpu':arrays['returned'].tobytes()==arrays['gpu'].tobytes(),
            'first_differences':[{'word':int(i),'expected':int(old[i]),'restored':int(new[i])} for i in different[:16]]}


def verify_diagnostic(directory):
    import json
    from pathlib import Path
    from check_plain_pair_diagnostic import SOURCES
    from compression_pair_cases import frame_specs,input_words
    root=Path(__file__).resolve().parent;directory=Path(directory)
    def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    record=json.loads((directory/'manifest.json').read_text())
    assert record['schema']=='plain-pair-failure-diagnostic-v1'
    assert set(record['sources'])==set(SOURCES)
    for name,expected in record['sources'].items():assert file_hash(root/name)==expected,name
    count=record['frames_requested'];assert isinstance(count,int) and 2<=count<=320
    assert record['total_bytes']==count*64*1024**2
    specs=frame_specs('<f4',record['total_bytes'],64*1024**2)
    assert record['frame_specs']==specs and len(record['source_guards'])==count
    route=record['route'];assert route['codec']=='plain' and route['policy']=='online' and route['cpu_threads']==1
    assert route['queue_depth'] in (1,2) and record['device']['major']==12 and record['device']['minor']==0
    assert record['device']['name']=='NVIDIA GeForce RTX 5060 Ti'
    dependencies=record['dependencies']
    for name,digest in dependencies['proofs'].items():assert file_hash(root/name)==digest
    for name,digest in dependencies['ordered_reference']['proofs'].items():assert file_hash(root/name)==digest
    observations=record['observations'];indices=[o['frame'] for o in observations]
    assert indices==sorted(set(indices)) and all(0<=i<count for i in indices)
    if record['complete']:
        assert record['mismatch'] is None and indices==list(range(count)) and record['all_host_sources_unchanged'] is True
    else:assert record['mismatch'] is not None
    indexed={o['frame']:o for o in observations};failure=record['mismatch'];replayed=None
    for frame,spec in enumerate(specs):
        source=input_words('smooth','<f4',spec['offset_words'],spec['bytes']//4,71).reshape(spec['shape'])
        guard=hashlib.sha256(source).hexdigest();assert record['source_guards'][frame]==guard
        if frame in indexed:
            witness=indexed[frame]['witness'];path=directory/witness['file']
            assert not path.is_symlink() and path.resolve().is_relative_to(directory.resolve())
            assert file_hash(path)==witness['sha256']
            table=np.load(path,allow_pickle=False)
            assert table.dtype==np.uint8 and list(table.shape)==witness['shape'] and table.nbytes==witness['array_bytes']
            assert hashlib.sha256(table).hexdigest()==witness['array_sha256']
            assert np.array_equal(table,standard_table(source)),'Retained diagnostic output differs from original bytes'
        if failure is not None and frame==failure['frame']:
            path=directory/'failure.npz';assert file_hash(path)==failure['failure_sha256']
            with np.load(path,allow_pickle=False) as archive:
                arrays={key:archive[key] for key in archive.files}
                assert arrays['regenerated'].dtype==source.dtype and arrays['regenerated'].shape==source.shape
                assert arrays['regenerated'].tobytes()==source.tobytes(),'False diagnostic regenerated input'
                replayed=replay_arrays(arrays,guard)
                assert replayed==failure['census'],'Failure census differs from independent array replay'
                assert not np.array_equal(arrays['returned'],arrays['expected']),'Claimed failed fingerprint comparison is actually equal'
    if failure is not None:assert replayed is not None
    return {'schema':'plain-pair-diagnostic-replay-v1','manifest_sha256':file_hash(directory/'manifest.json'),
            'requested_frames':count,'replayed_observations':len(observations),'total_logical_bytes':record['total_bytes'],
            'complete_plain_diagnostic':record['complete'],'failure_replayed':replayed,
            'verifiers':{name:file_hash(root/name) for name in ('verify_plain_pair_diagnostic.py','compression_pair_cases.py')},
            'scope':'Retained bytes/fingerprints and failure census; bounded repeat, not an original-failure cause or repair certificate'}


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--summary',type=Path,required=True);args=parser.parse_args()
    receipt=verify_diagnostic(args.directory);args.summary.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
