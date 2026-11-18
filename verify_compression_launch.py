"""Independent execution/cost-envelope and real GPU capture-selftest replay."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from compression_pair_cases import input_words
from verify_plain_pair_diagnostic import standard_table

ROOT=Path(__file__).resolve().parent
VERIFIERS=('verify_compression_launch.py','verify_plain_pair_diagnostic.py','compression_pair_cases.py')
HEX=re.compile(r'[0-9a-f]{64}\Z')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_check(mapping,expected):
    assert set(mapping)==set(expected),'Incorrect source closure'
    for name,digest in mapping.items():assert sha(ROOT/name)==digest,name


def verify_launch(directory):
    from check_compression_reference_pair import SOURCES,REFERENCE_FILES
    directory=Path(directory);record=json.loads((directory/'manifest.json').read_text())
    assert record['schema']=='compression-reference-launch-v1' and record['sealed'] is True
    assert directory.name.endswith('-launch')
    assert record['collection']=='results/'+directory.name.removesuffix('-launch')
    source_check(record['sources'],(*SOURCES,'launch_compression_reference.py'))
    assert record['explicit_environment']=={'CUDA_VISIBLE_DEVICES':'0','OPENBLAS_NUM_THREADS':'1',
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
    assert HEX.fullmatch(record['python_binary_sha256']) and isinstance(record['python_version'],str)
    assert record['ndzip_reference'] in ('ordered-copy','stock-serial')
    assert record['driver_arguments']==['--ndzip-reference',record['ndzip_reference'],'--suite',record['suite'],'--output',record['collection']]
    dependencies=record['dependencies']
    assert set(dependencies['proofs'])=={f'results/{name}/manifest.json' for name in
        ('compression-native-preflight','xbp-native-preflight','compression-integrity-preflight','compression-ndzip-cli-preflight')}
    assert set(dependencies['ordered_reference']['proofs'])==set(REFERENCE_FILES)
    for mapping in (dependencies['proofs'],dependencies['ordered_reference']['proofs']):
        for name,expected in mapping.items():assert sha(ROOT/name)==expected,name
    public=directory/'child.log';assert sha(public)==record['public_log_sha256']
    text=public.read_text();assert '/home/' not in text
    marker=any(value in text for value in ('Traceback (most recent call last):','Exception ignored in:','CUDA_ERROR_'))
    assert record['unhandled_error_marker']==marker and HEX.fullmatch(record['raw_private_log_sha256'])
    manifest=ROOT/record['collection']/'manifest.json'
    if record['collection_manifest_sha256'] is None:
        assert not manifest.exists();child={}
    else:
        assert sha(manifest)==record['collection_manifest_sha256'];child=json.loads(manifest.read_text())
        assert child['sources']=={name:record['sources'][name] for name in SOURCES}
        assert child['dependencies']==dependencies
        assert child['suite']==record['suite'] and child['ndzip_reference']==record['ndzip_reference']
    assert record['collection_complete']==(child.get('complete') is True)
    values=[]
    for case in child.get('cases',[]):
        values.extend(case[key] for key in ('setup_ns','prepare_ns','first_use_ns','control_replay_ns','post_guard_ns','allocator_cleanup_ns'))
        values.extend(trial['host_ns'] for trial in case['trials'])
    assert all(isinstance(value,int) and value>=0 for value in values)
    named=sum(values);wall=record['process_wall_ns'];other=record['other_child_wall_ns']
    assert isinstance(wall,int) and wall>0 and isinstance(other,int) and other>=0,'Invalid whole-child wall cost'
    assert record['summed_named_case_ns']==named and wall==named+other,'Whole-child wall decomposition differs'
    assert record['parent_envelope_ns']>=wall,'Parent wall is shorter than child wall'
    assert isinstance(record['exit_code'],int) and record['source_closure_valid'] is True
    accepted=record['exit_code']==0 and record['collection_complete'] and not marker
    assert record['child_accepted']==accepted
    assert record['launcher_exit_code']==(0 if accepted else (record['exit_code'] or 65))
    return {'schema':'compression-launch-replay-v1','collection':record['collection'],
            'child_accepted':accepted,'child_exit_code':record['exit_code'],'process_wall_ns':wall,
            'named_case_ns':named,'other_child_ns':other,'launch_manifest_sha256':sha(directory/'manifest.json'),
            'collection_manifest_sha256':record['collection_manifest_sha256'],
            'verifiers':{name:sha(ROOT/name) for name in VERIFIERS},
            'scope':'Execution/source/cost receipt readback and arithmetic closure; full content requires its separate streaming replay, not an independent clock trace'}


def verify_capture_selftest(directory):
    from check_compression_capture import SOURCES
    directory=Path(directory);meta=json.loads((directory/'manifest.json').read_text())
    assert meta['schema']=='compression-capture-selftest-v1' and meta['complete'] is True
    assert meta['cuda_device']==0 and meta['frame_bytes']==262144
    assert 'injected' in meta['purpose'].lower()
    source_check(meta['sources'],SOURCES)
    manifest=directory/'failure/manifest.json';assert sha(manifest)==meta['failure_manifest_sha256']
    record=json.loads(manifest.read_text());assert record['schema']=='compression-post-unwind-failure-v1'
    assert record['frame']==0 and record['spec']=={'frame':0,'offset_words':0,'shape':[512,128],'bytes':262144}
    assert record['dtype']=='<f4' and record['seed']==71 and record['profile']=='smooth'
    path=directory/'failure/arrays.npz';assert sha(path)==record['arrays_sha256']
    source=input_words('smooth','<f4',0,65536,71).reshape(512,128)
    good=standard_table(source);bad=good.copy();bad[0,0]^=1
    with np.load(path,allow_pickle=False) as archive:
        assert set(archive.files)=={'cached','regenerated','restored','expected','returned','device_digests','host_packet','device_packet'}
        for name in ('cached','regenerated','restored'):
            array=archive[name];assert array.dtype==source.dtype and array.shape==source.shape
            assert array.tobytes()==source.tobytes(),'Injected selftest GPU payload differs from independently regenerated original'
        for name,expected in (('expected',bad),('returned',good),('device_digests',good)):
            assert archive[name].dtype==np.uint8 and np.array_equal(archive[name],expected),'Wrong injected-oracle or device/returned hash evidence'
        for name in ('host_packet','device_packet'):
            assert archive[name].dtype==np.uint8 and archive[name].tobytes()==source.tobytes()
    assert record['prepared_cache_sha256']==hashlib.sha256(source).hexdigest()
    assert record['cache_matches_prepared_guard'] is True and record['restored_matches_regeneration'] is True
    assert record['expected_matches_regeneration'] is False and record['restored_different_words']==0
    assert record['gpu_digest_matches_cpu_restored'] is True and record['returned_digest_matches_device'] is True
    assert record['host_packet_matches_device'] is True and record['first_differences']==[]
    return {'schema':'compression-capture-selftest-replay-v1','manifest_sha256':sha(directory/'manifest.json'),
            'restored_words':int(source.size),'injected_oracle_only':True,
            'verifiers':{name:sha(ROOT/name) for name in VERIFIERS},
            'scope':'Real GPU arrays retained after frozen checker failure/finally; explicit injected oracle, not a natural failure or benchmark'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--kind',choices=('launch','selftest'),required=True);parser.add_argument('--summary',type=Path,required=True)
    args=parser.parse_args();receipt=(verify_launch if args.kind=='launch' else verify_capture_selftest)(args.directory)
    args.summary.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
