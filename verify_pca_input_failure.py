"""CPU-only audit of rejected input-integrity evidence, not a fault attribution.

Replay the ORIGINAL mean census by substituting the archived rejected collection
for its guarded replacement. Reconstruct every archived prepared source using
its retained center/scale and the phase-appropriate original or changed input.
These checks do not certify rejected covariance products, GPU hardware, a library
culprit, the incomplete sanitizer run or future absence of input corruption.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from pca_cases import DATASETS,inputs
from pca_provenance import validate_sources
from reference import digest

ROOT=Path(__file__).resolve().parent
FIXTURE=ROOT/'fixtures/pca-input-failure'
REJECTED='shifted-n32768-d256'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepared_source(data,method,stage,center,scale):
    rows=data[stage['begin']:stage['begin']+stage['count']].astype(np.float64)
    dtype=np.dtype(stage['source_dtype'])
    base=(rows if method['centering']=='raw' else rows-center).astype(dtype)
    source=np.zeros(stage['source_shape'],dtype=dtype)
    source[:,:stage['count']]=(base/scale).T
    return source


def audit():
    retained=json.loads((FIXTURE/'manifest.json').read_text())
    for name,expected in retained['files'].items():
        assert Path(name).name==name and sha(FIXTURE/name)==expected
    findings=json.loads((FIXTURE/'input-change-forensics.json').read_text())['findings']
    assert len(findings)==1
    finding=findings[0]
    data0=inputs(32768,256,'shifted','b')[0]
    negative=np.ascontiguousarray(-data0)
    changed=negative.copy()
    row,column=finding['row'],finding['column']
    assert float(negative[row,column])==finding['original']
    old=int(negative.view(np.uint32)[row,column])
    difference=int(finding['xor'],16)
    assert difference.bit_count()==finding['bit_count']==1 and hex(old)==finding['original_bits']
    changed.view(np.uint32)[row,column]^=np.uint32(difference)
    assert hex(int(changed.view(np.uint32)[row,column]))==finding['observed_bits']
    assert float(changed[row,column])==finding['observed']
    control=json.loads((FIXTURE/'focused-input-control-100.json').read_text())
    expected_inputs=(digest(data0),digest(negative))
    assert len(control['checks'])==100
    for index,check in enumerate(control['checks']):
        assert check['iteration']==index and check['variant']==1-index%2
        for field in ('host_before_sha256','host_after_sha256','device_before_sha256','device_after_sha256'):
            assert check[field]==expected_inputs[check['variant']]
        assert check['differing_mean_components']==0 and check['maximum_mean_error']==0
    original_counts=Counter()
    failures=[]
    source_counts=Counter()
    rejected_hash=None
    for scope,cohort in (('smoke','a'),('full','a'),('full','b')):
        logical=ROOT/'results'/f'pca-{scope}-{cohort}'
        names=[next(iter(DATASETS))] if scope=='smoke' else list(DATASETS)
        for name in names:
            directory=FIXTURE if scope=='full' and cohort=='b' and name==REJECTED else logical
            rejected=directory==FIXTURE
            manifest_path=directory/(name+'-collection.json')
            manifest=json.loads(manifest_path.read_text())
            assert not validate_sources(manifest['source_sha256'])
            if rejected:rejected_hash=sha(manifest_path)
            n,d,profile=DATASETS[name]
            data,_=inputs(n,d,profile,cohort)
            target=data.mean(axis=0,dtype=np.float64)
            hashes=[digest(data),digest(np.ascontiguousarray(-data))]
            gamma=n*2.**-53/(1-n*2.**-53)
            bound=8*gamma*np.mean(abs(data.astype(np.float64)),axis=0)+64*2.**-53*np.maximum(abs(target),1.)
            for item in manifest['cases']:
                path=directory/item['file']
                assert Path(item['file']).name==item['file'] and sha(path)==item['sha256']
                record=json.loads(path.read_text())
                assert record['input_sha256']==hashes
                assert record['source_sha256']==manifest['source_sha256']
                witness=directory/record['witness_file']
                assert Path(record['witness_file']).name==record['witness_file'] and sha(witness)==record['witness_sha256']
                with np.load(witness,allow_pickle=False) as archive:
                    for key,output in record['outputs'].items():
                        expected=target if output['variant']==0 else -target
                        actual=archive[output['prefix']+'_mean']
                        error=abs(actual-expected)
                        bad=np.flatnonzero(error>bound)
                        original_counts['outputs']+=1
                        if len(bad):
                            uses=[dict(boundary=b['name'],observation=i) for b in record['boundaries'] for i,observation in enumerate(b['observations']) if observation['output_key']==key]
                            failures.append(dict(case=logical.name+'/'+path.name,case_sha256=item['sha256'],witness_sha256=record['witness_sha256'],
                                                 output_key=key,variant=output['variant'],components=bad.tolist(),
                                                 actual=[float(actual[i]) for i in bad],expected=[float(expected[i]) for i in bad],
                                                 max_error=float(error.max()),max_bound=float(bound.max()),timing_uses=uses))
                            assert rejected and output['variant']==1 and bad.tolist()==[column]
                            assert np.all(abs(actual-changed.mean(axis=0,dtype=np.float64))<=bound)
                    if rejected:
                        for stage in record['stage_witnesses']:
                            source_data=changed if record['case_index']>35 and stage['variant']==1 else (negative if stage['variant']==1 else data0)
                            center=archive[stage['prefix']+'_center']
                            scale=archive[stage['prefix']+'_scale']
                            source=prepared_source(source_data,record['method'],stage,center,scale)
                            assert digest(source)==stage['operand_sha256'],'Rejected-stage input-change reconstruction differs'
                            source_counts['retained_prepared_sources']+=1
                            source_counts['changed_input_phase_sources']+=int(source_data is changed)
                            if path.name==finding['case_file'] and stage['variant']==1 and stage['begin']<=row<stage['begin']+stage['count']:
                                assert digest(source)==finding['source_sha256']
                                assert digest(prepared_source(negative,record['method'],stage,center,scale))!=finding['source_sha256']
                                source_counts['exact_localized_changed_word_matches']+=1
                original_counts['cases']+=1
            original_counts['collections']+=1
    original_counts['retained_output_mean_integrity_failures']=len(failures)
    census=json.loads((FIXTURE/'original-mean-census.json').read_text())
    assert dict(original_counts)==census['counts'] and failures==census['failures'],'Original mean census is not reproduced'
    assert source_counts['exact_localized_changed_word_matches']==1
    return dict(scope='Rejected original-input integrity evidence independently reconstructed; not hardware/vendor attribution or successful numerical PCA',
                original_census_counts=dict(original_counts),rejected_source_counts=dict(source_counts),
                fixture_manifest_sha256=sha(FIXTURE/'manifest.json'),rejected_collection_sha256=rejected_hash,
                verifier_source_sha256={'verify_pca_input_failure.py':sha(Path(__file__))},
                limits='Prepared-source byte fingerprints and original mean census only. Rejected products are not speed/quality evidence. The recorded 100-run diagnostic does not exclude rare faults; the sanitizer attempt did not finish.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--summary',type=Path)
    args=parser.parse_args()
    result=audit()
    if args.summary:args.summary.write_text(json.dumps(result,indent=2)+'\n')
    print('PCA_REJECTED_INPUT_EVIDENCE_RECONSTRUCTION_PASS',json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':main()
