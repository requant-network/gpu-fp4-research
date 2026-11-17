"""Two explicit PCA producer epochs sharing immutable arithmetic dependencies.

Original records bind the original producer. The guarded repeat additionally
binds its wrapper and input guard. No old source identifier/hash is rewritten and
no arbitrary third producer epoch is accepted.
"""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE_PRODUCERS=('pca_study.py','pca_native.py','pca_cases.py','pca_protocol.py','pca_reference.py','reference.py',
                'native/pca_bridge.cu','native/hardware_bridge.cu','native/application_bridge.cu',
                'native/optimized_bridge.cu','native/bridge.cu','build_pca.sh')
GUARDED_ADDITIONS=('pca_input_integrity.py','pca_study_guarded.py')


def producer_sources(guarded):
    assert type(guarded) is bool
    names=BASE_PRODUCERS+(GUARDED_ADDITIONS if guarded else ())
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def validate_sources(mapping):
    keys=set(mapping)
    if keys==set(BASE_PRODUCERS):guarded=False
    elif keys==set(BASE_PRODUCERS+GUARDED_ADDITIONS):guarded=True
    else:raise AssertionError('Unregistered PCA producer source set')
    assert mapping==producer_sources(guarded),'Executed PCA producer bytes differ from their registered epoch'
    return guarded


def verify_guarded_observations(record,guarded):
    if not guarded:
        assert 'input_integrity' not in record,'Unbound input-guard declaration'
        return 0
    guard=record['input_integrity']
    assert guard['expected_original_input_sha256']==record['input_sha256'] and guard['original_inputs_readonly'] is True
    observations=record['observations_per_boundary']
    warmups=record['warmups_per_boundary']
    assert guard['host_checks']==1+2*2+2*2*warmups+2*2*observations
    assert guard['device_checks']==1+2+2*warmups+3*observations
    checked=0
    for boundary in record['boundaries']:
        for row in boundary['observations']:
            expected=record['input_sha256'][row['variant']]
            for name in ('original_input_before_sha256','original_input_after_sha256','device_input_after_sha256'):
                assert row[name]==expected,'Timed original/device input fingerprint differs from declared observations'
            checked+=1
    return checked
