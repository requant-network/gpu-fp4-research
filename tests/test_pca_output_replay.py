"""Independent PCA final-output checks against executed GPU witnesses."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from pca_cases import inputs

ROOT=Path(__file__).resolve().parents[1]


class PcaOutputReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory=ROOT/'results/pca-smoke-a'
        path=directory/'uniform-n8192-d128-truth.npz'
        with np.load(path,allow_pickle=False) as archive:
            cls.target={field:archive['v0_'+field].copy() for field in ('mean','covariance','eigenvalues','eigenvectors')}
        cls.data=inputs(8192,128,'uniform','a')[0]

    def test_original_data_replays_the_entire_stored_truth(self):
        self.assertIsNotNone(importlib.util.find_spec('pca_output_replay'),'Independent final-output replay is missing')
        from pca_output_replay import replay_truth
        result=replay_truth(self.data,self.target)
        self.assertEqual(result['truth_covariance_entries'],128*128)
        self.assertEqual(result['truth_spectrum_entries'],128)
    def test_complete_gpu_output_is_bound_to_replayed_stages_and_truth(self):
        import pca_output_replay
        self.assertTrue(hasattr(pca_output_replay,'replay_case_outputs'),'Complete PCA output replay is missing')
        from pca_output_replay import replay_case_outputs
        directory=ROOT/'results/pca-smoke-a'
        path=directory/'uniform-n8192-d128-m000.json'
        record=json.loads(path.read_text())
        with np.load(directory/'uniform-n8192-d128-truth.npz',allow_pickle=False) as archive:
            targets=[{field:archive[f'v{index}_{field}'] for field in ('mean','covariance','eigenvalues','eigenvectors')} for index in range(2)]
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as archive:
            result=replay_case_outputs(self.data,record,archive,targets)
        self.assertGreater(result['output_covariance_entries'],0)
        self.assertEqual(result['timing_observations'],10)
        self.assertGreater(result['output_spectrum_entries'],0)
    def test_consistent_fake_covariance_spectrum_and_hashes_are_rejected(self):
        from pca_output_replay import replay_case_outputs
        from pca_protocol import array_binding
        from pca_reference import quality_leading
        directory=ROOT/'results/pca-smoke-a'
        path=directory/'uniform-n8192-d128-m000.json'
        record=json.loads(path.read_text())
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as archive:
            arrays={key:archive[key].copy() for key in archive.files}
        with np.load(directory/'uniform-n8192-d128-truth.npz',allow_pickle=False) as archive:
            targets=[{field:archive[f'v{index}_{field}'] for field in ('mean','covariance','eigenvalues','eigenvectors')} for index in range(2)]
        old_key=next(iter(record['outputs']))
        output=record['outputs'].pop(old_key)
        prefix=output['prefix']
        arrays[prefix+'_covariance']+=np.eye(128,dtype=np.float32)*np.float32(.25)
        values,vectors=np.linalg.eigh(arrays[prefix+'_covariance'])
        arrays[prefix+'_eigenvalues']=values
        arrays[prefix+'_leading']=vectors[:,-16:].copy()
        forged={key:arrays[prefix+'_'+key] for key in ('mean','covariance','eigenvalues','leading')}
        output['array_bindings']={key:array_binding(value) for key,value in forged.items()}
        output['quality']=quality_leading(forged['mean'],forged['covariance'],values,forged['leading'],targets[0])
        identity=hashlib.sha256(json.dumps(output['array_bindings'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        new_key='v0-'+identity
        record['outputs'][new_key]=output
        for boundary in record['boundaries']:
            for observation in boundary['observations']:
                if observation['output_key']==old_key:observation['output_key']=new_key
        with self.assertRaisesRegex(AssertionError,'Final covariance'):
            replay_case_outputs(self.data,record,arrays,targets)

    def test_stale_input_timing_binding_is_rejected(self):
        from pca_output_replay import replay_case_outputs
        directory=ROOT/'results/pca-smoke-a'
        path=directory/'uniform-n8192-d128-m000.json'
        record=json.loads(path.read_text())
        wrong=next(key for key,value in record['outputs'].items() if value['variant']==1)
        record['boundaries'][0]['observations'][0]['output_key']=wrong
        with np.load(directory/'uniform-n8192-d128-truth.npz',allow_pickle=False) as archive:
            targets=[{field:archive[f'v{index}_{field}'] for field in ('mean','covariance','eigenvalues','eigenvectors')} for index in range(2)]
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as archive:
            with self.assertRaisesRegex(AssertionError,'stale input'):
                replay_case_outputs(self.data,record,archive,targets)


if __name__=='__main__':unittest.main()
