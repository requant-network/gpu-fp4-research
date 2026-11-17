"""The retained source fingerprint distinguishes the actual one-bit input change."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from pca_cases import inputs
from reference import digest


class PcaInputFailureTests(unittest.TestCase):
    def test_real_rejected_operand_matches_one_changed_word_not_original_input(self):
        self.assertIsNotNone(importlib.util.find_spec('verify_pca_input_failure'),'Rejected-input evidence verifier is missing')
        from verify_pca_input_failure import prepared_source
        root=Path(__file__).resolve().parents[1]/'fixtures/pca-input-failure'
        record=json.loads((root/'shifted-n32768-d256-m036.json').read_text())
        stage=next(item for item in record['stage_witnesses'] if item['variant']==1 and item['begin']==6144)
        data=-inputs(32768,256,'shifted','b')[0]
        with np.load(root/record['witness_file'],allow_pickle=False) as archive:
            center=archive[stage['prefix']+'_center']
            scale=archive[stage['prefix']+'_scale']
        self.assertNotEqual(digest(prepared_source(data,record['method'],stage,center,scale)),stage['operand_sha256'])
        data.view(np.uint32)[6376,215]^=np.uint32(0x200000)
        self.assertEqual(digest(prepared_source(data,record['method'],stage,center,scale)),stage['operand_sha256'])


if __name__=='__main__':unittest.main()
