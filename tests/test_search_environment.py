"""Recorded search environment is source-bound and preserves actual runtime values."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest


class EnvironmentTests(unittest.TestCase):
    def test_source_and_record_hashes_cannot_be_silently_changed(self):
        self.assertIsNotNone(importlib.util.find_spec('verify_search_environment'),'Environment replay is missing')
        from verify_search_environment import verify
        root=Path(__file__).resolve().parents[1]
        record=json.loads((root/'results/search-environment.json').read_text())
        verify(record,root)
        broken=deepcopy(record)
        broken['source_sha256']['inspect_search_environment.py']='0'*64
        with self.assertRaises(AssertionError):verify(broken,root)
        broken=deepcopy(record)
        broken['packages']['faiss-gpu-cuvs']['record_sha256']='bad'
        with self.assertRaises(AssertionError):verify(broken,root)
        broken=deepcopy(record)
        broken['loaded_libraries'][0]['name']='../private/library.so'
        with self.assertRaises(AssertionError):verify(broken,root)


if __name__=='__main__':unittest.main()
