"""CPU source-binding/schema check for the recorded actual search environment.

This does not pretend to load or re-hash absent GPU libraries on a CPU machine;
package RECORD and loaded-library fingerprints are the executed snapshot.
"""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent


def verify(record,root=ROOT):
    assert record['gpu_index']==0 and record['gpu_exclusive'] is False
    assert record['compute_capability']==[12,0]
    assert record['gpu_name']=='NVIDIA GeForce RTX 5060 Ti'
    assert record['driver_api_version']>=record['runtime_api_version']>=13000
    assert record['source_sha256']=={'inspect_search_environment.py':hashlib.sha256((root/'inspect_search_environment.py').read_bytes()).hexdigest()}
    packages=record['packages']
    for name,version in {'numpy':'2.2.6','faiss-gpu-cuvs':'1.16.0','cuvs-cu13':'26.10.0','cupy-cuda13x':'14.2.0','nvidia-cuda-runtime':'13.3.29','nvidia-nvjitlink':'13.3.33'}.items():
        assert packages[name]['version']==version,(name,packages.get(name))
    for entry in packages.values():
        assert isinstance(entry['version'],str) and entry['version']
        assert re.fullmatch('[0-9a-f]{64}',entry['record_sha256'])
    names=[]
    for entry in record['loaded_libraries']:
        name=entry['name']
        assert name and Path(name).name==name and '/' not in name and '\\' not in name
        assert isinstance(entry['bytes'],int) and entry['bytes']>0
        assert re.fullmatch('[0-9a-f]{64}',entry['sha256'])
        names.append(name)
    assert len(names)==len(set(names)) and {'libcuvs.so','libcuvs_c.so','libfaiss.so'}<=set(names)
    return len(packages),len(names)


def main():
    record=json.loads((ROOT/'results/search-environment.json').read_text())
    packages,libraries=verify(record)
    print('Recorded search environment source/schema PASS',packages,'package RECORD fingerprints,',libraries,'loaded-library fingerprints;',
          'actual runtime',record['runtime_api_version'],'driver API',record['driver_api_version'])


if __name__=='__main__':main()
