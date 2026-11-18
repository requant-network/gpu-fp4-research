"""Capture byte-preserving public build proof without vendor binaries or private build paths."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import re
import subprocess


def sanitize_loader_text(text,root):
    text=text.replace(str(root),'<ORDERED_BUILD>').replace(str(Path.home()),'<HOME>')
    return re.sub(r'\s+\(0x[0-9a-fA-F]+\)','',text)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.build_root.resolve();manifest=root/'manifest.json';record=json.loads(manifest.read_text())
    assert record['complete'] and record['schema']=='ndzip-ordered-gpu-build-v1'
    assert all(sha(root/name)==digest for name,digest in record['binaries'].items())
    assert all(sha(root/'source'/name)==digest for name,digest in record['sources'].items())
    args.output.mkdir(parents=True,exist_ok=False)
    for name,source in (('manifest.json',manifest),('cpu_codec.inl',root/'source/src/ndzip/cpu_codec.inl'),
                        ('LICENSE',root/'source/LICENSE'),('ordered-copy.patch',root/'ordered-copy.patch')):
        shutil.copyfile(source,args.output/name);assert sha(source)==sha(args.output/name)
    cache={}
    for line in (root/'build/CMakeCache.txt').read_text().splitlines():
        if '=' in line and ':' in line.split('=',1)[0]:
            key,value=line.split('=',1);cache[key.split(':')[0]]=value
    keys=('CMAKE_BUILD_TYPE','CMAKE_CXX_FLAGS','CMAKE_CUDA_FLAGS','CMAKE_CUDA_ARCHITECTURES',
          'NDZIP_WITH_MT','NDZIP_WITH_CUDA','NDZIP_WITH_HIPSYCL')
    config={key:cache[key] for key in keys}
    assert config['NDZIP_WITH_CUDA']=='ON' and config['NDZIP_WITH_MT']=='ON' and config['CMAKE_CUDA_ARCHITECTURES']=='120'
    environment={'schema':'ndzip-ordered-gpu-environment-v1','configuration':config,
                 'capture_source_sha256':sha(__file__),
                 'cxx_version':subprocess.check_output([cache['CMAKE_CXX_COMPILER'],'--version'],text=True),
                 'cuda_compiler_version':subprocess.check_output([cache['CMAKE_CUDA_COMPILER'],'--version'],text=True),
                 'linked_libraries':{name:sanitize_loader_text(subprocess.check_output(['ldd',str(root/name)],text=True),root)
                                     for name in record['binaries']}}
    (args.output/'environment.json').write_text(json.dumps(environment,indent=2))
    print('ORDERED_GPU_BUILD_PROOF_CAPTURED_NO_BINARIES',len(record['binaries']),flush=True)
