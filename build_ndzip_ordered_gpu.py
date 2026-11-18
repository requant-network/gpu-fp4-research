"""Build a separately labelled CUDA-enabled ndzip ff4e6702 + ordered-copy reference."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from build_ndzip_cpu_controls import variant_source

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(upstream,output):
    upstream=Path(upstream).resolve();output=Path(output).resolve()
    pinned=json.loads((ROOT/'results/compression-ndzip-cli-preflight/manifest.json').read_text())
    assert pinned['upstream_commit']=='ff4e6702bf0abb86d4aeef8249bd30344dfeef75'
    assert all(sha(upstream/name)==expected for name,expected in pinned['vendor_sources'].items())
    assert all(sha(upstream/'build-sm120-module-boost'/name)==expected for name,expected in pinned['binaries'].items())
    causal=ROOT/'results/ndzip-cpu-causal-replay.json';cpu=json.loads(causal.read_text())
    assert all(cpu['variants'][name]['exact_parallel_outputs']==48 for name in ('ordered','ordered-delayed'))
    for name,expected in cpu['verifiers'].items():assert sha(ROOT/name)==expected
    cache={}
    for line in (upstream/'build-sm120-module-boost/CMakeCache.txt').read_text().splitlines():
        if '=' in line and ':' in line.split('=',1)[0]:
            name,value=line.split('=',1);cache[name.split(':')[0]]=value
    assert cache['CMAKE_CUDA_ARCHITECTURES']=='120'
    compiler=Path(cache['CMAKE_CUDA_COMPILER']);cuda=compiler.parent.parent
    output.mkdir(parents=True,exist_ok=False);source=output/'source';binary=output/'build';adapter=output/'adapter';adapter.mkdir()
    for name,expected in pinned['vendor_sources'].items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(upstream/name,path);assert sha(path)==expected
    codec=source/'src/ndzip/cpu_codec.inl';before=codec.read_text();after=variant_source(before,'ordered');codec.write_text(after)
    cpu_ordered=json.loads((ROOT/'results/ndzip-cpu-build-evidence/manifest.json').read_text())['variants']['ordered']
    assert sha(codec)==cpu_ordered['sources']['src/ndzip/cpu_codec.inl']
    patch=output/'ordered-copy.patch';patch.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='upstream/cpu_codec.inl',tofile='ordered/cpu_codec.inl')))
    native=ROOT/'native/compression_ndzip_bridge.cpp'
    old_native=json.loads((ROOT/'results/compression-native-preflight/manifest.json').read_text())
    assert sha(native)==old_native['sources']['native/compression_ndzip_bridge.cpp']
    record={'schema':'ndzip-ordered-gpu-build-v1','reference':'ndzip-ff4e6702+ordered-copy',
            'upstream_commit':pinned['upstream_commit'],'upstream_sources':pinned['vendor_sources'],
            'sources':{name:sha(source/name) for name in pinned['vendor_sources']},
            'builder_sha256':sha(__file__),'cpu_causal_receipt_sha256':sha(causal),
            'adapter_source_sha256':sha(native),'patch_sha256':sha(patch),
            'cuda_architectures':'120','cuda_flags':cache['CMAKE_CUDA_FLAGS'],
            'cuda_compiler_version':subprocess.check_output([str(compiler),'--version'],text=True),
            'binaries':{},'complete':False}
    manifest=output/'manifest.json';manifest.write_text(json.dumps(record,indent=2))
    options=['-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER='+cache['CMAKE_CXX_COMPILER'],
             '-DCMAKE_CXX_FLAGS='+cache['CMAKE_CXX_FLAGS'],
             '-DCMAKE_CUDA_COMPILER='+str(compiler),'-DCMAKE_CUDA_ARCHITECTURES=120',
             '-DCMAKE_CUDA_FLAGS='+cache['CMAKE_CUDA_FLAGS'],'-DNDZIP_WITH_CUDA=ON',
             '-DNDZIP_WITH_HIPSYCL=OFF','-DNDZIP_WITH_MT=ON','-DNDZIP_BUILD_BENCHMARK=OFF',
             '-DNDZIP_BUILD_TEST=OFF','-DNDZIP_WITH_3RDPARTY_BENCHMARKS=OFF','-DBoost_NO_BOOST_CMAKE=ON',
             '-DBoost_PROGRAM_OPTIONS_LIBRARY_RELEASE='+cache['Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE'],
             '-DBoost_PROGRAM_OPTIONS_LIBRARY_DEBUG='+cache['Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE']]
    commands=[('configure',['cmake','-S',str(source),'-B',str(binary),*options]),
              ('build',['cmake','--build',str(binary),'--target','compress','--parallel','4']),
              ('adapter',[cache['CMAKE_CXX_COMPILER'],'-std=c++17','-O3','-fPIC','-shared','-Wl,--no-undefined',str(native),
                          '-I'+str(source/'include'),'-I'+str(cuda/'include'),'-L'+str(binary),'-lndzip-cuda','-lndzip',
                          '-L'+str(cuda/'lib64'),'-lcudart','-Wl,-rpath,'+str(binary)+':'+str(cuda/'lib64'),
                          '-o',str(adapter/'libcompression_ndzip_ordered.so')])]
    for label,command in commands:
        with (output/(label+'.log')).open('w') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
        if completed.returncode:raise RuntimeError(f'Ordered GPU {label} exit {completed.returncode}')
        print('ORDERED_GPU_REFERENCE_BUILD_STAGE',label,flush=True)
    for name in ('build/compress','build/libndzip.so','build/libndzip-cuda.so','adapter/libcompression_ndzip_ordered.so'):
        record['binaries'][name]=sha(output/name)
    assert all(sha(upstream/name)==expected for name,expected in pinned['vendor_sources'].items())
    assert all(sha(upstream/'build-sm120-module-boost'/name)==expected for name,expected in pinned['binaries'].items())
    record['complete']=True;manifest.write_text(json.dumps(record,indent=2));return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=build(args.upstream,args.output);print('ORDERED_GPU_REFERENCE_BUILD_COMPLETE',result['reference'],flush=True)
