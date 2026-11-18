"""Build isolated CPU-only ndzip causal controls without modifying the pinned source."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
VARIANTS=('stock','delayed','ordered','ordered-delayed')
COPY='memcpy(stream.hypercube(write_task->first_hc_index), write_task->stream.data(),'
ORDERED='memcpy(stream.hypercube(0) + task_stream_offset, write_task->stream.data(),'
WRITE='            if (write_task) {\n'


def variant_source(source,name):
    if name not in VARIANTS:raise ValueError(name)
    if name=='stock':return source
    assert source.count(COPY)==1 and source.count(WRITE)==1
    if name in ('ordered','ordered-delayed'):source=source.replace(COPY,ORDERED)
    if name in ('delayed','ordered-delayed'):
        assert source.count('#pragma once')==1
        source=source.replace('#pragma once','#pragma once\n#include <chrono>\n#include <thread>',1)
        source=source.replace(WRITE,WRITE+
            '                if (write_task->first_hc_index == 0)\n'
            '                    std::this_thread::sleep_for(std::chrono::milliseconds(50));\n',1)
    return source


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_controls(upstream,destination):
    upstream=Path(upstream).resolve();destination=Path(destination).resolve()
    proof=json.loads((ROOT/'results/compression-ndzip-cli-preflight/manifest.json').read_text())
    assert proof['upstream_commit']=='ff4e6702bf0abb86d4aeef8249bd30344dfeef75'
    sources=proof['vendor_sources']
    assert all(sha(upstream/name)==expected for name,expected in sources.items())
    cache={}
    for line in (upstream/'build-sm120-module-boost/CMakeCache.txt').read_text().splitlines():
        if '=' in line and ':' in line.split('=',1)[0]:
            name,value=line.split('=',1);cache[name.split(':')[0]]=value
    boost=Path(cache['Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE']);assert boost.is_file()
    destination.mkdir(parents=True,exist_ok=False)
    record={'schema':'ndzip-cpu-causal-build-v1','upstream_commit':proof['upstream_commit'],
            'upstream_sources':sources,'builder_sha256':sha(__file__),'variants':{},'complete':False,
            'scope':'Diagnostic fork controls, not unmodified upstream or performance measurements'}
    manifest=destination/'manifest.json';manifest.write_text(json.dumps(record,indent=2))
    options=['-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_FLAGS=-march=native',
             '-DNDZIP_WITH_CUDA=OFF','-DNDZIP_WITH_HIPSYCL=OFF','-DNDZIP_WITH_MT=ON',
             '-DNDZIP_BUILD_BENCHMARK=OFF','-DNDZIP_BUILD_TEST=OFF',
             '-DNDZIP_WITH_3RDPARTY_BENCHMARKS=OFF','-DBoost_NO_BOOST_CMAKE=ON',
             '-DBoost_PROGRAM_OPTIONS_LIBRARY_RELEASE='+str(boost),
             '-DBoost_PROGRAM_OPTIONS_LIBRARY_DEBUG='+str(boost)]
    for name in VARIANTS:
        source=destination/name/'source';build=destination/name/'build'
        for relative,expected in sources.items():
            target=source/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(upstream/relative,target);assert sha(target)==expected
        codec=source/'src/ndzip/cpu_codec.inl';before=codec.read_text();after=variant_source(before,name)
        codec.write_text(after)
        patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='upstream/cpu_codec.inl',tofile=name+'/cpu_codec.inl'))
        patch_file=destination/name/'applied.patch';patch_file.write_text(patch)
        for phase,command in (('configure',['cmake','-S',str(source),'-B',str(build),*options]),
                              ('build',['cmake','--build',str(build),'--target','compress','--parallel','4'])):
            log=destination/name/(phase+'.log')
            with log.open('w') as output:
                result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,check=False)
            if result.returncode:raise RuntimeError(f'{name} {phase} failed: exit {result.returncode}')
        linked=subprocess.check_output(['ldd',str(build/'compress')],text=True)
        assert 'libcuda' not in linked and 'libcudart' not in linked,'CPU control must not link GPU runtime'
        (destination/name/'linked-libraries.txt').write_text(linked)
        record['variants'][name]={'source_subdir':str(source.relative_to(destination)),
                                  'build_subdir':str(build.relative_to(destination)),
                                  'sources':{relative:sha(source/relative) for relative in sources},
                                  'patch_sha256':sha(patch_file),
                                  'binaries':{binary:sha(build/binary) for binary in ('compress','libndzip.so')},
                                  'cuda_enabled':False,'openmp_enabled':True,'compile_jobs':4}
        manifest.write_text(json.dumps(record,indent=2));print('CPU_CONTROL_BUILT',name,flush=True)
    assert all(sha(upstream/name)==expected for name,expected in sources.items()),'Pinned vendor changed'
    record['complete']=True;manifest.write_text(json.dumps(record,indent=2));return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=build_controls(args.upstream,args.output);print('ALL_CPU_CONTROLS_BUILT',len(result['variants']),flush=True)
