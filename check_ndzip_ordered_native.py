"""Full-word GPU preflight for the explicitly patched ndzip reference; not timing evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compression_native_ordered import OrderedNdzipPlan
from compression_pair_cases import input_words
from ndzip_packet import inspect_packet

ROOT=Path(__file__).resolve().parent
SOURCES=('check_ndzip_ordered_native.py','compression_native_ordered.py','build_ndzip_ordered_gpu.py',
         'compression_native.py','compression_pair_cases.py','ndzip_packet.py',
         'native/compression_ndzip_bridge.cpp','build_ndzip_cpu_controls.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    import cupy as cp
    cp.cuda.Device(0).use();args.output.mkdir(parents=True,exist_ok=False)
    manifest=args.output/'manifest.json'
    props=cp.cuda.runtime.getDeviceProperties(0)
    device={'name':props['name'].decode() if isinstance(props['name'],bytes) else props['name'],
            'total_memory_bytes':int(props['totalGlobalMem']),'major':int(props['major']),'minor':int(props['minor']),
            'runtime_api':cp.cuda.runtime.runtimeGetVersion(),'driver_api':cp.cuda.runtime.driverGetVersion()}
    record={'schema':'ndzip-ordered-native-v1','reference':'ndzip-ff4e6702+ordered-copy',
            'device':device,'numpy':np.__version__,'cupy':cp.__version__,
            'sources':{name:sha(ROOT/name) for name in SOURCES},
            'build_manifest_sha256':sha(args.build_root/'manifest.json'),
            'build':json.loads((args.build_root/'manifest.json').read_text()),
            'cases':[],'complete':False,'scope':'All retained original/packet/restored words; no throughput, full-study or full-stream claim'}
    manifest.write_text(json.dumps(record,indent=2))
    for dtype in ('<f4','<f8'):
        count=64*1024**2//np.dtype(dtype).itemsize
        datasets=[((129,131),'random-bits',0),((17,18,19),'random-bits',7),
                  ((count//128,128),'smooth',0),((count//128,128),'smooth',54*count)]
        for shape,profile,offset in datasets:
            original=input_words(profile,dtype,offset,int(np.prod(shape)),71).reshape(shape)
            original_hash=hashlib.sha256(original).hexdigest()
            for threads in (1,4):
                for stream_name in ('default','nonblocking'):
                    stream=cp.cuda.Stream.null if stream_name=='default' else cp.cuda.Stream(non_blocking=True)
                    with stream,OrderedNdzipPlan(shape,dtype,threads,stream,build_root=args.build_root) as plan:
                        encoded=plan.encode(original)
                        assert inspect_packet(encoded,shape,dtype)['valid'],'Patched encoder produced structurally invalid packet'
                        device_packet=cp.asarray(encoded);device_output=cp.empty(shape,np.dtype(dtype))
                        plan.decode(device_packet,device_output);plan.synchronize()
                        restored=cp.asnumpy(device_output)
                        words='<u4' if np.dtype(dtype).itemsize==4 else '<u8'
                        assert np.array_equal(restored.view(words),original.view(words)),'Patched GPU decoder differs from original input'
                        assert hashlib.sha256(original).hexdigest()==original_hash,'Original host data changed'
                        signature=hashlib.sha256(original_hash.encode()+encoded.tobytes()+restored.tobytes()).hexdigest()
                        relative=f'case-{signature}.npz';path=args.output/relative
                        arrays={'original':original,'encoded':encoded,'restored':restored}
                        if path.exists():
                            with np.load(path,allow_pickle=False) as existing:
                                assert all(existing[name].dtype==array.dtype and existing[name].shape==array.shape and
                                           existing[name].tobytes()==array.tobytes() for name,array in arrays.items())
                        else:np.savez_compressed(path,**arrays)
                        record['cases'].append({'index':len(record['cases']),'dtype':dtype,'shape':list(shape),
                            'profile':profile,'offset_words':offset,'seed':71,'threads':threads,'stream':stream_name,
                            'original_sha256':original_hash,'encoded_bytes':encoded.nbytes,
                            'file':relative,'sha256':sha(path),'loaded_reference':plan.loaded_reference})
                    manifest.write_text(json.dumps(record,indent=2))
                    print('ORDERED_NDZIP_FULL_WORD_GPU_CASE',len(record['cases']),dtype,shape,profile,threads,stream_name,flush=True)
    assert record['sources']=={name:sha(ROOT/name) for name in SOURCES}
    assert record['build_manifest_sha256']==sha(args.build_root/'manifest.json')
    record['complete']=True;manifest.write_text(json.dumps(record,indent=2))
    print('ORDERED_NDZIP_NATIVE_PROCESS_COMPLETE',len(record['cases']),flush=True)
