"""Lossless complete-vector retention for physical-operator/solver observations.

Deduplication is exact-byte plus input identity; nondeterministic outputs retain
all IEEE bits as full vectors or reversible XOR deltas. No numeric sampling.
"""
import hashlib
import re
import numpy as np


def array_hash(value):return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


class OutputArchive:
    def __init__(self):
        self.arrays={};self.records={};self._bases={};self._seen={}

    def add(self,output,label,input_sha256):
        output=np.asarray(output)
        if output.ndim!=1 or output.dtype!=np.dtype('<f8'):raise ValueError('Full little-endian FP64 vector required')
        if not isinstance(label,str) or not label or not re.fullmatch('[0-9a-f]{64}',input_sha256):raise ValueError('Literal valid observation identity required')
        output_sha256=array_hash(output)
        identity=(label,input_sha256,output_sha256)
        if identity in self._seen:return self._seen[identity]
        record_id='o'+str(len(self.records));key='p'+str(len(self.arrays))
        record=dict(label=label,key=key,input_sha256=input_sha256,output_sha256=output_sha256,shape=list(output.shape),dtype='<f8')
        if label not in self._bases:
            self.arrays[key]=np.ascontiguousarray(output).copy()
            record.update(encoding='full',base=None)
            self._bases[label]=key
        else:
            base=self._bases[label]
            if self.arrays[base].shape!=output.shape:raise ValueError('Observation namespace changed geometry')
            self.arrays[key]=np.bitwise_xor(output.view(np.uint64),self.arrays[base].view(np.uint64))
            record.update(encoding='xor64',base=base,base_sha256=array_hash(self.arrays[base]))
        self.records[record_id]=record;self._seen[identity]=record_id
        return record_id


def decode_output(record,arrays,expected_input_sha256):
    assert record['input_sha256']==expected_input_sha256,'Output input identity changed'
    assert record['dtype']=='<f8' and len(record['shape'])==1,'Full vector schema changed'
    payload=np.asarray(arrays[record['key']])
    assert list(payload.shape)==record['shape'],'Full output shape changed'
    if record['encoding']=='full':
        assert payload.dtype==np.dtype('<f8') and record['base'] is None,'Full IEEE schema changed'
        value=payload.copy()
    else:
        assert record['encoding']=='xor64' and payload.dtype==np.dtype('<u8'),'Unknown lossless encoding'
        base=np.asarray(arrays[record['base']])
        assert base.dtype==np.dtype('<f8') and base.shape==payload.shape and array_hash(base)==record['base_sha256'],'XOR base bytes changed'
        value=np.bitwise_xor(payload,base.view(np.uint64)).view(np.float64)
    assert array_hash(value)==record['output_sha256'],'Retained output bytes changed'
    return value
