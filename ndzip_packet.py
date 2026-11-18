"""Independent ndzip framing/zero-bit-plane census; not a semantic decoder."""
import math
import numpy as np


def inspect_packet(packet,shape,dtype):
    dtype=np.dtype(dtype);shape=tuple(shape)
    if dtype not in (np.dtype('<f4'),np.dtype('<f8')) or not 1<=len(shape)<=3 or any(int(n)!=n or n<=0 for n in shape):
        raise ValueError('Expected positive 1-D/2-D/3-D FP32/64 geometry')
    raw=memoryview(packet).cast('B');item=dtype.itemsize
    side={1:4096,2:64,3:16}[len(shape)]
    cubes=math.prod(int(n)//side for n in shape)
    header_words=(cubes*4+item-1)//item
    border=math.prod(shape)-cubes*4096
    result={'packet_bytes':len(raw),'source_bytes':math.prod(shape)*item,'hypercubes':cubes,
            'header_words':header_words,'framing_errors':[],'bad_hypercubes':[]}
    errors=result['framing_errors']
    if len(raw)%item or len(raw)<header_words*item:
        errors.append('Truncated/misaligned stream header');return {**result,'valid':False}
    words=np.frombuffer(raw,'<u4' if item==4 else '<u8')
    offsets=np.frombuffer(raw,'<u4',count=cubes)
    last=int(offsets[-1]) if cubes else 0
    if header_words+last+border!=words.size:errors.append('Header/body/border word count differs from packet extent')
    previous=0;minimum=4096//(item*8)
    for index,end in enumerate(offsets):
        end=int(end);length=end-previous
        if length<minimum or length>minimum+4096 or header_words+end>words.size:
            errors.append(f'Invalid hypercube extent at {index}');previous=end;continue
        block=words[header_words+previous:header_words+end]
        predicted=minimum+sum(int(mask).bit_count() for mask in block[:minimum])
        if predicted!=length:
            result['bad_hypercubes'].append({'index':index,'observed_words':length,'predicted_words':predicted})
        previous=end
    result['valid']=not errors and not result['bad_hypercubes']
    return result
