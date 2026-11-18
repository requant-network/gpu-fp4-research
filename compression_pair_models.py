"""Independent equations for XBP1 encoded lengths and pinned ndzip 2-D bounds."""
import numpy as np


def xbp_size(original, block_words=256):
    dtype=np.dtype(original.dtype)
    assert dtype in (np.dtype('<f4'),np.dtype('<f8')) and original.flags.c_contiguous
    assert block_words>=16 and block_words<=4096 and block_words&(block_words-1)==0
    words=original.view('<u4' if dtype.itemsize==4 else '<u8').reshape(-1)
    full,tail=divmod(words.size,block_words)
    if full:
        rows=words[:full*block_words].reshape(full,block_words)
        unions=np.bitwise_or.reduce(rows^rows[:,:1],axis=1).astype(np.uint64)
    else:
        unions=np.zeros(0,np.uint64)
    counts=np.full(full,block_words,np.uint64)
    if tail:
        last=words[full*block_words:]
        unions=np.append(unions,np.uint64(np.bitwise_or.reduce(last^last[0])))
        counts=np.append(counts,np.uint64(tail))
    widths=np.zeros(unions.size,np.uint64)
    remainder=unions.copy()
    for step in (32,16,8,4,2,1):
        high=remainder>>step; mask=high!=0
        widths+=mask.astype(np.uint64)*step
        remainder=np.where(mask,high,remainder)
    widths+=(remainder!=0).astype(np.uint64)
    packed_words=(counts*widths+63)//64
    raw_words=(counts*dtype.itemsize+7)//8
    packed=32+16*counts.size+8*int(np.minimum(packed_words,raw_words).sum(dtype=np.uint64))
    return min(32+original.nbytes,packed)


def ndzip_bound_bytes(shape,dtype):
    assert len(shape)==2 and all(int(s)>0 for s in shape)
    dtype=np.dtype(dtype);assert dtype in (np.dtype('<f4'),np.dtype('<f8'))
    cubes=(int(shape[0])//64)*(int(shape[1])//64)
    header=(cubes+(dtype.itemsize//4)-1)//(dtype.itemsize//4)
    border=int(shape[0])*int(shape[1])-cubes*4096
    per_cube=4096+4096//(dtype.itemsize*8)
    return (header+cubes*per_cube+border)*dtype.itemsize
