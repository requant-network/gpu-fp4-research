"""Block SHA256 reference; byte digests certify content, not decoder internals."""
import hashlib
import numpy as np


def digest_chunks(raw, block_bytes=65536):
    assert isinstance(block_bytes, int) and block_bytes > 0
    raw = memoryview(raw).cast('B')
    digests = [hashlib.sha256(raw[offset:offset + block_bytes]).digest()
               for offset in range(0, len(raw), block_bytes)]
    if not digests:
        digests = [hashlib.sha256(b'').digest()]
    return np.frombuffer(b''.join(digests), np.uint8).reshape(-1, 32).copy()


class DeviceIntegrity:
    """GPU digest stage. The caller retains buffers until its CUDA stream completes."""
    def __init__(self, block_bytes=65536):
        import cupy as cp
        from pathlib import Path
        assert isinstance(block_bytes, int) and 0 < block_bytes < 2**32
        self.block_bytes = block_bytes
        code = (Path(__file__).resolve().parent / 'native/compression_integrity.cu').read_text()
        self.kernel = cp.RawKernel(code, 'compression_sha256', options=('--std=c++17',))

    def hash_into(self, data, output):
        import cupy as cp
        assert isinstance(data, cp.ndarray) and data.flags.c_contiguous
        assert isinstance(output, cp.ndarray) and output.flags.c_contiguous
        assert data.device.id == output.device.id == cp.cuda.Device().id
        count = max(1, (data.nbytes + self.block_bytes - 1) // self.block_bytes)
        assert output.shape == (count, 32) and output.dtype == np.dtype('u1')
        self.kernel(((count + 127) // 128,), (128,),
                    (data, np.uint64(data.nbytes), np.uint32(self.block_bytes), output, np.uint64(count)))
