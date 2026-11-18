"""Original native CPU XBP1 encoder, fast format validation and guarded CUDA decoding."""
import ctypes as C
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


class XbpNative:
    def __init__(self):
        self.lib = C.CDLL(str(ROOT / 'build/libxbp_cpu.so'))
        self.lib.xbp_error.argtypes = []; self.lib.xbp_error.restype = C.c_char_p
        self.lib.xbp_bound.argtypes = [C.c_uint64, C.c_uint32, C.c_uint32]
        self.lib.xbp_bound.restype = C.c_uint64
        self.lib.xbp_encode.argtypes = [C.c_void_p, C.c_uint64, C.c_uint32, C.c_uint32,
                                       C.c_uint32, C.c_void_p, C.c_uint64, C.POINTER(C.c_uint64)]
        self.lib.xbp_encode.restype = C.c_int
        self.lib.xbp_validate.argtypes = [C.c_void_p, C.c_uint64, C.POINTER(C.c_uint64)]
        self.lib.xbp_validate.restype = C.c_int
        self.kernel = None

    def _check(self, status):
        if status:
            raise ValueError(self.lib.xbp_error().decode())

    def bound(self, words, dtype, block_words=256):
        dtype = np.dtype(dtype)
        require(dtype in (np.dtype('<f4'), np.dtype('<f8')), 'Expected little-endian FP32/FP64')
        require(isinstance(words, int) and 0 <= words <= 2**32-1, 'Native XBP1 word-count limit')
        require(isinstance(block_words, int) and 16 <= block_words <= 4096 and
                block_words & (block_words-1) == 0, 'Invalid XBP1 block size')
        size = int(self.lib.xbp_bound(words, dtype.itemsize, block_words))
        if not size:
            raise ValueError(self.lib.xbp_error().decode())
        return size

    def encode(self, original, block_words=256, threads=1, out=None):
        require(isinstance(original, np.ndarray) and original.flags.c_contiguous, 'Expected contiguous input')
        require(isinstance(threads, int) and 1 <= threads <= 64, 'Invalid CPU thread count')
        size = self.bound(original.size, original.dtype, block_words)
        output = np.empty(size, np.uint8) if out is None else out
        require(isinstance(output, np.ndarray) and output.dtype == np.dtype('u1') and
                output.flags.c_contiguous and output.nbytes >= size and output.ctypes.data % 8 == 0,
                'Invalid/alignment-limited XBP1 candidate buffer')
        written = C.c_uint64()
        self._check(self.lib.xbp_encode(original.ctypes.data, original.size, original.dtype.itemsize,
                                        block_words, threads, output.ctypes.data, output.nbytes, C.byref(written)))
        require(32 <= written.value <= size, 'Native encoder returned invalid byte count')
        return output.reshape(-1)[:written.value]

    def validate(self, packet):
        if isinstance(packet, np.ndarray):
            require(packet.dtype == np.dtype('u1') and packet.flags.c_contiguous, 'Expected contiguous packet bytes')
            array = packet.reshape(-1)
        else:
            array = np.frombuffer(memoryview(packet).cast('B'), np.uint8)
        metadata = (C.c_uint64 * 8)()
        self._check(self.lib.xbp_validate(array.ctypes.data, array.nbytes, metadata))
        keys = ('words', 'itemsize', 'block_log2', 'flags', 'blocks', 'total_bytes', 'payload_begin', 'block_words')
        result = dict(zip(keys, map(int, metadata)))
        result.update(dtype='<f4' if result['itemsize'] == 4 else '<f8',
                      mode='raw' if result['flags'] else 'packed')
        return result

    def decode(self, packet, output, metadata, status):
        """Asynchronous. Caller retains all buffers and checks status after completion."""
        import cupy as cp
        require(isinstance(packet, cp.ndarray) and packet.dtype == np.dtype('u1') and
                packet.flags.c_contiguous and packet.nbytes >= 32 and packet.data.ptr % 8 == 0,
                'Invalid/alignment-limited device packet')
        require(isinstance(output, cp.ndarray) and output.flags.c_contiguous and
                output.dtype in (np.dtype('<f4'), np.dtype('<f8')), 'Invalid decoded output')
        require(isinstance(status, cp.ndarray) and status.dtype == np.dtype('<u4') and
                status.shape == (1,) and status.flags.c_contiguous, 'Invalid device status result')
        require(packet.device.id == output.device.id == status.device.id == cp.cuda.Device().id,
                'XBP1 device mismatch')
        n, size, log, flags, blocks = (metadata[key] for key in ('words', 'itemsize', 'block_log2', 'flags', 'blocks'))
        require(0 <= n <= 2**32-1 and size in (4, 8) and 4 <= log <= 12 and flags in (0, 1),
                'Invalid XBP1 host metadata')
        require(n == output.size and size == output.dtype.itemsize and packet.nbytes == metadata['total_bytes'],
                'XBP1 output or packet length differs from validated metadata')
        require((flags == 1 and blocks == 0 and packet.nbytes == 32+n*size) or
                (flags == 0 and n > 0 and blocks == (n+(1 << log)-1)//(1 << log)),
                'Invalid XBP1 host block geometry')
        if self.kernel is None:
            self.kernel = cp.RawKernel((ROOT / 'native/xbp_decode.cu').read_text(), 'xbp_decode',
                                       options=('--std=c++17',))
        status.fill(0)
        grid = max(1, (n+255)//256 if flags else blocks)
        self.kernel((grid,), (256,), (packet, np.uint64(packet.nbytes), output, np.uint64(n),
                                     np.uint32(size), np.uint32(log), np.uint32(flags), np.uint64(blocks), status))
