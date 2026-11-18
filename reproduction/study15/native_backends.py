"""Native NVML and CuPy/ctypes transport for the Study 15 pilot."""
import ctypes as C
import os
from pathlib import Path
import sys
import uuid

from request_plan import GPU_UUID, REQUEST_BYTES

CAPACITY = 8
OUT_BYTES = 8
CUDA_NOT_READY = 600


def bind(library, name, *argtypes):
    function = getattr(library, name)
    function.argtypes = argtypes
    function.restype = C.c_int
    return function


def timed_energy(call, now, *prefix):
    """Bracket the native call; allocate its output storage before the first stamp."""
    value = C.c_ulonglong()
    ptr = C.byref(value)
    start = now()
    status = call(*prefix, ptr)
    end = now()
    return {'start_ns': start, 'end_ns': end, 'nvml_status': status,
            'status': 'success' if status == 0 else 'nvml-error',
            'mJ': value.value if status == 0 else None}


class Utilization(C.Structure):
    _fields_ = [('gpu', C.c_uint), ('memory', C.c_uint)]


def load_nvml():
    if sys.platform.startswith('linux'):
        candidates = ['libnvidia-ml.so.1']  # Never load an SDK stub by unversioned name.
    elif sys.platform == 'win32':
        candidates = [str(Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'nvml.dll'),
                      str(Path(os.environ.get('ProgramW6432', r'C:\Program Files')) /
                          'NVIDIA Corporation' / 'NVSMI' / 'nvml.dll')]
    else:
        raise RuntimeError(f'Unsupported NVML platform: {sys.platform}')
    errors = []
    for name in candidates:
        try:
            return C.CDLL(name)
        except OSError as exc:
            errors.append(f'{name}: {exc}')
    raise OSError('Could not load the NVIDIA driver NVML library: ' + '; '.join(errors))


class NativeNvml:
    def __init__(self, gpu_uuid, *, library=None):
        self._library = load_nvml() if library is None else library
        lib = self._library
        self._closed = True
        self._handle = C.c_void_p()
        init = bind(lib, 'nvmlInit_v2')
        self._shutdown = bind(lib, 'nvmlShutdown')
        get_handle = bind(lib, 'nvmlDeviceGetHandleByUUID', C.c_char_p, C.POINTER(C.c_void_p))
        self._uuid = bind(lib, 'nvmlDeviceGetUUID', C.c_void_p, C.POINTER(C.c_char), C.c_uint)
        self._driver = bind(lib, 'nvmlSystemGetDriverVersion', C.POINTER(C.c_char), C.c_uint)
        self._version = bind(lib, 'nvmlSystemGetNVMLVersion', C.POINTER(C.c_char), C.c_uint)
        self._energy = bind(lib, 'nvmlDeviceGetTotalEnergyConsumption', C.c_void_p, C.POINTER(C.c_ulonglong))
        self._util = (bind(lib, 'nvmlDeviceGetUtilizationRates', C.c_void_p, C.POINTER(Utilization))
                      if hasattr(lib, 'nvmlDeviceGetUtilizationRates') else None)
        self._check('nvmlInit_v2', init())
        self._closed = False
        try:
            self._check('nvmlDeviceGetHandleByUUID', get_handle(gpu_uuid.encode('ascii'), C.byref(self._handle)))
            if not self._handle:
                raise RuntimeError('nvmlDeviceGetHandleByUUID returned a null handle')
        except (Exception, KeyboardInterrupt) as error:
            try:
                self.close()
            except (Exception, KeyboardInterrupt) as cleanup:
                error.add_note(str(cleanup))
            raise

    @staticmethod
    def _check(operation, status):
        if status != 0:
            raise RuntimeError(f'{operation} returned NVML status {status}')

    def _string(self, function, operation, *prefix):
        buf = C.create_string_buffer(128)
        self._check(operation, function(*prefix, buf, len(buf)))
        return buf.value.decode('ascii')

    def snapshot(self):
        row = {'gpu_uuid': self._string(self._uuid, 'nvmlDeviceGetUUID', self._handle),
               'driver_version': self._string(self._driver, 'nvmlSystemGetDriverVersion'),
               'nvml_version': self._string(self._version, 'nvmlSystemGetNVMLVersion'),
               'nvml_load_name': str(self._library._name)}
        if self._util is not None:
            value = Utilization()
            status = self._util(self._handle, C.byref(value))
            row['utilization_snapshot'] = {'nvml_status': status, 'gpu_percent': value.gpu if status == 0 else None,
                                           'memory_percent': value.memory if status == 0 else None}
        else:
            row['utilization_snapshot'] = {'status': 'api-unavailable'}
        return row

    def energy(self, now):
        if self._closed:
            raise RuntimeError('NVML session is closed')
        return timed_energy(self._energy, now, self._handle)

    def close(self):
        if not self._closed:
            self._closed = True
            self._check('nvmlShutdown', self._shutdown())


def cuda_uuid(raw):
    # CuPy exposes cudaDeviceProp.uuid.bytes. Do not infer identity from the PCI bus or model.
    if isinstance(raw, bytes) and len(raw) == 16:
        return 'GPU-' + str(uuid.UUID(bytes=raw))
    return None


class DriverUuid:
    """Retrieve binary UUIDs without Cython char-array conversion."""
    _buffer = C.c_ubyte * 16

    def __init__(self, library=None):
        if library is None:
            if sys.platform == 'win32':
                path = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'nvcuda.dll'
                library = C.WinDLL(str(path))
            elif sys.platform.startswith('linux'):
                library = C.CDLL('libcuda.so.1')
            else:
                raise RuntimeError(f'Unsupported CUDA Driver platform: {sys.platform}')
        self._library = library
        init = bind(library, 'cuInit', C.c_uint)
        self._device = bind(library, 'cuDeviceGet', C.POINTER(C.c_int), C.c_int)
        self._uuid = bind(library, 'cuDeviceGetUuid_v2', C.POINTER(self._buffer), C.c_int)
        self._check('cuInit', init(0))

    @staticmethod
    def _check(operation, status):
        if status != 0:
            raise RuntimeError(f'{operation} returned CUDA Driver status {status}')

    def get(self, ordinal):
        device, binary = C.c_int(), self._buffer()
        self._check('cuDeviceGet', self._device(C.byref(device), ordinal))
        self._check('cuDeviceGetUuid_v2', self._uuid(C.byref(binary), device.value))
        return cuda_uuid(bytes(binary))


class CudaDot:
    def __init__(self, gpu_uuid, kernel, *, cupy=None, library=None, driver_library=None):
        if cupy is None:
            if os.environ.get('CUDA_LAUNCH_BLOCKING') not in (None, '', '0'):
                raise ValueError('CUDA_LAUNCH_BLOCKING must be disabled for acquisition')
            import cupy
        if sys.byteorder != 'little':
            raise RuntimeError('Only little-endian CUDA hosts are supported')
        self._rt = rt = cupy.cuda.runtime
        if getattr(rt, 'is_hip', False):
            raise RuntimeError('Study 15 requires NVIDIA CUDA, not HIP')
        if not GPU_UUID.fullmatch(gpu_uuid):
            raise ValueError('CUDA selection requires a physical GPU UUID')
        visible = []
        matches = []
        driver_uuid = None
        for i in range(rt.getDeviceCount()):
            props = rt.getDeviceProperties(i)
            observed = cuda_uuid(props.get('uuid'))
            source = 'cudaDeviceProp.uuid'
            if observed is None:
                if driver_uuid is None:
                    driver_uuid = DriverUuid(driver_library)
                observed = driver_uuid.get(i)  # CUDA-visible ordinal on both API paths.
                source = 'cuDeviceGetUuid_v2'
            visible.append(observed)
            if observed is not None and observed.casefold() == gpu_uuid.casefold():
                matches.append((i, observed, source, props))
        if len(matches) != 1 or None in visible:
            raise ValueError(f'CUDA GPU UUID could not be uniquely resolved from visible UUIDs: {visible}')
        ordinal, observed, source, props = matches[0]
        self._device = cupy.cuda.Device(ordinal)
        self._device.use()
        self._library = C.CDLL(str(kernel)) if library is None else library
        self._launch = bind(self._library, 'study15_dot_launch', C.c_void_p, C.c_void_p, C.c_uint, C.c_void_p)
        self._stream = cupy.cuda.Stream(non_blocking=True)
        self._event = cupy.cuda.Event(disable_timing=True)
        self._h_in = cupy.cuda.alloc_pinned_memory(CAPACITY * REQUEST_BYTES)
        self._h_out = cupy.cuda.alloc_pinned_memory(CAPACITY * OUT_BYTES)
        self._d_in = cupy.cuda.alloc(CAPACITY * REQUEST_BYTES)
        self._d_out = cupy.cuda.alloc(CAPACITY * OUT_BYTES)
        self._active_count = 0
        self._event_recorded = False
        self._closed = False
        name = props['name']
        self._identity = {'gpu_uuid': observed, 'visible_uuids': visible, 'ordinal': ordinal,
                          'uuid_source': source,
                          'name': name.decode('utf-8', errors='replace') if isinstance(name, bytes) else name,
                          'compute_capability': [props['major'], props['minor']],
                          'pci_bus_id': rt.deviceGetPCIBusId(ordinal), 'cupy_version': cupy.__version__,
                          'cupy_module': getattr(cupy, '__file__', None),
                          'cupy_cuda_runtime_version': rt.runtimeGetVersion(),
                          'cuda_driver_api_version': rt.driverGetVersion(),
                          'dot_library_path': str(kernel), 'capacity': CAPACITY,
                          'stream': 'explicit nonblocking', 'pinned_host_bytes': CAPACITY * (REQUEST_BYTES + OUT_BYTES),
                          'device_buffer_bytes': CAPACITY * (REQUEST_BYTES + OUT_BYTES)}

    def snapshot(self):
        return dict(self._identity)

    def submit(self, records):
        if self._closed or self._active_count:
            raise RuntimeError('CUDA transport is closed or has work in flight')
        n = len(records)
        if not 1 <= n <= CAPACITY or any(type(r) is not bytes or len(r) != REQUEST_BYTES for r in records):
            raise ValueError('CUDA batch must contain 1..8 complete immutable input records')
        for i, data in enumerate(records):
            C.memmove(self._h_in.ptr + i * REQUEST_BYTES, data, REQUEST_BYTES)
        # Hold every pointer even if enqueue succeeds only partly and throws.
        self._active_count = n
        rt, stream = self._rt, self._stream
        rt.memcpyAsync(self._d_in.ptr, self._h_in.ptr, n * REQUEST_BYTES, rt.memcpyHostToDevice, stream.ptr)
        status = self._launch(self._d_in.ptr, self._d_out.ptr, n, stream.ptr)
        if status != 0:
            raise RuntimeError(f'study15_dot_launch returned CUDA launch status {status}')
        rt.memcpyAsync(self._h_out.ptr, self._d_out.ptr, n * OUT_BYTES, rt.memcpyDeviceToHost, stream.ptr)
        self._event.record(stream)
        self._event_recorded = True

    def poll(self):
        if self._closed or not self._event_recorded:
            raise RuntimeError('No CUDA completion event is in flight')
        status = self._rt.eventQuery(self._event.ptr)
        if status == CUDA_NOT_READY:
            return None
        if status != 0:
            raise RuntimeError(f'CUDA event completion query returned status {status}')
        raw = C.string_at(self._h_out.ptr, self._active_count * OUT_BYTES)
        self._active_count = 0
        self._event_recorded = False
        return tuple(raw[i:i + OUT_BYTES] for i in range(0, len(raw), OUT_BYTES))

    def warmup(self, records):
        self.submit(records)
        self._stream.synchronize()  # Never used on the active-window owner loop.
        result = self.poll()
        if result is None:
            raise RuntimeError('CUDA event is not ready after warmup stream synchronization')
        return result

    def close(self):
        if not self._closed:
            self._closed = True
            self._stream.synchronize()  # Drain even after partial enqueue; do not reset the device.
            self._event = self._h_in = self._h_out = self._d_in = self._d_out = self._stream = None
