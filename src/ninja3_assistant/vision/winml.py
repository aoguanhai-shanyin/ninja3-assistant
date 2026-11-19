"""Select the installed WinML TensorRT RTX provider for Python ONNX Runtime."""
import ctypes as ct
from functools import cache
import logging
import os
from pathlib import Path
import sys
from threading import Lock

import onnxruntime as ort

from ninja3_assistant.paths import project_root


_LOG = logging.getLogger(__name__)
_PROVIDER = 'NvTensorRTRTXExecutionProvider'
_DLL = 'Microsoft.Windows.AI.MachineLearning.dll'
_GPU_BACKEND = 'GPU: Windows ML / TensorRT RTX'
_DEVICE_LOCK = Lock()


def _provider_library(path):
    api = ct.WinDLL(str(path))

    def bind(name, *arguments, result=ct.c_long):
        function = getattr(api, name)
        function.argtypes = arguments
        function.restype = result
        return function

    def check(function, *arguments):
        status = function(*arguments)
        if status < 0:
            raise OSError(f'{function.__name__}: HRESULT 0x{status & 0xffffffff:08X}')

    handle, number, size = ct.c_void_p, ct.c_int, ct.c_size_t
    create = bind('WinMLEpCatalogCreate', ct.POINTER(handle))
    release = bind('WinMLEpCatalogRelease', handle, result=None)
    find = bind('WinMLEpCatalogFindProvider', handle, ct.c_char_p, ct.c_char_p, ct.POINTER(handle))
    certification = bind('WinMLEpGetCertification', handle, ct.POINTER(number))
    ready_state = bind('WinMLEpGetReadyState', handle, ct.POINTER(number))
    activate = bind('WinMLEpEnsureReady', handle)
    path_size = bind('WinMLEpGetLibraryPathSize', handle, ct.POINTER(size))
    path_read = bind('WinMLEpGetLibraryPath', handle, size, ct.c_char_p, ct.POINTER(size))

    catalog, provider = handle(), handle()
    check(create, ct.byref(catalog))
    try:
        check(find, catalog, _PROVIDER.encode('ascii'), None, ct.byref(provider))
        if not provider:
            return None
        certified, state = number(), number()
        check(certification, provider, ct.byref(certified))
        check(ready_state, provider, ct.byref(state))
        # Certified=1, Ready=0, NotReady=1. Activate only an installed provider.
        if certified.value != 1 or state.value not in (0, 1):
            return None
        if state.value == 1:
            check(activate, provider)
        length = size()
        check(path_size, provider, ct.byref(length))
        path = ct.create_string_buffer(length.value)
        check(path_read, provider, length.value, path, None)
        return path.value.decode('utf-8') or None
    finally:
        release(catalog)


def winml_gpu():
    """Return an ORT device supplied by WinML, if usable in this worker."""
    with _DEVICE_LOCK:
        return _winml_gpu()


@cache
def _winml_gpu():
    if sys.platform != 'win32':
        return None
    runtime = (Path(sys.executable).parent if getattr(sys, 'frozen', False) else
               project_root()/'dist'/'Ninja3Assistant')
    try:
        existing = next((device for device in ort.get_ep_devices() if device.ep_name == _PROVIDER), None)
        if existing is not None:
            return existing
        dll = runtime/_DLL
        library = _provider_library(dll) if dll.is_file() else None
        if library is None:
            return None
        ort.register_execution_provider_library(_PROVIDER, library)
        return next((device for device in ort.get_ep_devices() if device.ep_name == _PROVIDER), None)
    except (AttributeError, OSError, RuntimeError) as error:
        _LOG.warning('Windows ML GPU unavailable: %s', error)
        return None


def _provider_options():
    local = os.environ.get('LOCALAPPDATA')
    if not local:
        return {}
    directory = Path(local)/'Ninja3Assistant'/'winml-runtime-cache'
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return {}
    return {'nv_runtime_cache_path': str(directory)}


def model_session(model, *, use_gpu=True, require_gpu=False, options=None):
    """Create one ONNX session with WinML GPU or the caller's permitted CPU path."""
    source = model if isinstance(model, bytes) else os.fspath(model)
    device = winml_gpu() if use_gpu else None
    if device is None:
        if require_gpu:
            raise RuntimeError('当前环境没有可用的 Windows ML GPU 推理设备')
        return ort.InferenceSession(source, sess_options=options,
                                    providers=['CPUExecutionProvider']), 'CPU'
    if options is None:
        options = ort.SessionOptions()
    options.add_provider_for_devices([device], _provider_options())
    options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    return ort.InferenceSession(source, sess_options=options, enable_fallback=False), _GPU_BACKEND
