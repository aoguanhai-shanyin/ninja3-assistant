"""One input owner per MuMu instance across worker and diagnostic processes."""

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from hashlib import sha256
from pathlib import Path
import sys


class InputBusy(RuntimeError):
    pass


@contextmanager
def input_owner(folder, index):
    if sys.platform != 'win32':
        yield
        return
    instance = f'{Path(folder).resolve()}\n{index}'.casefold()
    name = 'Local\\Ninja3Assistant.Input.' + sha256(instance.encode('utf-8')).hexdigest()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.ReleaseMutex.argtypes = kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.ReleaseMutex.restype = kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateMutexW(None, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        status = kernel.WaitForSingleObject(handle, 0)
        if status == 258:
            raise InputBusy('当前模拟器的输入已由另一个助手或导航任务占用')
        if status not in (0, 128):  # Acquired or reclaimed from a dead owner.
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            kernel.ReleaseMutex(handle)
    finally:
        kernel.CloseHandle(handle)
