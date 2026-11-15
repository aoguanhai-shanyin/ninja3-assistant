import ctypes
from ctypes import wintypes
from uuid import UUID

class Guid(ctypes.Structure):
    _fields_ = [('data1', wintypes.DWORD), ('data2', wintypes.WORD), ('data3', wintypes.WORD), ('data4', ctypes.c_ubyte * 8)]
class Luid(ctypes.Structure):
    _fields_ = [('low', wintypes.DWORD), ('high', wintypes.LONG)]
class Desc(ctypes.Structure):
    _fields_ = [('name', wintypes.WCHAR * 128), ('vendor', wintypes.UINT), ('device', wintypes.UINT),
                ('subsys', wintypes.UINT), ('revision', wintypes.UINT), ('dedicated', ctypes.c_size_t),
                ('dedicated_system', ctypes.c_size_t), ('shared', ctypes.c_size_t), ('luid', Luid), ('flags', wintypes.UINT)]

def invoke(obj, index, restype, argtypes, *args):
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    function = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[index])
    return function(obj, *args)

def adapters():
    factory = ctypes.c_void_p()
    guid = Guid.from_buffer_copy(UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
    result = ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(guid), ctypes.byref(factory))
    if result != 0: raise OSError('CreateDXGIFactory1 failed')
    try:
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            result = invoke(factory, 12, ctypes.c_long, [wintypes.UINT, ctypes.POINTER(ctypes.c_void_p)], index, ctypes.byref(adapter))
            if result != 0: break
            try:
                desc = Desc()
                result = invoke(adapter, 10, ctypes.c_long, [ctypes.POINTER(Desc)], ctypes.byref(desc))
                if result != 0: raise OSError('GetDesc1 failed')
                yield index, desc.name, desc.vendor, desc.dedicated
            finally:
                invoke(adapter, 2, wintypes.ULONG, [])
            index += 1
    finally:
        invoke(factory, 2, wintypes.ULONG, [])

if __name__ == '__main__':
    for index, name, vendor, dedicated in adapters():
        print(f'adapter={index} name={name} vendor={vendor:#x} dedicated_mb={dedicated // 1024 // 1024}')
