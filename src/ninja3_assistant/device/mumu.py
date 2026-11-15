"""MuMu's local capture and touch SDK; no mouse cursor control is required."""
import ctypes
import json
import os
from pathlib import Path
import subprocess

import numpy as np

GAME = 'com.pandadastudio.ninjamustdie3.mi'


def set_window_visibility(directory,index,*,hidden):
    result = subprocess.run([str(Path(directory)/'nx_main/MuMuManager.exe'),'control',
        '--vmindex',str(index),'hide_window' if hidden else 'show_window'],
        capture_output=True,timeout=8,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    payload = json.loads(result.stdout.decode('utf-8-sig'))
    if result.returncode or payload.get('errcode',0) or payload.get('error_code',0):
        raise RuntimeError('无法设置所选 MuMu 窗口的显示状态')

def instance_serial(payload, index):
    info = payload if 'adb_port' in payload else payload.get(str(index), {})
    if 'adb_host_ip' not in info or 'adb_port' not in info:
        raise RuntimeError('无法读取 MuMu 实例的 ADB 地址')
    return f"{info['adb_host_ip']}:{info['adb_port']}"


def prepare_game(directory, index, control, status):
    """Start only the selected instance and launch the game if it is stopped."""
    manager = Path(directory) / 'nx_main/MuMuManager.exe'
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

    def call(*args):
        result = subprocess.run([str(manager), *args], capture_output=True, timeout=8,
                                creationflags=flags)
        try:
            payload = json.loads(result.stdout.decode('utf-8-sig'))
        except (ValueError, UnicodeDecodeError) as error:
            raise RuntimeError('无法读取 MuMu 的启动状态') from error
        if result.returncode or payload.get('errcode', 0) or payload.get('error_code', 0):
            raise RuntimeError('MuMu 操作失败：' + str(payload.get('errmsg') or payload.get('launch_err_msg') or '请检查实例状态'))
        return payload

    index = str(index)
    deadline = control.now() + 120
    requested = False
    def apply_visibility():
        hidden = getattr(control,'hide_game_window',None)
        if isinstance(hidden,bool) and not control.stop.is_set():
            set_window_visibility(directory,index,hidden=hidden)
    while control.wait_active():
        info = call('info', '--vmindex', index)
        if info.get('is_process_started') and info.get('is_android_started'):
            break
        if not info.get('is_process_started') and not requested:
            status('正在启动所选模拟器…')
            launched = control.run_if_active(lambda: call('control', '--vmindex', index, 'launch'))
            if launched is False:
                continue
            requested = True
        else:
            status('等待模拟器启动完成…')
        if control.now() >= deadline:
            raise RuntimeError('等待模拟器启动超时，请检查 MuMu')
        control.stop.wait(.5)
    else:
        return None

    if not control.wait_active():
        return None
    state = call('control', '--vmindex', index, 'app', 'info', '--package', GAME).get('state')
    if state == 'not_installed':
        raise RuntimeError('所选模拟器未安装当前版本的忍者必须死3')
    if state == 'running':
        apply_visibility()
        return requested
    if state != 'stopped':
        raise RuntimeError('无法确认游戏是否已安装或启动')
    while control.wait_active():
        status('正在打开忍者必须死3…')
        launched = control.run_if_active(lambda: call('control', '--vmindex', index, 'app',
                                                       'launch', '--package', GAME))
        if launched is not False:
            apply_visibility()
            return True
    return None

class MuMu:
    def __init__(self, directory, index):
        self.root = Path(directory)
        self.index = int(index)
        candidates = [self.root / 'nx_device/12.0/shell/sdk', self.root / 'nx_main/sdk']
        folder = next((path for path in candidates if (path / 'external_renderer_ipc.dll').is_file()), None)
        if not folder:
            raise RuntimeError('找不到 MuMu 的截图接口，请检查安装目录')
        self.dll_directory = os.add_dll_directory(str(folder))
        self.dll = ctypes.CDLL(str(folder / 'external_renderer_ipc.dll'))
        self.dll.nemu_connect.argtypes = [ctypes.c_wchar_p, ctypes.c_int]
        self.dll.nemu_connect.restype = ctypes.c_int
        self.dll.nemu_disconnect.argtypes = [ctypes.c_int]
        self.dll.nemu_get_display_id.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.dll.nemu_get_display_id.restype = ctypes.c_int
        self.dll.nemu_capture_display.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ubyte)]
        self.dll.nemu_input_event_touch_down.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.dll.nemu_input_event_touch_up.argtypes = [ctypes.c_int, ctypes.c_int]
        self.handle = self.dll.nemu_connect(str(self.root), self.index)
        if self.handle <= 0:
            raise RuntimeError('无法连接 MuMu 实例，请确认实例正在运行')
        try:
            self.display = self.dll.nemu_get_display_id(self.handle, GAME.encode(), 0)
            if self.display != 0:
                raise RuntimeError('当前游戏显示屏不支持安全触控，请关闭模拟器后台保活后重试')
            self.width, self.height = ctypes.c_int(), ctypes.c_int()
            result = self.dll.nemu_capture_display(self.handle, self.display, 0,
                    ctypes.byref(self.width), ctypes.byref(self.height), None)
            if result or (self.width.value, self.height.value) != (1280, 720):
                raise RuntimeError('请将 MuMu 游戏分辨率设置为 1280×720 横屏')
            self.size = self.width.value * self.height.value * 4
            self.pixels = (ctypes.c_ubyte * self.size)()
        except Exception:
            self.close()
            raise

    def capture(self):
        result = self.dll.nemu_capture_display(self.handle, self.display, self.size,
                ctypes.byref(self.width), ctypes.byref(self.height), self.pixels)
        if result or (self.width.value, self.height.value) != (1280, 720):
            raise RuntimeError('取图失败或屏幕尺寸发生变化')
        # The SDK provides bottom-up RGBA pixels.
        return np.frombuffer(self.pixels, dtype=np.uint8).reshape(720, 1280, 4)[::-1, :, :3].copy()

    def tap(self, point, stopped, hold_seconds=.07, on_pressed=None):
        x, y = point
        if stopped.is_set():
            return False
        if not (0 <= x < 1280 and 0 <= y < 720):
            raise RuntimeError('识别坐标超出屏幕')
        down = self.dll.nemu_input_event_touch_down(self.handle, self.display, x, y)
        try:
            if not down and on_pressed is not None:
                on_pressed()
            stopped.wait(hold_seconds)
        finally:
            up = self.dll.nemu_input_event_touch_up(self.handle, self.display)
        if down or up:
            raise RuntimeError('MuMu 触控失败')
        return not stopped.is_set()

    def swipe(self, start, end, stopped, *, step_seconds=.005, steps=4, on_pressed=None):
        """Send a continuous SDK contact, releasing it even if cancelled."""
        if stopped.is_set():
            return False
        if steps < 1 or step_seconds < 0:
            raise ValueError('invalid swipe timing')
        if any(not (0 <= x < 1280 and 0 <= y < 720) for x, y in (start, end)):
            raise RuntimeError('识别坐标超出屏幕')
        down = self.dll.nemu_input_event_touch_down(self.handle, self.display, *start)
        try:
            if down:
                raise RuntimeError('MuMu 触控失败')
            if on_pressed is not None:
                on_pressed()
            for i in range(1, steps + 1):
                if stopped.wait(step_seconds):
                    return False
                point = tuple(round(start[j] + (end[j] - start[j]) * i / steps) for j in (0, 1))
                if self.dll.nemu_input_event_touch_down(self.handle, self.display, *point):
                    raise RuntimeError('MuMu 滑动失败')
        finally:
            up = self.dll.nemu_input_event_touch_up(self.handle, self.display)
            if up:
                raise RuntimeError('MuMu 触控释放失败')
        return not stopped.is_set()

    def reopen_game(self, stopped):
        manager, adb = self.root / 'nx_main/MuMuManager.exe', self.root / 'nx_main/adb.exe'
        flags = subprocess.CREATE_NO_WINDOW
        result = subprocess.run([str(manager), 'info', '--vmindex', str(self.index)],
                capture_output=True, timeout=5, check=True, creationflags=flags)
        serial = instance_serial(json.loads(result.stdout.decode('utf-8')), self.index)
        if stopped.is_set():
            return
        connection = subprocess.run([str(adb), 'connect', serial], capture_output=True, timeout=8,
                                    check=True, creationflags=flags)
        if any(word in connection.stdout.lower() for word in (b'failed', b'unable')):
            raise RuntimeError('无法连接所选 MuMu 实例的 ADB 地址')
        if stopped.is_set():
            return
        subprocess.run([str(adb), '-s', serial, 'shell', 'am', 'start',
                        '-a', 'android.intent.action.MAIN', '-c', 'android.intent.category.LAUNCHER', '-p', GAME],
                capture_output=True, timeout=8, check=True, creationflags=flags)
        if stopped.wait(1):
            return
        self.display = self.dll.nemu_get_display_id(self.handle, GAME.encode(), 0)
        if self.display != 0:
            raise RuntimeError('游戏重启后的显示屏不支持安全触控')

    def close(self):
        if getattr(self, 'handle', 0) > 0:
            self.dll.nemu_disconnect(self.handle)
            self.handle = 0
        if getattr(self, 'dll_directory', None):
            self.dll_directory.close()
            self.dll_directory = None
