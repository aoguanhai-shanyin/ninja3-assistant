"""Bounded hold of the explicitly selected, validated MuMu skate macro."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from ninja3_assistant.device.mumu import GAME


SCHEME = 'Codex滑步测试'
VK_F8, SCAN_F8 = 0x77, 0x42
PRESS_ACTIONS = [
    'start_loop:until_release', 'curve_first_point_sleep_time:10',
    'curve_last_point_sleep_time:until_release_cmd',
    'curve_rel:(0.148438,0.861111);(0.148438,0.972222)',
    'press_rel:(0.851562,0.875000)', 'sleep:30',
    'release_rel:(0.851562,0.875000)', 'curve_release', 'sleep:20',
    'curve_first_point_sleep_time:10', 'curve_last_point_sleep_time:until_release_cmd',
    'curve_rel:(0.148438,0.861111);(0.148438,0.972222)',
    'sleep:30', 'curve_release', 'stop_loop',
]


def selected_macro(root):
    """Accept only the named local scheme and its exact, unique F8 gesture."""
    root = Path(root)
    expected = root / (GAME + '-' + SCHEME + '.json')
    try:
        metadata = json.loads((root / (GAME + '.json')).read_text(encoding='utf-8-sig'))
        current = metadata['Current']
        if set(current) != {SCHEME} or current[SCHEME].get('configType') != 'Customer':
            return False
        if Path(current[SCHEME]['path']).resolve() != expected.resolve():
            return False
        data = json.loads(expected.read_text(encoding='utf-8-sig'))
        candidates = [item for item in data['keymaps'] if item.get('key', {}).get('device') == 'keyboard'
                      and (item['key'].get('virtual_key') == VK_F8 or item['key'].get('scan_code') == SCAN_F8
                           or item['key'].get('text') == 'F8')]
        if len(candidates) != 1:
            return False
        item = candidates[0]
        return (set(item) == {'editor_icon_scale', 'icon', 'key', 'press_actions', 'release_actions',
                              'rel_work_position', 'type'} and item.get('type') == 'Macro' and item.get('key') == {
            'device': 'keyboard', 'scan_code': SCAN_F8, 'text': 'F8', 'virtual_key': VK_F8}
            and item.get('press_actions') == PRESS_ACTIONS and item.get('release_actions') == [])
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


class SkateMacro:
    """A held F8 expires unless fresh ground observations renew its lease."""
    def __init__(self, hwnd, root, stopped, post, emit, *, clock=time.perf_counter, lease=.15):
        self.hwnd, self.root, self.stopped = hwnd, Path(root), stopped
        self.post, self.emit, self.clock, self.lease = post, emit, clock, lease
        self.lock = threading.Condition()
        self.held = self.closed = False
        self.available = True
        self.deadline = 0.
        self.error = None
        self.thread = threading.Thread(target=self._watch, daemon=True, name='mumu-skate-lease')
        self.thread.start()

    def _send(self, down):
        message = 0x100 if down else 0x101
        param = 1 | (SCAN_F8 << 16) | (0 if down else (3 << 30))
        if not self.post(self.hwnd, message, VK_F8, param):
            raise RuntimeError('MuMu 宏按键消息发送失败')
        self.emit('opening_action', {'kind': 'native_skate', 'stage': 'down' if down else 'up',
                                     'time': self.clock()})

    def _release_locked(self):
        if self.held:
            self._send(False)
            self.held = False

    def renew(self):
        with self.lock:
            if self.error:
                raise RuntimeError('MuMu 宏停止失败') from self.error
            if self.closed or self.stopped.is_set():
                self._release_locked()
                return False
            if not self.available or not selected_macro(self.root):
                self._release_locked()
                self.available = False
                return False
            if not self.held:
                self.held = True
                try:
                    self._send(True)
                except Exception:
                    self._release_locked()
                    raise
            self.deadline = self.clock() + self.lease
            self.lock.notify_all()
            return True

    def release(self):
        with self.lock:
            self._release_locked()
            self.lock.notify_all()

    def _watch(self):
        with self.lock:
            while not self.closed:
                if self.held and (self.stopped.is_set() or self.clock() >= self.deadline):
                    try:
                        self._release_locked()
                    except Exception as error:
                        self.error = error
                        self.available = False
                        self.lock.wait(.02)
                        continue
                self.lock.wait(min(.02, max(0., self.deadline-self.clock())) if self.held else .02)

    def close(self):
        error = None
        with self.lock:
            try:
                self._release_locked()
            except Exception as failure:
                error = failure
            self.closed = True
            self.lock.notify_all()
        self.thread.join(1.)
        if self.thread.is_alive():
            raise RuntimeError('MuMu 宏线程未停止')
        if error is not None:
            raise RuntimeError('MuMu 宏停止失败') from error


class GroundMacroGate:
    """Allow a short missing-button grace only after confirmed ground."""
    def __init__(self, macro, control, *, clock=time.perf_counter, grace=.15):
        self.macro, self.control, self.clock, self.grace = macro, control, clock, grace
        self.last_ground = None

    def release(self, *, reason='gate_release', captured_at=None):
        self.last_ground = None
        note = getattr(self.macro, 'note_stop_reason', None)
        if note is not None:
            note(reason, captured_at)
        self.macro.release()

    def forget(self):
        """Invalidate ground permission when a priority native action takes over."""
        self.last_ground = None

    def update(self, *, allowed, ground_frames, captured_at):
        now = self.clock()
        if not allowed or now-captured_at > .2:
            self.release(reason='gate_disallowed' if not allowed else 'gate_stale_frame', captured_at=captured_at)
            return False
        if ground_frames >= 2 or (ground_frames and self.last_ground is not None):
            self.last_ground = now
        if self.last_ground is not None and now-self.last_ground <= self.grace:
            active = bool(self.control.run_if_active(self.macro.renew))
            if active:
                return True
        self.release(reason='gate_ground_unconfirmed' if self.last_ground is None else 'gate_ground_expired',
                     captured_at=captured_at)
        return False


def create_skate_macro(folder, index, stopped, emit):
    appdata = os.environ.get('APPDATA')
    if not appdata:
        return None
    root = Path(appdata) / 'Netease' / 'MuMuPlayer' / 'data' / 'keymapConfig'
    if not selected_macro(root):
        return None
    try:
        result = subprocess.run([str(Path(folder) / 'nx_main' / 'MuMuManager.exe'), 'info',
                                 '--vmindex', str(index)], capture_output=True, check=True, timeout=5,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        hwnd = int(json.loads(result.stdout.decode('utf-8-sig'))['render_wnd'], 16)
        user32 = ctypes.windll.user32
        post = user32.PostMessageW
        post.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
        post.restype = ctypes.c_int
        user32.IsWindow.argtypes = [ctypes.c_void_p]
        user32.IsWindow.restype = ctypes.c_int
        if not hwnd or not user32.IsWindow(hwnd):
            return None
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, AttributeError):
        return None
    return SkateMacro(hwnd, root, stopped, post, emit)
