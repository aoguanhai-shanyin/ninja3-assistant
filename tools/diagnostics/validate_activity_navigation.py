"""Bounded live check of activity navigation and pause/resume, never matches."""
from ninja3_assistant.paths import project_root
import json
import ctypes
import time

ROOT = project_root()
from tools.diagnostics.assistant_control import AssistantController


def check_hidden_windows():
    user32, windows = ctypes.windll.user32, []
    user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
    callback = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(handle, _):
        title = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(handle, title, 512)
        if 'mumu' in title.value.lower():
            windows.append({'title': title.value, 'visible': bool(user32.IsWindowVisible(handle))})
        return True
    user32.EnumWindows(callback(visit), 0)
    return windows


def main():
    controller = AssistantController('D:/Program Files/Netease/MuMu', 1)
    output = ROOT/'data/intelligence-simulation'/f'navigation-{time.strftime("%Y%m%d-%H%M%S")}.json'
    history = []
    passed = False
    try:
        controller.start('run', activity_enabled=True)
        deadline = time.monotonic()+35
        for cycle in range(2):
            while time.monotonic() < deadline:
                state = controller.status()
                if state['state'] == 'error':
                    raise RuntimeError(state['error'])
                if '等待情报模拟战开放' in (state['status'] or ''):
                    break
                time.sleep(.2)
            else:
                raise RuntimeError('navigation did not reach verified closed-window wait')
            history.append(controller.status())
            if cycle == 0:
                paused = controller.pause()
                if paused['state'] != 'paused':
                    raise RuntimeError('pause was not acknowledged')
                history.append(paused)
                time.sleep(.4)
                history.append(controller.status())
                controller.resume()
                # Wait for a fresh task-page inspection, not the previous status.
                while time.monotonic() < deadline and controller.status()['status'] == state['status']:
                    time.sleep(.05)
        passed = True
    finally:
        history.append(controller.stop())
        output.write_text(json.dumps({'passed': passed, 'history': history}, ensure_ascii=False, indent=2),
                          encoding='utf-8')
        print(json.dumps({'passed': passed, 'evidence': str(output), 'final': controller.status()['state']},
                         ensure_ascii=False))


if __name__ == '__main__':
    main()
