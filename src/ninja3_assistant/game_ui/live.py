"""Use the page API with an already running MuMu instance."""

from contextlib import closing
import subprocess

from ninja3_assistant.device.input_owner import InputBusy, input_owner
from ninja3_assistant.device.mumu import MuMu
from ninja3_assistant.device.native_touch import create_native_touch, executable_path
from ninja3_assistant.game_ui import GameUI, NavigationResult
from ninja3_assistant.game_ui.pages import MenuVision, NAVIGABLE_PAGES, ROUTES


def observe_game(folder, index, control):
    with closing(MuMu(folder, index)) as device:
        ui = GameUI(control, device.capture, recognize=MenuVision(control.stop), routes=ROUTES)
        return ui.observe()


def navigate_game(folder, index, control, target):
    if not isinstance(target, str) or target not in NAVIGABLE_PAGES:
        raise ValueError('unknown navigation page: ' + str(target))
    try:
        with input_owner(folder, index), closing(MuMu(folder, index)) as device:
            native = create_native_touch(folder, index, control.stop, lambda *args: None)
            if native is None:
                raise RuntimeError('界面导航需要 NinjaTouch.exe')
            with closing(native):
                control.set_release_input(native.stop_all)

                def swipe(start, end):
                    native.stop_all()
                    if control.stop.is_set():
                        return False
                    subprocess.run([str(executable_path()), '--folder', str(folder), '--index', str(index),
                                    '--swipe', *map(str, (*start, *end))], capture_output=True, timeout=8,
                                   check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    return not control.stop.is_set()

                try:
                    ui = GameUI(control, device.capture, native, recognize=MenuVision(control.stop),
                                routes=ROUTES, swipe=swipe)
                    return ui.navigate(target)
                finally:
                    control.set_release_input(None)
    except InputBusy:
        return NavigationResult('blocked', None)
