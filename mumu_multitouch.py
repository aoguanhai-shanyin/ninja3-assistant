"""Isolated MuMu two-finger experiment; not wired into the assistant.

The SDK header documents finger down/up. MaaXYZ/MaaFramework's
source/MaaAdbControlUnit/EmulatorExtras/MuMuPlayerExtras.cpp uses finger down
for both down and move. This DLL's same-ID behavior still needs live proof.
The coordinates and millisecond waits are experimental and not calibrated.
"""
import ctypes
import sys
import time


class ExperimentalMultiTouch:
    """Own two finger IDs for a single gesture; caller must serialize SDK input."""

    def __init__(self, device, *, clock=time.perf_counter):
        self.device = device
        self.clock = clock
        self.down = device.dll.nemu_input_event_finger_touch_down
        self.up = device.dll.nemu_input_event_finger_touch_up
        self.down.argtypes = [ctypes.c_int] * 5
        self.down.restype = ctypes.c_int
        self.up.argtypes = [ctypes.c_int] * 3
        self.up.restype = ctypes.c_int

    def run(self, stopped, emit):
        """Left down/move, right tap while left is held, then left down/move.

        SDK return 0 means only that a request was accepted, not that the game
        generated a move or performed a dodge. Event times use ``clock``.
        """
        if stopped.is_set():
            return False
        held = set()

        def down(finger, point, phase):
            # Treat failure as uncertain contact and attempt to release it too.
            held.add(finger)
            start = self.clock()
            try:
                result = self.down(self.device.handle, self.device.display, finger, *point)
            except Exception as error:
                emit({'phase': phase, 'finger_id': finger, 'point': point,
                      'sdk_started_at': start, 'sdk_returned_at': self.clock(),
                      'sdk_error': str(error), 'move_is_assumed': phase.endswith('_move')})
                raise
            end = self.clock()
            emit({'phase': phase, 'finger_id': finger, 'point': point,
                  'sdk_started_at': start, 'sdk_returned_at': end, 'sdk_result': result,
                  'move_is_assumed': phase.endswith('_move')})
            if result:
                raise RuntimeError(f'MuMu finger down failed: {finger} ({result})')

        def up(finger, phase):
            start = self.clock()
            try:
                result = self.up(self.device.handle, self.device.display, finger)
            except Exception as error:
                emit({'phase': phase, 'finger_id': finger, 'point': None,
                      'sdk_started_at': start, 'sdk_returned_at': self.clock(),
                      'sdk_error': str(error), 'move_is_assumed': False})
                raise
            end = self.clock()
            if not result:
                held.discard(finger)
            emit({'phase': phase, 'finger_id': finger, 'point': None,
                  'sdk_started_at': start, 'sdk_returned_at': end, 'sdk_result': result,
                  'move_is_assumed': False})
            if result:
                raise RuntimeError(f'MuMu finger up failed: {finger} ({result})')

        def wait(seconds):
            return not stopped.wait(seconds)

        try:
            down(1, (190, 620), 'left_first_down')
            for step in range(1, 5):
                if not wait(.01):
                    return False
                down(1, (190, 620 + 16 * step), 'left_first_move')
            down(2, (1090, 630), 'right_down')
            # Finish the first swipe while the jump contact is still held.
            if not wait(.01):
                return False
            down(1, (190, 700), 'left_first_move')
            if not wait(.02):
                return False
            up(2, 'right_up')
            if stopped.is_set():
                return False
            up(1, 'left_first_up')
            if not wait(.02):
                return False
            down(1, (190, 620), 'left_second_down')
            for step in range(1, 6):
                if not wait(.012):
                    return False
                down(1, (190, 620 + 16 * step), 'left_second_move')
            up(1, 'left_second_up')
            return not stopped.is_set()
        finally:
            # Release both even when the first release itself raises/fails.
            # Preserve an earlier exception; otherwise expose a cleanup failure.
            prior_error = sys.exc_info()[1]
            failures = []
            for finger in (2, 1):
                if finger in held:
                    try:
                        up(finger, 'cleanup_up')
                    except Exception as error:
                        failures.append(error)
            if failures:
                if prior_error is not None:
                    prior_error.add_note('MuMu finger cleanup failed: '+str(failures[0]))
                else:
                    raise RuntimeError('MuMu finger cleanup failed') from failures[0]
