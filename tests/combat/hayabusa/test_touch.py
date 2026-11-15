from types import SimpleNamespace
import threading
import unittest

from ninja3_assistant.device.mumu import MuMu


class TouchReportTest(unittest.TestCase):
    def test_press_is_reported_and_released_even_when_stop_arrives_during_hold(self):
        calls = []
        device = object.__new__(MuMu)
        device.handle, device.display = 1, 0
        device.dll = SimpleNamespace(
            nemu_input_event_touch_down=lambda *args: (calls.append('down') or 0),
            nemu_input_event_touch_up=lambda *args: (calls.append('up') or 0))
        stopped = threading.Event()
        def pressed():
            calls.append('pressed')
            stopped.set()
        self.assertFalse(device.tap((190, 630), stopped, hold_seconds=.025, on_pressed=pressed))
        self.assertEqual(calls, ['down', 'pressed', 'up'])

    def test_press_callback_failure_still_releases_the_touch(self):
        calls = []
        device = object.__new__(MuMu)
        device.handle, device.display = 1, 0
        device.dll = SimpleNamespace(
            nemu_input_event_touch_down=lambda *args: 0,
            nemu_input_event_touch_up=lambda *args: (calls.append('up') or 0))
        def pressed():
            raise RuntimeError('report failed')
        with self.assertRaisesRegex(RuntimeError, 'report failed'):
            device.tap((190, 630), threading.Event(), on_pressed=pressed)
        self.assertEqual(calls, ['up'])


if __name__ == '__main__':
    unittest.main()
