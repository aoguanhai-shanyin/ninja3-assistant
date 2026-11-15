import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from flow import Decision
from runner import RunControl, probe, run_flow
from vision import Screen


class RunnerTest(unittest.TestCase):
    def setUp(self):
        starter = patch('runner.prepare_game', return_value=False)
        self.startup = starter.start()
        self.addCleanup(starter.stop)
        native = patch('runner.create_native_touch',return_value=None)
        native.start()
        self.addCleanup(native.stop)

    def test_new_game_waits_for_display_before_capturing(self):
        class Device:
            def __init__(self):
                self.closed = False
            def capture(self):
                return object()
            def close(self):
                self.closed = True

        class Detector:
            backend = 'test'
            def detect(self, frame):
                return Screen()

        class StopFlow:
            successful_kills = 0
            def step(self, screen, now):
                return Decision('stop')

        self.startup.return_value = True
        device = Device()
        events = []
        with patch('runner.Vision', Detector), patch('runner.MuMu', side_effect=[RuntimeError('not ready'), device]), \
                patch('runner.Flow', StopFlow):
            run_flow('test', 0, RunControl(), lambda kind, value: events.append((kind, value)))
        self.assertTrue(device.closed)
        self.assertIn(('status', '等待游戏画面就绪…'), events)

    def test_pause_does_not_advance_flow_time(self):
        value = [100.0]
        control = RunControl(clock=lambda: value[0])
        self.assertEqual(control.now(), 100.0)
        self.assertTrue(control.pause())
        value[0] = 145.0
        self.assertEqual(control.now(), 100.0)
        self.assertTrue(control.resume())
        value[0] = 147.0
        self.assertEqual(control.now(), 102.0)

    def test_pause_acknowledges_only_after_inflight_action_finishes(self):
        control = RunControl()
        started, release, paused = threading.Event(), threading.Event(), threading.Event()

        def action():
            started.set()
            release.wait(2)
            return True

        action_thread = threading.Thread(target=lambda: control.run_if_active(action))
        action_thread.start()
        self.assertTrue(started.wait(1))
        pause_thread = threading.Thread(target=lambda: (control.pause(), paused.set()))
        pause_thread.start()
        self.assertFalse(paused.wait(.1))
        release.set()
        action_thread.join(1)
        pause_thread.join(1)
        self.assertTrue(paused.is_set())
        self.assertFalse(control.run_if_active(lambda: self.fail('action after pause')))

    def test_paused_runner_waits_for_resume_before_capturing_or_tapping(self):
        connected = threading.Event()
        captured = threading.Event()
        events = []

        class Device:
            def __init__(self, folder, index):
                self.taps = []
                self.closed = False
                connected.set()
            def capture(self):
                captured.set()
                return object()
            def tap(self, point, stop, hold_seconds=.07):
                self.taps.append(point)
                return True
            def close(self):
                self.closed = True

        class Detector:
            backend = 'test'
            def detect(self, frame):
                return Screen()

        class Flow:
            successful_kills = 0
            def __init__(self):
                self.completed_once = False
            def step(self, screen, now):
                return Decision('stop') if self.completed_once else Decision('join', (100, 200), '加入')
            def completed(self, decision, now):
                self.completed_once = True

        devices = []
        def create_device(*args):
            device = Device(*args)
            devices.append(device)
            return device

        control = RunControl()
        control.pause()
        with patch('runner.Vision', Detector), patch('runner.MuMu', create_device), patch('runner.Flow', Flow):
            thread = threading.Thread(target=run_flow, args=('test', 0, control, lambda k, v: events.append((k, v))))
            thread.start()
            try:
                self.assertFalse(connected.wait(.15))
                self.assertFalse(captured.is_set())
            finally:
                control.resume()
            self.assertTrue(connected.wait(2))
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(devices[0].taps, [(100, 200)])
        self.assertTrue(devices[0].closed)
        self.assertEqual(events[-1][0], 'finished')

    def test_join_recognizes_between_clicks_and_honors_interval(self):
        sequence = []
        class Device:
            def __init__(self, *args):
                self.presses = []
            def capture(self):
                return object()
            def tap(self, point, stop, hold_seconds=.07):
                sequence.append('tap')
                self.presses.append((time.perf_counter(), hold_seconds))
                return not stop.wait(hold_seconds)
            def close(self):
                pass

        class Detector:
            backend = 'test'
            def __init__(self):
                self.frames = 0
            def detect(self, frame):
                self.frames += 1
                sequence.append('detect')
                return Screen()

        class Flow:
            successful_kills = 0
            def __init__(self):
                self.clicks = 0
            def step(self, screen, now):
                return Decision('join', (100, 200), '加入') if self.clicks < 5 else Decision('stop')
            def completed(self, decision, now):
                self.clicks += 1

        for interval in (30, 60):
            with self.subTest(interval=interval):
                sequence.clear()
                device, detector = Device(), Detector()
                with patch('runner.Vision', return_value=detector), patch('runner.MuMu', return_value=device), patch('runner.Flow', Flow):
                    run_flow('test', 0, RunControl(), lambda *_: None, join_interval_ms=interval)
                self.assertEqual(len(device.presses), 5)
                self.assertGreaterEqual(detector.frames, 6)
                self.assertTrue(all(hold == .015 for _, hold in device.presses))
                gaps = [b[0] - a[0] for a, b in zip(device.presses, device.presses[1:])]
                self.assertTrue(all(gap >= interval / 1000 - .002 for gap in gaps), gaps)
                previous = -1
                for index, kind in enumerate(sequence):
                    if kind == 'tap':
                        self.assertIn('detect', sequence[previous + 1:index])
                        previous = index

    def test_stop_prevents_further_join_clicks(self):
        control = RunControl()

        class Device:
            def __init__(self, *args):
                self.count = 0
            def capture(self):
                return object()
            def tap(self, point, stop, hold_seconds=.07):
                self.count += 1
                if self.count == 2:
                    stop.set()
                return True
            def close(self):
                pass

        class Detector:
            backend = 'test'
            def detect(self, frame):
                return Screen()

        class Flow:
            successful_kills = 0
            def step(self, screen, now):
                return Decision('join', (100, 200), '加入')
            def completed(self, decision, now):
                pass

        device = Device()
        with patch('runner.Vision', Detector), patch('runner.MuMu', return_value=device), patch('runner.Flow', Flow):
            run_flow('test', 0, control, lambda *_: None, join_interval_ms=30)
        self.assertEqual(device.count, 2)

    def test_room_is_recognized_before_a_second_join_click(self):
        control = RunControl()

        class Device:
            def __init__(self, *args):
                self.points = []
            def capture(self):
                return object()
            def tap(self, point, stop, hold_seconds=.07):
                self.points.append(point)
                if point == (1100, 660):
                    stop.set()
                return True
            def close(self):
                pass

        class Detector:
            backend = 'test'
            def __init__(self):
                self.frames = 0
            def detect(self, frame):
                self.frames += 1
                if self.frames == 1:
                    return Screen(board=True, a='off', s='on', ss='off', join=(400, 300))
                return Screen(ready=(1100, 660))

        device, detector = Device(), Detector()
        with patch('runner.Vision', return_value=detector), patch('runner.MuMu', return_value=device):
            run_flow('test', 0, control, lambda *_: None, join_interval_ms=30)
        self.assertEqual(device.points, [(400, 300), (1100, 660)])

    def test_ready_is_clicked_once_without_fast_burst(self):
        class Device:
            def __init__(self, *args):
                self.presses = []
            def capture(self):
                return object()
            def tap(self, point, stop, hold_seconds=.07):
                self.presses.append((point, hold_seconds))
                return True
            def close(self):
                pass

        class Detector:
            backend = 'test'
            def detect(self, frame):
                return Screen()

        class Flow:
            successful_kills = 0
            def __init__(self):
                self.done = False
            def step(self, screen, now):
                return Decision('stop') if self.done else Decision('ready', (1100, 660), '准备')
            def completed(self, decision, now):
                self.done = True

        device = Device()
        with patch('runner.Vision', Detector), patch('runner.MuMu', return_value=device), patch('runner.Flow', Flow):
            run_flow('test', 0, RunControl(), lambda *_: None)
        self.assertEqual(device.presses, [((1100, 660), .07)])

    def test_resume_after_pause_recaptures_before_clicking(self):
        control = RunControl()
        pressed, release, paused = threading.Event(), threading.Event(), threading.Event()

        class Device:
            def __init__(self, *args):
                self.count = 0
            def capture(self):
                return object()
            def tap(self, point, stop, hold_seconds=.07):
                self.count += 1
                pressed.set()
                release.wait(2)
                return True
            def close(self):
                pass

        class Detector:
            backend = 'test'
            def __init__(self):
                self.frames = 0
            def detect(self, frame):
                self.frames += 1
                return Screen()

        class Flow:
            successful_kills = 0
            def __init__(self):
                self.calls = 0
            def step(self, screen, now):
                self.calls += 1
                return Decision('join', (100, 200), '加入') if self.calls == 1 else Decision('stop')
            def completed(self, decision, now):
                pass

        device, detector = Device(), Detector()
        with patch('runner.Vision', return_value=detector), patch('runner.MuMu', return_value=device), patch('runner.Flow', Flow):
            thread = threading.Thread(target=run_flow, args=('test', 0, control, lambda *_: None))
            thread.start()
            self.assertTrue(pressed.wait(2))
            pause_thread = threading.Thread(target=lambda: (control.pause(), paused.set()))
            pause_thread.start()
            release.set()
            self.assertTrue(paused.wait(2))
            self.assertEqual(device.count, 1)
            control.resume()
            thread.join(2)
            pause_thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(device.count, 1)
        self.assertGreaterEqual(detector.frames, 2)

    def test_probe_reads_screen_without_clicking(self):
        class Device:
            def __init__(self, folder, index):
                self.taps = 0
                self.closed = False
            def capture(self):
                return SimpleNamespace(shape=(720, 1280, 3))
            def tap(self, *args):
                self.taps += 1
            def close(self):
                self.closed = True

        class Detector:
            backend = 'test GPU'
            def detect(self, frame):
                return Screen(board=True)

        device = Device('test', 0)
        events = []
        with patch('runner.Vision', Detector), patch('runner.MuMu', return_value=device):
            probe('test', 0, lambda kind, value: events.append((kind, value)))
        self.assertEqual(device.taps, 0)
        self.assertTrue(device.closed)
        self.assertEqual(events[0][1]['page'], '逆袭悬赏列表')
        self.assertEqual(events[-1][0], 'finished')

    def test_probe_reports_network_error_without_retrying(self):
        device = unittest.mock.Mock()
        device.capture.return_value = SimpleNamespace(shape=(720, 1280, 3))
        detector = unittest.mock.Mock(backend='test GPU')
        detector.detect.return_value = Screen(network_error=True, network_retry=(638, 538))
        events = []
        with patch('runner.Vision', return_value=detector), patch('runner.MuMu', return_value=device):
            probe('test', 0, lambda kind, value: events.append((kind, value)))
        self.assertEqual(events[0][1]['page'], '游戏网络异常')
        device.tap.assert_not_called()


if __name__ == '__main__':
    unittest.main()
