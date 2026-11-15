import unittest
from types import SimpleNamespace

from mumu_multitouch import ExperimentalMultiTouch


class Function:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class FakeStop:
    def __init__(self, stop_at_wait=None):
        self.stop_at_wait = stop_at_wait
        self.waits = 0
        self.durations = []
        self.stopped = False

    def is_set(self):
        return self.stopped

    def wait(self, _seconds):
        self.waits += 1
        self.durations.append(_seconds)
        if self.waits == self.stop_at_wait:
            self.stopped = True
        return self.stopped


class FakeSDK:
    def __init__(self):
        self.calls = []
        self.active = set()
        self.fail_down = None
        self.raise_down = None
        self.fail_up = None
        self.right_saw_left = False
        self.handles = []
        self.nemu_input_event_finger_touch_down = Function(self.down)
        self.nemu_input_event_finger_touch_up = Function(self.up)

    def down(self, handle, display, finger, x, y):
        self.handles.append((handle, display))
        self.calls.append(('down', finger, x, y))
        if finger == 2:
            self.right_saw_left = 1 in self.active
        # Simulate uncertain contact even if an error is returned.
        self.active.add(finger)
        if self.raise_down == (finger, x, y):
            raise OSError('transport lost')
        return 7 if self.fail_down == (finger, x, y) else 0

    def up(self, handle, display, finger):
        self.handles.append((handle, display))
        self.calls.append(('up', finger))
        if self.fail_up == finger:
            return 9
        self.active.discard(finger)
        return 0


class MultiTouchTest(unittest.TestCase):
    def setUp(self):
        self.sdk = FakeSDK()
        self.events = []
        clock = iter(range(1000))
        self.gesture = ExperimentalMultiTouch(SimpleNamespace(dll=self.sdk, handle=7, display=0),
                                              clock=lambda: next(clock) / 1000)

    def test_two_fingers_overlap_and_second_left_swipe_follows_release(self):
        stop = FakeStop()
        self.assertTrue(self.gesture.run(stop, self.events.append))
        self.assertEqual(self.sdk.calls, [
            ('down', 1, 190, 620),
            *[('down', 1, 190, y) for y in (636, 652, 668, 684)],
            ('down', 2, 1090, 630), ('down', 1, 190, 700),
            ('up', 2), ('up', 1), ('down', 1, 190, 620),
            *[('down', 1, 190, y) for y in (636, 652, 668, 684, 700)],
            ('up', 1)])
        self.assertTrue(self.sdk.right_saw_left)
        self.assertEqual(self.sdk.active, set())
        self.assertEqual(set(self.sdk.handles), {(7, 0)})
        self.assertEqual([event['phase'] for event in self.events], [
            'left_first_down', *['left_first_move'] * 4, 'right_down',
            'left_first_move', 'right_up', 'left_first_up', 'left_second_down',
            *['left_second_move'] * 5, 'left_second_up'])
        self.assertTrue(all(event['sdk_started_at'] < event['sdk_returned_at'] for event in self.events))
        self.assertEqual(sum(event['move_is_assumed'] for event in self.events), 10)
        self.assertEqual(stop.durations, [.01] * 5 + [.02, .02] + [.012] * 5)

    def test_cancel_before_start_never_touches(self):
        stop = FakeStop()
        stop.stopped = True
        self.assertFalse(self.gesture.run(stop, self.events.append))
        self.assertEqual(self.sdk.calls, [])

    def test_cancel_releases_left_and_both_fingers(self):
        for stop_at, expected_tail in ((1, [('up', 1)]), (5, [('up', 2), ('up', 1)]),
                                       (6, [('up', 2), ('up', 1)]), (8, [('up', 1)])):
            with self.subTest(stop_at=stop_at):
                sdk = FakeSDK()
                gesture = ExperimentalMultiTouch(SimpleNamespace(dll=sdk, handle=7, display=0))
                self.assertFalse(gesture.run(FakeStop(stop_at), lambda event: None))
                self.assertEqual(sdk.calls[-len(expected_tail):], expected_tail)
                self.assertEqual(sdk.active, set())

    def test_failed_second_finger_down_releases_uncertain_and_held_contacts(self):
        self.sdk.fail_down = (2, 1090, 630)
        with self.assertRaisesRegex(RuntimeError, 'finger down failed'):
            self.gesture.run(FakeStop(), self.events.append)
        self.assertEqual(self.sdk.calls[-2:], [('up', 2), ('up', 1)])
        self.assertEqual(self.sdk.active, set())
        self.assertEqual(next(e['sdk_result'] for e in self.events if e['phase'] == 'right_down'), 7)

    def test_release_failure_does_not_skip_other_finger(self):
        self.sdk.fail_up = 2
        with self.assertRaisesRegex(RuntimeError, 'finger up failed') as caught:
            self.gesture.run(FakeStop(), self.events.append)
        self.assertEqual(self.sdk.calls[-3:], [('up', 2), ('up', 2), ('up', 1)])
        self.assertEqual(self.sdk.active, {2})
        self.assertTrue(any('cleanup failed' in note for note in caught.exception.__notes__))

    def test_native_exception_is_logged_and_all_contacts_released(self):
        self.sdk.raise_down = (2, 1090, 630)
        with self.assertRaisesRegex(OSError, 'transport lost'):
            self.gesture.run(FakeStop(), self.events.append)
        self.assertEqual(self.sdk.calls[-2:], [('up', 2), ('up', 1)])
        self.assertEqual(self.sdk.active, set())
        self.assertEqual(next(e['sdk_error'] for e in self.events if e['phase'] == 'right_down'), 'transport lost')

    def test_observer_failure_still_releases_fingers(self):
        def emit(event):
            if event['phase'] == 'right_down':
                raise ValueError('observer unavailable')
        with self.assertRaisesRegex(ValueError, 'observer unavailable'):
            self.gesture.run(FakeStop(), emit)
        self.assertEqual(self.sdk.calls[-2:], [('up', 2), ('up', 1)])
        self.assertEqual(self.sdk.active, set())


if __name__ == '__main__':
    unittest.main()
