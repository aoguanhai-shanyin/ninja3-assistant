import threading
import time
import unittest
from types import SimpleNamespace

from experimental_input_scheduler import ExperimentalInputScheduler
from mumu import MuMu


def wait_for(predicate, timeout=1.):
    end = time.perf_counter()+timeout
    while time.perf_counter() < end:
        if predicate():
            return True
        threading.Event().wait(.002)
    return bool(predicate())


class InputSchedulerTests(unittest.TestCase):
    def test_one_active_one_pending_duplicate_and_full_are_rejected(self):
        scheduler = ExperimentalInputScheduler()
        entered, release, second = threading.Event(), threading.Event(), threading.Event()
        def first(token, pressed):
            pressed('down')
            entered.set()
            release.wait(1)
            return not token.is_set()
        def next_action(token, pressed):
            pressed('second')
            second.set()
            return True
        try:
            deadline = time.perf_counter()+3
            self.assertTrue(scheduler.submit('slide', first, generation=0, deadline=deadline))
            self.assertTrue(entered.wait(1))
            self.assertFalse(scheduler.submit('slide', next_action, generation=0, deadline=deadline))
            self.assertTrue(scheduler.submit('jump', next_action, generation=0, deadline=deadline))
            self.assertFalse(scheduler.submit('skill', next_action, generation=0, deadline=deadline))
            self.assertFalse(second.is_set())
            release.set()
            self.assertTrue(second.wait(1))
            self.assertTrue(wait_for(lambda: sum(e['event']=='finished' for e in scheduler.events()) == 2))
            events = scheduler.events()
            self.assertEqual([e['key'] for e in events if e['event']=='sdk_pressed'], ['slide', 'jump'])
            self.assertEqual(len([e for e in events if e['event']=='queued']), 2)
        finally:
            release.set()
            self.assertTrue(scheduler.close(1))

    def test_pause_cancels_active_and_pending_and_rejects_old_generation(self):
        scheduler = ExperimentalInputScheduler()
        entered, released, called = threading.Event(), threading.Event(), threading.Event()
        def first(token, pressed):
            try:
                pressed('touch_down')
                entered.set()
                return not token.wait(3)
            finally:
                released.set()  # Represents the SDK gesture's finally-touch-up.
        def pending(token, pressed):
            called.set()
            return True
        try:
            deadline = time.perf_counter()+3
            self.assertTrue(scheduler.submit('slide', first, generation=0, deadline=deadline))
            self.assertTrue(entered.wait(1))
            self.assertTrue(scheduler.submit('jump', pending, generation=0, deadline=deadline))
            self.assertEqual(scheduler.pause(), 1)
            self.assertTrue(released.wait(1))
            self.assertTrue(scheduler.wait_idle(1))
            self.assertFalse(called.is_set())
            self.assertFalse(scheduler.submit('jump', pending, generation=0, deadline=deadline))
            self.assertTrue(scheduler.resume())
            self.assertFalse(scheduler.submit('jump', pending, generation=0, deadline=deadline))
            self.assertTrue(scheduler.submit('jump', pending, generation=1, deadline=deadline))
            self.assertTrue(called.wait(1))
            self.assertTrue(any(e['event']=='cancelled' and e['key']=='jump' for e in scheduler.events()))
        finally:
            self.assertTrue(scheduler.close(1))

    def test_exception_in_sdk_gesture_still_releases_contact(self):
        scheduler = ExperimentalInputScheduler()
        calls, released = [], threading.Event()
        device = MuMu.__new__(MuMu)
        device.handle, device.display = 1, 0
        def down(*args):
            calls.append('down')
            if len(calls) == 2:
                raise RuntimeError('simulated SDK move failure')
            return 0
        def up(*args):
            calls.append('up')
            released.set()
            return 0
        device.dll = SimpleNamespace(nemu_input_event_touch_down=down, nemu_input_event_touch_up=up)
        try:
            def action(token, pressed):
                return device.swipe((190,620),(190,700),token,step_seconds=0,steps=4,
                                    on_pressed=lambda: pressed('slide_down'))
            self.assertTrue(scheduler.submit('slide', action, generation=0, deadline=time.perf_counter()+2))
            self.assertTrue(released.wait(1))
            self.assertTrue(wait_for(lambda: any(e['event']=='failed' for e in scheduler.events())))
            self.assertEqual(calls, ['down', 'down', 'up'])
            self.assertEqual([e['stage'] for e in scheduler.events() if e['event']=='sdk_pressed'], ['slide_down'])
        finally:
            self.assertTrue(scheduler.close(1))

    def test_expired_pending_input_is_never_executed(self):
        tick = [0.]
        scheduler = ExperimentalInputScheduler(clock=lambda: tick[0])
        entered, release, called = threading.Event(), threading.Event(), threading.Event()
        def blocker(token, pressed):
            entered.set()
            release.wait(1)
            return True
        try:
            self.assertTrue(scheduler.submit('first', blocker, generation=0, deadline=10.))
            self.assertTrue(entered.wait(1))
            self.assertTrue(scheduler.submit('late', lambda *_: called.set(), generation=0, deadline=1.))
            tick[0] = 2.
            release.set()
            self.assertTrue(wait_for(lambda: any(e.get('reason')=='expired_before_execution' for e in scheduler.events())))
            self.assertFalse(called.is_set())
        finally:
            release.set()
            self.assertTrue(scheduler.close(1))

    def test_close_is_bounded_even_if_action_ignores_cancellation(self):
        scheduler = ExperimentalInputScheduler()
        entered, release = threading.Event(), threading.Event()
        def stuck(token, pressed):
            entered.set()
            release.wait(1)
            return True
        self.assertTrue(scheduler.submit('stuck', stuck, generation=0, deadline=time.perf_counter()+2))
        self.assertTrue(entered.wait(1))
        try:
            scheduler.pause()
            self.assertFalse(scheduler.wait_idle(.01))
            self.assertFalse(scheduler.close(.01))
            self.assertFalse(scheduler.submit('new', stuck, generation=scheduler.generation,
                                              deadline=time.perf_counter()+2))
        finally:
            release.set()
            self.assertTrue(scheduler.close(1))


if __name__ == '__main__':
    unittest.main()
