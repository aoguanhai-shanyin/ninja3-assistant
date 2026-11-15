import subprocess
import threading
import time
import unittest

from ninja3_assistant.device.native_touch import NativeTouch, TimedTap, executable_path


@unittest.skipUnless(executable_path().is_file(),'build native_touch/build.cmd first')
class NativePlanTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.stop = threading.Event()
        def popen(_args,**options):
            return subprocess.Popen([str(executable_path()),'--mock'],**options)
        self.touch = NativeTouch('mock','folder',1,self.stop,
            lambda k,v:self.events.append((k,v)),popen=popen,observation_timeout=.5,lease_ms=1000)

    def tearDown(self):
        self.touch.close()

    def wait(self,predicate):
        deadline = time.perf_counter()+2
        while time.perf_counter() < deadline:
            if predicate():
                return
            time.sleep(.002)
        self.fail(str(self.events[-15:]))

    def pressed(self,revision):
        return [v for k,v in self.events if k == 'native_plan' and
                v['event'] == 'pressed' and v['revision'] == revision]

    def test_one_frame_sends_three_delayed_commands(self):
        captured = time.perf_counter()
        revision = self.touch.plan(4,(TimedTap(30,(380,630)),TimedTap(90,(900,630)),
                                    TimedTap(150,(740,630))),
                                   captured_at=captured,deadline=captured+.2)
        self.wait(lambda:len(self.pressed(revision)) == 3)
        starts = [v['down_start_us']/1e6 for v in self.pressed(revision)]
        self.assertGreater(starts[2]-starts[0],.11)

    def test_cancel_future_plan_does_not_stop_skate(self):
        self.touch.note_frame(time.perf_counter())
        self.touch.renew()
        captured = time.perf_counter()
        revision = self.touch.plan(4,(TimedTap(120,(900,630)),),
                                   captured_at=captured,deadline=captured+.2)
        cancel = self.touch.cancel_plan(4)
        self.wait(lambda:any(k == 'native_plan' and v['event']=='cleared' and
                             v['revision']==cancel for k,v in self.events))
        self.wait(lambda:any(k == 'native_touch' for k,v in self.events))
        self.assertEqual(self.pressed(revision),[])
        self.assertTrue(self.touch.desired)
        self.assertFalse(any(k == 'native_control' and v['event']=='stop_requested'
                             for k,v in self.events))

    def test_stop_clears_future_plan_and_new_plan_can_run(self):
        captured = time.perf_counter()
        old = self.touch.plan(4,(TimedTap(120,(900,630)),),
                              captured_at=captured,deadline=captured+.2)
        self.touch.stop_all()
        captured = time.perf_counter()
        new = self.touch.plan(4,(TimedTap(50,(380,630)),),
                              captured_at=captured,deadline=captured+.2)
        self.wait(lambda:len(self.pressed(new)) == 1)
        self.assertGreater(new,old)
        self.assertEqual(self.pressed(old),[])

    def test_rejects_overlapping_actions_before_sending(self):
        captured = time.perf_counter()
        with self.assertRaises(ValueError):
            self.touch.plan(4,(TimedTap(20,(380,630),100),TimedTap(40,(900,630))),
                            captured_at=captured,deadline=captured+.2)
        self.assertFalse(self.touch.plan_pending)

    def test_stop_event_cancels_plan_without_active_movement(self):
        captured = time.perf_counter()
        revision = self.touch.plan(4,(TimedTap(150,(900,630)),),
                                   captured_at=captured,deadline=captured+.2)
        self.stop.set()
        self.wait(lambda:self.touch.stop_ack > 0)
        self.assertEqual(self.pressed(revision),[])
