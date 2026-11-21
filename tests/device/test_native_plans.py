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
        pressed = self.pressed(revision)
        self.assertEqual([v['index'] for v in pressed],[0,1,2])
        captured_us = int(captured*1_000_000)
        # Each command is scheduled from the captured frame. A late first
        # command can shorten the observed span without moving later commands.
        for event,offset_ms in zip(pressed,(30,90,150)):
            due_us = captured_us+offset_ms*1000
            self.assertGreaterEqual(event['down_start_us'],due_us)
            self.assertLess(event['down_start_us'],due_us+20000)

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

    def test_board_timeline_executes_beyond_200ms_independently_of_skill(self):
        captured = time.perf_counter()
        board = self.touch.plan(5,(TimedTap(50,(480,460),100),TimedTap(380,(720,470))),
                                captured_at=captured,deadline=captured+1.)
        skill = self.touch.plan(4,(TimedTap(100,(900,630)),),
                                captured_at=captured,deadline=captured+.2)
        self.wait(lambda:len(self.pressed(board)) == 2 and len(self.pressed(skill)) == 1)
        first, second = self.pressed(board)
        self.assertEqual([first['finger'],second['finger'],self.pressed(skill)[0]['finger']],
                         [5,5,4])
        self.assertLess(first['down_start_us'],self.pressed(skill)[0]['down_start_us'])
        self.assertGreaterEqual(second['down_start_us'],int(captured*1_000_000)+380_000)

    def test_board_cancel_releases_active_contact_and_drops_remaining_taps(self):
        captured = time.perf_counter()
        board = self.touch.plan(5,(TimedTap(30,(480,460),175),TimedTap(420,(720,470))),
                                captured_at=captured,deadline=captured+1.)
        self.wait(lambda:len(self.pressed(board)) == 1)
        cancel = self.touch.cancel_plan(5)
        skill = self.touch.plan(4,(TimedTap(110,(900,630)),),
                                captured_at=captured,deadline=captured+.2)
        self.wait(lambda:len(self.pressed(skill)) == 1 and
                  any(k == 'native_plan' and v['event'] == 'cleared' and v['revision'] == cancel
                      for k,v in self.events))
        self.wait(lambda:any(k == 'native_plan' and v['event'] == 'dropped' and
                            v['revision'] == board and v['index'] == 0 for k,v in self.events))
        self.assertTrue(any(k == 'native_plan' and v['event'] == 'dropped' and
                            v['revision'] == board and v['index'] == 1 for k,v in self.events))
        self.assertEqual(len(self.pressed(board)),1)

    def test_board_rejects_stale_capture_in_python_and_native(self):
        captured = time.perf_counter()-.25
        actions = (TimedTap(450,(720,470)),)
        self.assertIsNone(self.touch.plan(5,actions,captured_at=captured,deadline=captured+1.))
        captured_us = int(captured*1_000_000)
        # Bypass the Python admission guard to check the native boundary too.
        wire = (f'PLAN 5 50 {captured_us} {captured_us+1_000_000} 1 '
                f'{captured_us+450_000} 720 470 15 {captured_us+470_000}\n')
        with self.touch.cv:
            self.touch.proc.stdin.write(wire)
            self.touch.proc.stdin.flush()
        self.wait(lambda:any(k == 'native_plan' and v['event'] == 'rejected' and
                             v['revision'] == 50 for k,v in self.events))
        self.assertEqual(self.pressed(50),[])

    def test_stop_barrier_blocks_older_board_revision_after_fresh_capture(self):
        captured = time.perf_counter()
        board = self.touch.plan(5,(TimedTap(700,(720,470)),),
                                captured_at=captured,deadline=captured+1.)
        self.stop.set()
        self.wait(lambda:self.touch.stop_ack > 0)
        captured_us = int(time.perf_counter()*1_000_000)
        wire = (f'PLAN 5 {board} {captured_us} {captured_us+1_000_000} 1 '
                f'{captured_us+350_000} 720 470 15 {captured_us+370_000}\n')
        with self.touch.cv:
            self.touch.proc.stdin.write(wire)
            self.touch.proc.stdin.flush()
        self.wait(lambda:any(k == 'native_plan' and v['event'] == 'rejected' and
                             v['revision'] == board for k,v in self.events))
        self.assertEqual(self.pressed(board),[])
