import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from native_touch import NativeTouch, executable_path


CHILD = r'''
import sys,time
log = open(sys.argv[1], 'a', encoding='utf-8', buffering=1)
print('READY', flush=True)
for raw in sys.stdin:
    line = raw.strip()
    print(line, file=log, flush=True)
    parts = line.split()
    if parts[0] == 'START':
        now = time.perf_counter_ns()//1000
        print('STARTED '+parts[2]+' '+str(now)+' '+str(now), flush=True)
    elif parts[0] == 'STOP':
        if len(sys.argv) > 2 and sys.argv[2] == 'delay':
            print('EXPIRED', flush=True)
            print('STOPPED', flush=True)
            time.sleep(.15)
        elif len(sys.argv) > 2 and sys.argv[2] == 'silent':
            time.sleep(.15)
        print('STOPPED '+parts[1], flush=True)
    elif parts[0] == 'TAP2':
        now = time.perf_counter_ns()//1000
        if not (len(sys.argv) > 2 and sys.argv[2] == 'drop'):
            print('PRESSED '+parts[1]+' '+str(now)+' '+str(now), flush=True)
        time.sleep(.05)
        verb = 'TAP_DROPPED2' if len(sys.argv) > 2 and sys.argv[2] == 'drop' else 'TAPPED2'
        print(verb+' '+parts[1]+' '+parts[2]+' '+parts[3], flush=True)
'''


class NativeTouchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name)/'commands.txt'
        self.child = Path(self.tmp.name)/'child.py'
        self.child.write_text(CHILD, encoding='utf-8')
        self.events = []
        def popen(_args, **kwargs):
            return subprocess.Popen([sys.executable, '-u', str(self.child), str(self.log)], **kwargs)
        self.stopped = threading.Event()
        self.touch = NativeTouch('fake.exe', 'folder', 1, self.stopped,
                                 lambda kind, value:self.events.append((kind,value)), popen=popen)

    def tearDown(self):
        self.touch.close()
        self.tmp.cleanup()

    def wait_for(self, predicate):
        deadline = time.monotonic()+2
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        self.fail('condition not met before deadline')

    def commands(self):
        return self.log.read_text(encoding='utf-8').splitlines() if self.log.exists() else []

    def test_three_second_movement_is_one_submission_and_can_be_cancelled(self):
        revision=self.touch.schedule_movement(3,captured_at=time.perf_counter())
        self.assertIsNotNone(revision)
        self.wait_for(lambda:any(x.startswith('START ') for x in self.commands()))
        for _ in range(25):
            self.touch.note_frame(time.perf_counter())
            time.sleep(.01)
        self.assertEqual(sum(x.startswith('START ') for x in self.commands()),1)
        self.assertFalse(any(x.startswith('RENEW ') for x in self.commands()))
        self.assertTrue(self.touch.desired)
        self.touch.release()
        self.assertTrue(self.commands()[-1].startswith('STOP '))
        self.assertIsNone(self.touch.movement_until)

    def test_new_movement_submission_replaces_remaining_horizon(self):
        self.touch.schedule_movement(3,captured_at=time.perf_counter())
        self.wait_for(lambda:any(x.startswith('START ') for x in self.commands()))
        self.touch.schedule_movement(.1,captured_at=time.perf_counter())
        self.wait_for(lambda:any(x.startswith('RENEW ') for x in self.commands()))
        for _ in range(20):
            self.touch.note_frame(time.perf_counter())
            time.sleep(.01)
        self.assertFalse(self.touch.desired)
        self.assertTrue(any(x.startswith('STOP ') for x in self.commands()))

    def test_stale_frame_cannot_create_three_second_movement(self):
        self.assertIsNone(self.touch.schedule_movement(3,captured_at=time.perf_counter()-.3))
        self.assertFalse(self.touch.desired)

    def test_observation_renews_and_release_waits_for_ack(self):
        self.assertTrue(self.touch.renew())
        self.wait_for(lambda:any(x.startswith('START ') for x in self.commands()))
        self.touch.release()
        self.assertTrue(self.commands()[-1].startswith('STOP '))
        self.assertEqual(self.touch.stop_ack, self.touch.stop_requested)
        self.assertFalse(self.touch.running_sent)

    def test_stale_observation_stops_without_replaying_old_start(self):
        self.touch.renew()
        self.wait_for(lambda:any(x.startswith('START ') for x in self.commands()))
        self.wait_for(lambda:any(x.startswith('STOP ') for x in self.commands()))
        time.sleep(.12)
        self.assertEqual(sum(x.startswith('START ') for x in self.commands()), 1)
        self.assertFalse(self.touch.desired)
        self.wait_for(lambda:any(k == 'native_control' and v.get('reason') == 'observation_timeout'
                                 for k,v in self.events))
        event = next(v for k,v in self.events if k == 'native_control' and v.get('reason') == 'observation_timeout')
        self.assertGreaterEqual(event['frame_age_ms'], 150)

    def test_stop_reason_only_emits_for_an_active_transition(self):
        self.touch.renew()
        self.wait_for(lambda:any(x.startswith('START ') for x in self.commands()))
        self.touch.note_stop_reason('pause', time.perf_counter()-.02)
        self.touch.release()
        self.wait_for(lambda:any(k == 'native_control' and v.get('reason') == 'pause' for k,v in self.events))
        self.touch.note_stop_reason('pause')
        self.touch.release()
        time.sleep(.05)
        self.assertEqual(sum(k == 'native_control' and v.get('reason') == 'pause' for k,v in self.events), 1)

    def test_priority_stale_reason_is_reported(self):
        capture = time.perf_counter()-.25
        self.assertIsNone(self.touch.priority_tap((1090,630),captured_at=capture,deadline=capture+.2))
        self.wait_for(lambda:any(k == 'native_control' and v.get('reason') == 'stale_before_enqueue'
                                 for k,v in self.events))

    def test_capture_age_is_not_reset_by_slow_recognition(self):
        self.touch.note_frame(time.perf_counter()-.3)
        self.touch.renew()
        time.sleep(.08)
        self.assertFalse(any(x.startswith('START ') for x in self.commands()))

    def test_skill_taps_are_bounded_and_completion_is_reported(self):
        self.assertTrue(self.touch.tap((70,330)))
        self.assertFalse(self.touch.tap((55,60)))
        self.wait_for(lambda:any(k == 'opening_action' for k,v in self.events))
        self.assertEqual(self.events[-1][1]['point'], (70,330))
        taps = [x.split() for x in self.commands() if x.startswith('TAP2 ')]
        self.assertEqual([(x[2],x[3]) for x in taps], [('70','330')])

    def test_stale_tap_is_dropped_before_sending(self):
        self.assertTrue(self.touch.tap((70,330),deadline=time.perf_counter()-1))
        self.wait_for(lambda:any(v.get('stage') == 'dropped' for k,v in self.events))
        self.assertFalse(any(x.startswith('TAP2 ') for x in self.commands()))
        self.assertFalse(self.touch.tap_inflight)

    def test_preserving_jump_does_not_block_weapon_mailbox(self):
        capture = time.perf_counter()
        self.assertIsNotNone(self.touch.priority_tap((1090,630),captured_at=capture,
            deadline=capture+.5,hold_seconds=.1,preserve_slide=True))
        self.assertTrue(self.touch.tap((900,630),deadline=capture+.5))
        self.wait_for(lambda:any(k == 'opening_action' and v.get('point') == (900,630)
                                 for k,v in self.events))
        taps = [x.split() for x in self.commands() if x.startswith('TAP2 ')]
        self.assertEqual([(x[2],x[3],x[-1]) for x in taps],
                         [('1090','630','2'),('900','630','0')])

    def test_quiesce_stops_inflight_touch_before_return(self):
        self.touch.renew()
        self.wait_for(lambda:any(x.startswith('START ') for x in self.commands()))
        self.touch.quiesce()
        self.assertTrue(self.commands()[-1].startswith('STOP '))
        self.assertEqual(self.touch.stop_ack, self.touch.stop_requested)

    def test_process_exit_is_not_silently_ignored(self):
        self.touch.proc.terminate()
        self.touch.proc.wait(timeout=2)
        self.wait_for(lambda:self.touch.error is not None)
        with self.assertRaises(RuntimeError):
            self.touch.renew()

    def test_expired_reply_does_not_acknowledge_later_stop(self):
        def popen(_args, **kwargs):
            return subprocess.Popen([sys.executable, '-u', str(self.child), str(self.log), 'delay'], **kwargs)
        touch = NativeTouch('fake.exe', 'folder', 1, threading.Event(), lambda *_:None, popen=popen)
        try:
            touch.renew()
            self.wait_for(lambda:touch.running_sent)
            before = time.perf_counter()
            touch.release()
            self.assertGreaterEqual(time.perf_counter()-before,.12)
            self.assertEqual(touch.stop_ack,touch.stop_requested)
        finally:
            touch.close()

    def test_stop_event_does_not_flood_while_release_ack_is_pending(self):
        def popen(_args, **kwargs):
            return subprocess.Popen([sys.executable,'-u',str(self.child),str(self.log),'silent'], **kwargs)
        stopped = threading.Event()
        touch = NativeTouch('fake.exe','folder',1,stopped,lambda *_:None,
                            popen=popen,observation_timeout=1.)
        try:
            touch.renew()
            self.wait_for(lambda:any(x.startswith('START ') for x in self.commands()))
            stopped.set()
            self.wait_for(lambda:any(x.startswith('STOP ') for x in self.commands()))
            time.sleep(.08)
            self.assertEqual(sum(x.startswith('STOP ') for x in self.commands()),1)
            self.wait_for(lambda:touch.stop_ack >= 1)
        finally:
            touch.close()

    def test_dropped_tap_clears_inflight_and_reports_drop(self):
        def popen(_args, **kwargs):
            return subprocess.Popen([sys.executable, '-u', str(self.child), str(self.log), 'drop'], **kwargs)
        events = []
        touch = NativeTouch('fake.exe', 'folder', 1, threading.Event(),
                            lambda kind,value:events.append((kind,value)), popen=popen)
        try:
            self.assertTrue(touch.tap((70,330)))
            self.wait_for(lambda:any(v.get('stage') == 'dropped' for k,v in events))
            self.assertFalse(touch.tap_inflight)
            self.assertTrue(touch.tap((55,60)))
        finally:
            touch.close()

    def test_frozen_worker_uses_adjacent_executable(self):
        with patch.object(sys, 'frozen', True, create=True), \
                patch.object(sys, 'executable', str(Path(self.tmp.name)/'Ninja3Worker.exe')):
            self.assertEqual(executable_path(), Path(self.tmp.name)/'NinjaTouch.exe')

    @unittest.skipUnless(executable_path().is_file(), 'native touch executable not built')
    def test_native_executable_mock_start_tap_and_stop(self):
        executable = executable_path()
        events = []
        def popen(args, **kwargs):
            return subprocess.Popen([*args, '--mock'], **kwargs)
        touch = NativeTouch(executable, 'test', 0, threading.Event(),
                            lambda kind,value:events.append((kind,value)), popen=popen)
        try:
            self.assertTrue(touch.renew())
            self.assertTrue(touch.tap((70,330)))
            self.wait_for(lambda:any(kind == 'opening_action' for kind,_ in events))
            self.wait_for(lambda:any(kind == 'native_timing' and value['kind'] == 'start_down'
                                     for kind,value in events))
            timing = next(value for kind,value in events if kind == 'native_timing' and value['kind'] == 'start_down')
            self.assertGreaterEqual(timing['capture_to_down_ms'],0)
            touch.release()
            self.assertEqual(touch.stop_ack, touch.stop_requested)
        finally:
            touch.close()

    @unittest.skipUnless(executable_path().is_file(), 'native touch executable not built')
    def test_priority_action_ack_is_actual_native_down(self):
        events = []
        def popen(args, **kwargs):
            return subprocess.Popen([*args,'--mock'],**kwargs)
        touch = NativeTouch(executable_path(),'test',0,threading.Event(),
                            lambda kind,value:events.append((kind,value)),popen=popen)
        try:
            capture = time.perf_counter()
            pressed = touch.priority_tap((1090,630),captured_at=capture,deadline=capture+.2)
            self.assertIsNotNone(pressed)
            self.assertGreaterEqual(pressed[0],capture)
            self.assertGreaterEqual(pressed[1],pressed[0])
            self.wait_for(lambda:any(k=='native_press' and v['kind']=='boss' for k,v in events))
        finally:
            touch.close()

    @unittest.skipUnless(executable_path().is_file(), 'native touch executable not built')
    def test_late_native_press_is_recorded_but_not_returned_for_commit(self):
        events = []
        def popen(args, **kwargs):
            env = os.environ.copy()
            env['NINJA_TOUCH_MOCK_DELAY_MS'] = '100'
            return subprocess.Popen([*args,'--mock'],env=env,**kwargs)
        touch = NativeTouch(executable_path(),'test',0,threading.Event(),
                            lambda kind,value:events.append((kind,value)),popen=popen)
        try:
            capture = time.perf_counter()
            pressed = touch.priority_tap((1090,630),captured_at=capture,deadline=capture+.2,wait_seconds=.05)
            self.assertIsNone(pressed)
            self.wait_for(lambda:any(k=='native_press' and v['kind']=='boss' for k,v in events))
            self.assertEqual(touch.stop_ack,touch.stop_requested)
        finally:
            touch.close()

    @unittest.skipUnless(executable_path().is_file(), 'native touch executable not built')
    def test_concurrent_stop_prevents_press_from_committing(self):
        events, result = [], []
        def popen(args, **kwargs):
            env = os.environ.copy()
            env['NINJA_TOUCH_MOCK_DELAY_MS'] = '100'
            return subprocess.Popen([*args,'--mock'],env=env,**kwargs)
        touch = NativeTouch(executable_path(),'test',0,threading.Event(),
                            lambda kind,value:events.append((kind,value)),popen=popen)
        try:
            capture = time.perf_counter()
            thread = threading.Thread(target=lambda: result.append(touch.priority_tap(
                (1090,630),captured_at=capture,deadline=capture+.5,wait_seconds=.3)))
            thread.start()
            time.sleep(.03)
            touch.stop_all()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result,[None])
            self.wait_for(lambda:any(k=='native_press' and v['kind']=='boss' for k,v in events))
            self.assertEqual(touch.stop_ack,touch.stop_requested)
        finally:
            touch.close()


if __name__ == '__main__':
    unittest.main()
