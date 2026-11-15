"""Process-level native input protocol tests using the built-in mock SDK."""
from ninja3_assistant.paths import project_root
import queue
import os
import subprocess
import threading
import time
import unittest

EXE = project_root()/'build/ninja-touch/NinjaTouch.exe'


@unittest.skipUnless(EXE.is_file(), 'build native_touch/build.cmd first')
class NativeTouchProtocol(unittest.TestCase):
    def setUp(self):
        self.start_process()

    def start_process(self, extra=()):
        self.p = subprocess.Popen([str(EXE),'--mock',*extra],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,text=True,bufsize=1)
        self.lines = []
        self.queue = queue.Queue()
        def read():
            for line in self.p.stdout:
                text = line.strip()
                self.lines.append(text)
                self.queue.put(text)
        self.reader = threading.Thread(target=read,daemon=True)
        self.reader.start()
        self.wait('READY')

    def test_normal_slide_jump_slide_uses_slower_cycles_with_parallel_skills(self):
        self.tearDown()
        self.start_process(['--slide'])
        self.send('START 1000 1')
        self.wait('STARTED 1 ')
        self.plan(4,1,[(20,380,630,15),(100,900,630,15)])
        self.wait('PLAN_RELEASED 4 1 1 ')
        first_line = self.wait('CYCLE 1 ')
        second_line = self.wait('CYCLE 2 ')
        # Let the next cycle finish: reader scheduling must not change which
        # events belong to the two completed cycles under examination.
        self.wait('CYCLE 3 ')
        events = self.lines[:self.lines.index(second_line)]
        first, second = first_line.split(), second_line.split()
        self.assertEqual(sum(' DOWN 2 ' in line for line in events),2)
        self.assertGreaterEqual(int(second[2])-int(first[2]),280000)
        self.assertEqual(sum(' DOWN 1 190 700 ' in line for line in events),4)
        self.assertTrue(any(' DOWN 4 380 630 ' in line for line in events))
        self.assertTrue(any(' DOWN 4 900 630 ' in line for line in events))
        self.send('STOP')
        self.wait('STOPPED')
        self.assert_released()

    def test_two_jumps_then_slide_repeat_with_parallel_skills_and_cancel(self):
        self.tearDown()
        self.start_process(['--jump-two-slide'])
        self.send('START 2500 1')
        self.wait('STARTED 1 ')
        self.plan(4,1,[(20,380,630,15),(100,900,630,15)])
        self.wait('PLAN_RELEASED 4 1 1 ')
        first = self.wait('CYCLE 1 ')
        events = self.lines[:self.lines.index(first)]
        movement = [line.split()[3:] for line in events if line.startswith('EVENT ') and
                    line.split()[4] in ('1','2')]
        presses = [parts for parts in movement if parts[0] == 'DOWN']
        self.assertEqual([parts[1:] for parts in presses],
                         [['2','1090','630','0']]*2+
                         [['1','190','620','0'],['1','190','700','0']])
        second = self.wait('CYCLE 2 ')
        self.assertGreaterEqual(int(second.split()[2])-int(first.split()[2]),460000)
        self.send('STOP')
        self.wait('STOPPED')
        self.assert_released()

    def test_two_jump_avoidance_does_not_hold_slide_against_the_jump(self):
        self.tearDown()
        self.start_process(['--jump-two-slide'])
        self.send('START 2000 1')
        self.wait('STARTED 1 ')
        # Wait for the slide portion of the first cycle.
        deadline = time.monotonic()+1.5
        while not any(' DOWN 1 190 700 ' in line for line in self.lines):
            self.assertLess(time.monotonic(),deadline)
            time.sleep(.005)
        marker = len(self.lines)
        self.send(f'TAP2 2 1090 630 30 {time.perf_counter_ns()//1000+200000} 2')
        self.wait('PRESSED 2 ')
        self.wait('TAPPED2 2 ')
        after = self.lines[marker:]
        self.assertTrue(any(' UP 1 ' in line for line in after))
        self.assertFalse(any(' DOWN 1 ' in line for line in after))
        self.send('STOP')
        self.wait('STOPPED')
        self.assert_released()

    def test_two_jump_pattern_releases_priority_down_before_resuming_jumps(self):
        self.tearDown()
        self.start_process(['--jump-two-slide'])
        self.send('START 2500 1')
        self.wait('STARTED 1 ')
        self.send(f'TAP2 2 190 630 30 {time.perf_counter_ns()//1000+200000} 2')
        self.wait('TAPPED2 2 ')
        self.send('RENEW 2500')
        self.wait('CYCLE 1 ')
        held = set()
        for line in self.lines:
            parts = line.split()
            if parts[:1] != ['EVENT']:
                continue
            action,finger = parts[3],int(parts[4])
            if action == 'DOWN':
                if finger == 2:
                    self.assertNotIn(1,held)
                held.add(finger)
            else:
                held.discard(finger)
        self.send('STOP')
        self.wait('STOPPED')
        self.assert_released()

    def tearDown(self):
        if self.p.poll() is None:
            try:
                self.send('QUIT')
                self.p.wait(timeout=2)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.p.kill()
                self.p.wait(timeout=2)
        self.p.stdout.close()
        self.p.stderr.close()
        if not self.p.stdin.closed:
            self.p.stdin.close()

    def send(self, text):
        self.p.stdin.write(text+'\n')
        self.p.stdin.flush()

    def wait(self, prefix, timeout=2):
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            try:
                line = self.queue.get(timeout=max(.001,deadline-time.monotonic()))
            except queue.Empty:
                break
            if line.startswith(prefix):
                return line
        self.fail(f'{prefix} missing: {self.lines[-20:]}')

    def assert_released(self):
        held = set()
        for line in self.lines:
            parts = line.split()
            if parts[:1] != ['EVENT']:
                continue
            action, finger = parts[3], int(parts[4])
            if action == 'DOWN': held.add(finger)
            elif action == 'UP': held.discard(finger)
        self.assertEqual(held,set(),self.lines[-20:])

    def plan(self, finger, revision, actions, *, captured=None):
        captured = time.perf_counter_ns()//1000 if captured is None else captured
        fields = []
        for offset,x,y,hold in actions:
            at = captured+offset*1000
            fields.extend((at,x,y,hold,min(captured+200000,at+20000)))
        self.send(f'PLAN {finger} {revision} {captured} {captured+200000} {len(actions)} '+
                  ' '.join(map(str,fields)))
        self.wait(f'PLAN_ACCEPTED {finger} {revision}')
        return captured

    def test_native_timeline_executes_multiple_frames_without_parent_renewals(self):
        captured = self.plan(4,1,[(30,380,630,10),(95,900,630,10),(160,740,630,10)])
        starts = []
        for i,offset in enumerate((30,95,160)):
            fields = self.wait(f'PLAN_PRESSED 4 1 {i} ').split()
            started = int(fields[4])
            self.assertGreaterEqual(started,captured+offset*1000)
            self.assertLess(started,captured+(offset+20)*1000)
            starts.append(started)
        self.wait('PLAN_RELEASED 4 1 2 ')
        self.assertGreater(starts[-1]-starts[0],120000)
        self.send('STOP')
        self.wait('STOPPED')
        self.assert_released()

    def test_newer_plan_replaces_queued_old_frame(self):
        captured = self.plan(4,1,[(120,100,100,10)])
        self.plan(4,2,[(150,200,200,10)],captured=captured)
        self.wait('PLAN_PRESSED 4 2 0 ')
        self.assertFalse(any(x.startswith('PLAN_PRESSED 4 1 ') for x in self.lines))
        self.assertIn('PLAN_DROPPED 4 1 0',self.lines)

    def test_cancel_one_finger_preserves_other_plan_and_skating(self):
        self.send('START 1000 301')
        self.wait('STARTED 301 ')
        captured = self.plan(4,1,[(120,380,630,10)])
        self.plan(3,2,[(150,1090,630,10)],captured=captured)
        self.send('CANCEL 4 3')
        self.wait('PLAN_CLEARED 4 3')
        self.wait('PLAN_PRESSED 3 2 0 ')
        self.assertFalse(any(x.startswith('PLAN_PRESSED 4 1 ') for x in self.lines))
        self.assertTrue(any(x.startswith('CYCLE ') for x in self.lines))
        self.assertFalse(any(x.startswith('STOPPED') for x in self.lines))
        self.send('STOP')
        self.wait('STOPPED')
        self.assert_released()

    def test_cancel_releases_active_plan_contact_without_stopping_movement(self):
        self.send('START 1000 302')
        self.wait('STARTED 302 ')
        self.plan(4,1,[(20,900,630,150)])
        self.wait('PLAN_PRESSED 4 1 0 ')
        self.send('CANCEL 4 2')
        self.wait('PLAN_DROPPED 4 1 0')
        self.assertTrue(any(' UP 4 ' in x for x in self.lines))
        self.wait('CYCLE ')
        self.assertFalse(any(x.startswith('STOPPED') for x in self.lines))
        self.send('STOP')
        self.wait('STOPPED')
        self.assert_released()

    def test_expired_action_is_not_replayed_and_next_action_still_runs(self):
        captured = time.perf_counter_ns()//1000-50000
        self.plan(4,1,[(10,100,100,10),(120,200,200,10)],captured=captured)
        self.wait('PLAN_PRESSED 4 1 1 ')
        self.assertIn('PLAN_DROPPED 4 1 0',self.lines)
        self.assertFalse(any(x.startswith('PLAN_PRESSED 4 1 0 ') for x in self.lines))

    def test_stop_revision_barrier_rejects_pre_stop_plan(self):
        self.send('STOP 1 100')
        self.wait('STOPPED 1')
        captured = time.perf_counter_ns()//1000
        at = captured+50000
        self.send(f'PLAN 4 99 {captured} {captured+200000} 1 {at} 100 100 10 {at+20000}')
        self.wait('PLAN_REJECTED 4 99')
        self.plan(4,101,[(50,200,200,10)])
        self.wait('PLAN_PRESSED 4 101 0 ')

    def test_overlapping_plan_is_rejected_atomically(self):
        captured = time.perf_counter_ns()//1000
        self.send(f'PLAN 4 1 {captured} {captured+200000} 2 '+
                  f'{captured+50000} 100 100 100 {captured+70000} '+
                  f'{captured+80000} 200 200 10 {captured+100000}')
        self.wait('ERROR invalid_plan')
        self.assertFalse(any(x.startswith('PLAN_PRESSED ') for x in self.lines))

    def test_idle_tap_completes(self):
        self.send('TAP 70 330')
        self.wait('TAPPED 70 330')
        self.send('STOP')
        self.wait('STOPPED')
        self.assert_released()

    def test_glide_jump_keeps_slide_finger_until_explicit_stop(self):
        self.send('START 1000 123')
        self.wait('STARTED 123 ')
        deadline = time.perf_counter_ns()//1000 + 500_000
        self.send(f'TAP2 88 1090 630 100 {deadline} 2')
        pressed = self.wait('PRESSED 88 ').split()
        started = int(pressed[2])
        time.sleep(.2)
        stopping = time.perf_counter_ns()//1000
        self.send('STOP 89')
        self.wait('STOPPED 89')
        early = [line for line in self.lines if line.startswith('EVENT ') and
                 ' UP 1 ' in line and started < int(line.split()[1]) < stopping]
        self.assertEqual(early,[])
        self.assert_released()

    def test_skill_and_glide_jump_overlap_on_separate_fingers(self):
        deadline = time.perf_counter_ns()//1000 + 500_000
        self.send(f'TAP2 101 1090 630 200 {deadline} 2')
        self.wait('PRESSED 101 ')
        self.send(f'TAP2 102 900 630 100 {deadline} 0')
        self.wait('PRESSED 102 ')
        # Skill must press on finger 4 while the jump on finger 3 is held.
        self.assertTrue(any(' DOWN 4 900 630 ' in x for x in self.lines))
        self.assertFalse(any(x.startswith('TAPPED2 101 ') for x in self.lines))
        self.send('STOP 103')
        self.wait('STOPPED 103')
        self.assert_released()
        self.assertTrue(any(x.startswith('TAP_DROPPED2 101 ') for x in self.lines))
        self.assertTrue(any(x.startswith('TAP_DROPPED2 102 ') for x in self.lines))

    def test_skate_cycles_continue_while_skill_finger_is_held(self):
        self.send('START 1000 201')
        self.wait('STARTED 201 ')
        deadline = time.perf_counter_ns()//1000 + 500_000
        self.send(f'TAP2 202 380 630 200 {deadline} 0')
        pressed = int(self.wait('PRESSED 202 ').split()[2])
        cycle = self.wait('CYCLE ').split()
        self.assertGreater(int(cycle[3]),pressed)
        self.assertFalse(any(x.startswith('TAPPED2 202 ') for x in self.lines))
        self.wait('TAPPED2 202 ')
        self.send('STOP 203')
        self.wait('STOPPED 203')
        self.assert_released()

    def test_numbered_start_reports_actual_first_down_qpc(self):
        before = time.perf_counter_ns()//1000
        self.send('START 500 123')
        parts = self.wait('STARTED 123 ').split()
        self.assertGreaterEqual(int(parts[2]),before)
        self.assertGreaterEqual(int(parts[3]),int(parts[2]))
        self.send('STOP 9')
        self.wait('STOPPED 9')
        self.assert_released()

    def test_priority_dodge_preempts_skate_and_skill(self):
        self.send('START 1000 8')
        self.wait('STARTED 8 ')
        deadline = time.perf_counter_ns()//1000 + 500_000
        self.send(f'TAP2 1 70 330 200 {deadline} 0')
        self.wait('PRESSED 1 ')
        self.send(f'TAP2 2 1090 630 25 {deadline} 1')
        self.wait('TAP_DROPPED2 1 ')
        pressed = self.wait('PRESSED 2 ').split()
        self.assertGreaterEqual(int(pressed[3]),int(pressed[2]))
        self.wait('TAPPED2 2 1090 630')
        self.send('STOP 10')
        self.wait('STOPPED 10')
        self.assert_released()

    def test_expired_numbered_tap_is_not_pressed(self):
        deadline = time.perf_counter_ns()//1000 - 1
        self.send(f'TAP2 12 190 630 25 {deadline} 1')
        self.wait('TAP_DROPPED2 12 190 630')
        self.assertFalse(any(x.startswith('PRESSED 12 ') for x in self.lines))

    def test_renew_tap_and_interrupting_stop_release_every_contact(self):
        self.send('START 500')
        self.wait('STARTED')
        self.send('RENEW 500')
        self.send('TAP 70 330')
        self.wait('TAPPED 70 330')
        self.wait('CYCLE ')
        self.send('STOP')
        self.wait('STOPPED')
        self.assert_released()
        event_count = sum(x.startswith('EVENT ') for x in self.lines)
        time.sleep(.18)
        self.assertEqual(sum(x.startswith('EVENT ') for x in self.lines),event_count)

    def test_expired_lease_releases_all_contacts(self):
        self.send('START 100')
        self.wait('STARTED')
        self.wait('EXPIRED')
        self.wait('STOPPED')
        self.assert_released()

    def test_numbered_stop_ack_is_distinct_from_unsolicited_expiry(self):
        self.send('START 100')
        self.wait('STARTED')
        self.wait('EXPIRED')
        self.assertEqual(self.wait('STOPPED'),'STOPPED')
        self.send('STOP 123')
        self.assertEqual(self.wait('STOPPED 123'),'STOPPED 123')
        self.assert_released()

    def test_cancel_inflight_tap_reports_drop_before_stop_ack(self):
        self.send('TAP 70 330')
        while True:
            event = self.wait('EVENT ')
            if ' DOWN 3 ' in event:
                break
        self.send('STOP 42')
        self.assertEqual(self.wait('TAP_DROPPED 70 330'),'TAP_DROPPED 70 330')
        self.assertEqual(self.wait('STOPPED 42'),'STOPPED 42')
        self.assert_released()
        after = len(self.lines)
        time.sleep(.1)
        self.assertFalse(any(x.startswith(('TAPPED ','TAP_DROPPED ')) for x in self.lines[after:]))

    def test_queued_tap_expiry_reports_drop(self):
        env = os.environ.copy()
        env['NINJA_TOUCH_MOCK_DELAY_MS'] = '300'
        process = subprocess.Popen([str(EXE),'--mock'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL,text=True,bufsize=1,env=env)
        lines = queue.Queue()
        reader = threading.Thread(target=lambda: [lines.put(x.strip()) for x in process.stdout],daemon=True)
        reader.start()
        try:
            self.assertEqual(lines.get(timeout=2),'READY')
            process.stdin.write('TAP 11 22\n')
            process.stdin.flush()
            time.sleep(.1)
            process.stdin.write('TAP 33 44\n')
            process.stdin.flush()
            deadline = time.monotonic()+2
            seen = []
            while time.monotonic()<deadline:
                event = lines.get(timeout=max(.01,deadline-time.monotonic()))
                seen.append(event)
                if event == 'TAP_DROPPED 33 44':
                    break
            self.assertIn('TAP_DROPPED 33 44',seen)
            process.stdin.write('STOP 7\nQUIT\n')
            process.stdin.flush()
            process.wait(timeout=2)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
            process.stdin.close()
            process.stdout.close()

    def test_eof_releases_and_exits(self):
        self.send('START 1000')
        self.wait('STARTED')
        self.p.stdin.close()
        self.assertEqual(self.p.wait(timeout=2),0)
        self.reader.join(timeout=1)
        self.assertIn('STOPPED',self.lines)
        self.assert_released()

    def test_stop_when_idle_and_invalid_input(self):
        self.send('STOP')
        self.wait('STOPPED')
        self.send('TAP -1 4')
        self.wait('ERROR invalid_tap')
        self.send('RENEW 250')
        self.wait('ERROR not_started')

    def test_slow_stdout_cannot_block_touch_and_stop(self):
        process = subprocess.Popen([str(EXE),'--mock'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL,text=True,bufsize=1)
        try:
            self.assertEqual(process.stdout.readline().strip(),'READY')
            process.stdin.write('START 1000\n')
            process.stdin.flush()
            end = time.monotonic()+2.5
            while time.monotonic()<end:
                process.stdin.write('RENEW 1000\n')
                process.stdin.flush()
                time.sleep(.2)
            # With nobody reading stdout the OS pipe fills.  Releasing it must
            # not depend on the telemetry writer making progress.
            process.stdin.write('STOP\nQUIT\n')
            process.stdin.flush()
            output, _ = process.communicate(timeout=3)
            self.assertEqual(process.returncode,0)
            self.assertIn('STOPPED',output)
            starts = [int(line.split()[2]) for line in output.splitlines() if line.startswith('CYCLE ')]
            self.assertGreater(len(starts),5)
            self.assertGreater(starts[-1]-starts[0],1_000_000)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)


if __name__ == '__main__':
    unittest.main()
