"""Bounded parent-side control for the optional MuMu native touch process."""
from pathlib import Path
from collections import deque
import queue
import subprocess
import sys
import threading
import time
import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimedTap:
    offset_ms: int
    point: tuple[int,int]
    hold_ms: int = 15
    slack_ms: int = 20


def executable_path():
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent / 'NinjaTouch.exe'
    return Path(__file__).resolve().parent / 'build' / 'ninja-touch' / 'NinjaTouch.exe'


class NativeTouch:
    """Coalesce observations, drain child output and bound pending skill taps."""

    def __init__(self, executable, folder, index, stopped, emit, *, popen=subprocess.Popen,
                 lease_ms=250, observation_timeout=.15, ready_timeout=6., movement='skate'):
        if movement not in ('skate', 'slide', 'jump-two-slide'):
            raise ValueError('unknown movement pattern')
        self.stopped, self.emit = stopped, emit
        self.lease_ms, self.observation_timeout = lease_ms, observation_timeout
        self.cv = threading.Condition()
        self.ready = False
        self.closed = False
        self.error = None
        self.desired = self.running_sent = False
        self.seen = 0.
        self.frame_time = None
        self.stop_hint = None
        self.last_renew = 0.
        self.movement_until = None
        self.movement_revision = self.movement_sent = 0
        self.stop_requested = self.stop_sent = self.stop_ack = 0
        self.stop_times = {}
        self.start_seq = 0
        self.start_times = {}
        self.tap_seq = 0
        self.requests = {}
        self.priority_pending = None
        self.pending_tap = None
        self.tap_inflight = False
        self.plan_supported = False
        self.plan_seq = 0
        self.plan_pending = {}
        self.plan_live_until = {3:0.,4:0.}
        self.telemetry = deque(maxlen=512)
        self.outbox = queue.Queue(maxsize=128)
        self.dropped_events = 0
        self.proc = popen([str(executable), '--folder', str(folder), '--index', str(index),
                          *(['--'+movement] if movement != 'skate' else [])],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True, encoding='utf-8', errors='replace', bufsize=1,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.reader = threading.Thread(target=self._read, daemon=True, name='native-touch-reader')
        self.writer = threading.Thread(target=self._write, daemon=True, name='native-touch-writer')
        self.notifier = threading.Thread(target=self._notify, daemon=True, name='native-touch-events')
        self.notifier.start()
        self.reader.start()
        self.writer.start()
        with self.cv:
            self.cv.wait_for(lambda: self.ready or self.error or self.proc.poll() is not None,
                             timeout=ready_timeout)
            reason = self.error or (None if self.ready else '原生触控进程未就绪')
            if reason:
                self.closed = True
                self.cv.notify_all()
        if reason:
            self.writer.join(1.)
            self._shutdown_process()
            self.reader.join(1.)
            self.notifier.join(.2)
            if not self.reader.is_alive():
                self.proc.stdout.close()
            raise RuntimeError(str(reason))

    def _check(self):
        if self.error:
            raise RuntimeError('原生触控进程错误: ' + str(self.error))
        if self.closed or self.proc.poll() is not None:
            raise RuntimeError('原生触控进程已退出')

    def renew(self):
        with self.cv:
            self._check()
            if self.stopped.is_set():
                return False
            self.desired = True
            self.movement_until = None
            self.seen = self.frame_time if self.frame_time is not None else time.perf_counter()
            self.cv.notify_all()
            return True

    def schedule_movement(self, seconds=3., *, captured_at):
        """Replace a bounded native movement run; fresh captures keep it valid."""
        if not 0 < seconds <= 3:
            raise ValueError('movement duration must be within 3 seconds')
        with self.cv:
            self._check()
            now = time.perf_counter()
            if self.stopped.is_set() or not 0 <= now-captured_at <= .2:
                return None
            self.movement_revision += 1
            self.movement_until = captured_at+seconds
            self.seen = captured_at
            self.desired = True
            self.cv.notify_all()
            return self.movement_revision

    def movement_remaining(self):
        with self.cv:
            if not self.desired or self.movement_until is None:
                return 0.
            return max(0.,self.movement_until-time.perf_counter())

    def note_frame(self, captured_at):
        with self.cv:
            self.frame_time = captured_at
            if self.desired and self.movement_until is not None:
                self.seen = max(self.seen,captured_at)

    def plan(self, finger, actions, *, captured_at, deadline):
        """Replace one finger's bounded future timeline; return its revision.

        Offsets refer to the original capture, not enqueue time. Native plans
        never change the continuous movement gesture on fingers 1 and 2.
        """
        actions = tuple(actions)
        if finger not in (3,4) or not 1 <= len(actions) <= 16:
            raise ValueError('a plan requires finger 3/4 and 1..16 taps')
        deadline = min(deadline,captured_at+.2)
        captured_us = int(captured_at*1_000_000)
        deadline_us = min(int(deadline*1_000_000),captured_us+200000)
        previous_up = 0
        fields = []
        for action in actions:
            x,y = action.point
            if (not isinstance(action.offset_ms,int) or not isinstance(action.hold_ms,int) or
                not isinstance(action.slack_ms,int) or not 1 <= action.slack_ms <= 200 or
                not isinstance(x,int) or not isinstance(y,int) or
                action.offset_ms < previous_up or not 0 <= x < 1280 or not 0 <= y < 720 or
                not 1 <= action.hold_ms <= 200 or captured_us+action.offset_ms*1000 >= deadline_us):
                raise ValueError('invalid, overlapping or expired timeline action')
            previous_up = action.offset_ms+action.hold_ms
            at_us = captured_us+action.offset_ms*1000
            fields.extend((at_us,x,y,action.hold_ms,min(deadline_us,at_us+action.slack_ms*1000)))
        with self.cv:
            self._check()
            if not self.plan_supported:
                self.cv.wait_for(lambda:self.plan_supported or self.error,timeout=.1)
                self._check()
                if not self.plan_supported:
                    raise RuntimeError('NinjaTouch 缺少 PLAN1 协议，请同步更新运行组件')
            if self.stopped.is_set() or self.stop_requested > self.stop_ack or time.perf_counter() >= deadline:
                return None
            self.plan_seq += 1
            revision = self.plan_seq
            command = f'PLAN {finger} {revision} {captured_us} {deadline_us} {len(actions)} '
            command += ' '.join(map(str,fields))
            self.plan_pending[finger] = (command,deadline)
            self.plan_live_until[finger] = captured_at+previous_up/1000
            self.cv.notify_all()
            return revision

    def cancel_plan(self, finger):
        """Revoke queued actions and release that plan's active contact only."""
        if finger not in (3,4):
            raise ValueError('planned finger must be 3 or 4')
        with self.cv:
            self._check()
            self.plan_seq += 1
            self.plan_pending[finger] = (f'CANCEL {finger} {self.plan_seq}',float('inf'))
            self.plan_live_until[finger] = 0.
            self.cv.notify_all()
            return self.plan_seq

    def _clear_plan_intents(self):
        self.plan_seq += 1
        self.plan_pending.clear()
        self.plan_live_until[3] = self.plan_live_until[4] = 0.

    def _cancel_queued(self):
        self.movement_until = None
        self._clear_plan_intents()
        if self.pending_tap is not None:
            seq,point,_ = self.pending_tap
            self.requests.pop(seq,None)
            self._publish('opening_action',{'kind':'ninjutsu','stage':'dropped',
                'point':point,'id':seq,'reason':'cancelled_before_send'})
            self.pending_tap = None
        if self.priority_pending is not None:
            request = self.requests.get(self.priority_pending)
            if request is not None:
                request['dropped'] = request['done'] = True
                if request.get('consumed'):
                    self.requests.pop(self.priority_pending,None)
            self.priority_pending = None

    def note_stop_reason(self, reason, captured_at=None):
        with self.cv:
            self.stop_hint = (reason, captured_at)

    def _log_stop(self, reason, captured_at=None):
        now = time.perf_counter()
        capture = self.frame_time if captured_at is None else captured_at
        self._publish('native_control', {'event':'stop_requested', 'reason':reason,
            'at_perf':now, 'frame_age_ms':None if capture is None else (now-capture)*1000})

    def _request_stop(self, *, all_touches, reason=None):
        with self.cv:
            self._check()
            now = time.perf_counter()
            hint = self.stop_hint
            self.stop_hint = None
            busy = bool(self.desired or self.running_sent or self.pending_tap is not None or
                        self.priority_pending is not None or self.tap_inflight or self.stop_sent > self.stop_ack or
                        self.plan_pending or any(v > now for v in self.plan_live_until.values()))
            if not all_touches and not (self.desired or self.running_sent or self.stop_sent > self.stop_ack):
                return
            if busy and self.stop_requested == self.stop_ack:
                self._log_stop(reason or (hint[0] if hint else 'explicit_stop'), hint[1] if hint else None)
            self.desired = False
            self._cancel_queued()
            self.stop_requested += 1
            target = self.stop_requested
            self.cv.notify_all()
            if not self.cv.wait_for(lambda: self.stop_ack >= target or self.error or self.proc.poll() is not None,
                                    timeout=.8):
                raise RuntimeError('等待原生触控释放超时')
            self._check()

    def release(self):
        self._request_stop(all_touches=False)

    def stop_all(self):
        self._request_stop(all_touches=True)

    def quiesce(self, *, reason='quiesce', captured_at=None):
        with self.cv:
            self._check()
            busy = bool(self.desired or self.running_sent or self.pending_tap is not None or self.priority_pending is not None or
                        self.tap_inflight or self.stop_sent > self.stop_ack or self.plan_pending or
                        any(v > time.perf_counter() for v in self.plan_live_until.values()))
            if busy:
                self.stop_hint = (reason, captured_at)
        if busy:
            self.stop_all()

    def tap(self, point, *, deadline=None, captured_at=None):
        with self.cv:
            self._check()
            if (self.stopped.is_set() or self.pending_tap is not None or self.tap_inflight or
                    any(v['kind'] == 'boss' and v['priority'] != 2 and not v['done'] for v in self.requests.values())):
                return False
            if self.stop_requested > self.stop_ack:
                return False
            self.tap_seq += 1
            seq = self.tap_seq
            self.requests[seq] = {'kind':'ninjutsu', 'point':tuple(point), 'done':False,
                                 'capture':self.frame_time if captured_at is None else captured_at}
            self.pending_tap = (seq, tuple(point), deadline if deadline is not None else time.perf_counter()+.2)
            self.cv.notify_all()
            return True

    def priority_tap(self, point, *, captured_at, deadline, hold_seconds=.025, wait_seconds=.05, preserve_slide=False):
        """Preempt skating/skill and wait briefly for the real SDK press reply."""
        with self.cv:
            self._check()
            now = time.perf_counter()
            if self.stopped.is_set() or now >= deadline or self.stop_requested > self.stop_ack:
                self._publish('native_control', {'event':'priority_dropped',
                    'reason':'stopped' if self.stopped.is_set() else 'stale_before_enqueue' if now >= deadline else 'stop_pending',
                    'frame_age_ms':(now-captured_at)*1000})
                return None
            if self.desired or self.running_sent or self.tap_inflight or self.pending_tap is not None:
                self._publish('native_control', {'event':'priority_preempt', 'reason':'movement_hop' if preserve_slide else 'boss_action',
                    'frame_age_ms':(now-captured_at)*1000})
            if self.pending_tap is not None and not preserve_slide:
                old, _, _ = self.pending_tap
                self.requests.pop(old, None)
                self.pending_tap = None
            if self.priority_pending is not None:
                old = self.requests.get(self.priority_pending)
                if old:
                    old['dropped'] = True
            self.tap_seq += 1
            seq = self.tap_seq
            request = {'kind':'boss','point':tuple(point),'capture':captured_at,'deadline':deadline,
                       'hold_ms':max(1,min(200,round(hold_seconds*1000))),
                       'pressed':None,'dropped':False,'done':False,'consumed':False,
                       'priority':2 if preserve_slide else 1}
            self.requests[seq] = request
            self.priority_pending = seq
            self.desired = bool(preserve_slide)
            if self.movement_until is not None:
                if preserve_slide:
                    self.movement_revision += 1
                else:
                    self.movement_until = None
            if preserve_slide:
                self.seen = captured_at
            self.cv.notify_all()
            wait = max(0.,min(wait_seconds,deadline-time.perf_counter()))
            self.cv.wait_for(lambda:request['pressed'] is not None or request['dropped'] or self.error or
                             self.proc.poll() is not None, timeout=wait)
            self._check()
            pressed = request['pressed']
            valid = (pressed is not None and not request['dropped'] and not self.stopped.is_set() and
                     self.stop_requested == self.stop_ack and
                     captured_at <= pressed[0] <= pressed[1] <= deadline and time.perf_counter() <= deadline)
            request['consumed'] = True
            if request['done']:
                self.requests.pop(seq, None)
        if valid:
            return pressed
        # A press that arrived after the bounded wait is logged by the reader,
        # but it is not committed to the stale scene. Cancel and await release.
        self._publish('native_control', {'event':'priority_dropped',
            'reason':'native_dropped' if request['dropped'] else 'press_wait_timeout' if pressed is None else 'late_or_cancelled_press',
            'frame_age_ms':(time.perf_counter()-captured_at)*1000, 'id':seq})
        self._request_stop(all_touches=True, reason='priority_cancel')
        return None

    def _write(self):
        try:
            while True:
                with self.cv:
                    if self.closed or self.error:
                        return
                    now = time.perf_counter()
                    if self.desired and self.movement_until is not None and now >= self.movement_until:
                        self.desired = False
                        self._log_stop('movement_plan_finished', self.seen)
                        self._cancel_queued()
                        self.stop_requested += 1
                    if self.desired and now-self.seen > self.observation_timeout:
                        self.desired = False
                        self._log_stop('observation_timeout', self.seen)
                        self._cancel_queued()
                        self.stop_requested += 1
                    if self.stopped.is_set() and (self.desired or self.running_sent or self.pending_tap is not None or
                        self.priority_pending is not None or self.tap_inflight or self.plan_pending or
                        any(v > now for v in self.plan_live_until.values())):
                        self.desired = False
                        if self.stop_requested == self.stop_ack:
                            self._log_stop('stop_event')
                            self.stop_requested += 1
                            self._cancel_queued()
                    if self.stop_requested > self.stop_sent:
                        self.stop_sent = self.stop_requested
                        self.stop_times[self.stop_sent] = now
                        command = f'STOP {self.stop_sent}'
                        if self.plan_supported:
                            command += f' {self.plan_seq}'
                    elif self.stop_sent > self.stop_ack:
                        self.cv.wait(.02)
                        continue
                    elif self.plan_pending:
                        finger = next(iter(self.plan_pending))
                        command,deadline = self.plan_pending.pop(finger)
                        if now >= deadline:
                            self._publish('native_plan',{'event':'expired_before_send','finger':finger})
                            continue
                    elif self.priority_pending is not None:
                        seq = self.priority_pending
                        self.priority_pending = None
                        request = self.requests.get(seq)
                        if request is None or now >= request['deadline']:
                            if request is not None:
                                request['dropped'] = request['done'] = True
                                self._publish('native_control', {'event':'priority_dropped',
                                    'reason':'stale_before_send','id':seq,
                                    'frame_age_ms':(now-request['capture'])*1000})
                                self.cv.notify_all()
                            continue
                        x, y = request['point']
                        self.running_sent = request['priority'] == 2
                        command = f"TAP2 {seq} {x} {y} {request['hold_ms']} {int(request['deadline']*1_000_000)} {request['priority']}"
                    elif self.pending_tap is not None:
                        seq, (x, y), deadline = self.pending_tap
                        self.pending_tap = None
                        if now >= deadline:
                            self.requests.pop(seq, None)
                            self._publish('opening_action', {'kind':'ninjutsu','stage':'dropped',
                                                             'point':(x,y),'reason':'stale_before_send'})
                            continue
                        self.tap_inflight = True
                        command = f'TAP2 {seq} {x} {y} 15 {int(deadline*1_000_000)} 0'
                    elif self.desired and (not self.running_sent or
                            (self.movement_until is None and now-self.last_renew >= .05) or
                            (self.movement_until is not None and self.movement_revision != self.movement_sent)):
                        lease = (self.lease_ms if self.movement_until is None else
                                 max(50,min(3000,math.ceil((self.movement_until-now)*1000))))
                        if self.running_sent:
                            command = f'RENEW {lease}'
                        else:
                            self.start_seq += 1
                            self.start_times = {self.start_seq:self.seen}
                            command = f'START {lease} {self.start_seq}'
                        self.movement_sent = self.movement_revision
                        self.running_sent = True
                        self.last_renew = now
                    else:
                        self.cv.wait(.02)
                        continue
                self.proc.stdin.write(command + '\n')
                self.proc.stdin.flush()
        except (OSError, ValueError, BrokenPipeError) as error:
            with self.cv:
                if not self.closed:
                    self.error = str(error)
                self.cv.notify_all()

    def _read(self):
        try:
            for raw in self.proc.stdout:
                line = raw.strip()
                with self.cv:
                    if line == 'READY':
                        self.ready = True
                    elif line == 'CAPS PLAN1':
                        self.plan_supported = True
                    elif line.startswith('PLAN_'):
                        fields = line.split()
                        value = {'event':fields[0][5:].lower(),'finger':int(fields[1]),'revision':int(fields[2])}
                        if len(fields) >= 4:
                            value['index'] = int(fields[3])
                        if fields[0] == 'PLAN_PRESSED':
                            value['down_start_us'],value['down_end_us'] = int(fields[4]),int(fields[5])
                        elif fields[0] == 'PLAN_RELEASED':
                            value['released_us'] = int(fields[4])
                        self._publish('native_plan',value)
                    elif line.startswith('STARTED '):
                        fields = line.split()
                        if len(fields) == 4:
                            seq, down_start, down_end = int(fields[1]),int(fields[2]),int(fields[3])
                            capture = self.start_times.pop(seq, None)
                            if capture is not None:
                                delay = (down_start/1_000_000-capture)*1000
                                self._publish('native_timing', {'kind':'start_down', 'id':seq,
                                    'capture_perf':capture, 'down_start_us':down_start, 'down_end_us':down_end,
                                    'capture_to_down_ms':delay if delay >= 0 and down_end >= down_start else None})
                    elif line.startswith('STOPPED '):
                        stop_id = int(line.split()[1])
                        if 0 < stop_id <= self.stop_sent:
                            self.stop_ack = max(self.stop_ack, stop_id)
                            self.running_sent = False
                            self.tap_inflight = False
                            sent = self.stop_times.pop(stop_id, None)
                            if sent is not None:
                                self._publish('native_timing', {'kind':'stop_ack','id':stop_id,
                                    'roundtrip_ms':(time.perf_counter()-sent)*1000})
                    elif line.startswith('PRESSED '):
                        fields = line.split()
                        if len(fields) == 4:
                            seq = int(fields[1])
                            request = self.requests.get(seq)
                            if request is not None:
                                start,end = int(fields[2])/1_000_000,int(fields[3])/1_000_000
                                request['pressed'] = (start,end)
                                self._publish('native_press', {'id':seq,'kind':request['kind'],
                                    'point':request['point'],'down_start_us':int(fields[2]),
                                    'down_end_us':int(fields[3]), 'capture_perf':request.get('capture')})
                    elif line.startswith(('TAPPED2 ', 'TAP_DROPPED2 ')):
                        fields = line.split()
                        if len(fields) >= 4:
                            seq = int(fields[1])
                            request = self.requests.get(seq)
                            if request is not None:
                                dropped = fields[0] == 'TAP_DROPPED2'
                                request['dropped'] = dropped
                                request['done'] = True
                                if request['kind'] == 'ninjutsu':
                                    self.tap_inflight = False
                                self._publish('opening_action', {'kind':request['kind'],
                                    'stage':'dropped' if dropped else 'sdk_completed',
                                    'point':request['point'],'id':seq,'reported_at':time.monotonic()})
                                if request['kind'] == 'ninjutsu' or request.get('consumed'):
                                    self.requests.pop(seq, None)
                    elif line.startswith('ERROR'):
                        self.error = line
                    elif line == 'EXPIRED':
                        self.running_sent = False
                    self.cv.notify_all()
                if line.startswith('EVENT '):
                    self.telemetry.append(line)
                elif line.startswith('CYCLE '):
                    self.telemetry.append(line)
                    self._publish('native_touch', line)
        except (OSError, ValueError) as error:
            with self.cv:
                if not self.closed:
                    self.error = str(error)
                self.cv.notify_all()
        finally:
            with self.cv:
                if not self.closed and not self.error:
                    self.error = '原生触控进程输出已关闭'
                self.cv.notify_all()

    def _publish(self, kind, value):
        try:
            self.outbox.put_nowait((kind, value))
        except queue.Full:
            self.dropped_events += 1

    def _notify(self):
        while True:
            try:
                item = self.outbox.get(timeout=.1)
            except queue.Empty:
                if self.closed:
                    return
                continue
            self.emit(*item)

    def _shutdown_process(self):
        try:
            self.proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=1.)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=1.)

    def close(self):
        failure = None
        try:
            if not self.closed and self.proc.poll() is None and not self.error:
                self.stop_all()
        except RuntimeError as error:
            failure = error
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        self.writer.join(1.)
        if self.writer.is_alive() and self.proc.poll() is None:
            # A blocked pipe writer must not hold shutdown forever. The lease
            # has already expired while stop_all waited; report its failure.
            self.proc.terminate()
            self.proc.wait(timeout=1.)
            self.writer.join(1.)
        self._shutdown_process()
        self.reader.join(1.)
        self.notifier.join(.2)
        if not self.reader.is_alive():
            self.proc.stdout.close()
        if failure:
            raise failure


def create_native_touch(folder, index, stopped, emit, **options):
    executable = executable_path()
    if not executable.is_file():
        return None
    return NativeTouch(executable, folder, index, stopped, emit, **options)
