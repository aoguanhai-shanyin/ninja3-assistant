"""Shared unbuffered capture and bounded C++ dispatch for battle strategies.

Strategies own perception and movement decisions. This kernel never queues
frames or replays expired actions; the native process owns gesture timing.
"""
import time
import cv2
import queue
import threading
from dataclasses import dataclass

from ninja3_assistant.device.native_touch import PLAN_FRAME_MAX_AGE, PLAN_HORIZONS


@dataclass(frozen=True, slots=True)
class FrameStamp:
    sequence: int
    generation: int
    captured_at: float


class RecognitionCadence:
    """Bounded 30..60 Hz recognition; urgency and generation bypass skipping."""
    __slots__ = ('next_at','generation','cost_ms','recognized','skipped')

    def __init__(self):
        self.next_at = float('-inf')
        self.generation = -1
        self.cost_ms = 0.
        self.recognized = self.skipped = 0

    def due(self, now, generation, *, urgent=False):
        if urgent or generation != self.generation or now+1e-9 >= self.next_at:
            self.generation = generation
            return True
        self.skipped += 1
        return False

    def completed(self, captured_at, finished_at, *, stable=False):
        cost = max(0.,finished_at-captured_at)
        self.cost_ms = cost*1000 if not self.recognized else self.cost_ms*.8+cost*200
        interval = 1/30 if stable else max(1/60,min(1/30,self.cost_ms/500))
        self.next_at = max(finished_at,captured_at+interval)
        self.recognized += 1


class FrameDiagnostics:
    """Drop old debug frames rather than block the battle loop on image I/O."""
    def __init__(self):
        self.queue = queue.Queue(maxsize=2)
        self.thread = threading.Thread(target=self._write,daemon=True,name='battle-diagnostics')
        self.thread.start()

    def save(self, path, frame):
        try:
            self.queue.put_nowait((path,frame))
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait((path,frame))
            except queue.Full:
                pass

    def _write(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            path,frame = item
            cv2.imwrite(str(path),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

    def close(self):
        self.queue.put(None)
        self.thread.join()


class RealtimeBattleKernel:
    def __init__(self, control, *, clock=None, movement_horizon=None):
        cv2.setNumThreads(1)
        self.control, self.clock = control, clock or (lambda:time.perf_counter())
        self.movement_horizon=movement_horizon
        self.driver = None
        self.captured_at = float('-inf')
        self.generation = control.generation()
        self.capture_ms = 0.
        self.stale_drops = 0
        self.sequence = 0
        self.cadence = RecognitionCadence()
        self.plan_frames = dict.fromkeys(PLAN_HORIZONS,0)
        self.movement_frame = 0

    def attach(self, driver):
        self.driver = driver
        return self

    def capture(self, device):
        self.generation = self.control.generation()
        self.captured_at = self.clock()
        self.sequence += 1
        frame = device.capture()
        self.capture_ms = (self.clock()-self.captured_at)*1000
        if self.driver is not None:
            self.driver.note_frame(self.captured_at)
        return frame

    def fresh(self, captured_at=None, deadline=None):
        captured_at = self.captured_at if captured_at is None else min(self.captured_at,captured_at)
        expires = captured_at+.2
        if deadline is not None:
            expires = min(expires,deadline)
        return (not self.control.stop.is_set() and not self.control.paused() and
                self.control.generation() == self.generation and
                self.clock() <= expires)

    def note_frame(self, captured_at):
        self.driver.note_frame(captured_at)

    def renew(self):
        if self.fresh():
            if self.movement_horizon is not None:
                if self.driver.movement_remaining() <= .3:
                    return self.schedule_movement(self.movement_horizon)
                return True
            return self.control.run_if_active(self.driver.renew)
        self.stale_drops += 1
        if (self.movement_horizon is not None and not self.control.stop.is_set() and
                not self.control.paused() and self.control.generation() == self.generation and
                self.driver.movement_remaining() > 0):
            return False
        self.driver.release()
        return False

    def tap(self, point, *, deadline=None):
        deadline = min(self.captured_at+.2,deadline) if deadline is not None else self.captured_at+.2
        if not self.fresh(deadline=deadline):
            self.stale_drops += 1
            return False
        return self.control.run_if_active(lambda:self.driver.tap(point,deadline=deadline))

    def schedule_movement(self, seconds=3., *, frame=None):
        frame = self.stamp() if frame is None else frame
        def commit():
            if (not 1 <= frame.sequence <= self.sequence or frame.sequence < self.movement_frame or
                    frame.captured_at > self.captured_at or
                    frame.generation != self.control.generation() or
                    not self.fresh(frame.captured_at)):
                self.stale_drops += 1
                return None
            revision=self.driver.schedule_movement(seconds,captured_at=frame.captured_at)
            if revision is not None:
                self.movement_frame=frame.sequence
            return revision
        return self.control.run_if_active(commit)

    def cancel_movement(self):
        self.movement_frame=self.sequence
        return self.control.run_if_active(self.driver.release)

    def priority_tap(self, point, *, captured_at, deadline, **options):
        captured_at = min(self.captured_at,captured_at)
        deadline = min(captured_at+.2,deadline)
        if not self.fresh(captured_at,deadline):
            self.stale_drops += 1
            return None
        return self.control.run_if_active(lambda:self.driver.priority_tap(
            point,captured_at=captured_at,deadline=deadline,**options))

    def stamp(self):
        return FrameStamp(self.sequence,self.generation,self.captured_at)

    def recognition_due(self, *, urgent=False):
        return self.cadence.due(self.captured_at,self.generation,urgent=urgent)

    def recognized(self, *, stable=False):
        self.cadence.completed(self.captured_at,self.clock(),stable=stable)

    def plan(self, finger, actions, *, frame=None, deadline=None):
        if finger not in PLAN_HORIZONS:
            raise ValueError('planned finger must be 3, 4, or 5')
        frame = self.stamp() if frame is None else frame
        expires = frame.captured_at+PLAN_HORIZONS[finger]
        if deadline is not None:
            expires = min(deadline,expires)
        admission_deadline = min(expires,frame.captured_at+PLAN_FRAME_MAX_AGE)
        def commit():
            if (not 1 <= frame.sequence <= self.sequence or frame.captured_at > self.captured_at or
                frame.sequence < self.plan_frames[finger] or frame.generation != self.control.generation() or
                self.clock() >= admission_deadline or self.control.paused() or self.control.stop.is_set()):
                self.stale_drops += 1
                return None
            revision = self.driver.plan(finger,actions,captured_at=frame.captured_at,deadline=expires)
            if revision is not None:
                self.plan_frames[finger] = frame.sequence
            return revision
        result = self.control.run_if_active(commit)
        return None if result is False else result

    def cancel_plan(self, finger):
        if finger not in PLAN_HORIZONS:
            raise ValueError('planned finger must be 3, 4, or 5')
        def cancel():
            # A cancelled board sequence must wait for a new capture.
            self.plan_frames[finger] = self.sequence+1 if finger == 5 else self.sequence
            return self.driver.cancel_plan(finger)
        return self.control.run_if_active(cancel)

    def metrics(self):
        return {'capture_ms':round(self.capture_ms,2),
                'frame_age_ms':round((self.clock()-self.captured_at)*1000,2),
                'stale_drops':self.stale_drops,'recognitions':self.cadence.recognized,
                'recognition_skips':self.cadence.skipped}

    def __getattr__(self, name):
        return getattr(self.driver,name)
