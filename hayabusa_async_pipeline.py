"""Experimental latest-frame vision worker. It never generates game inputs."""
from dataclasses import dataclass
from threading import Condition, Thread
import time


@dataclass(frozen=True)
class Observation:
    generation: int
    frame_id: int
    captured_at: float
    started_at: float
    completed_at: float
    value: object


class LatestVision:
    """One waiting frame and one output; consumers advance once per frame ID.

    Capture timestamps MUST come from timestamp(), using this exact clock.
    The caller must also apply battle/stall/pause and input-generation guards.
    A slow analyze call is cooperative: Python cannot safely stop its thread.
    """
    def __init__(self, analyze, *, reset_analyzer=None, clock=time.perf_counter):
        self.analyze = analyze
        self.reset_analyzer = reset_analyzer
        self.clock = clock
        self.lock = Condition()
        self.generation = 0
        self.next_id = 0
        self.frame = None
        self.output = None
        self.error = None
        self.stopped = False
        self.processed = 0
        self.dropped = 0
        self.thread = Thread(target=self._run, daemon=True, name='hayabusa-latest-vision')
        self.thread.start()

    def timestamp(self):
        return self.clock()

    def submit(self, frame, captured_at):
        """The producer must retain an immutable frame until processing ends."""
        with self.lock:
            if self.stopped:
                raise RuntimeError('vision worker stopped')
            self.next_id += 1
            if self.frame is not None:
                self.dropped += 1
            self.frame = (self.generation, self.next_id, captured_at, frame)
            self.lock.notify()
            return self.next_id

    def reset(self):
        with self.lock:
            self.generation += 1
            self.frame = self.output = None
            self.lock.notify()

    def poll(self, after_id, max_age=.2):
        """Return each recent result at most once; never revive a stale frame."""
        with self.lock:
            if self.error is not None:
                raise RuntimeError('vision worker failed') from self.error
            result = self.output
            if (result is None or result.generation != self.generation or
                    result.frame_id <= after_id or not 0 <= self.clock()-result.captured_at <= max_age):
                return None
            return result

    def close(self, timeout=1.):
        """Stop admission; return false if the analyzer has not exited yet."""
        with self.lock:
            self.stopped = True
            self.frame = self.output = None
            self.lock.notify()
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def _run(self):
        analysis_generation = 0
        while True:
            with self.lock:
                self.lock.wait_for(lambda: self.stopped or self.frame is not None)
                if self.stopped:
                    return
                generation, frame_id, captured_at, frame = self.frame
                self.frame = None
            start = self.clock()
            try:
                if generation != analysis_generation:
                    if self.reset_analyzer is not None:
                        self.reset_analyzer()
                    analysis_generation = generation
                value = self.analyze(frame, captured_at)
            except Exception as error:
                with self.lock:
                    self.error = error
                    self.stopped = True
                return
            end = self.clock()
            with self.lock:
                self.processed += 1
                if not self.stopped and generation == self.generation:
                    self.output = Observation(generation,frame_id,captured_at,start,end,value)
