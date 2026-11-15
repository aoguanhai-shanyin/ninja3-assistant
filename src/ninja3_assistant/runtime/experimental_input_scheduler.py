"""Unwired, bounded input prototype; concurrent use of a MuMu handle is unverified.

Actions must use the supplied stop token for every SDK operation and must release
any pressed contacts in a finally block, as MuMu.tap/swipe do. A timed-out close
does not make an in-flight SDK handle safe to close or share.
"""
from collections import deque
from dataclasses import dataclass
from threading import Condition, Event, Thread
import time


class InputToken:
    """Event-like cancellation token with an absolute same-clock deadline."""
    def __init__(self, clock, deadline):
        self._clock, self._deadline = clock, deadline
        self._cancel = Event()

    def cancel(self):
        self._cancel.set()

    def is_set(self):
        return self._cancel.is_set() or self._clock() >= self._deadline

    def wait(self, seconds):
        remaining = max(0., self._deadline-self._clock())
        self._cancel.wait(min(max(0., seconds), remaining))
        return self.is_set()


@dataclass
class _Job:
    key: str
    generation: int
    action: object
    token: InputToken


class ExperimentalInputScheduler:
    """Single worker, at most one pending job; refuses duplicate or stale work."""
    def __init__(self, *, clock=time.perf_counter):
        self.clock = clock
        self._lock = Condition()
        self._generation = 0
        self._pending = self._active = None
        self._paused = self._closed = False
        self._events = deque(maxlen=256)
        self.dropped_events = 0
        self._thread = Thread(target=self._run, daemon=True, name='experimental-input')
        self._thread.start()

    @property
    def generation(self):
        with self._lock:
            return self._generation

    def _record(self, kind, job, **fields):
        # The condition uses an RLock so it is safe both inside and outside it.
        with self._lock:
            if len(self._events) == self._events.maxlen:
                self.dropped_events += 1
            self._events.append({'event':kind, 'key':job.key, 'generation':job.generation,
                                 'time':self.clock(), **fields})

    def events(self):
        with self._lock:
            return list(self._events)

    def submit(self, key, action, *, generation, deadline):
        """Queue action(token, on_pressed), rejecting full, duplicate or old work."""
        with self._lock:
            if (self._closed or self._paused or generation != self._generation or
                    deadline <= self.clock() or self._pending is not None or
                    (self._active is not None and self._active.key == key)):
                return False
            job = _Job(key, generation, action, InputToken(self.clock, deadline))
            self._pending = job
            self._record('queued', job)
            self._lock.notify()
            return True

    def pause(self):
        """Signal cancellation; use wait_idle before acknowledging the pause."""
        with self._lock:
            self._paused = True
            self._generation += 1
            if self._active:
                self._active.token.cancel()
            if self._pending:
                self._pending.token.cancel()
                self._record('cancelled', self._pending, reason='pause')
                self._pending = None
            self._lock.notify_all()
            return self._generation

    def resume(self):
        with self._lock:
            if self._closed:
                return False
            self._paused = False
            self._lock.notify_all()
            return True

    def wait_idle(self, timeout):
        """A pause is complete only after the in-flight gesture has returned."""
        with self._lock:
            return self._lock.wait_for(lambda: self._active is None, timeout=max(0., timeout))

    def close(self, timeout):
        """Cancel and wait bounded time; False means the owner must retain resources."""
        with self._lock:
            self._closed = True
            self._generation += 1
            if self._active:
                self._active.token.cancel()
            if self._pending:
                self._pending.token.cancel()
                self._record('cancelled', self._pending, reason='close')
                self._pending = None
            self._lock.notify_all()
        self._thread.join(max(0., timeout))
        return not self._thread.is_alive()

    def _run(self):
        while True:
            with self._lock:
                self._lock.wait_for(lambda: self._closed or (not self._paused and self._pending is not None))
                if self._closed:
                    return
                job = self._active = self._pending
                self._pending = None
            try:
                if job.token.is_set():
                    self._record('cancelled', job, reason='expired_before_execution')
                else:
                    self._record('started', job)
                    def pressed(stage, target=job):
                        # Callback means an SDK down really occurred, even if a
                        # cancellation raced with that call; never hide that fact.
                        self._record('sdk_pressed', target, stage=stage, cancelled=target.token.is_set())
                    success = job.action(job.token, pressed)
                    self._record('finished', job, success=bool(success), cancelled=job.token.is_set())
            except Exception as error:
                self._record('failed', job, error=repr(error))
            finally:
                with self._lock:
                    self._active = None
                    self._lock.notify_all()
