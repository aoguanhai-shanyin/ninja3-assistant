"""Page observations, guarded menu actions, and navigation over recognized pages."""

from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Mapping


Point = tuple[int, int]


@dataclass(frozen=True)
class PageAction:
    name: str
    point: Point
    end: Point | None = None
    guard: Callable[[object], bool] | None = field(default=None, kw_only=True)
    hold_seconds: float = field(default=.025, kw_only=True)
    max_age: float = field(default=.2, kw_only=True)
    wait_seconds: float = field(default=.2, kw_only=True)


@dataclass(frozen=True)
class PageObservation:
    page: str | None
    frame: object
    generation: int
    captured_at: float
    controls: Mapping[str, PageAction] = field(default_factory=dict)
    detail: object = None
    blocking: bool = False


@dataclass(frozen=True)
class Transition:
    source: str
    target: str
    action: str
    search: str | None = None
    search_limit: int = 0
    attempts: int = 2
    reverse_search: str | None = None


@dataclass(frozen=True)
class NavigationStep:
    source: str
    action: str
    expected: str


@dataclass(frozen=True)
class NavigationResult:
    status: str
    page: str | None
    steps: tuple[NavigationStep, ...] = ()


class NavigationError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(f'Navigation {result.status} at {result.page or "unknown"}')


class GamePage:
    """A recognized view bound to the session that owns its input."""

    page_id: str
    related_pages: tuple[str, ...] = ()

    def __init__(self, ui, observation):
        self.ui = ui
        self.observation = observation

    def _perform(self, name):
        return self.ui.perform(self.observation, name)


def plan_route(routes, source, target):
    """Return the shortest declared path, or None if the target is disconnected."""
    queue = deque([(source, ())])
    seen = {source}
    while queue:
        page, path = queue.popleft()
        if page == target:
            return path
        for edge in routes:
            if edge.source == page and edge.target not in seen:
                seen.add(edge.target)
                queue.append((edge.target, (*path, edge)))
    return None


class GameUI:
    """Borrow the caller's capture and NinjaTouch session for ordinary actions."""

    def __init__(self, control, capture, touch=None, *, recognize=None, routes=(), swipe=None, clock=perf_counter):
        self.control = control
        self.capture = capture
        self.tap = touch if callable(touch) and not hasattr(touch, 'priority_tap') else (
            touch.priority_tap if touch is not None else None)
        self.swipe = swipe
        self.recognize = recognize
        self.routes = tuple(routes)
        self.clock = clock

    def observe(self):
        if self.recognize is None:
            raise ValueError('a page recognizer is required to observe the game')
        generation = self.control.generation()
        captured_at = self.clock()
        return self.recognize(self.capture(), generation, captured_at)

    @staticmethod
    def _inside(point):
        return 0 <= point[0] < 1280 and 0 <= point[1] < 720

    def perform(self, observation, action, *, refresh=True, before_send=None):
        """Return the native acknowledgement, or None if the action cannot be sent."""
        action = observation.controls.get(action) if isinstance(action, str) else action
        if action is None or not self._inside(action.point) or (action.end and not self._inside(action.end)):
            return None

        def active():
            return (observation.generation == self.control.generation()
                    and not self.control.stop.is_set())

        def send():
            if not active():
                return None
            captured_at, frame = observation.captured_at, observation.frame
            if refresh:
                captured_at = self.clock()
                frame = self.capture()
            deadline = captured_at + action.max_age
            if action.guard is not None and not action.guard(frame):
                return None
            if not active() or self.clock() >= deadline:
                return None
            if before_send is not None and before_send() is False:
                return None
            if not active() or self.clock() >= deadline:
                return None
            if action.end is not None:
                return self.swipe(action.point, action.end) if self.swipe is not None else None
            if self.tap is not None:
                return self.tap(action.point, captured_at=captured_at, deadline=deadline,
                                hold_seconds=action.hold_seconds, wait_seconds=action.wait_seconds)
            return None

        return self.control.run_if_active(send) or None

    def plan_route(self, source, target):
        return plan_route(self.routes, source, target)

    def read_page(self, page_type):
        """Read a typed page from the current frame, including its own dialogs."""
        observation = self.observe()
        if observation.page in (page_type.page_id, *page_type.related_pages):
            return page_type(self, observation)
        return None

    def navigate(self, target):
        """Navigate to an ID or return a bound page when given its type."""
        if isinstance(target, type) and issubclass(target, GamePage):
            result, observation = self._navigate(target.page_id)
            if result.status != 'arrived':
                raise NavigationError(result)
            return target(self, observation)
        if not isinstance(target, str) or not target:
            raise ValueError('navigation needs a destination page ID')
        return self._navigate(target)[0]

    def _navigate(self, target):
        deadline = self.clock() + 20
        steps = []
        attempts = {}
        observed = None

        def finish(status):
            page = observed.page if observed is not None else None
            return NavigationResult(status, page, tuple(steps)), observed

        while self.clock() < deadline:
            if self.control.stop.is_set():
                return finish('stopped')
            if self.control.paused():
                return finish('blocked')
            observed = self.observe()
            page = observed.page
            if self.control.stop.is_set():
                return finish('stopped')
            if self.clock() >= deadline:
                break
            if observed.generation != self.control.generation():
                continue
            if observed.blocking:
                return finish('blocked')
            if page == target:
                return finish('arrived')
            if page is not None:
                route = self.plan_route(page, target)
                if route is None:
                    return finish('unreachable')
                edge = route[0]
                searches = attempts.get((page, edge.search), 0)
                search = edge.reverse_search if edge.reverse_search and searches >= edge.search_limit // 2 else edge.search
                action = edge.action if edge.action in observed.controls else search
                if action in observed.controls:
                    searching = action == search
                    key = (page, action)
                    count = attempts.get(key, 0)
                    limit = (edge.search_limit // 2 if edge.reverse_search else edge.search_limit) if searching else edge.attempts
                    if count >= limit:
                        if searching:
                            return finish('unreachable')
                    elif self.perform(observed, action, before_send=lambda: self.clock() < deadline):
                        attempts[key] = count + 1
                        steps.append(NavigationStep(page, action, page if searching else edge.target))
            self.control.stop.wait(max(0, min(.6, deadline - self.clock())))
        return finish('stopped' if self.control.stop.is_set() else 'timeout')
