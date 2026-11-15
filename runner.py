"""Shared, UI-independent runner for the reverse-S workflow."""

import os
import threading
import time
from collections import deque

from flow import Decision, Flow
from hayabusa_scene import HayabusaScene
from mumu import MuMu, prepare_game
from opening_actions import OpeningActions, execute_opening
from mumu_macro import GroundMacroGate, create_skate_macro
from native_touch import create_native_touch
from realtime_battle import RealtimeBattleKernel
from vision import Vision


class RunControl:
    def __init__(self, stop=None, clock=time.monotonic, *, hide_game_window=None):
        self.stop = stop if stop is not None else threading.Event()
        self._clock = clock
        self._lock = threading.RLock()
        self._paused_at = None
        self._paused_total = 0.0
        self._generation = 0
        self._release_input = None
        self.hide_game_window = hide_game_window

    def set_release_input(self, callback):
        with self._lock:
            self._release_input = callback

    def request_stop(self):
        self.stop.set()
        with self._lock:
            if self._release_input is not None:
                self._release_input()

    def pause(self):
        with self._lock:
            if self._paused_at is None and not self.stop.is_set():
                self._paused_at = self._clock()
                self._generation += 1
                if self._release_input is not None:
                    self._release_input()
                return True
        return False

    def resume(self):
        with self._lock:
            if self._paused_at is not None:
                self._paused_total += self._clock() - self._paused_at
                self._paused_at = None
                return True
        return False

    def paused(self):
        with self._lock:
            return self._paused_at is not None

    def now(self):
        with self._lock:
            return (self._paused_at if self._paused_at is not None else self._clock()) - self._paused_total

    def generation(self):
        with self._lock:
            return self._generation

    def run_if_active(self, action):
        # A pause is acknowledged only after an in-flight touch has completed.
        with self._lock:
            if self._paused_at is not None or self.stop.is_set():
                return False
            return action()

    def wait_active(self):
        while self.paused() and not self.stop.wait(.05):
            pass
        return not self.stop.is_set()


def run_flow(folder, index, control, emit, join_interval_ms=30):
    kernel = RealtimeBattleKernel(control)
    device = None
    scene = None
    native = macro = gate = None
    last_status = None

    def status(message):
        nonlocal last_status
        if message != last_status:
            last_status = message
            emit('status', message)

    try:
        if not 20 <= join_interval_ms <= 1000:
            raise ValueError('抢队点击间隔须在 20 到 1000 毫秒之间')
        launched = prepare_game(folder, index, control, status)
        if launched is None:
            return
        vision = Vision()
        emit('backend', vision.backend)
        if control.stop.is_set():
            return
        deadline = control.now() + 40
        while control.wait_active():
            try:
                device = MuMu(folder, index)
                break
            except RuntimeError:
                if not launched or control.now() >= deadline:
                    raise
                status('等待游戏画面就绪…')
                control.stop.wait(.5)
        if device is None:
            return
        flow = Flow()
        flow.entering = launched
        scene = HayabusaScene(clock=control.now)
        opening = OpeningActions()
        native = create_native_touch(folder, index, control.stop, emit)
        if native is not None:
            native = kernel.attach(native)
        macro = native
        if macro is None and os.environ.get('NINJA_USE_MUMU_MACRO') == '1':
            macro = create_skate_macro(folder, index, control.stop, emit)
        status('C++ 触控进程已就绪' if native is not None else
               '使用 MuMu 宏对照模式' if macro is not None else '未找到 C++ 触控组件，使用 SDK 回退')
        if macro is not None:
            gate = GroundMacroGate(macro, control)
            def release_inputs():
                gate.release(reason='stop' if control.stop.is_set() else 'pause')
                if native is not None:
                    note = getattr(native, 'note_stop_reason', None)
                    if note is not None:
                        note('stop' if control.stop.is_set() else 'pause')
                    native.stop_all()
            control.set_release_input(release_inputs)
        scene_generation = control.generation()
        scene_owned = False
        current_backend = vision.backend
        clicks = frames = reported_kills = 0
        last_join_press = None
        recent_join_intervals = deque(maxlen=20)
        join_interval = join_interval_ms / 1000.0
        interval_start = control.now()
        emit('state', 'paused' if control.paused() else 'running')
        while control.wait_active():
            generation = control.generation()
            if generation != scene_generation:
                if gate is not None:
                    gate.release(reason='generation_changed')
                scene.reset()
                opening.reset()
                scene_generation = generation
            captured = time.monotonic()
            captured_precise = time.perf_counter()
            frame = kernel.capture(device)
            capture_seconds = time.perf_counter()-captured_precise
            scene_capture_time = control.now()
            screen = vision.detect(frame)
            if control.paused() or control.stop.is_set() or control.generation() != generation:
                continue
            # Navigation and modal controls take priority over a previously owned
            # battle, including when its HUD is still visible behind the popup.
            navigation = bool(screen.board or screen.menu or screen.ready or screen.prepared
                or screen.home or screen.login or screen.hall or screen.network_error or screen.invite)
            if navigation and gate is not None:
                gate.release(reason='navigation', captured_at=captured_precise)
                if native is not None:
                    native.quiesce(reason='navigation', captured_at=captured_precise)
            scene_decision = None if navigation else scene.step(frame, scene_capture_time)
            if scene_decision is not None:
                scene.capture_duration(capture_seconds)
                scene.frame_age(time.perf_counter()-captured_precise)
            if control.paused() or control.stop.is_set() or control.generation() != generation:
                continue
            claimed = bool(scene_decision is not None and scene_decision.claimed)
            if claimed:
                scene_owned = True
            boss_active = bool(claimed and scene_decision.active)
            # Count every processed frame, including frames that immediately
            # dispatch an opening action and otherwise continue the loop.
            frames += 1
            active_now = control.now()
            if active_now - interval_start >= 1:
                actual = sum(recent_join_intervals) / len(recent_join_intervals) if recent_join_intervals else None
                emit('metrics', {'fps': frames / (active_now - interval_start), 'clicks': clicks,
                                 'join_interval_ms': actual})
                frames, interval_start = 0, active_now
            priority_action = bool(native is not None and claimed and scene_decision.action is not None and
                                   (scene_decision.active or scene_decision.result) and
                                   not scene_decision.stalled and time.perf_counter()-captured_precise <= .2)
            if gate is not None and (boss_active or (claimed and
                    (scene_decision.action or scene_decision.result or scene_decision.stalled))):
                if priority_action:
                    gate.forget()
                else:
                    gate.release(reason='boss_or_scene_action' if boss_active or scene_decision.action else
                                 'scene_result' if scene_decision.result else 'scene_stalled',
                                 captured_at=captured_precise)
                    if native is not None and (scene_decision.action or scene_decision.result or scene_decision.stalled):
                        native.quiesce(reason='scene_action_or_result', captured_at=captured_precise)
            if (not navigation and not (claimed and scene_decision.action)
                    and not (claimed and scene_decision.result) and not (claimed and scene_decision.stalled)):
                opening_action = opening.decide(frame, scene_capture_time, boss_active=boss_active,
                    battle_allowed=boss_active or (not scene_owned and screen.battle_active))
                macro_active = False
                if gate is not None:
                    if native is not None:
                        native.note_frame(captured_precise)
                    macro_active = gate.update(
                        allowed=not (boss_active or scene_owned or screen.result or screen.training or
                                     screen.game_loading or screen.battle_loading),
                        ground_frames=opening.ground_frames, captured_at=captured_precise)
                if opening_action is not None:
                    if opening_action.kind == 'ground_jump_slide' and macro_active:
                        opening.completed(opening_action, control.now())
                        continue
                    def do_opening():
                        if control.generation() != generation or time.perf_counter()-captured_precise > .2:
                            return
                        if native is not None and opening_action.kind == 'ninjutsu':
                            if native.tap(opening_action.point, deadline=captured_precise+.2):
                                opening.scheduled(opening_action, control.now())
                        elif execute_opening(opening_action, device, control, emit):
                            opening.completed(opening_action, control.now())
                    control.run_if_active(do_opening)
                    continue
            elif gate is not None:
                if not priority_action:
                    gate.release(reason='opening_not_allowed', captured_at=captured_precise)
            # Generic battle/loading markers can also occur in obscured Hayabusa
            # frames; they cannot establish that a different scene owns the frame.
            release_scene = scene_owned and (navigation or bool(screen.result and
                not (claimed and (scene_decision.active or scene_decision.result))))
            if claimed and not release_scene:
                def process_scene():
                    nonlocal clicks, current_backend
                    if control.generation() != generation:
                        return
                    status(scene_decision.status)
                    if scene_decision.backend and scene_decision.backend != current_backend:
                        current_backend = scene_decision.backend
                        emit('backend', current_backend)
                    if ((scene_decision.active or scene_decision.result) and
                            scene_decision.action is not None and
                            control.generation() == generation and
                            not control.paused() and not control.stop.is_set() and
                            time.perf_counter()-captured_precise <= .2):
                        def pressed():
                            nonlocal clicks
                            clicks += 1
                            scene.completed(scene_decision.action, control.now())
                        input_started = time.perf_counter()
                        try:
                            if native is not None:
                                ack = native.priority_tap(scene_decision.action.point, captured_at=captured_precise,
                                                          deadline=captured_precise+.2, hold_seconds=.025)
                                if ack is not None and control.generation() == generation and not control.stop.is_set():
                                    clicks += 1
                                    pressed_at = scene_capture_time + max(0., ack[0]-(captured_precise+capture_seconds))
                                    scene.completed(scene_decision.action, pressed_at)
                                else:
                                    scene.failed_touch(RuntimeError('native press expired or cancelled'))
                            else:
                                device.tap(scene_decision.action.point, control.stop,
                                           hold_seconds=.025, on_pressed=pressed)
                        except Exception as error:
                            scene.failed_touch(error)
                            raise
                        finally:
                            scene.input_duration(time.perf_counter()-input_started)
                control.run_if_active(process_scene)
                continue
            def process_screen():
                nonlocal clicks, reported_kills, current_backend, scene_owned
                if control.generation() != generation:
                    return 'stale'
                if release_scene:
                    scene.release()
                    opening.reset()
                    scene_owned = False
                    if current_backend != vision.backend:
                        current_backend = vision.backend
                        emit('backend', current_backend)
                decision = flow.step(screen, control.now())
                if flow.successful_kills > reported_kills:
                    emit('kill', flow.successful_kills - reported_kills)
                    reported_kills = flow.successful_kills
                status(decision.status)
                if decision.action in ('stop', 'done'):
                    return 'stop'
                if decision.action == 'reopen':
                    if native is not None:
                        native.quiesce()
                    device.reopen_game(control.stop)
                    return 'reopen'
                if decision.action != 'wait' and decision.point and time.monotonic() - captured <= .5:
                    if decision.action == 'join':
                        return decision
                    if native is not None:
                        native.quiesce()
                    if device.tap(decision.point, control.stop):
                        clicks += 1
                        flow.completed(decision, control.now())
                return 'continue'

            result = control.run_if_active(process_screen)
            if result == 'stop':
                break
            if isinstance(result, Decision):
                if last_join_press is not None:
                    remaining = last_join_press + join_interval - time.perf_counter()
                    if remaining > 0:
                        control.stop.wait(min(remaining, .02))
                        continue

                def join_once():
                    nonlocal clicks
                    if control.generation() != generation or time.perf_counter() - captured_precise > .5:
                        return None
                    pressed = time.perf_counter()
                    if native is not None:
                        native.quiesce()
                    if device.tap(result.point, control.stop, hold_seconds=.015):
                        clicks += 1
                        flow.completed(result, control.now())
                        return pressed
                    return None

                pressed = control.run_if_active(join_once)
                if pressed:
                    if last_join_press is not None and pressed - last_join_press <= max(1, join_interval * 2):
                        recent_join_intervals.append((pressed - last_join_press) * 1000)
                    last_join_press = pressed
            elif result != 'reopen':
                control.stop.wait(.03)
        if control.stop.is_set():
            status('已停止')
    except Exception as error:
        status('已停止：' + str(error))
    finally:
        try:
            if macro is not None:
                control.set_release_input(None)
                try:
                    macro.close()
                except Exception as error:
                    status('触控进程停止失败：' + str(error))
        finally:
            try:
                if scene:
                    scene.close()
            finally:
                if device:
                    device.close()
                emit('finished', '')


def probe(folder, index, emit):
    device = None
    try:
        vision = Vision()
        device = MuMu(folder, index)
        frame = device.capture()
        screen = vision.detect(frame)
        if screen.network_error:
            page = '游戏网络异常'
        elif screen.login:
            page = '游戏登录页'
        elif screen.home:
            page = '游戏主页'
        elif screen.hall:
            page = '集会所'
        elif screen.board:
            page = '逆袭悬赏列表'
        elif screen.battle_loading or screen.battle_active:
            page = '战斗画面'
        elif screen.ready or screen.prepared:
            page = '队伍房间'
        elif screen.result:
            page = '结算画面'
        else:
            page = '其他或未识别画面'
        emit('probe', {'backend': vision.backend, 'page': page, 'width': frame.shape[1], 'height': frame.shape[0]})
    except Exception as error:
        emit('error', str(error))
    finally:
        if device:
            device.close()
        emit('finished', '')
