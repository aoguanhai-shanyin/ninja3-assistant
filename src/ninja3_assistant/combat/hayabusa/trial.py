"""Manual Hayabusa diagnostics, using the assistant's automatic scene engine."""
from pathlib import Path
import os
import time

from ninja3_assistant.combat.hayabusa.policy import HayabusaPolicy
from ninja3_assistant.combat.hayabusa.scene import HayabusaScene, RECORDINGS
from ninja3_assistant.device.mumu import MuMu, set_window_visibility
from ninja3_assistant.combat.opening_actions import OpeningActions, execute_opening
from ninja3_assistant.device.mumu_macro import GroundMacroGate, create_skate_macro
from ninja3_assistant.device.native_touch import create_native_touch
from ninja3_assistant.runtime.realtime_battle import RealtimeBattleKernel


def run_hayabusa(folder, index, control, emit, duration=600, observe_only=True):
    scene = device = native = macro = gate = None
    kernel = RealtimeBattleKernel(control)
    last_status = None

    def status(message):
        nonlocal last_status
        if message != last_status:
            last_status = message
            emit('status', message)

    try:
        if not 0 < duration <= 3600:
            raise ValueError('场景调试时长须在 0 到 3600 秒之间')
        if not control.wait_active():
            return
        scene = HayabusaScene(clock=control.now, debug=True, record_root=RECORDINGS,
                              policy=HayabusaPolicy())
        opening = None if observe_only else OpeningActions()
        emit('backend', scene.backend)
        device = MuMu(folder, index)
        if isinstance(control.hide_game_window,bool):
            set_window_visibility(folder,index,hidden=control.hide_game_window)
        if opening is not None:
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
        emit('state', 'paused' if control.paused() else 'running')
        status('隼白场景调试：仅观察，不发送动作' if observe_only else '隼白场景调试：执行动作')
        started = interval_start = control.now()
        interval_frames = 0
        generation = control.generation()
        while control.wait_active() and control.now()-started < duration:
            if generation != control.generation():
                if gate is not None:
                    gate.release(reason='generation_changed')
                scene.reset()
                if opening is not None:
                    opening.reset()
                generation = control.generation()
                interval_frames, interval_start = 0, control.now()
            before = time.perf_counter()
            frame = kernel.capture(device)
            capture_seconds = time.perf_counter()-before
            captured = control.now()
            decision = scene.step(frame, captured)
            scene.capture_duration(capture_seconds)
            elapsed = time.perf_counter()-before
            scene.frame_age(elapsed)
            if control.stop.is_set() or control.paused() or control.generation() != generation:
                continue
            interval_frames += 1
            if decision is not None:
                status(decision.status)
                priority_action = bool(native is not None and not observe_only and decision.action is not None and
                                       not decision.stalled and elapsed <= .2)
                if gate is not None and (decision.active or decision.action is not None or
                                          decision.result or decision.stalled or elapsed > .2):
                    if priority_action:
                        gate.forget()
                    else:
                        gate.release(reason='boss_or_scene_action' if decision.active or decision.action is not None else
                                     'scene_result' if decision.result else 'scene_stalled' if decision.stalled else 'stale_frame',
                                     captured_at=before)
                        if native is not None and (decision.action is not None or decision.result or decision.stalled):
                            native.quiesce(reason='scene_action_or_result', captured_at=before)
                if opening is not None and decision.action is None and not decision.result and not decision.stalled and elapsed <= .2:
                    opening_action = opening.decide(frame, captured, boss_active=decision.active,
                                                    battle_allowed=decision.active or not scene.owned)
                    macro_active = False
                    if gate is not None:
                        if native is not None:
                            native.note_frame(before)
                        macro_active = gate.update(allowed=not decision.active and not scene.owned,
                                                   ground_frames=opening.ground_frames, captured_at=before)
                    if opening_action is not None:
                        if opening_action.kind == 'ground_jump_slide' and macro_active:
                            opening.completed(opening_action, control.now())
                            continue
                        def do_opening():
                            if control.generation() != generation or time.perf_counter()-before > .2:
                                return
                            if native is not None and opening_action.kind == 'ninjutsu':
                                if native.tap(opening_action.point, deadline=before+.2):
                                    opening.scheduled(opening_action, control.now())
                            elif execute_opening(opening_action, device, control, emit):
                                opening.completed(opening_action, control.now())
                        control.run_if_active(do_opening)
                if decision.action is not None and elapsed <= .2:
                    def process():
                        if control.generation() != generation or time.perf_counter()-before > .2:
                            return
                        if observe_only:
                            scene.observe(decision.action, control.now())
                        else:
                            input_started = time.perf_counter()
                            try:
                                if native is not None:
                                    ack = native.priority_tap(decision.action.point, captured_at=before,
                                                              deadline=before+.2, hold_seconds=.025)
                                    if ack is not None and control.generation() == generation and not control.stop.is_set():
                                        scene.completed(decision.action, captured+max(0.,ack[0]-(before+capture_seconds)))
                                    else:
                                        scene.failed_touch(RuntimeError('native press expired or cancelled'))
                                else:
                                    device.tap(decision.action.point, control.stop, hold_seconds=.025,
                                        on_pressed=lambda: scene.completed(decision.action, control.now()))
                            finally:
                                scene.input_duration(time.perf_counter()-input_started)
                    try:
                        control.run_if_active(process)
                    except Exception as error:
                        scene.failed_touch(error)
                        raise
                if decision.result:
                    status('隼白场景调试：识别到结算，调试结束')
                    break
            elif gate is not None:
                gate.release(reason='scene_unrecognized', captured_at=before)
            if captured-interval_start >= 1:
                emit('metrics', {'fps':interval_frames/(captured-interval_start),
                    'clicks':scene.clicks,'suggestions':scene.suggestions})
                interval_frames, interval_start = 0, captured
        else:
            status('场景调试已停止' if control.stop.is_set() else '场景调试到达时长上限')
    except Exception as error:
        status('隼白场景调试已停止：'+str(error))
        emit('error', str(error))
    finally:
        if macro is not None:
            control.set_release_input(None)
            try:
                macro.close()
            except Exception as error:
                emit('error', '触控进程停止失败：'+str(error))
        if device is not None:
            device.close()
        if scene is not None:
            scene.close()
            if scene.log is not None and Path(scene.log).is_file():
                emit('status', '隼白场景调试记录：'+str(scene.log))
        emit('finished', '')
