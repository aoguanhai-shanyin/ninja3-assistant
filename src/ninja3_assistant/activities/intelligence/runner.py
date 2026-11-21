"""Separate, conservatively gated activity runner using the existing SDK and controls."""
from ninja3_assistant.paths import project_root
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import time
import subprocess

import cv2

from ninja3_assistant.activities.intelligence.flow import ActivityFlow, HALL_SEARCH, daily_key
from ninja3_assistant.activities.intelligence.evidence import UnknownCardRecorder
from ninja3_assistant.activities.intelligence.vision import ActivityVision
from ninja3_assistant.device.mumu import MuMu, prepare_game
from ninja3_assistant.device.native_touch import create_native_touch, executable_path
from ninja3_assistant.game_ui import GameUI, PageAction, PageObservation
from ninja3_assistant.scheduling.task_outcome import TaskOutcome
from ninja3_assistant.scheduling.invitation_handoff import bounty_handoff, clear_handled_invitation
from ninja3_assistant.vision.invitation import SENDER_BOX, TITLE_BOX


PAGE_IDS = {
    'home': 'game.home',
    'hall': 'game.hall',
    'hall_search': 'game.hall',
    'login': 'game.login',
    'network_retry': 'game.network_retry',
    'invite': 'game.invite',
    'event_home': 'intelligence.lobby',
}
OPERATION_NAMES = {
    'home': 'open_hall',
    'hall': 'enter_intelligence',
    'invite': 'decline_invitation',
    'guide': 'continue_guide',
    'tasks': 'open_tasks',
    'claim': 'claim_task',
    'match': 'start_match',
    'place': 'place_card',
    'result_confirm': 'confirm_result',
}


def _observe_action(screen, frame, generation, captured_at, action):
    page = PAGE_IDS.get(screen.kind, f'intelligence.{screen.kind}')
    return PageObservation(page, frame, generation, captured_at,
                           controls={action.name: action}, detail=screen)


def _coin_digit_changes(before, after):
    """Compare only the observed brown coin digits, excluding icon/paper animation.

    This is the fixed digit area in the actual 1280x720 placement samples. The
    OCR crop also contains a moving gold icon; its full-area average could both
    reject that animation and accept a small 3-to-5 digit change.
    """
    masks = [cv2.inRange(cv2.cvtColor(frame[164:188, 1195:1221], cv2.COLOR_RGB2GRAY), 0, 119)
             for frame in (before, after)]
    if any(cv2.countNonZero(mask) < 8 for mask in masks):
        return None
    return cv2.countNonZero(cv2.absdiff(*masks))


def run_activity(folder, index, control, emit, mode='daily', *, checkpoint=False):
    from ninja3_assistant.activities.intelligence.ocr import WindowsOcr

    device = None
    native = None
    ocr = None
    battle_active = False
    safe_invite = False
    last_status = None
    root = Path(sys.executable).parent if getattr(sys, 'frozen', False) else project_root()/'data'/'runtime'
    output = root/'intelligence-simulation'/'runs'/(time.strftime('%Y%m%d-%H%M%S')+f'-{time.time_ns()%1000000:06d}')
    log = None
    discards, discard_written = {}, {}

    def discard(reason, frame, refreshed=None, **details):
        discards[reason] = discards.get(reason, 0)+1
        precise = time.perf_counter()
        if reason in discard_written and precise-discard_written[reason] < 5:
            return
        first = reason not in discard_written
        discard_written[reason] = precise
        record = {'event': 'discard', 'time': time.time(), 'reason': reason,
                  'count': discards[reason], **details}
        log.write(json.dumps(record, ensure_ascii=False)+'\n')
        log.flush()
        # One pair per reason per run is enough to inspect animation and stale
        # state without accumulating screenshots on every rejected frame.
        if first:
            cv2.imwrite(str(output/f'discard-{reason}-before.png'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if refreshed is not None:
                cv2.imwrite(str(output/f'discard-{reason}-after.png'), cv2.cvtColor(refreshed, cv2.COLOR_RGB2BGR))

    def status(message):
        nonlocal last_status
        if message != last_status:
            last_status = message
            emit('status', message)

    try:
        output.mkdir(parents=True, exist_ok=False)
        log = (output/'events.jsonl').open('w', encoding='utf-8')
        flow = ActivityFlow(mode)
        if checkpoint and mode == 'continuous':
            # The plan only schedules this phase after all daily rewards are claimed.
            flow.checked_day = daily_key()
            flow.daily_complete = True
        unknown_cards = UnknownCardRecorder(output)
        vision = ActivityVision(use_winml=True)
        if control.stop.is_set():
            return TaskOutcome.STOPPED
        vision.battle.warmup()
        ocr = WindowsOcr(persistent=True)
        ocr.warmup()
        launched = prepare_game(folder, index, control, status)
        if launched is None:
            return TaskOutcome.STOPPED
        backend = None

        def report_backend():
            nonlocal backend
            if vision.backend != backend:
                backend = vision.backend
                emit('backend', backend+' / Windows OCR')
                warmup_ms = vision.battle.portrait_warmup_ms
                if warmup_ms is not None:
                    log.write(json.dumps({'event': 'portrait_backend', 'time': time.time(),
                                          'backend': backend, 'warmup_ms': round(warmup_ms, 2)},
                                         ensure_ascii=False)+'\n')
                    log.flush()

        report_backend()
        if not control.wait_active():
            return TaskOutcome.STOPPED
        device = MuMu(folder, index)
        native = create_native_touch(folder, index, control.stop, emit)
        if native is None:
            raise RuntimeError('情报模拟战需要 NinjaTouch.exe')
        control.set_release_input(native.stop_all)
        emit('state', 'running')
        status('情报模拟战记录：'+str(output))
        generation = control.generation()
        clicks = frames = sequence = reported_matches = 0
        last_page = None
        hall_searches = 0
        metrics_at = control.now()

        def swipe(start, end):
            # The one-shot swipe owns the input only after native taps are released.
            native.stop_all()
            if control.stop.is_set() or time.perf_counter() >= captured+.5:
                return False
            subprocess.run([str(executable_path()), '--folder', str(folder), '--index', str(index),
                            '--swipe', *map(str, (*start, *end))], capture_output=True,
                           timeout=8, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            return not control.stop.is_set()

        ui = GameUI(control, device.capture, native, swipe=swipe, clock=time.perf_counter)
        while control.wait_active():
            current_generation = control.generation()
            if current_generation != generation:
                flow.checked_day = None
                flow.reset_inputs()
                generation = current_generation
            vision.battle.warmup()
            captured = time.perf_counter()
            frame = device.capture()
            checked_frame = frame
            capture_ms = (time.perf_counter()-captured)*1000
            vision_started = time.perf_counter()
            screen = vision.detect(frame)
            vision_ms = (time.perf_counter()-vision_started)*1000
            report_backend()
            placement_timing = None
            if screen.kind == 'battle':
                battle_active = True
                safe_invite = False
            elif screen.kind in ('event_home', 'home', 'hall', 'hall_search', 'tasks', 'login', 'guide'):
                battle_active = False
                safe_invite = True
                clear_handled_invitation(control)
            elif (screen.kind == 'invite' and safe_invite and not battle_active
                  and getattr(control, 'bounty_invite_targets', None)
                  and not getattr(control, 'bounty_invite_suppressed', False)):
                handoff = bounty_handoff(frame, vision.navigation.invitations, ocr, control, safe=True)
                if handoff:
                    return handoff
                if control.stop.is_set() or control.paused() or control.generation() != generation:
                    continue
                captured = time.perf_counter()
                refreshed = device.capture()
                check = vision.detect(refreshed)
                if check.kind != 'invite' or check.point != screen.point:
                    continue
                # A different invitation can reuse the same buttons. Check the
                # sender and title pixels before declining the OCR result's invite.
                if any(cv2.norm(frame[y1:y2, x1:x2], refreshed[y1:y2, x1:x2], cv2.NORM_INF)
                       for x1, y1, x2, y2 in (SENDER_BOX, TITLE_BOX)):
                    continue
                frame, screen = refreshed, check
                checked_frame = refreshed
            if (screen.kind in ('hall', 'unknown') and flow.last_action == 'hall'
                    and control.now()-flow.last_action_at <= 1.5):
                toast = cv2.resize(frame[225:310, 220:1060], None, fx=2, fy=2)
                notice = ocr.read(toast, stop=control.stop)
                text = ''.join((notice or {}).get('text', '').split())
                if ('服务器维护中' in text and '维护结束后' in text
                        and re.search(r'该[活括]动[暂暫哲]时关闭', text)
                        and not control.stop.is_set() and not control.paused()
                        and control.generation() == generation):
                    cv2.imwrite(str(output/'maintenance.png'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    log.write(json.dumps({'event': 'maintenance', 'time': time.time(), 'text': text},
                                         ensure_ascii=False)+'\n')
                    status('游戏提示情报模拟战因维护暂时关闭')
                    return TaskOutcome.MAINTENANCE
            if screen.kind == 'battle' and screen.battle.kind == 'result':
                result = ocr.read(vision.battle.result_numbers_image(frame), stop=control.stop)
                if result is None:
                    continue
                screen = replace(screen, battle=vision.battle.with_numbers(screen.battle, result))
                captured = time.perf_counter()
                refreshed = device.capture()
                check = vision.detect(refreshed)
                if check.kind != 'battle' or check.battle.kind != 'result':
                    continue
                checked_frame = refreshed
            if screen.kind == 'battle' and screen.battle.kind == 'placement':
                ocr_started = time.perf_counter()
                result = ocr.read(vision.battle.numbers_image(frame), stop=control.stop)
                ocr_ms = (time.perf_counter()-ocr_started)*1000
                if result is None:
                    discard('battle_ocr_cancelled', frame, capture_ms=capture_ms, vision_ms=vision_ms, ocr_ms=ocr_ms)
                    continue
                verify_started = time.perf_counter()
                screen = replace(screen, battle=vision.battle.with_numbers(screen.battle, result))
                captured = time.perf_counter()
                refreshed = device.capture()
                check = vision.detect(refreshed)
                def signature(value):
                    if value.kind != 'battle' or value.battle.kind != 'placement':
                        return None
                    observation = value.battle
                    return (observation.round_number, observation.selected_shop,
                            tuple(card.id if card else None for card in observation.board),
                            tuple(card.id if card else None for card in observation.shop),
                             observation.unknown_board, observation.unknown_shop, observation.place_points)
                verify_ms = (time.perf_counter()-verify_started)*1000
                placement_timing = {'capture_ms': round(capture_ms, 2), 'vision_ms': round(vision_ms, 2),
                                    'ocr_ms': round(ocr_ms, 2), 'verify_ms': round(verify_ms, 2)}
                if signature(check) != signature(screen):
                    discard('battle_state_changed', frame, refreshed, **placement_timing)
                    continue
                digit_changes = _coin_digit_changes(frame, refreshed)
                # Actual same-value 3 samples differ by four antialias pixels;
                # actual 3-to-5 samples differ by 43. Unknown/blank is rejected.
                if digit_changes is None or digit_changes > 9:
                    discard('coin_digits_changed', frame, refreshed, digit_changes=digit_changes, **placement_timing)
                    continue
                checked_frame = refreshed
            if screen.kind == 'tasks':
                task = vision.tasks.read(frame, ocr, stop=control.stop)
                screen = vision.task_screen(task)
                # OCR can outlive a page transition; revalidate its anchor and pixels.
                captured = time.perf_counter()
                refreshed = device.capture()
                current = vision.tasks.detect(refreshed)
                region = vision.tasks.ocr_region(task)
                if current.kind != 'tasks' or current.header_bounds != task.header_bounds or current.present != task.present:
                    continue
                if region:
                    x, y, w, h = region
                    differences = cv2.absdiff(frame[y:y+h, x:x+w], refreshed[y:y+h, x:x+w])
                    if any(differences[top:top+60].mean() > 3 for top in range(0, h, 60)):
                        continue
                checked_frame = refreshed
            if control.paused() or control.stop.is_set() or control.generation() != generation:
                continue
            now = control.now()
            decision = flow.step(screen, now)
            if flow.completed_matches > reported_matches:
                emit('kill', flow.completed_matches-reported_matches)
                reported_matches = flow.completed_matches
                if checkpoint:
                    return TaskOutcome.MATCH_DONE
            if checkpoint and screen.kind == 'hall_search':
                if hall_searches >= 3:
                    status('集会所未找到情报模拟战，继续其他任务')
                    return TaskOutcome.UNAVAILABLE
                action = PageAction('search_intelligence', *HALL_SEARCH, max_age=.5)
                observed = _observe_action(screen, checked_frame, generation, captured, action)
                if ui.perform(observed, action.name, refresh=False):
                    hall_searches += 1
                control.stop.wait(.75)
                continue
            if (checkpoint and decision.action == 'wait' and screen.kind == 'event_home'
                    and flow.checked_day is not None):
                from ninja3_assistant.activities.intelligence.flow import matching_open
                if not matching_open():
                    return TaskOutcome.WAITING
            battle = screen.battle
            message = decision.status
            if screen.task and screen.task.rows:
                pending=[f'{row.progress[0]}/{row.progress[1]}' for row in screen.task.rows
                         if row.state=='incomplete' and row.progress]
                if pending:
                    message+='（日常未完成：'+', '.join(pending)+'）'
            if battle and battle.kind == 'result' and battle.score is not None and battle.score_gain is not None:
                message += f'（本局 +{battle.score_gain}，当前积分 {battle.score}）'
            status(message)
            page = (screen.kind, screen.daily, battle.kind if battle else None,
                    battle.round_number if battle else None)
            record = None
            if page != last_page or decision.action != 'wait' or decision.point:
                record = {'time': time.time(), 'page': screen.kind, 'daily': screen.daily,
                          'action': decision.action, 'point': decision.point, 'status': decision.status}
                if decision.selection:
                    record['selection'] = decision.selection
                if screen.task and screen.task.rows:
                    record['daily_progress']=[row.progress for row in screen.task.rows]
                if decision.end_point:
                    record['end_point'] = decision.end_point
                if placement_timing is not None:
                    record.update(placement_timing)
                if battle:
                    record['battle'] = {'kind': battle.kind, 'round': battle.round_number, 'coins': battle.coins,
                        'selected_shop': battle.selected_shop,
                        'board': [card.id if card else None for card in battle.board],
                        'shop': [card.id if card else None for card in battle.shop],
                        'unknown_board': battle.unknown_board, 'unknown_shop': battle.unknown_shop}
                    if battle.kind == 'result':
                        record['battle'].update(score=battle.score, score_gain=battle.score_gain)
            terminal = decision.action in ('done', 'stop', 'retry')
            try:
                if decision.point and not terminal:
                    name = OPERATION_NAMES.get(decision.action, decision.action)
                    action = PageAction(name, decision.point, decision.end_point,
                                        hold_seconds=.07, max_age=.5)
                    observed = _observe_action(screen, checked_frame, generation, captured, action)

                    def act():
                        nonlocal clicks, battle_active
                        started = time.perf_counter()
                        record['frame_age_ms'] = round((started-captured)*1000, 2)
                        deadline = captured+.5
                        if control.generation() != generation or started >= deadline:
                            return
                        pressed = ui.perform(observed, name, refresh=False)
                        acted = pressed is not None
                        if not decision.end_point:
                            if acted:
                                record['capture_to_press_ms'] = round((pressed[0]-captured)*1000, 2)
                                # The native acknowledgement is sent on press. Allow the
                                # scheduled release before observing the button response.
                                acted = not control.stop.wait(max(0., pressed[1]+.07-time.perf_counter()))
                        record['touch_ms'] = round((time.perf_counter()-started)*1000, 2)
                        record['touch_completed'] = bool(acted)
                        if acted:
                            clicks += 1
                            flow.completed(decision, control.now())
                            if decision.action == 'match':
                                battle_active = True
                    control.run_if_active(act)
            finally:
                if record is not None:
                    log.write(json.dumps(record, ensure_ascii=False)+'\n')
                    log.flush()
                    cv2.imwrite(str(output/f'{sequence:06d}-{screen.kind}-{decision.action}.png'),
                                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    sequence += 1
                    last_page = page
            if terminal:
                if decision.action == 'done':
                    return TaskOutcome.DAILY_DONE
                return TaskOutcome.RECOVER_IN_PLACE if battle_active else TaskOutcome.ERROR
            if battle and battle.kind == 'placement':
                evidence = unknown_cards.save(frame,battle)
                if evidence:
                    log.write(json.dumps(evidence,ensure_ascii=False)+'\n')
                    log.flush()
            frames += 1
            if now-metrics_at >= 1:
                emit('metrics', {'fps': frames/(now-metrics_at), 'clicks': clicks})
                frames, metrics_at = 0, now
            if battle and battle.kind == 'placement':
                control.stop.wait(.2)
            else:
                control.stop.wait(.25 if screen.kind == 'tasks' else .5)
    except Exception as error:
        emit('error', str(error))
        status('情报模拟战等待恢复：'+str(error))
        return TaskOutcome.RECOVER_IN_PLACE if battle_active else TaskOutcome.ERROR
    finally:
        if native is not None:
            control.set_release_input(None)
            try:
                native.close()
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                emit('error', str(error))
        if device:
            device.close()
        if ocr is not None:
            ocr.close()
        if log:
            if discards:
                log.write(json.dumps({'event': 'discard_summary', 'time': time.time(),
                                      'counts': discards}, ensure_ascii=False)+'\n')
            log.close()
        emit('finished', '')
    return TaskOutcome.STOPPED
