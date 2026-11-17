"""Separate, conservatively gated activity runner using the existing SDK and controls."""
from ninja3_assistant.paths import project_root
import json
from pathlib import Path
import sys
import time
import subprocess

import cv2

from ninja3_assistant.activities.intelligence.flow import ActivityFlow
from ninja3_assistant.activities.intelligence.evidence import UnknownCardRecorder
from ninja3_assistant.activities.intelligence.vision import ActivityVision
from ninja3_assistant.device.mumu import MuMu, prepare_game
from ninja3_assistant.scheduling.task_outcome import TaskOutcome
from ninja3_assistant.scheduling.invitation_handoff import bounty_handoff, clear_handled_invitation


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
        unknown_cards = UnknownCardRecorder(output)
        launched = prepare_game(folder, index, control, status)
        if launched is None:
            return TaskOutcome.STOPPED
        vision, ocr = ActivityVision(), WindowsOcr()
        emit('backend', vision.backend+' / Windows OCR')
        if not control.wait_active():
            return TaskOutcome.STOPPED
        device = MuMu(folder, index)
        emit('state', 'running')
        status('情报模拟战记录：'+str(output))
        generation = control.generation()
        clicks = frames = sequence = reported_matches = 0
        last_page = None
        hall_searches = 0
        metrics_at = control.now()
        while control.wait_active():
            current_generation = control.generation()
            if current_generation != generation:
                flow.checked_day = None
                flow.reset_inputs()
                generation = current_generation
            captured = time.perf_counter()
            frame = device.capture()
            capture_ms = (time.perf_counter()-captured)*1000
            vision_started = time.perf_counter()
            screen = vision.detect(frame)
            if screen.kind == 'battle':
                battle_active = True
                safe_invite = False
            elif screen.kind in ('event_home', 'home', 'hall', 'hall_search', 'tasks', 'login'):
                battle_active = False
                safe_invite = True
                clear_handled_invitation(control)
            elif screen.kind == 'invite' and safe_invite and not battle_active:
                handoff = bounty_handoff(frame,vision.navigation.invitations,ocr,control,safe=True)
                if handoff:
                    return handoff
            vision_ms = (time.perf_counter()-vision_started)*1000
            if screen.kind == 'battle' and screen.battle.kind == 'result':
                result = ocr.read(vision.battle.result_numbers_image(frame), stop=control.stop)
                if result is None:
                    continue
                screen = vision.detect(frame, result)
                refreshed = device.capture()
                check = vision.detect(refreshed)
                if check.kind != 'battle' or check.battle.kind != 'result':
                    continue
                captured = time.perf_counter()
            if screen.kind == 'battle' and screen.battle.kind == 'placement':
                ocr_started = time.perf_counter()
                result = ocr.read(vision.battle.numbers_image(frame), stop=control.stop)
                ocr_ms = (time.perf_counter()-ocr_started)*1000
                if result is None:
                    discard('battle_ocr_cancelled', frame, capture_ms=capture_ms, vision_ms=vision_ms, ocr_ms=ocr_ms)
                    continue
                verify_started = time.perf_counter()
                screen = vision.detect(frame, result)
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
                timing = {'capture_ms': round(capture_ms, 2), 'vision_ms': round(vision_ms, 2),
                          'ocr_ms': round(ocr_ms, 2), 'verify_ms': round(verify_ms, 2)}
                if signature(check) != signature(screen):
                    discard('battle_state_changed', frame, refreshed, **timing)
                    continue
                digit_changes = _coin_digit_changes(frame, refreshed)
                # Actual same-value 3 samples differ by four antialias pixels;
                # actual 3-to-5 samples differ by 43. Unknown/blank is rejected.
                if digit_changes is None or digit_changes > 9:
                    discard('coin_digits_changed', frame, refreshed, digit_changes=digit_changes, **timing)
                    continue
            if screen.kind == 'tasks':
                task = vision.tasks.read(frame, ocr, stop=control.stop)
                screen = vision.task_screen(task)
                # OCR can outlive a page transition; revalidate its anchor and pixels.
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
                captured = time.perf_counter()
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
                from ninja3_assistant.device.native_touch import executable_path
                control.run_if_active(lambda: subprocess.run(
                    [str(executable_path()), '--folder', str(folder), '--index', str(index),
                     '--swipe', '300', '410', '1150', '410'], capture_output=True,
                    timeout=8, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)))
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
            if page != last_page or decision.action != 'wait':
                record = {'time': time.time(), 'page': screen.kind, 'daily': screen.daily,
                          'action': decision.action, 'point': decision.point, 'status': decision.status}
                if decision.selection:
                    record['selection'] = decision.selection
                if screen.task and screen.task.rows:
                    record['daily_progress']=[row.progress for row in screen.task.rows]
                if decision.end_point:
                    record['end_point'] = decision.end_point
                if battle:
                    record['battle'] = {'kind': battle.kind, 'round': battle.round_number, 'coins': battle.coins,
                        'selected_shop': battle.selected_shop,
                        'board': [card.id if card else None for card in battle.board],
                        'shop': [card.id if card else None for card in battle.shop],
                        'unknown_board': battle.unknown_board, 'unknown_shop': battle.unknown_shop}
                    if battle.kind == 'result':
                        record['battle'].update(score=battle.score, score_gain=battle.score_gain)
                log.write(json.dumps(record, ensure_ascii=False)+'\n')
                log.flush()
                cv2.imwrite(str(output/f'{sequence:06d}-{screen.kind}-{decision.action}.png'),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                sequence += 1
                last_page = page
            if decision.action in ('done', 'stop', 'retry'):
                if decision.action == 'done':
                    return TaskOutcome.DAILY_DONE
                return TaskOutcome.RECOVER_IN_PLACE if battle_active else TaskOutcome.ERROR
            if decision.point:
                def act():
                    nonlocal clicks, battle_active
                    if control.generation() != generation or time.perf_counter()-captured > .5:
                        return
                    acted = (device.swipe(decision.point, decision.end_point, control.stop,
                                          steps=12, step_seconds=.025) if decision.end_point else
                             device.tap(decision.point, control.stop))
                    if acted:
                        clicks += 1
                        flow.completed(decision, control.now())
                        if decision.action == 'match':
                            battle_active = True
                control.run_if_active(act)
            if battle and battle.kind == 'placement':
                evidence = unknown_cards.save(frame,battle)
                if evidence:
                    log.write(json.dumps(evidence,ensure_ascii=False)+'\n')
                    log.flush()
            frames += 1
            if now-metrics_at >= 1:
                emit('metrics', {'fps': frames/(now-metrics_at), 'clicks': clicks})
                frames, metrics_at = 0, now
            control.stop.wait(.25 if screen.kind == 'tasks' else .5)
    except Exception as error:
        emit('error', str(error))
        status('情报模拟战等待恢复：'+str(error))
        return TaskOutcome.RECOVER_IN_PLACE if battle_active else TaskOutcome.ERROR
    finally:
        if device:
            device.close()
        if log:
            if discards:
                log.write(json.dumps({'event': 'discard_summary', 'time': time.time(),
                                      'counts': discards}, ensure_ascii=False)+'\n')
            log.close()
        emit('finished', '')
    return TaskOutcome.STOPPED
