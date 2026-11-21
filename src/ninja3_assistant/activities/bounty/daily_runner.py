"""Daily bounty menus and realtime combat share one native input owner."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

import cv2

from ninja3_assistant.activities.bounty.combat import BountyBattleState, BountyCombat, BountyCombatVision
from ninja3_assistant.activities.bounty.daily_flow import BountyProgress, DailyBountyFlow
from ninja3_assistant.activities.bounty.daily_menu import menu_decision
from ninja3_assistant.activities.bounty.daily_vision import DailyBountyScreen, DailyBountyVision
from ninja3_assistant.activities.daily.vision import DailyScreen
from ninja3_assistant.activities.intelligence.ocr import WindowsOcr
from ninja3_assistant.device.mumu import MuMu, prepare_game
from ninja3_assistant.device.native_touch import TimedTap, create_native_touch, executable_path
from ninja3_assistant.game_ui import GameUI, PageAction, PageObservation
from ninja3_assistant.game_ui.pages import page_id
from ninja3_assistant.paths import project_root
from ninja3_assistant.runtime.realtime_battle import FrameDiagnostics, RealtimeBattleKernel
from ninja3_assistant.scheduling.invitation_handoff import bounty_handoff, clear_handled_invitation
from ninja3_assistant.scheduling.task_outcome import TaskOutcome
from ninja3_assistant.vision.game_update import game_update_banner, game_update_point


def _battle_overlay(frame, vision):
    """Only the observed overlays that can cover still-visible battle controls."""
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    if vision.common.invitations.detect(gray):
        return 'invite'
    if vision.tasks_panel(frame):
        return 'bounty_tasks'
    if vision.order_result_confirm(frame) is not None or vision.defeat_confirm(frame) is not None:
        return 'result'
    for name,page,box in (
        ('result-title','result',(540,40,820,135)),
        ('grade-title','bounty_grade',(517,172,756,221)),
        ('free-title','bounty_free',(546,176,731,224)),
        ('own-empty','bounty_own_empty',(493,294,796,351)),
        ('order-locked','bounty_locked',(442,332,870,393)),
        ('room-footer','bounty_room',(1045,585,1200,627))):
        if vision.match(frame,name,box):
            return page
    if (vision.match(frame,'leave-title',(540,348,729,397)) and
            vision.match(frame,'leave-room-context',(1045,585,1200,627))):
        return 'bounty_leave'
    if (vision.match(frame,'reward-blank-title',(569,124,716,181),.95) and
            vision.match(frame,'reward-blank-close',(530,621,742,672),.95)):
        return 'reward'
    if (vision.common.match(frame,'reward',(540,45,750,145)) and
            vision.common.match(frame,'confirm',(200,550,900,715))):
        return 'reward'
    return None


def _recognize_frame(frame, vision, battle_vision):
    state = battle_vision.detect(frame)
    if state.active:
        page = _battle_overlay(frame,vision)
    else:
        page,_ = vision.navigation(frame)
    if page is not None:
        battle_vision.reset()
        return page,BountyBattleState()
    return None,state


class _InvitationRun:
    """Own an invited match only after a room, battle or result is visible."""

    def __init__(self, resume):
        self.phase = 'owned' if resume else 'offered'
        self.sent_at = None

    @property
    def owns_match(self):
        return self.phase in ('owned','settling')

    def sent(self, now):
        self.phase, self.sent_at = 'joining',now

    def saw_match(self):
        if self.phase != 'settling':
            self.phase = 'owned'

    def checkpoint(self, page, now):
        if page == 'intelligence_tasks':
            if self.phase == 'joining' and now-self.sent_at < 5:
                return None
            self.phase = 'finished'
            return TaskOutcome.SCENE_CHANGED
        if self.phase == 'offered':
            return None if page == 'invite' else TaskOutcome.WAITING
        if page in ('bounty_room','result'):
            self.saw_match()
        if self.phase == 'joining':
            return TaskOutcome.WAITING if now-self.sent_at >= 5 else None
        if page in ('bounty_daily','bounty_order','bounty_hub','home','hall'):
            outcome = TaskOutcome.MATCH_DONE if self.phase == 'settling' else TaskOutcome.WAITING
            self.phase = 'finished'
            return outcome
        return None


class _OrderJoinBurst:
    """Repeat one freshly verified S card through the shared native input owner."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.point = None
        self.next_at = 0.

    def stop(self):
        if self.point is not None:
            self.kernel.cancel_plan(3)
            self.point = None
        self.next_at = 0.

    def submit(self, point):
        now = self.kernel.clock()
        if self.point != point:
            self.stop()
        captured_at = self.kernel.captured_at
        age = now-captured_at
        if not 0 <= age < .075:
            self.stop()
            return None
        if now < self.next_at:
            return None
        first = math.ceil(age*1000)+2
        deadline = min(captured_at+.14,now+.08)
        actions = tuple(TimedTap(offset,point) for offset in range(first,first+61,30)
                        if captured_at+(offset+15)/1000 < deadline)
        revision = self.kernel.plan(3,actions,deadline=deadline)
        if revision is not None:
            self.point = point
            self.next_at = captured_at+(actions[-1].offset_ms+30)/1000
        return revision


def run_daily_bounty(folder, index, control, emit, *, grade='D', active_invite=True, passive_invite=True,
                     invite_only=False, invite_resume=False, recover_in_place=False, progress=None):
    if grade not in ('D','C','B','A','S'):
        raise ValueError('expected bounty grade D/C/B/A/S')
    root = Path(sys.executable).parent if getattr(sys,'frozen',False) else project_root()/'data/runtime'
    output = root/'daily-cleanup/runs'/(time.strftime('%Y%m%d-%H%M%S')+
             f'-{time.time_ns()%1000000:06d}-bounty-{grade}')
    output.mkdir(parents=True,exist_ok=False)
    device = native = combat = diagnostics = ocr = burst = None
    progress = progress if progress is not None else BountyProgress()
    in_battle = recover_in_place
    outcome = TaskOutcome.ERROR
    previous_status = None
    invitation = _InvitationRun(invite_resume) if invite_only else None
    lock = threading.Lock()
    with (output/'events.jsonl').open('w',encoding='utf-8') as log:
        def record(value):
            with lock:
                log.write(json.dumps({'time':time.time(),**value},ensure_ascii=False)+'\n')
                log.flush()

        def status(message):
            nonlocal previous_status
            if message != previous_status:
                previous_status = message
                emit('status',message)

        try:
            ocr = WindowsOcr(persistent=True)
            ocr.warmup()
            if prepare_game(folder,index,control,status) is None:
                return TaskOutcome.STOPPED
            flow = DailyBountyFlow(kind='bounty_s' if grade == 'S' else 'bounty',
                                   grade='D' if grade == 'S' else grade,
                                   active_invite=active_invite,passive_invite=passive_invite,
                                   clock=control.now,progress=progress)
            flow.battle_seen = recover_in_place
            vision = DailyBountyVision(grade=grade)
            battle_vision = BountyCombatVision()
            kernel = RealtimeBattleKernel(control,movement_horizon=3.)
            device = MuMu(folder,index)
            native = create_native_touch(folder,index,control.stop,
                lambda event,value:record({'event':event,'value':value}))
            if native is None:
                raise RuntimeError('每日悬赏需要 NinjaTouch.exe')
            kernel.attach(native)

            def swipe(start, end):
                native.stop_all()
                subprocess.run([str(executable_path()),'--folder',str(folder),'--index',str(index),
                    '--swipe',*map(str,(*start,*end))],capture_output=True,check=True,timeout=8,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                return True

            ui = GameUI(control,lambda:kernel.capture(device),kernel,swipe=swipe)
            combat = BountyCombat(kernel)
            burst = _OrderJoinBurst(kernel) if grade == 'S' and not invite_only else None
            control.set_release_input(native.stop_all)
            diagnostics = FrameDiagnostics()
            record({'event':'start','grade':grade,
                    'active_invite':active_invite,'passive_invite':passive_invite,
                    'progress':progress.tracking(control.now()),
                    'timing_scope':'input_ms measures dispatch, not visible game response'})
            emit('state','running')
            status('每日悬赏记录：'+str(output))
            generation = control.generation()
            started = last_progress = control.now()
            next_menu = 0.
            inactive_since = None
            previous_menu = previous_battle = None
            order_poll_at = None
            order_poll_remaining = None
            order_reference = None
            invite_board = 'bounty_order' if grade == 'S' else 'bounty_daily'
            safe_invite_board = False
            sequence = 0
            metrics_at = time.perf_counter()
            capture_count = recognition_count = input_count = 0
            capture_ms = recognition_ms = input_ms = menu_ocr_ms = 0.
            while control.wait_active() and (grade == 'S' or control.now()-started < 1800):
                loop_at = time.perf_counter()
                if generation != control.generation():
                    combat.stop()
                    if burst:
                        burst.stop()
                    order_reference = None
                    safe_invite_board = False
                    battle_vision.reset()
                    flow.invalidate_tasks()
                    generation = control.generation()
                    next_menu = 0.
                    previous_battle = None
                    inactive_since = None
                frame = kernel.capture(device)
                capture_count += 1
                capture_ms += kernel.capture_ms
                observed_generation = generation
                observed_at = kernel.captured_at
                if game_update_banner(frame):
                    combat.stop()
                    if burst:
                        burst.stop()
                    order_reference = None
                    safe_invite_board = False
                    result = ocr.read(frame,stop=control.stop)
                    if (result is not None and generation == control.generation() and not control.stop.is_set()
                            and game_update_point(DailyScreen(frame,result))):
                        status('识别到游戏更新提示，交回调度重启客户端')
                        record({'event':'checkpoint','page':'game_update','outcome':'scene_changed'})
                        outcome = TaskOutcome.SCENE_CHANGED
                        break
                    control.stop.wait(.5)
                    continue
                before = time.perf_counter()
                # Active battle uses only the bounded overlay probes. Full menu
                # navigation is deliberately absent from the per-frame fast path.
                page,state = _recognize_frame(frame,vision,battle_vision)
                if page not in (invite_board,'invite'):
                    safe_invite_board = False
                rapid = None
                if burst and order_reference is not None:
                    if page == 'bounty_order' and flow.day_clock() == progress.day:
                        rapid = vision.fast_order(frame,order_reference)
                    if rapid is None:
                        order_reference = None
                        next_menu = 0.
                    elif not (rapid.join or rapid.join_failed or rapid.join_pending or
                              flow.order_join_failed or flow.no_owned_orders):
                        rapid = None
                recognition_count += 1
                recognition_ms += (time.perf_counter()-before)*1000
                if control.stop.is_set() or control.paused() or observed_generation != control.generation():
                    continue
                if burst and (rapid is None or not rapid.join or flow.order_join_failed):
                    burst.stop()
                now = control.now()
                if state.active:
                    inactive_since = None
                    in_battle = flow.battle_seen = True
                    if invitation:
                        invitation.saw_match()
                    last_progress = now
                    before = time.perf_counter()
                    message = combat.update(state,now)
                    input_count += 1
                    input_ms += (time.perf_counter()-before)*1000
                    status(message)
                    battle = (state.ground,state.on_rope,state.boss_active)
                    if battle != previous_battle:
                        image = f'{sequence:04d}-battle.png'
                        diagnostics.save(output/image,frame)
                        record({'event':'battle','state':asdict(state),'frame':image})
                        sequence += 1
                        previous_battle = battle
                else:
                    if inactive_since is None:
                        inactive_since = now
                    # Only recognized menus/results release ownership. An
                    # obscured HUD or unknown transition keeps the current match.
                    if page is not None and page != 'invite':
                        in_battle = False
                    unknown_grace = in_battle and page is None and now-inactive_since < .5
                    if unknown_grace:
                        # A pressed control or brief effect can hide the HUD.
                        # Keep only the already bounded plan; do not renew or
                        # block capture with OCR during this short transition.
                        status('悬赏战斗画面暂未识别，保留当前对局')
                    else:
                        combat.stop()
                    if not unknown_grace and (rapid is not None or time.perf_counter() >= next_menu):
                        if page == 'hall':
                            result = {}
                            screen = DailyBountyScreen(frame,result,'hall',entry=vision.hall_entry(frame))
                            next_menu = time.perf_counter()+.12
                        elif rapid is None:
                            before = time.perf_counter()
                            result = ocr.read(frame,stop=control.stop)
                            menu_ocr_ms += (time.perf_counter()-before)*1000
                            next_menu = time.perf_counter()+.5
                            if result is None or control.stop.is_set() or control.paused() or observed_generation != control.generation():
                                continue
                            screen = vision.observe(frame,result)
                        else:
                            screen,result = rapid,{}
                        if not invite_only and passive_invite and getattr(control,'bounty_invite_targets',None):
                            if screen.page != 'invite':
                                safe_invite_board = page == screen.page == invite_board and not screen.join_pending
                                if safe_invite_board:
                                    clear_handled_invitation(control)
                            elif (safe_invite_board and not (flow.in_room or in_battle or flow.battle_seen) and
                                  flow.order_join_pending_at is None and screen.invitation and
                                  screen.invitation.grade != grade):
                                handoff = bounty_handoff(frame,vision.common.invitations,ocr,control,
                                                         safe=True,result=result)
                                if handoff:
                                    record({'event':'invitation_handoff','target':handoff.target})
                                    outcome = handoff
                                    break
                        checkpoint = invitation.checkpoint(screen.page,now) if invitation else (
                            TaskOutcome.SCENE_CHANGED if screen.page == 'intelligence_tasks' else None)
                        if checkpoint is not None:
                            in_battle = False
                            if checkpoint == TaskOutcome.SCENE_CHANGED:
                                status('识别到情报模拟战任务面板，结束当前悬赏流程并继续调度')
                            record({'event':'checkpoint','page':screen.page,'outcome':checkpoint.value})
                            outcome = checkpoint
                            break
                        if invitation and invitation.phase == 'joining':
                            control.stop.wait(.15)
                            continue
                        if screen.page == 'bounty_order' and screen.remaining is None:
                            # Limit the fallback to the unlocked order page and
                            # the S field; SS+ starts to the right of this crop.
                            counter_frame = cv2.copyMakeBorder(
                                cv2.resize(frame[150:194,105:390],None,fx=3,fy=3),
                                40,40,40,40,cv2.BORDER_CONSTANT,value=(0,0,0))
                            before = time.perf_counter()
                            counter = ocr.read(counter_frame,stop=control.stop)
                            menu_ocr_ms += (time.perf_counter()-before)*1000
                            next_menu = time.perf_counter()+.5
                            if counter is None or control.stop.is_set() or control.paused() or observed_generation != control.generation():
                                continue
                            result['bounty_order_counter'] = counter
                            screen = vision.observe(frame,result)
                        if screen.page in ('result','bounty_daily','bounty_order','bounty_hub','home','hall'):
                            in_battle = False
                        if (invite_only and screen.page == 'bounty_room' and
                                screen.grade is not None and screen.grade != grade):
                            decision = menu_decision(screen,'leave_wrong_grade','悬赏队伍等级不符，离开后继续原任务')
                        else:
                            decision = flow.step(screen,vision)
                        if (burst and rapid is None and page == screen.page == 'bounty_order' and
                                isinstance(screen.remaining,int) and screen.remaining > 0 and
                                (screen.s_filter,screen.ss_filter) == ('on','off') and flow.tasks_checked and
                                decision.action != 'retry'):
                            order_reference = (screen.remaining,vision.order_counter(frame).copy())
                        accepts_invite = (screen.page == 'invite' and screen.invitation is not None and
                                          decision.operation == 'accept_invite' and decision.point == screen.invitation.accept)
                        if invitation and invitation.phase == 'offered' and not accepts_invite:
                            outcome = TaskOutcome.WAITING
                            break
                        remaining = screen.remaining if isinstance(screen.remaining,int) else None
                        observed_grade = screen.grade if isinstance(screen.grade,str) else None
                        tracking = progress.tracking(now)
                        tracking['tasks_reused'] = flow.tasks_cached is True
                        menu = (screen.page,decision.action,decision.status,decision.point,remaining,observed_grade,tracking)
                        order_poll = screen.page == 'bounty_order' and decision.operation in ('refresh','poll_orders')
                        order_burst = burst and decision.operation == 'join_order'
                        if burst and not order_burst:
                            burst.stop()
                        if order_poll:
                            save_frame = (order_poll_at is None or now-order_poll_at >= 15 or
                                          order_poll_remaining != remaining)
                            if save_frame:
                                order_poll_at,order_poll_remaining = now,remaining
                        else:
                            order_poll_at = None
                            save_frame = menu != previous_menu or (decision.point is not None and not order_burst)
                        if save_frame:
                            image = f'{sequence:04d}-{screen.page}-{decision.action}.png'
                            diagnostics.save(output/image,frame)
                            record({'event':'decision','page':screen.page,'action':decision.action,
                                    'operation':decision.operation,
                                    'point':decision.point,'status':decision.status,'free_kind':decision.free_kind,
                                    'remaining':remaining,'observed_grade':observed_grade,'progress':tracking,
                                    'guard':decision.guard,'frame':image,'ocr':result,
                                    'join_failed':bool(screen.join_failed),'fast_order':rapid is not None})
                            sequence += 1
                        if menu != previous_menu:
                            last_progress = now
                        previous_menu = menu
                        status('持续刷新，等待S级队伍' if order_poll else decision.status)
                        if decision.action in ('done','unavailable'):
                            outcome = (TaskOutcome.DAILY_DONE if decision.action == 'done' else
                                       TaskOutcome.RECOVER_IN_PLACE if in_battle or
                                       (invitation and invitation.owns_match) else TaskOutcome.UNAVAILABLE)
                            break
                        if decision.action == 'retry':
                            outcome = TaskOutcome.WAITING
                            break
                        if order_burst:
                            if rapid is not None:
                                before = time.perf_counter()
                                first = burst.point != decision.point
                                revision = burst.submit(decision.point)
                                input_count += 1
                                input_ms += (time.perf_counter()-before)*1000
                                if revision is not None:
                                    last_progress = control.now()
                                    if first:
                                        flow.completed(decision,screen)
                                        record({'event':'order_join_burst','point':decision.point,
                                                'revision':revision,'remaining':remaining,
                                                'frame_age_ms':(before-kernel.captured_at)*1000})
                        elif decision.point is not None:
                            def verify(fresh):
                                valid = vision.current_action(screen,decision,fresh)
                                if not valid:
                                    record({'event':'input_rejected','reason':'button_changed_or_not_free'})
                                return valid

                            action = PageAction(decision.operation,decision.point,decision.end,guard=verify,hold_seconds=.07)
                            observation = PageObservation(page_id('bounty',screen.page),frame,observed_generation,observed_at,
                                                          {action.name:action},detail=screen)

                            def act():
                                acted = ui.perform(observation,action.name)
                                if acted and not decision.end:
                                    control.stop.wait(.09)
                                if acted:
                                    flow.completed(decision,screen)
                                    if invitation and accepts_invite:
                                        invitation.sent(control.now())
                                    if invitation and decision.operation == 'confirm_settlement':
                                        invitation.phase = 'settling'
                                    record({'event':'input_sent','point':decision.point,'free_kind':decision.free_kind,
                                            'status':decision.status})
                                return bool(acted)
                            before = time.perf_counter()
                            if control.run_if_active(act):
                                last_progress = control.now()
                            input_count += 1
                            input_ms += (time.perf_counter()-before)*1000
                elapsed = time.perf_counter()-metrics_at
                if elapsed >= 1:
                    metrics = {'fps':recognition_count/elapsed,'capture_fps':capture_count/elapsed,
                               'recognition_fps':recognition_count/elapsed,
                               'capture_ms':capture_ms/max(1,capture_count),
                               'recognition_ms':recognition_ms/max(1,recognition_count),
                               'input_ms':input_ms/max(1,input_count),'input_samples':input_count,
                               'menu_ocr_ms':menu_ocr_ms,'stale_drops':kernel.stale_drops}
                    emit('metrics',metrics)
                    record({'event':'metrics',**metrics})
                    metrics_at = time.perf_counter()
                    capture_count = recognition_count = input_count = 0
                    capture_ms = recognition_ms = input_ms = menu_ocr_ms = 0.
                if now-last_progress >= (30 if in_battle else 90):
                    status('悬赏画面未恢复，保留未完成状态并稍后重试')
                    outcome = TaskOutcome.RECOVER_IN_PLACE if in_battle or (invitation and invitation.owns_match) else TaskOutcome.ERROR
                    break
                control.stop.wait(max(0.,1/60-(time.perf_counter()-loop_at)))
            else:
                outcome = (TaskOutcome.STOPPED if control.stop.is_set() else
                            TaskOutcome.RECOVER_IN_PLACE if in_battle or (invitation and invitation.owns_match) else TaskOutcome.UNAVAILABLE)
        except Exception as error:
            record({'event':'error','message':str(error),'in_battle':in_battle})
            emit('error','每日悬赏：'+str(error))
            outcome = TaskOutcome.STOPPED if control.stop.is_set() else (
                TaskOutcome.RECOVER_IN_PLACE if in_battle or (invitation and invitation.owns_match) else TaskOutcome.ERROR)
        finally:
            control.set_release_input(None)
            for resource in (burst,combat,native,device,diagnostics,ocr):
                if resource is None:
                    continue
                try:
                    resource.stop() if resource in (burst,combat) else resource.close()
                except Exception as error:
                    record({'event':'cleanup_error','message':str(error)})
                    emit('error','每日悬赏释放资源：'+str(error))
                    if not control.stop.is_set():
                        outcome = TaskOutcome.RECOVER_IN_PLACE if in_battle else TaskOutcome.ERROR
            emit('finished','')
    return outcome
