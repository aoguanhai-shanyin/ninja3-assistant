"""Xumi trials with the shared capture, input and combat kernel."""
from contextlib import ExitStack
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time

import cv2

from ninja3_assistant.activities.daily.flow import DailyDecision
from ninja3_assistant.activities.daily.vision import DailyScreen
from ninja3_assistant.activities.intelligence.ocr import WindowsOcr
from ninja3_assistant.activities.xumi.combat import XumiCombat, XumiCombatVision
from ninja3_assistant.activities.xumi.tasks import read_task_panel, tasks_claimed
from ninja3_assistant.activities.xumi.vision import TITLE, XumiVision
from ninja3_assistant.device.mumu import MuMu, prepare_game
from ninja3_assistant.device.native_touch import create_native_touch, executable_path
from ninja3_assistant.paths import project_root
from ninja3_assistant.game_ui import GameUI
from ninja3_assistant.runtime.realtime_battle import FrameDiagnostics, RealtimeBattleKernel
from ninja3_assistant.scheduling.invitation_handoff import bounty_handoff, clear_handled_invitation
from ninja3_assistant.scheduling.task_outcome import TaskOutcome
from ninja3_assistant.vision.game_update import game_update_banner


TRIALS = {'fangcun': '方寸试炼', 'daqian': '大千试炼'}
TEAM_PAGES = ('team', 'roster', 'own_roster')


def tap(operation, point, status, guard=None):
    return DailyDecision.tap(operation, point, status, guard=guard) if point else DailyDecision(status=status)


@dataclass
class XumiProgress:
    last_mode: str | None = None
    settled_mode: str | None = None
    scores: dict[str, int] = field(default_factory=dict)
    no_progress: dict[str, int] = field(default_factory=lambda: dict.fromkeys(TRIALS,0))
    blocked_modes: dict[str, str] = field(default_factory=dict)


class XumiFlow:
    def __init__(self, progress=None, *, trial=None, difficulty=None):
        self.mode = self.trial = trial
        self.difficulty = difficulty
        self.progress = progress if progress is not None else XumiProgress()
        self.hall_swipes = 0

    def skip_trial(self, scene, reason):
        label = TRIALS[self.mode]+'：'+reason
        self.progress.blocked_modes[self.mode] = label
        self.mode = None
        if self.trial is not None or scene.back is None:
            return DailyDecision('defer', status=label)
        return tap('back', scene.back, label)

    def step(self, frame, scene, rows=()):
        kind = scene.kind
        if kind == 'score_confirm':
            return DailyDecision('unavailable', status='确认编队变更会重置积分，保留当前积分并交回调度')
        if kind in TEAM_PAGES or kind.startswith('equipment'):
            return DailyDecision('defer', status='当前处于手动编队或装备界面，跳过须弥幻境')
        if kind == 'home':
            return tap('open_hall', scene.point, '进入集会所寻找须弥幻境')
        if kind == 'hall':
            if scene.point:
                return tap('open_xumi', scene.point, '进入须弥幻境', (scene.point[0]-100,130,scene.point[0]+100,195))
            if self.hall_swipes >= 6:
                return DailyDecision('unavailable', status='集会所未确认须弥幻境入口')
            points = ((1080,400),(280,400)) if self.hall_swipes < 3 else ((280,400),(1080,400))
            return DailyDecision.swipe('find_xumi',points[0],'在集会所寻找须弥幻境',points[1])
        if kind == 'guide':
            return tap('advance_guide', scene.point, '查看须弥幻境说明')
        if kind == 'lobby':
            if self.mode is None:
                return tap('open_tasks', scene.point, '打开须弥幻境每日任务')
            return tap('open_'+self.mode, getattr(scene, self.mode), '进入'+TRIALS[self.mode], TITLE)
        if kind == 'tasks':
            if tasks_claimed(rows):
                return DailyDecision('done', status='须弥幻境方寸和大千每日奖励均已领取')
            pending = next((row for row in rows if row.claim_point is not None), None)
            if pending is not None:
                x,y = pending.claim_point
                return tap('claim_task_reward', pending.claim_point, '领取'+TRIALS[pending.mode]+'每日奖励',
                           (x-48,y-22,x+48,y+22))
            if len(rows) != 4 or any(not row.claimed and (row.current is None or row.current >= row.goal)
                                     for row in rows):
                return DailyDecision(status='等待确认须弥幻境四项每日任务进度')
            progress = self.progress
            scores = {row.mode: row.goal if row.claimed else row.current for row in rows if row.kind == 'score'}
            settled = progress.settled_mode
            if settled is not None:
                progress.no_progress[settled] = (progress.no_progress[settled]+1
                                                 if scores[settled] == progress.scores.get(settled) else 0)
                progress.settled_mode = None
            progress.scores = scores
            score_order = sorted(TRIALS,key=lambda mode: (mode == progress.last_mode,scores[mode]))
            mode = next((mode for kind,order in (('match',TRIALS),('score',score_order)) for mode in order if any(
                row.mode == mode and row.kind == kind and not row.claimed and row.current < row.goal
                for row in rows) and mode not in progress.blocked_modes and progress.no_progress[mode] < 2), None)
            if mode is None:
                blocked = [progress.blocked_modes[mode] for mode in TRIALS if mode in progress.blocked_modes and any(
                    row.mode == mode and not row.claimed and row.current < row.goal for row in rows)]
                stalled = [TRIALS[mode] for mode in TRIALS if progress.no_progress[mode] >= 2 and any(
                    row.mode == mode and not row.claimed and row.current < row.goal for row in rows)]
                if stalled:
                    blocked.append('、'.join(stalled)+'连续两轮未增加每日积分，保留未完成进度')
                if blocked:
                    return DailyDecision('defer', status='；'.join(blocked))
                return DailyDecision(status='等待确认须弥幻境每日奖励领取状态')
            if settled is not None:
                return DailyDecision('checkpoint', status='须弥幻境整轮已结算，日常进度已复查，继续调度')
            self.mode = mode
            return tap('back', scene.back, '继续'+TRIALS[mode]+'未完成日常')
        if kind == 'result_next' and scene.point is not None:
            return DailyDecision('continue', scene.point, '继续第二场须弥幻境试炼',
                                 guard=(870,610,1075,652), operation='continue_trial')
        if kind == 'result_done' and scene.point is not None:
            return DailyDecision('settle', scene.point, '完成整轮试炼，返回核对每日进度',
                                 guard=(778,611,1000,652), operation='settle_trial')
        if kind.endswith('_setup'):
            if self.mode is None or kind != self.mode+'_setup':
                return tap('back', scene.back, '返回须弥幻境大厅检查每日进度')
            if scene.notice == 'missing_weapon':
                return self.skip_trial(scene, '队伍缺少武器，请手动装备，跳过本轮')
            setup = scene.setup
            if setup is not None and 'empty' in setup.leads:
                return self.skip_trial(scene, '请先手动配好两支队伍，跳过本轮')
            if setup is None or setup.leads != ('filled','filled'):
                return DailyDecision(status='等待确认两支手动配置的队伍')
            current = setup.selected_difficulty
            if current is not None:
                if self.difficulty is not None and current != self.difficulty:
                    return self.skip_trial(scene, f'当前开放第{current}重，无法进入第{self.difficulty}重')
                if self.trial is None and current == 3:
                    return self.skip_trial(scene, '已通过前两重，当前为第三重')
                if setup.can_start:
                    return DailyDecision('start', setup.start, f'开始{TRIALS[self.mode]}第{current}重',
                                         operation='start_trial', value=current)
            return DailyDecision(status='等待确认试炼难度与开始按钮')
        return DailyDecision(status='等待识别须弥幻境画面')

    def completed(self, decision):
        if decision.action == 'swipe':
            self.hall_swipes += 1
        if decision.action == 'settle':
            self.progress.last_mode = self.progress.settled_mode = self.mode
            self.mode = None


def run_xumi(folder, index, control, emit, *, progress, trial=None, difficulty=None):
    if trial is not None and trial not in TRIALS:
        raise ValueError('须弥试炼须指定方寸或大千')
    if difficulty is not None and (trial is None or difficulty not in (1,2,3)):
        raise ValueError('须弥实测须指定试炼及一至三重难度')
    device = native = combat = ocr = diagnostics = None
    root = Path(sys.executable).parent if getattr(sys, 'frozen', False) else project_root()/'data/runtime'
    output = root/'xumi'/('runs')/(time.strftime('%Y%m%d-%H%M%S')+f'-{time.time_ns()%1000000:06d}')
    output.mkdir(parents=True, exist_ok=False)
    last_status = None

    def status(message):
        nonlocal last_status
        if message != last_status:
            last_status = message
            emit('status', message)

    with (output/'events.jsonl').open('w', encoding='utf-8') as log:
        log_lock = Lock()

        def record(kind, **fields):
            with log_lock:
                log.write(json.dumps({'event':kind, 'time':time.time(), **fields}, ensure_ascii=False)+'\n')
                log.flush()

        def record_input(kind, fields):
            if (kind == 'native_plan' and fields['finger'] in (3,4,5) and
                    fields['event'] in ('pressed','expired_before_send','dropped')):
                details = dict(fields)
                stage = details.pop('event')
                record('movement_input' if fields['finger'] == 5 else 'skill_input', stage=stage, **details)

        try:
            if prepare_game(folder, index, control, status) is None:
                return TaskOutcome.STOPPED
            vision = XumiVision()
            battle_vision = XumiCombatVision()
            flow = XumiFlow(progress, trial=trial, difficulty=difficulty)
            ocr = WindowsOcr(persistent=True)
            ocr.warmup()
            device = MuMu(folder, index)
            native = create_native_touch(folder, index, control.stop, record_input)
            if native is None:
                raise RuntimeError('须弥幻境需要 NinjaTouch.exe')
            kernel = RealtimeBattleKernel(control)
            kernel.attach(native)
            combat = XumiCombat(kernel,battle_vision)
            diagnostics = FrameDiagnostics()

            def swipe(point, end):
                native.stop_all()
                if control.stop.is_set():
                    return False
                subprocess.run([str(executable_path()),'--folder',str(folder),'--index',str(index),
                    '--swipe',*map(str,(*point,*end))],capture_output=True,check=True,timeout=8,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                return True

            ui = GameUI(control,lambda:kernel.capture(device),native,swipe=swipe,clock=kernel.clock)
            control.set_release_input(native.stop_all)
            emit('state','running')
            emit('status','须弥幻境记录：'+str(output))
            sequence = 0
            battle_start = None
            battle_seen = False
            boss_number = None
            current_difficulty = None
            boss_entered_at = None
            seen_bosses = set()
            relay_bosses = set()
            failed_starts = 0
            next_snapshot = 0.
            generation = control.generation()
            last_progress = control.now()
            finished = False
            last_frame = None
            last_scores = None
            safe_invite = False
            previous_skill_points = None
            while control.wait_active():
                if generation != control.generation():
                    generation = control.generation()
                    combat.stop()
                    battle_vision.reset()
                    last_progress = control.now()
                frame = kernel.capture(device)
                if control.stop.is_set() or control.generation() != generation:
                    continue
                now = control.now()
                if game_update_banner(frame):
                    status('须弥幻境遇到客户端更新，交回调度重启')
                    return TaskOutcome.SCENE_CHANGED
                if battle_start is not None and not vision.relay(frame):
                    state = battle_vision.detect(frame)
                    if state.active:
                        safe_invite = False
                        if not battle_seen:
                            last_frame = None
                            previous_skill_points = None
                            failed_starts = 0
                        if state.boss_active and boss_entered_at is None:
                            boss_entered_at = now
                            seen_bosses.add(boss_number)
                            next_snapshot = 0.
                            record('battle_start', mode=flow.mode, difficulty=current_difficulty, boss=boss_number)
                        skill_points = (state.ninjutsu_point,state.weapon_points)
                        if skill_points != previous_skill_points:
                            record('battle_controls',mode=flow.mode,ninjutsu=state.ninjutsu_point,
                                   weapons=state.weapon_points)
                            previous_skill_points = skill_points
                        boss_elapsed = None if boss_entered_at is None else now-boss_entered_at
                        if now >= next_snapshot:
                            diagnostics.save(output/f'battle-{sequence:04d}.png',frame)
                            sequence += 1
                            capture_detail = (trial is not None and boss_elapsed is not None
                                              and (boss_elapsed < 15 or 27 < boss_elapsed < 40))
                            next_snapshot = now+(.25 if capture_detail else 3.)
                        battle_seen = True
                        message, dodge = combat.update(frame,state,now,boss_elapsed)
                        if dodge is not None:
                            action, point, taps = dodge
                            filename = f'{sequence:04d}-boss-{boss_number}-dodge-{action}.png'
                            diagnostics.save(output/filename,frame)
                            sequence += 1
                            record('dodge', mode=flow.mode, difficulty=current_difficulty,
                                   boss=boss_number, action=action, point=point, frame=filename,
                                   taps=[(item.offset_ms,item.point) for item in taps],
                                   capture_ms=round(kernel.capture_ms,2), frame_age_ms=kernel.metrics()['frame_age_ms'])
                        status('须弥幻境 · '+TRIALS.get(flow.mode,'试炼')+' · '+message)
                        last_progress = now
                        control.stop.wait(.012)
                        continue
                combat.cancel_board()
                scene = vision.scene(frame)
                if battle_start is not None:
                    if scene.kind == 'relay':
                        combat.stop()
                        if boss_number not in relay_bosses:
                            relay_bosses.add(boss_number)
                            record('relay', mode=flow.mode, difficulty=current_difficulty, boss=boss_number)
                            diagnostics.save(output/f'{sequence:04d}-boss-{boss_number}-relay.png', frame)
                            sequence += 1
                        status('队伍已倒下，等待接力倒计时结束')
                        control.stop.wait(.15)
                        continue
                    if scene.kind == 'unknown' and battle_seen and now-last_progress < .5:
                        # Keep the existing bounded plan through brief effects.
                        control.stop.wait(.03)
                        continue
                    combat.stop()
                invitation = vision.daily.invitations.detect(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY))
                if invitation:
                    handoff = bounty_handoff(frame,vision.daily.invitations,ocr,control,safe=safe_invite)
                    if handoff:
                        return handoff
                else:
                    safe_invite = scene.kind in ('home','hall','lobby','tasks','fangcun_setup','daqian_setup')
                    if safe_invite:
                        clear_handled_invitation(control)
                if battle_start is not None and scene.kind in ('lobby','fangcun_setup','daqian_setup'):
                    if not battle_seen and now-battle_start < 8 and scene.notice is None:
                        control.stop.wait(.2)
                        continue
                    if not battle_seen:
                        failed_starts += 1
                        record('start_unconfirmed',mode=flow.mode,notice=scene.notice)
                        if scene.notice != 'missing_weapon' and (failed_starts >= 2 or trial is not None):
                            status('须弥幻境未确认成功开战，稍后重试')
                            return TaskOutcome.WAITING
                    elif trial is not None:
                        status('须弥幻境提前退出，未确认整轮结算')
                        return TaskOutcome.WAITING
                    battle_start = None
                    battle_seen = False
                    battle_vision.reset()
                    if scene.notice is None:
                        flow.mode = None
                rows = ()
                if scene.kind == 'tasks' and not finished:
                    result = ocr.read(frame,stop=control.stop)
                    if result is not None:
                        rows = read_task_panel(DailyScreen(frame,result,'xumi-tasks'),ocr,control.stop)
                    if control.stop.is_set() or control.generation() != generation:
                        continue
                    record('tasks',rows=[vars(row) for row in rows])
                    scores = {row.mode: row.goal if row.claimed else row.current for row in rows
                              if row.kind == 'score' and (row.current is not None or row.claimed)}
                    if len(scores) == len(TRIALS) and scores != last_scores:
                        last_scores = scores
                        status('须弥幻境今日积分 · 方寸 '+str(scores['fangcun'])+'/50000 · 大千 '+str(scores['daqian'])+'/50000')
                if invitation:
                    x,y = invitation
                    decision = tap('decline_invite', invitation,'拒绝当前不接取的悬赏邀请',(x-58,y-24,x+58,y+24))
                elif finished:
                    if scene.kind == 'home':
                        return TaskOutcome.DAILY_DONE
                    decision = tap('back', scene.back,'返回主页，继续调度')
                else:
                    decision = flow.step(frame,scene,rows)
                    if scene.kind == 'result_next' and boss_number == 2:
                        decision = DailyDecision(status='已确认继续试炼，等待第二个 Boss')
                    if scene.kind == 'unknown':
                        reward, point = vision.daily.navigation(frame)
                        if reward in ('reward','reward_continue'):
                            decision = tap('dismiss_reward', point,'确认已获得的须弥幻境奖励')
                if scene.kind == 'unknown' and battle_start is not None and now-last_progress < 1:
                    control.stop.wait(.05)
                    continue
                status(decision.status)
                key = (scene.kind, decision.action, decision.status, decision.point)
                if key != last_frame:
                    filename = f'{sequence:04d}-{scene.kind}-{decision.action}.png'
                    cv2.imwrite(str(output/filename),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                    record('decision',page=scene.kind,action=decision.action,operation=decision.operation,
                           point=decision.point,status=decision.status,frame=filename)
                    sequence += 1
                    last_frame = key
                if decision.action == 'unavailable':
                    return TaskOutcome.UNAVAILABLE
                if decision.action == 'defer':
                    return TaskOutcome.DEFERRED_TODAY
                if decision.action == 'checkpoint':
                    return TaskOutcome.MATCH_DONE
                if decision.action == 'done':
                    finished = True
                    last_progress = now
                    continue
                if decision.point is not None and decision.operation:
                    if trial is not None and decision.action == 'settle':
                        result = ocr.read(frame, stop=control.stop)
                        if control.stop.is_set() or generation != control.generation():
                            continue
                        scores = DailyScreen(frame,result).text((90,315,545,447)) if result is not None else ''
                        record('trial_result', mode=trial, difficulty=current_difficulty, scores=scores,
                               bosses=sorted(seen_bosses - {None}),
                               relay_bosses=sorted(relay_bosses - {None}))
                    observed = vision.menu_observation(frame,scene,decision,invitation,generation,kernel.captured_at)
                    if ui.perform(observed,decision.operation):
                        flow.completed(decision)
                        record('input_sent',point=decision.point,action=decision.action,operation=decision.operation)
                        if decision.action in ('start','continue'):
                            if decision.action == 'start':
                                current_difficulty = decision.value
                                seen_bosses.clear()
                                relay_bosses.clear()
                            boss_number = 1 if decision.action == 'start' else 2
                            boss_entered_at = None
                            combat.reset_boss()
                            battle_vision.reset()
                            battle_start = control.now()
                            battle_seen = False
                            next_snapshot = 0.
                        if decision.action == 'settle':
                            battle_start = None
                            battle_seen = False
                            battle_vision.reset()
                            if trial is not None:
                                return TaskOutcome.MATCH_DONE if seen_bosses == {1, 2} and not relay_bosses else TaskOutcome.WAITING
                        last_progress = now
                        if decision.action == 'start':
                            if not control.stop.wait(.28):
                                response = kernel.capture(device)
                                if vision.scene(response).kind == scene.kind:
                                    diagnostics.save(output/f'{sequence:04d}-start-feedback.png',response)
                                    sequence += 1
                        else:
                            control.stop.wait(.5)
                        continue
                if now-last_progress >= (45 if battle_start is not None else 22):
                    status('须弥幻境未确认下一步，保留未完成状态')
                    return TaskOutcome.WAITING
                control.stop.wait(.3)
            return TaskOutcome.STOPPED
        finally:
            control.set_release_input(None)
            with ExitStack() as cleanup:
                for resource in (diagnostics,ocr,device,native):
                    if resource is not None:
                        cleanup.callback(resource.close)
                if combat is not None:
                    cleanup.callback(combat.stop)
