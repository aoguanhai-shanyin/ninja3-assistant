"""Daily-only Xumi trials with the common capture, input and combat kernel."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time

import cv2

from ninja3_assistant.activities.bounty.combat import BountyCombat, BountyCombatVision
from ninja3_assistant.activities.daily.flow import DailyDecision
from ninja3_assistant.activities.daily.vision import DailyScreen
from ninja3_assistant.activities.intelligence.ocr import WindowsOcr
from ninja3_assistant.activities.xumi.equipment import EquipmentSetup
from ninja3_assistant.activities.xumi.tasks import read_task_panel, tasks_claimed
from ninja3_assistant.activities.xumi.teams import TeamSetup
from ninja3_assistant.activities.xumi.vision import TITLE, XumiVision
from ninja3_assistant.device.mumu import MuMu, prepare_game
from ninja3_assistant.device.native_touch import create_native_touch, executable_path
from ninja3_assistant.paths import project_root
from ninja3_assistant.runtime.realtime_battle import FrameDiagnostics, RealtimeBattleKernel
from ninja3_assistant.scheduling.invitation_handoff import bounty_handoff, clear_handled_invitation
from ninja3_assistant.scheduling.task_outcome import TaskOutcome
from ninja3_assistant.vision.game_update import game_update_banner


TRIALS = {'fangcun': '方寸试炼', 'daqian': '大千试炼'}
TEAM_PAGES = ('team', 'roster', 'own_roster')


def tap(point, status, guard=None):
    return DailyDecision('tap', point, status, guard=guard) if point else DailyDecision(status=status)


@dataclass
class XumiProgress:
    last_mode: str | None = None
    settled_mode: str | None = None
    scores: dict[str, int] = field(default_factory=dict)
    no_progress: dict[str, int] = field(default_factory=lambda: dict.fromkeys(TRIALS,0))


class XumiFlow:
    def __init__(self, progress=None):
        self.mode = None
        self.progress = progress if progress is not None else XumiProgress()
        self.teams = TeamSetup()
        self.gear = EquipmentSetup()
        self.equipping = False
        self.gear_team = None
        self.gear_required = False
        self.gear_ready = set()
        self.gear_saved = False
        self.hall_swipes = 0

    def step(self, frame, scene, rows=()):
        kind = scene.kind
        if kind == 'score_confirm':
            return DailyDecision('unavailable', status='确认编队变更会重置积分，保留当前积分并交回调度')
        if self.gear_saved:
            if kind != 'team' or scene.team_number != self.gear_team:
                return DailyDecision(status='等待确认大千装备搭配已保存')
            self.gear_saved = self.equipping = False
            self.gear_ready.add(self.gear_team)
            self.gear_team = None
        if kind == 'home':
            return tap(scene.point, '进入集会所寻找须弥幻境')
        if kind == 'hall':
            if scene.point:
                return tap(scene.point, '进入须弥幻境', (scene.point[0]-100,130,scene.point[0]+100,195))
            if self.hall_swipes >= 6:
                return DailyDecision('unavailable', status='集会所未确认须弥幻境入口')
            points = ((1080,400),(280,400)) if self.hall_swipes < 3 else ((280,400),(1080,400))
            return DailyDecision('swipe',points[0],'在集会所寻找须弥幻境',points[1])
        if kind == 'guide':
            return tap(scene.point, '查看须弥幻境说明')
        if kind == 'lobby':
            if self.mode is None:
                return tap(scene.point, '打开须弥幻境每日任务')
            return tap(getattr(scene, self.mode), '进入'+TRIALS[self.mode], TITLE)
        if kind == 'tasks':
            if tasks_claimed(rows):
                return DailyDecision('done', status='须弥幻境方寸和大千每日奖励均已领取')
            pending = next((row for row in rows if row.claim_point is not None), None)
            if pending is not None:
                x,y = pending.claim_point
                return tap(pending.claim_point, '领取'+TRIALS[pending.mode]+'每日奖励', (x-48,y-22,x+48,y+22))
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
                for row in rows) and progress.no_progress[mode] < 2), None)
            if mode is None:
                stalled = [TRIALS[mode] for mode in TRIALS if progress.no_progress[mode] >= 2 and any(
                    row.mode == mode and not row.claimed and row.current < row.goal for row in rows)]
                if stalled:
                    return DailyDecision('defer', status='、'.join(stalled)+'连续两轮未增加每日积分，保留未完成进度')
                return DailyDecision(status='等待确认须弥幻境每日奖励领取状态')
            if settled is not None:
                return DailyDecision('checkpoint', status='须弥幻境整轮已结算，日常进度已复查，继续调度')
            self.mode = mode
            return tap(scene.back, '继续'+TRIALS[mode]+'未完成日常')
        if kind == 'result_next' and scene.point is not None:
            return DailyDecision('continue', scene.point, '继续第二场须弥幻境试炼', guard=(870,610,1075,652))
        if kind == 'result_done' and scene.point is not None:
            return DailyDecision('settle', scene.point, '完成整轮试炼，返回核对每日进度', guard=(778,611,1000,652))
        if kind == 'equipment_confirm' and self.equipping and scene.point is not None:
            return DailyDecision('gear_recheck',scene.point,'保存当前装备并重新核对',guard=(730,505,886,550))
        if kind.startswith('equipment'):
            if self.equipping and self.mode == 'daqian':
                return self.gear.step(frame,scene)
            return tap(scene.back,'返回小队搭配，继续核对须弥日常')
        if kind.endswith('_setup') or kind in TEAM_PAGES:
            if self.mode is None or (kind.endswith('_setup') and kind != self.mode+'_setup'):
                return tap(scene.back, '返回须弥幻境大厅检查每日进度')
            if kind == 'team' and self.equipping:
                if scene.team_number != self.gear_team:
                    if scene.team_number is None:
                        return DailyDecision(status='等待确认当前装备队伍')
                    return tap((120,188 if self.gear_team == 1 else 277),
                               f'切换大千试炼队伍{self.gear_team}',(80,155,195,302))
                return tap((1250,363),'打开大千试炼装备搭配',(1228,214,1276,508))
            decision = self.teams.step(frame, kind, self.mode)
            if (kind == 'daqian_setup' and decision.action == 'start' and
                    self.gear_required and len(self.gear_ready) != 2):
                team = 1 if 1 not in self.gear_ready else 2
                y = 565 if team == 1 else 647
                return DailyDecision('check_gear',(563,y),f'核对大千试炼队伍{team}的武器',
                                     guard=(532,y-34,595,y+32))
            return decision
        return DailyDecision(status='等待识别须弥幻境画面')

    def completed(self, decision):
        if decision.action == 'swipe':
            self.hall_swipes += 1
        if decision.action == 'check_gear':
            self.equipping = True
            self.gear_team = 1 if decision.point[1] == 565 else 2
            self.gear = EquipmentSetup(expected_team=self.gear_team)
        elif decision.action == 'gear_done':
            self.gear_saved = True
        elif decision.action == 'gear_recheck':
            self.gear = EquipmentSetup(expected_team=self.gear_team)
        elif decision.action == 'settle':
            self.progress.last_mode = self.progress.settled_mode = self.mode
            self.mode = None
        self.gear.completed(decision)
        self.teams.completed(decision)


def run_xumi(folder, index, control, emit, *, progress):
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
            if (kind == 'native_plan' and fields['finger'] in (3,4) and
                    fields['event'] in ('pressed','expired_before_send','dropped')):
                details = dict(fields)
                stage = details.pop('event')
                record('skill_input', stage=stage, **details)

        try:
            if prepare_game(folder, index, control, status) is None:
                return TaskOutcome.STOPPED
            vision = XumiVision()
            battle_vision = BountyCombatVision()
            flow = XumiFlow(progress)
            ocr = WindowsOcr(persistent=True)
            ocr.warmup()
            device = MuMu(folder, index)
            native = create_native_touch(folder, index, control.stop, record_input)
            if native is None:
                raise RuntimeError('须弥幻境需要 NinjaTouch.exe')
            kernel = RealtimeBattleKernel(control)
            kernel.attach(native)
            combat = BountyCombat(kernel)
            diagnostics = FrameDiagnostics()
            control.set_release_input(native.stop_all)
            emit('state','running')
            emit('status','须弥幻境记录：'+str(output))
            sequence = 0
            battle_start = None
            battle_seen = False
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
                if battle_start is not None:
                    state = battle_vision.detect(frame)
                    if state.active:
                        safe_invite = False
                        if not battle_seen:
                            record('battle_start',mode=flow.mode)
                            last_frame = None
                            previous_skill_points = None
                            failed_starts = 0
                        skill_points = (state.ninjutsu_point,state.weapon_points)
                        if skill_points != previous_skill_points:
                            record('battle_controls',mode=flow.mode,ninjutsu=state.ninjutsu_point,
                                   weapons=state.weapon_points)
                            previous_skill_points = skill_points
                        if now >= next_snapshot:
                            diagnostics.save(output/f'battle-{sequence:04d}.png',frame)
                            sequence += 1
                            next_snapshot = now+3
                        battle_seen = True
                        status('须弥幻境 · '+TRIALS.get(flow.mode,'试炼')+' · '+combat.update(state,now))
                        last_progress = now
                        control.stop.wait(.012)
                        continue
                    combat.stop()
                scene = vision.scene(frame)
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
                        if failed_starts >= 2:
                            status('须弥幻境未确认成功开战，稍后重试')
                            return TaskOutcome.WAITING
                        if scene.notice == 'missing_weapon':
                            flow.gear_required = True
                            flow.gear_ready.clear()
                            flow.gear = EquipmentSetup()
                    battle_start = None
                    battle_seen = False
                    battle_vision.reset()
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
                    decision = tap(invitation,'拒绝当前不接取的悬赏邀请',(x-58,y-24,x+58,y+24))
                elif finished:
                    if scene.kind == 'home':
                        return TaskOutcome.DAILY_DONE
                    decision = tap(scene.back,'返回主页，继续调度')
                else:
                    decision = flow.step(frame,scene,rows)
                    if scene.kind == 'unknown':
                        reward, point = vision.daily.navigation(frame)
                        if reward in ('reward','reward_continue'):
                            decision = tap(point,'确认已获得的须弥幻境奖励')
                if scene.kind == 'unknown' and battle_start is not None and now-last_progress < 1:
                    control.stop.wait(.05)
                    continue
                status(decision.status)
                key = (scene.kind, decision.action, decision.status, decision.point)
                if key != last_frame:
                    filename = f'{sequence:04d}-{scene.kind}-{decision.action}.png'
                    cv2.imwrite(str(output/filename),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                    record('decision',page=scene.kind,action=decision.action,point=decision.point,status=decision.status,frame=filename)
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
                if decision.point is not None:
                    def act():
                        if control.generation() != generation:
                            return False
                        fresh = kernel.capture(device)
                        fresh_scene = vision.scene(fresh)
                        if game_update_banner(fresh) or fresh_scene.kind != scene.kind:
                            return False
                        if scene.team_number is not None and fresh_scene.team_number != scene.team_number:
                            return False
                        if scene.kind in ('home','hall','guide') and fresh_scene.point != scene.point:
                            return False
                        if scene.kind == 'equipment_target' and decision.point not in fresh_scene.equipment_targets:
                            return False
                        if invitation and vision.daily.invitations.detect(
                                cv2.cvtColor(fresh,cv2.COLOR_RGB2GRAY)) != invitation:
                            return False
                        if decision.status == '确认已获得的须弥幻境奖励':
                            reward, point = vision.daily.navigation(fresh)
                            if reward not in ('reward','reward_continue') or point != decision.point:
                                return False
                        if decision.guard:
                            x1,y1,x2,y2 = decision.guard
                            if cv2.absdiff(frame[y1:y2,x1:x2],fresh[y1:y2,x1:x2]).mean() > 14:
                                return False
                        if decision.action == 'swipe':
                            native.stop_all()
                            subprocess.run([str(executable_path()),'--folder',str(folder),'--index',str(index),
                                '--swipe',*map(str,(*decision.point,*decision.end))],capture_output=True,
                                check=True,timeout=8,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                            return True
                        captured = kernel.captured_at
                        return native.priority_tap(decision.point,captured_at=captured,deadline=captured+.2,
                                                   wait_seconds=.2,hold_seconds=.055)
                    if control.run_if_active(act):
                        flow.completed(decision)
                        record('input_sent',point=decision.point,action=decision.action)
                        if decision.action in ('start','continue'):
                            battle_vision.reset()
                            battle_start = control.now()
                            battle_seen = False
                            next_snapshot = 0.
                        if decision.action == 'settle':
                            battle_start = None
                            battle_seen = False
                            battle_vision.reset()
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
            if combat is not None:
                combat.stop()
            if native is not None:
                native.close()
            if device is not None:
                device.close()
            if ocr is not None:
                ocr.close()
            if diagnostics is not None:
                diagnostics.close()
