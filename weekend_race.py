"""Observed weekend race navigation and generic native-input racing loop."""
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import cv2
import numpy as np

from activity_ocr import WindowsOcr
from activity_vision import ActivityVision
from mumu import MuMu, prepare_game
from native_touch import create_native_touch, executable_path, TimedTap
from opening_actions import OpeningActions
from realtime_battle import RealtimeBattleKernel, FrameDiagnostics
from task_outcome import TaskOutcome
from invitation import InvitationVision
from race_obstacles import RaceObstacleVision
from race_rope import RaceRopeTraversal

ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))


def remaining_matches(text):
    match = re.search(r'剩余\s*(\d{1,2})\s*次', ''.join(text.split()))
    if not match and '积分场次' in ''.join(text.split()):
        match = re.search(r'(\d{1,2})次', ''.join(text.split()))
    return int(match[1]) if match and int(match[1]) <= 25 else None


class RaceVision:
    def __init__(self):
        self.invitations = InvitationVision()
        self.images = {p.stem: cv2.imread(str(p), 0) for p in (ROOT/'assets/weekend-race').glob('*.png')}
        self.opening = OpeningActions()
        self.rope_down = cv2.inRange(self.images['rope-down'],200,255)
        self.rope_up = cv2.flip(self.rope_down,0)
        self.dash = cv2.inRange(self.images['dash'],230,255)
        self.effect = cv2.inRange(self.images['race-effect'],180,255)
        self.on_rope = False
        self.confirmed_race = False
        self.obscured = False

    def match(self, frame, name, box):
        x,y,w,h = box
        image = cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_RGB2GRAY)
        template = self.images[name]
        return cv2.minMaxLoc(cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED))[1] >= (.77 if name == 'match' else .93)

    def detect(self, frame, gray=None):
        self.confirmed_race = False
        self.obscured = False
        if gray is None:
            gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        if self.invitations.detect(gray):
            return 'invite'
        if self.match(frame, 'finish', (500,370,300,80)):
            return 'finish'
        if self.match(frame, 'result', (1030,645,230,65)):
            return 'result'
        if self.match(frame, 'standings', (1190,145,90,65)):
            return 'standings'
        if gray is None:
            gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        mask = cv2.inRange(gray,200,255)
        if cv2.minMaxLoc(cv2.matchTemplate(cv2.inRange(gray[350:450,990:1130],230,255),self.dash,cv2.TM_CCOEFF_NORMED))[1] >= .85:
            return 'dash'
        rope = mask[565:700,120:270]
        arrow = max(cv2.minMaxLoc(cv2.matchTemplate(rope,image,cv2.TM_CCOEFF_NORMED))[1]
                    for image in (self.rope_down,self.rope_up))
        self.on_rope = arrow >= .88
        self.opening.ground_frames = 0 if self.on_rope else 1
        # The jump glyph changes posture during skating and in the air. Its
        # connected white silhouette is stable; matching one pose caused the
        # native slide lease to be cancelled on every icon transition.
        button = mask[579:664,1045:1140]
        count = cv2.countNonZero(button)
        movement = False
        if 250 < count < 4500:
            for contour in cv2.findContours(button,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
                x,y,w,h = cv2.boundingRect(contour)
                if w > 20 and h > 32:
                    movement = True
                    break
        effect = cv2.minMaxLoc(cv2.matchTemplate(cv2.inRange(gray[0:65,875:940],180,255),self.effect,cv2.TM_CCOEFF_NORMED))[1]
        board = (self.match(frame,'race-distance',(1130,25,90,38)) and
                 self.match(frame,'race-distance',(1140,72,90,38)))
        self.confirmed_race = effect >= .93 or board
        if self.confirmed_race:
            region = frame[200:580,300:1000]
            neutral = region.max(axis=2)-region.min(axis=2) < 18
            brightness = gray[200:580,300:1000]
            self.obscured = float(np.mean(neutral & (brightness>45) & (brightness<230))) > .45
            return 'race' if movement or self.on_rope or self.obscured else 'respawn'
        if self.match(frame, 'reward', (510,70,260,90)):
            return 'reward'
        if self.match(frame, 'tasks', (280,40,210,60)):
            return 'tasks'
        if self.match(frame, 'ready', (510,540,270,80)):
            return 'ready'
        if self.match(frame, 'home', (90,5,340,90)):
            return 'home'
        if self.match(frame, 'hall', (100,0,260,100)):
            return 'hall'
        return 'unknown'

    def claimed(self, frame):
        return all(self.match(frame, 'claimed', (925,y,145,55)) for y in (125,225,325,425,525))

    def claimable(self, frame):
        return any(self.match(frame, 'claim', (925,y,145,55)) and
                   not self.match(frame, 'claimed', (925,y,145,55)) for y in (125,225,325,425,525))


class RaceAvoider:
    """Use the current road edge, including stationary walls and missing road."""
    def __init__(self):
        self.last_jump = float('-inf')
        self.air_until = 0.
        self.hazard = None
        self.ground = None
        self.road_confirmed = False
        self.candidate = None
        self.confirmed_frames = 0
        self.player_x, self.player_y = 600,350
        self.player_tracked = False
        self.blocked = False
        self.obstacles = RaceObstacleVision()
        self.detected = ()
        self.last_player = float('-inf')
        self.previous_obstacle = None
        self.approach_speed = 0.

    def track_player(self, frame, now):
        # The local player's small health bar moves horizontally when blocked
        # or overtaken. Obstacles must be measured from that player, rather
        # than from a fixed screen coordinate occupied by another racer.
        region = frame[160:590,220:800]
        mask = cv2.inRange(region,(150,40,0),(255,240,125))
        r,g,b = region.astype(np.int16).transpose(2,0,1)
        mask[(r*20 <= g*21) | (g <= b)] = 0
        _,_,stats,_ = cv2.connectedComponentsWithStats(mask)
        bars = []
        for x,y,w,h,a in stats[1:]:
            if not (6 <= w <= 30 and 2 <= h <= 5 and w >= h*1.5 and 0 < y and y+h < region.shape[0]):
                continue
            # The health fill is bounded by two dark horizontal borders.
            # Gold scenery and particles can have the same fill colour.
            border = region[[y-1,y+h],x:x+w]
            if np.mean(border.max(axis=2)<145) >= .8:
                bars.append((int(x)+234,int(y)+160,int(w)))
        if bars:
            self.player_x,self.player_y,_ = min(bars,key=lambda p:
                abs(p[0]-self.player_x)+abs(p[1]-self.player_y)*.3-p[2]*2)
            self.player_tracked = True
            self.last_player = now
        elif now-self.last_player > .4:
            self.player_tracked = False

    def decide(self, frame, now, gray=None):
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.track_player(frame,now)
        # Find the broad road/cliff boundary. A narrow reference patch can be
        # covered by another player or a skill effect and jump to background
        # edges, repeatedly inventing pits on an otherwise continuous road.
        sums = np.vstack((np.zeros((1,320),np.float32),
                          np.cumsum(gray[:,::4],axis=0,dtype=np.float32)))
        rows = np.arange(300,580,2)
        contrast = ((sums[rows-5]-sums[rows-20])/15-
                    (sums[rows+40]-sums[rows+15])/25)
        behind = max(38,(self.player_x-240)//4)
        profile = np.median(contrast[:,behind:max(behind+12,(self.player_x-70)//4)],axis=1)
        if self.player_tracked:
            profile[rows-15 < self.player_y+45] = -1
        best = int(profile.argmax())
        reference = float(profile[best])
        previous_ground = self.ground
        ground = int(rows[best])-15 if reference > 15 else previous_ground or 540
        self.ground = ground
        self.hazard = None
        self.road_confirmed = reference > 15
        self.blocked = False
        self.detected = self.obstacles.detect(frame, self.player_x, ground)
        # Missing terrain under a full-screen effect is not evidence of a pit.
        # A gap requires the current road plus a missing continuation ahead.
        if reference > 15:
            missing = 0
            ahead = min(1000,self.player_x+100)
            for x in range(ahead,ahead+211,30):
                # A sloping/stepped road is still road: search nearby heights
                # before treating disappearance of one horizontal row as a gap.
                nearby = contrast[max(0,best-35):best+36, x//4:(x+30)//4]
                edge = float(np.max(np.median(nearby, axis=1)))
                missing = missing+1 if edge < max(15,reference*.25) else 0
                if missing >= 2:
                    self.hazard = 'pit'
                    break
            left = self.player_x+35
            area = gray[max(150,ground-150):ground+12,left:min(1280,ahead+240)]
            lines = cv2.HoughLinesP(cv2.Canny(area,70,150),1,np.pi/180,30,
                                    minLineLength=30,maxLineGap=7)
            if lines is not None:
                for line in lines.reshape(-1,4):
                    x1,y1,x2,y2 = map(int,line)
                    near = min(x1,x2) < ahead-left
                    if (abs(x2-x1)<8 and min(x1,x2)<ahead-left+180 and
                        max(y1,y2)>area.shape[0]-35 and (not near or abs(y2-y1)>=60)):
                        edge = left+min(x1,x2)
                        solid = max(float(np.mean(gray[max(0,ground-65):ground+5,
                                                        max(0,edge+dx):min(1280,edge+dx+24)] < 85))
                                    for dx in (-12,0))
                        if solid < .65 or self.hazard == 'pit':
                            continue
                        self.hazard = 'wall'
                        self.blocked = near
                        break
        action = (1090,630)
        upcoming = [item for item in self.detected if item.box[0]+item.box[2] > self.player_x+15 and
                    (item.kind != 'bomb' or item.box[1]+item.box[3] > ground-170)]
        if upcoming:
            obstacle = min(upcoming, key=lambda item:item.box[0])
            distance = obstacle.box[0]-self.player_x
            previous = self.previous_obstacle
            if previous and previous[0] == obstacle.kind and 0 < now-previous[2] < .25:
                speed = (previous[1]-distance)/(now-previous[2])
                if 0 <= speed < 2500:
                    self.approach_speed = self.approach_speed*.5+speed*.5
            self.previous_obstacle = (obstacle.kind, distance, now)
            lead = min(360, max(200, self.approach_speed*.28+60))
            if distance < lead and self.hazard != 'pit':
                if obstacle.kind == 'bomb':
                    # Do not jump into a hanging trap. An overhead trap above
                    # the play corridor needs no extra movement input.
                    if obstacle.box[1]+obstacle.box[3] > ground-170:
                        self.hazard, action = 'overhead', (190,630)
                else:
                    self.hazard = obstacle.kind
                    self.blocked = distance < 70
        else:
            self.previous_obstacle = None
            self.approach_speed = 0.
        candidate = self.hazard
        self.confirmed_frames = self.confirmed_frames+1 if candidate == self.candidate else 1
        self.candidate = candidate
        if self.confirmed_frames < 2:
            self.hazard = None
        # Let at least one complete skate cycle run after a wall hop. A wall
        # visible for many frames must not keep extending the native air hold.
        retry = .55 if self.hazard != 'pit' else .32
        if self.hazard and now-self.last_jump >= retry:
            self.last_jump = now
            self.air_until = now+.42
            return action
        return None

class RaceSkills:
    # Include both weapons. The centre cards use one or two rows as items are
    # acquired; tapping an empty/cooling slot does not interrupt C++ skating.
    points = ((380,630),(900,630),(520,630),(635,630),(740,630))
    actions = tuple(TimedTap(20+i*35,point) for i,point in enumerate(points))
    def __init__(self):
        self.last = float('-inf')

    def due(self, now):
        return now-self.last >= .6

    def scheduled(self, now):
        self.last = now


def run_weekend(folder, index, control, emit, mode='daily', record=False, practice=False, *, checkpoint=False):
    if mode not in ('daily','25'):
        raise ValueError('weekend mode must be daily or 25')
    device = native = recorder = None
    kernel = RealtimeBattleKernel(control,movement_horizon=3.)
    diagnostics = FrameDiagnostics()
    output = (Path(sys.executable).parent if getattr(sys,'frozen',False) else ROOT/'data'/'runtime')/'weekend-race'/'runs'/(time.strftime('%Y%m%d-%H%M%S')+f'-{time.time_ns()%1000000:06d}')
    output.mkdir(parents=True,exist_ok=False)
    last = None
    checked = False
    last_action = float('-inf')
    skills = RaceSkills()
    armed = False
    armed_at = None
    hall_swipes = 0
    last_race = float('-inf')
    last_ground = float('-inf')
    last_hazard = None
    sampled_start = float('-inf')
    observed_race = False
    recovering = False
    recovery_until, recovery_sampled = 0.,float('-inf')
    frames = 0
    measured_at = control.now()
    result_confirmed = False
    unknown_since = None
    rope = RaceRopeTraversal()
    try:
        if prepare_game(folder,index,control,lambda s:emit('status',s)) is None:
            return TaskOutcome.STOPPED
        vision, navigation, ocr, avoider = RaceVision(), ActivityVision(), WindowsOcr(), RaceAvoider()
        native = create_native_touch(folder,index,control.stop,emit,observation_timeout=.5,lease_ms=500,movement='jump-two-slide')
        if native is None:
            raise RuntimeError('同屏竞速需要 NinjaTouch.exe C++ 触控组件')
        native = kernel.attach(native)
        control.set_release_input(native.stop_all)
        device = MuMu(folder,index)
        emit('backend','C++ NinjaTouch / 起跳2次→下滑循环 / Windows OCR')
        if record:
            recorder = cv2.VideoWriter(str(output/'race.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1280,720))
        emit('state','running')
        generation = control.generation()
        with (output/'events.jsonl').open('w',encoding='utf-8') as log:
            while control.wait_active():
                captured = time.perf_counter()
                frame = kernel.capture(device)
                if armed and last is not None and last[0] == 'race' and not recovering:
                    if not kernel.recognition_due(urgent=avoider.hazard is not None):
                        if not rope.active:
                            kernel.renew()
                        control.stop.wait(max(0,1/60-(time.perf_counter()-captured)))
                        continue
                gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                kind = vision.detect(frame,gray)
                now = control.now()
                if not armed and (kind in ('ready','dash') or
                                  (kind in ('race','respawn') and vision.confirmed_race)):
                    armed = True
                    armed_at = now
                    recovering = kind == 'respawn'
                if kind == 'race' and (not armed or not observed_race) and not vision.confirmed_race:
                    kind = 'unknown'
                if armed and not observed_race and armed_at is not None and now-armed_at >= 60:
                    emit('error','竞速开局等待超时，交回计划重试')
                    return TaskOutcome.ERROR
                on_rope = kind == 'race' and vision.on_rope
                if kind == 'unknown':
                    if unknown_since is None:
                        unknown_since = now
                    elif now-unknown_since >= (60 if armed and not observed_race else 15):
                        emit('error','同屏竞速画面等待超时，交回计划重试')
                        return TaskOutcome.ERROR
                else:
                    unknown_since = None
                if armed and not observed_race and kind == 'unknown' and now-sampled_start >= .25:
                    diagnostics.save(output/f'{time.time_ns()}-startup.png',frame)
                    sampled_start = now
                if kind == 'race':
                    observed_race = True
                    last_race = now
                elif kind == 'unknown' and armed and now-last_race < .35:
                    kind = 'race'
                if generation != control.generation():
                    checked = False
                    avoider = RaceAvoider()
                    rope = RaceRopeTraversal()
                    last_ground = last_race = float('-inf')
                    generation = control.generation()
                point, message = None, '等待识别同屏竞速页面'
                if kind == 'dash' and armed:
                    native.tap((1050,385),deadline=captured+.2)
                    message = '倒计时连续点击超级起步'
                elif kind == 'respawn' and armed:
                    native.quiesce(reason='respawn')
                    recovering = observed_race
                    message = '等待恢复操作，恢复后先跳出危险区' if observed_race else '等待超级起步完成'
                elif kind == 'race' and armed:
                    if vision.opening.ground_frames >= 1:
                        last_ground = now
                    ground = not vision.on_rope and not vision.obscured and now-last_ground < .25
                    point = avoider.decide(frame,now,gray) if ground else None
                    kernel.recognized(stable=point is None and avoider.hazard is None and not recovering)
                    if recovering and not vision.on_rope and not vision.obscured:
                        point = (1090,630)
                        avoider.last_jump,avoider.air_until = now,now+.42
                        recovering = False
                        recovery_until = now+1.2
                        emit('recovery',{'phase':'jump','player_x':avoider.player_x,'hazard':avoider.hazard})
                    if now < recovery_until and now-recovery_sampled >= .1:
                        diagnostics.save(output/f'{time.time_ns()}-recovery.png',frame)
                        recovery_sampled = now
                    if not ground:
                        avoider.hazard = None
                    frames += 1
                    if now-measured_at >= 1:
                        emit('metrics',{'fps':round(frames/(now-measured_at),2),'player_x':avoider.player_x,'ground':avoider.ground,'hazard':avoider.hazard,'blocked':avoider.blocked,**kernel.metrics()})
                        measured_at,frames = now,0
                    if avoider.hazard != last_hazard:
                        log.write(json.dumps({'time':time.time(),'hazard':avoider.hazard,'ground':avoider.ground,
                            'player_x':avoider.player_x,'approach_speed':avoider.approach_speed,
                            'unknown_objects':[{'kind':item.kind,'box':item.box} for item in avoider.detected]})+'\n')
                        diagnostics.save(output/f'{time.time_ns()}-terrain.png',frame)
                        last_hazard = avoider.hazard
                    if recorder:
                        recorder.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                    if on_rope:
                        if not rope.active:
                            kernel.cancel_movement()
                            rope.active = True
                        avoider.hazard = None
                        if not vision.obscured:
                            avoider.track_player(frame,now)
                            actions = rope.decide(gray,now,avoider.player_x,avoider.player_y)
                            if actions and kernel.plan(3,actions) is not None:
                                rope.scheduled(now)
                    elif point:
                        if rope.active:
                            kernel.cancel_plan(3)
                            rope = RaceRopeTraversal()
                        control.run_if_active(lambda:native.priority_tap(point,captured_at=captured,deadline=captured+.3,hold_seconds=.1,preserve_slide=True))
                    else:
                        if rope.active:
                            kernel.cancel_plan(3)
                            rope = RaceRopeTraversal()
                        def skate():
                            native.note_frame(captured)
                            native.renew()
                        control.run_if_active(skate)
                    if skills.due(now) and kernel.plan(4,skills.actions) is not None:
                        skills.scheduled(now)
                    message = (('绳上方有地板，起跳登板后接回下滑' if rope.platform is not None else
                                '绳上跳跃前进，避免贴墙停住') if on_rope else
                               '起跳2次→下滑，识别障碍并释放卡牌')
                else:
                    native.quiesce()
                    rope = RaceRopeTraversal()
                    if now-last_action < 1:
                        control.stop.wait(.05)
                        continue
                    if kind == 'finish':
                        point,message = (640,680),'比赛结束，继续查看结算'
                    elif kind == 'standings':
                        message = '等待成绩展示完成，不发送战斗动作'
                    elif kind == 'invite':
                        point,message = vision.invitations.detect(gray),'拒绝队友邀请'
                    elif kind == 'result':
                        point,message = (1145,681),'确认同屏竞速结算'
                        checked,armed = False,False
                        recovering = False
                    elif kind == 'reward':
                        point,message = (640,644),'确认周末任务奖励'
                    elif kind == 'tasks':
                        if vision.claimable(frame):
                            point,message = (1055,679),'一键领取周末日常奖励'
                        elif vision.claimed(frame) and mode == 'daily':
                            emit('status','同屏竞速五项日常已全部领取，任务结束')
                            return TaskOutcome.DAILY_DONE
                        else:
                            point,message = (1190,400),'关闭任务页，准备继续竞速'
                            checked = True
                    elif kind == 'home':
                        if checkpoint and result_confirmed:
                            return TaskOutcome.MATCH_DONE
                        if armed and observed_race:
                            emit('error','竞速已返回主页但未确认结算，交回计划重试')
                            return TaskOutcome.ERROR
                        if checkpoint and checked:
                            from datetime import datetime
                            from activity_flow import CHINA
                            if not 9 <= datetime.now(CHINA).hour < 23:
                                return TaskOutcome.WAITING
                        if armed:
                            message = '等待竞速比赛加载'
                        elif not checked:
                            point,message = (885,47),'检查周末每日任务'
                        elif vision.match(frame,'match',(1030,655,230,55)):
                            result = ocr.read(cv2.resize(frame[638:718,995:1270],None,fx=3,fy=3),stop=control.stop)
                            left = remaining_matches((result or {}).get('text',''))
                            if left == 0 and not practice:
                                emit('status','今日25场积分比赛已用完，任务结束')
                                return TaskOutcome.LIMIT_REACHED
                            if left is not None:
                                point,message = (1140,677),(f'开始同屏竞速，今日剩余 {left} 场积分比赛' if left else '积分次数已用完，仅实测避障动作')
                            else:
                                message = '等待读取今日剩余积分场次'
                    elif kind == 'ready':
                        point,message = (640,579),'准备同屏竞速'
                        armed = True
                        observed_race = False
                        recovering = False
                    elif kind == 'hall':
                        if vision.match(frame,'entry',(900,110,380,120)):
                            point,message = (1130,390),'进入已确认的同屏竞速'
                        else:
                            if hall_swipes >= 3:
                                emit('status','当前周末入口未确认同屏竞速，任务结束')
                                return TaskOutcome.UNAVAILABLE
                            native.quiesce()
                            control.run_if_active(lambda:subprocess.run([str(executable_path()),'--folder',str(folder),'--index',str(index),'--swipe','1150','410','300','410'],capture_output=True,timeout=8,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
                            last_action = now
                            hall_swipes += 1
                            message = '向右查找周末活动；仅进入同屏竞速'
                    else:
                        nav = navigation.detect(frame)
                        if nav.kind in ('login','home'):
                            point,message = nav.point,'前往集会所'
                    if point:
                        def tap():
                            refreshed = kernel.capture(device)
                            if vision.detect(refreshed) != kind:
                                return None
                            return native.priority_tap(point,captured_at=kernel.captured_at,
                                                       deadline=kernel.captured_at+.2,wait_seconds=.2)
                        if control.run_if_active(tap):
                            last_action = now
                            if kind == 'result':
                                result_confirmed = True
                state = (kind,message)
                if state != last and (kind not in ('race','unknown') or now-getattr(avoider,'logged_at',-1e9) >= 1):
                    avoider.logged_at = now
                    emit('status',message)
                    log.write(json.dumps({'time':time.time(),'kind':kind,'message':message,'point':point},ensure_ascii=False)+'\n')
                    log.flush()
                    diagnostics.save(output/f'{time.time_ns()}-{kind}.png',frame)
                    last = state
                control.stop.wait(max(0,1/60-(time.perf_counter()-captured)) if kind in ('race','dash','respawn') else .15)
    except Exception as error:
        emit('error',str(error))
        return TaskOutcome.ERROR
    finally:
        control.set_release_input(None)
        if native:
            native.close()
        if recorder:
            recorder.release()
        if device:
            device.close()
        diagnostics.close()
        emit('finished','')
    return TaskOutcome.STOPPED
