"""Bounty running and Boss skills on the shared realtime input kernel."""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from ninja3_assistant.combat.opening_actions import OpeningActions
from ninja3_assistant.device.native_touch import TimedTap
from ninja3_assistant.paths import resource_root
from ninja3_assistant.vision.vision import ASSETS


@dataclass(frozen=True, slots=True)
class BountyBattleState:
    active: bool = False
    ground: bool = False
    on_rope: bool = False
    boss_active: bool = False
    ninjutsu_point: tuple | None = None
    weapon_points: tuple = ()


class BountyCombatVision:
    """Recognize control silhouettes and the Boss portrait before its bar fills."""
    # The same left/right weapon slots are used by the existing race runner.
    weapon_points = ((380,630),(900,630))

    def __init__(self):
        self.opening = OpeningActions()
        down = cv2.imread(str(ASSETS/'battle-down.png'),0)
        if down is None:
            raise RuntimeError('缺少悬赏绳索控制识别模板')
        self.down = cv2.inRange(down,200,255)
        self.up = cv2.flip(self.down,0)
        self.ground_controls = tuple(cv2.inRange(self.opening.images[name],200,255)
                                     for name in ('battle-slide','battle-jump'))
        assets = resource_root()/'assets/bounty-combat'
        self.boss_frame = cv2.imread(str(assets/'boss-portrait-frame.png'),0) if (assets/'boss-portrait-frame.png').is_file() else None
        self.boss_level = cv2.imread(str(assets/'boss-level-label.png'),0) if (assets/'boss-level-label.png').is_file() else None
        if self.boss_frame is not None:
            y,x = np.indices(self.boss_frame.shape)
            radius = np.sqrt((x-36)**2+(y-35)**2)
            self.boss_mask = np.where((radius>=28)&(radius<=37)&~((x>40)&(y>40)),255,0).astype(np.uint8)
        self.boss_seen = False

    def boss_portrait(self, gray):
        if self.boss_frame is None or self.boss_level is None:
            return False
        label = gray[32:70,444:490]
        if np.count_nonzero(label > 205) < 20:
            return False
        level = cv2.minMaxLoc(cv2.matchTemplate(label,self.boss_level,
                                               cv2.TM_CCOEFF_NORMED))[1]
        if level < .94:
            return False
        score = cv2.minMaxLoc(cv2.matchTemplate(gray[15:105,365:460],self.boss_frame,
            cv2.TM_CCORR_NORMED,mask=self.boss_mask))[1]
        return math.isfinite(score) and score >= .96

    @staticmethod
    def boss_bar(frame):
        # Observed HUD: white Boss name above a coloured 14px bar, with
        # dark borders above and below. Brown scenery beams and the red
        # WARNING transition have neither this position nor both borders.
        name = frame[35:65,440:790]
        if np.count_nonzero(name.min(axis=2) > 205) < 40:
            return False
        hsv = cv2.cvtColor(frame[55:95,400:840],cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv,(0,110,75),(179,255,255))
        for contour in cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
            x,y,w,h = cv2.boundingRect(contour)
            if w < 24 or not 6 <= h <= 20 or cv2.contourArea(contour)/(w*h) < .65:
                continue
            x,y = x+400,y+55
            top = frame[y-4:y,x:x+w].max(axis=2)
            bottom = frame[y+h:y+h+4,x:x+w].max(axis=2)
            if (top < 55).mean() >= .85 and (bottom < 55).mean() >= .85:
                return True
        return False

    def reset(self):
        self.boss_seen = False

    def detect(self, frame, *, battle_allowed=True):
        if not battle_allowed:
            self.reset()
            return BountyBattleState()
        if not isinstance(frame,np.ndarray) or frame.shape != (720,1280,3):
            return BountyBattleState()
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        rope = cv2.inRange(gray[565:700,120:270],200,255)
        on_rope = max(cv2.minMaxLoc(cv2.matchTemplate(rope,image,cv2.TM_CCOEFF_NORMED))[1]
                      for image in (self.down,self.up)) >= .88
        ground = False
        if not on_rope:
            # Bright glyphs retain their shape while translucent button
            # backgrounds change with the scenery and pressed animation.
            ground = all(cv2.minMaxLoc(cv2.matchTemplate(
                cv2.inRange(gray[y1:y2,x1:x2],200,255),template,cv2.TM_CCOEFF_NORMED))[1] >= .92
                for template,(x1,y1,x2,y2) in zip(self.ground_controls,
                    ((80,550,310,715),(1010,550,1190,715))))
        boss_signal = self.boss_portrait(gray) or self.boss_bar(frame)
        point = self.opening._ninjutsu_point(gray) if self.boss_seen or boss_signal else None
        # The Boss portrait can arrive while a control is still obscured.
        # Its fixed border+Lv layout and a visible skill HUD suffice; do not
        # wait for both movement controls or for the coloured health fill.
        active = on_rope or ground or (boss_signal and point is not None)
        if not active:
            return BountyBattleState()
        self.boss_seen = self.boss_seen or boss_signal
        return BountyBattleState(True,ground,on_rope,self.boss_seen,point,
                                 self.weapon_points if self.boss_seen else ())


class BountyCombat:
    """Caller captures through the kernel and installs native.stop_all on control.

    Movement stays on native fingers 1/2. Ninjutsu and weapons use independent
    bounded plans on fingers 3/4; no Python input loop or worker is created.
    """
    def __init__(self, kernel):
        self.kernel = kernel
        self.kernel.movement_horizon = 3.
        self.moving = False
        self.skills = set()
        self.last_ninjutsu = self.last_weapon = float('-inf')
        self.generation = kernel.control.generation()

    def _stop_movement(self):
        if self.moving:
            self.kernel.cancel_movement()
            self.moving = False

    def _stop_skills(self):
        for finger in self.skills:
            self.kernel.cancel_plan(finger)
        self.skills.clear()

    def stop(self):
        self._stop_movement()
        self._stop_skills()
        self.last_ninjutsu = self.last_weapon = float('-inf')

    def update(self, state, now):
        generation = self.kernel.control.generation()
        if generation != self.generation:
            self.stop()
            self.generation = generation
        if not self.kernel.fresh() or not state.active:
            self.stop()
            return '等待悬赏战斗画面'
        if state.boss_active:
            self._stop_movement()
            if state.ninjutsu_point is not None and now-self.last_ninjutsu >= .2:
                # PLAN times refer to capture, not dispatch. Begin just after
                # the current frame age instead of adding a fixed 30ms delay.
                first = max(0,math.ceil((self.kernel.clock()-self.kernel.captured_at)*1000))+2
                actions = tuple(TimedTap(offset,state.ninjutsu_point) for offset in range(first,186,70))
                if actions and self.kernel.plan(3,actions) is not None:
                    self.skills.add(3)
                    self.last_ninjutsu = now
            if state.weapon_points and now-self.last_weapon >= .4:
                actions = tuple(TimedTap(30+i*65,point) for i,point in enumerate(state.weapon_points))
                if self.kernel.plan(4,actions) is not None:
                    self.skills.add(4)
                    self.last_weapon = now
            return 'Boss头像或血条已出现，释放忍术并持续攻击'
        self._stop_skills()
        if state.on_rope:
            self._stop_movement()
            return '绳索上保持不动，等待进入地面或Boss战'
        if state.ground:
            if self.kernel.renew():
                self.moving = True
            return '地面滑步收集忍币'
        self._stop_movement()
        return '等待地面或绳索状态'
