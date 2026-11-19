"""Evidence-backed activity anchors; no coordinates are used without page recognition."""
from ninja3_assistant.paths import resource_root

import cv2
import numpy as np

from ninja3_assistant.activities.daily.vision import DailyVision
from ninja3_assistant.activities.intelligence.flow import ActivityScreen
from ninja3_assistant.activities.intelligence.battle_vision import BattleVision
from ninja3_assistant.activities.intelligence.tasks import TaskVision
from ninja3_assistant.vision.vision import Vision

ROOT = resource_root()


class ActivityVision:
    def __init__(self, *, use_winml=False):
        self.navigation = Vision(use_gpu=False)
        self.daily_navigation = DailyVision()
        self.battle = BattleVision(use_winml=use_winml)
        self.tasks = TaskVision()
        self.templates = {}
        for name in ('hall-title', 'home-title', 'start', 'navigation-home', 'navigation-home-summon'):
            template = cv2.imread(str(ROOT/'assets/intelligence'/f'{name}.png'), 0)
            if template is None:
                raise RuntimeError('找不到情报模拟战识别图片：'+name)
            self.templates[name] = template

    @property
    def backend(self):
        return self.battle.backend

    def match(self, gray, name, area, threshold=.94, max_error=22):
        x, y, w, h = area
        template = self.templates[name]
        region = gray[y:y+h, x:x+w]
        if min(region.shape[:2]) == 0 or any(a < b for a, b in zip(region.shape, template.shape)):
            return False
        _, score, _, at = cv2.minMaxLoc(cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED))
        patch = region[at[1]:at[1]+template.shape[0], at[0]:at[0]+template.shape[1]]
        return score >= threshold and (max_error is None or np.abs(patch.astype(np.int16)-template).mean() < max_error)

    def detect(self, frame, ocr=None):
        if frame.shape != (720, 1280, 3):
            return ActivityScreen()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        invite = self.navigation.invitations.detect(gray)
        if invite:
            return ActivityScreen('invite', invite)
        battle = self.battle.detect(frame, ocr)
        if battle.kind != 'unknown':
            return ActivityScreen('battle', battle=battle)
        task = self.tasks.detect(frame, ocr)
        if task.kind in ('tasks', 'reward'):
            return self.task_screen(task)
        if (self.daily_navigation.guide_chrome(frame) and
                self.match(gray, 'home-title', (70, 0, 400, 110), threshold=.98, max_error=None)):
            return ActivityScreen('guide', self.daily_navigation.guide_point(frame))
        if (self.match(gray, 'home-title', (70, 0, 400, 110)) and
                self.match(gray, 'start', (480, 520, 330, 90))):
            return ActivityScreen('event_home')
        if (self.match(gray, 'navigation-home', (750, 585, 140, 60)) and
                self.match(gray, 'navigation-home-summon', (625, 659, 110, 55))):
            return ActivityScreen('home', (864, 612))
        nav = self.navigation.detect(frame)
        if nav.login:
            return ActivityScreen('login', nav.login)
        if nav.home:
            return ActivityScreen('home', nav.home)
        if nav.hall and self.match(gray, 'hall-title', (220, 110, 310, 100)):
            return ActivityScreen('hall', (373, 389))
        if self.navigation.match(gray, 'startup-hall-title', 80, 0, 300, 110,
                                 threshold=.93, max_error=30):
            if self.match(gray, 'hall-title', (220, 110, 310, 100)):
                return ActivityScreen('hall', (373, 389))
            return ActivityScreen('hall_search')
        if nav.network_error and nav.network_retry:
            return ActivityScreen('network_retry', nav.network_retry)
        if self.daily_navigation.home(frame):
            return ActivityScreen('home', (864, 612))
        return ActivityScreen()

    @staticmethod
    def task_screen(task):
        if task.kind == 'reward':
            return ActivityScreen('reward', point=task.confirm_point, task=task)
        if task.rows:
            return ActivityScreen('tasks',daily=tuple(row.state for row in task.rows),
                                  claim_points=tuple(row.point for row in task.rows),task=task,
                                  task_ids=tuple('visible_'+str(i) for i in range(len(task.rows))))
        task_ids = tuple(task.daily_states)
        return ActivityScreen('tasks', daily=tuple(task.daily_states[key] for key in task_ids),
                              claim_points=tuple(task.claim_points[key] for key in task_ids), task=task,
                              task_ids=task_ids)
