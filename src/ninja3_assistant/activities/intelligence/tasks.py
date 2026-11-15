"""Dynamic daily-task and medal-reward recognition from actual activity samples.

Missing task rows remain unknown. Only the caller's confirmed claim lifecycle
may interpret a disappearing row as received; an unobserved row is never done.
"""
from ninja3_assistant.paths import resource_root
from dataclasses import dataclass, field
import re

import cv2
import numpy as np

from ninja3_assistant.activities.intelligence.ocr import compact_text, region_words

ROOT = resource_root()
TASK_IDS = ('games', 'damage')
TARGETS = {'games': 2, 'damage': 200, 'characters': 5}


@dataclass(frozen=True)
class DailyRow:
    state: str = 'unknown'
    point: tuple | None = None
    progress: tuple | None = None
    signature: bytes = b''


@dataclass(frozen=True)
class TaskObservation:
    kind: str = 'unknown'
    daily_states: dict = field(default_factory=lambda: dict.fromkeys(TASK_IDS, 'unknown'))
    claim_points: dict = field(default_factory=lambda: dict.fromkeys(TASK_IDS))
    present: tuple = ()
    header_bounds: tuple = ()  # Daily/achievement header bboxes, x,y,w,h.
    confirm_point: tuple | None = None
    progress: dict = field(default_factory=lambda: dict.fromkeys(TASK_IDS))
    rows: tuple = ()


class TaskVision:
    def __init__(self):
        self.templates = {}
        for name in ('daily-title', 'daily-games', 'daily-damage', 'daily-characters', 'achievement-title'):
            self.templates[name] = self._load(ROOT/'assets/intelligence'/f'{name}.png')
        for name in ('reward-title', 'reward-medal', 'reward-medal-daily', 'reward-confirm', 'claim-label', 'claimed-label'):
            self.templates[name] = self._load(ROOT/'assets/intelligence-tasks'/f'{name}.png')

    @staticmethod
    def _load(path):
        template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise RuntimeError('找不到活动任务识别图片：'+path.name)
        return template

    def _match(self, gray, name, region, threshold=.94):
        x, y, w, h = map(int, region)
        image, template = gray[y:y+h, x:x+w], self.templates[name]
        if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
            return None
        _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED))
        dx, dy = location
        patch = image[dy:dy+template.shape[0], dx:dx+template.shape[1]]
        if score < threshold or np.abs(patch.astype(np.int16)-template).mean() > 22:
            return None
        return (x+dx, y+dy, template.shape[1], template.shape[0])

    def detect(self, frame, ocr=None):
        if not isinstance(frame, np.ndarray) or frame.shape != (720, 1280, 3):
            return TaskObservation()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if (self._match(gray, 'reward-title', (520, 60, 260, 120)) and
                any(self._match(gray, name, (550, 230, 190, 190))
                    for name in ('reward-medal', 'reward-medal-daily')) and
                self._match(gray, 'reward-confirm', (570, 590, 160, 100))):
            return TaskObservation('reward', confirm_point=(640, 644))
        daily = self._match(gray, 'daily-title', (500, 95, 280, 90))
        achievement = self._match(gray, 'achievement-title', (500, 155, 280, 400))
        if not daily or not achievement or achievement[1] <= daily[1]+daily[3]:
            return TaskObservation()
        top = daily[1]+daily[3]+2
        bottom = achievement[1]-6
        states, points, progress = dict.fromkeys(TASK_IDS, 'unknown'), dict.fromkeys(TASK_IDS), dict.fromkeys(TASK_IDS)
        present = []
        for task_id in TARGETS:
            label = self._match(gray, 'daily-'+task_id, (285, top, 330, bottom-top))
            if not label:
                continue
            states[task_id], points[task_id], progress[task_id] = 'unknown', None, None
            present.append(task_id)
            center = label[1]+label[3]/2
            row_top, row_bottom = max(top, int(center-46)), min(bottom, int(center+46))
            row_region = (820, row_top, 270, row_bottom-row_top)
            claim = self._match(gray, 'claim-label', row_region)
            text = compact_text(region_words(ocr, *row_region)) if ocr else ''
            ratio = re.search(r'(\d+)/(\d+)', text.replace('／', '/'))
            if ratio and int(ratio[2]) == TARGETS[task_id] and int(ratio[1]) <= TARGETS[task_id]:
                progress[task_id] = (int(ratio[1]), int(ratio[2]))
            if claim:
                states[task_id] = 'claimable'
                points[task_id] = (claim[0]+claim[2]//2, claim[1]+claim[3]//2)
            elif '已领取' in text:
                states[task_id] = 'claimed'
            elif progress[task_id] and progress[task_id][0] < progress[task_id][1]:
                states[task_id] = 'incomplete'
        rows = []
        for row_top in range(top, bottom-60, 102):
            height=min(92,bottom-row_top)
            region=(820,row_top,270,height)
            text=compact_text(region_words(ocr,*region)) if ocr else ''
            claim=self._match(gray,'claim-label',region)
            claimed=self._match(gray,'claimed-label',region)
            ratio=re.search(r'(\d+)/(\d+)',text.replace('／','/'))
            value=None
            if ratio and 0 <= int(ratio[1]) <= int(ratio[2]) and int(ratio[2])>0:
                value=(int(ratio[1]),int(ratio[2]))
            state=('claimable' if claim else 'claimed' if claimed or '已领取' in text else
                   'incomplete' if value and value[0]<value[1] else 'unknown')
            point=(claim[0]+claim[2]//2,claim[1]+claim[3]//2) if claim else None
            label=gray[row_top+8:min(row_top+75,bottom),300:820]
            ink=(label<120).astype(np.uint8)
            ys,xs=np.nonzero(ink)
            if len(xs):
                ink=ink[ys.min():ys.max()+1,xs.min():xs.max()+1]
            signature=np.packbits(cv2.resize(ink,(200,24),
                                             interpolation=cv2.INTER_NEAREST)).tobytes()
            rows.append(DailyRow(state,point,value,signature))
        return TaskObservation('tasks', states, points, tuple(present), (daily, achievement),
                               progress=progress,rows=tuple(rows))

    @staticmethod
    def ocr_region(observation):
        """Source-image crop for the small daily digits and claim button text."""
        if observation.kind != 'tasks' or len(observation.header_bounds) != 2:
            return None
        daily, achievement = observation.header_bounds
        top, bottom = daily[1]+daily[3]+2, achievement[1]-6
        return (800, top, 290, bottom-top) if bottom > top else None

    def read(self, frame, engine, stop=None):
        """OCR only the confirmed dynamic daily area, mapping 3x words back."""
        observation = self.detect(frame)
        region = self.ocr_region(observation)
        if region is None or not (observation.present or observation.rows):
            return observation
        x, y, w, h = region
        crop = cv2.resize(frame[y:y+h, x:x+w], (w*3, h*3), interpolation=cv2.INTER_LINEAR)
        result = engine.read(crop, stop=stop)
        if result is None:
            return observation
        mapped = {'words': [dict(word, bbox=[word['bbox'][0]/3+x, word['bbox'][1]/3+y,
                                            word['bbox'][2]/3, word['bbox'][3]/3])
                            for word in result.get('words', [])]}
        return self.detect(frame, mapped)
