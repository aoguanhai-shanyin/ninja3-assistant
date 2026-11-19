"""Read the four Xumi daily requirements and their independent reward states."""
from dataclasses import dataclass
import re

import cv2

from ninja3_assistant.activities.daily.vision import DailyScreen


_TITLE = (470, 90, 760, 150)
_LABELS = (365, 170, 760, 590)
_CONTROLS = (780, 171, 1060, 590)
_MODES = {'方寸': 'fangcun', '大千': 'daqian'}
_REQUIREMENTS = (
    ('match', r'完[成咸]1次(?:方寸|大千)试炼', 1),
    ('score', r'(?:方寸|大千)试炼累计获得50000分', 50000),
)
_EXPECTED = {(mode, kind) for mode in _MODES.values() for kind, _, _ in _REQUIREMENTS}
_PROGRESS = r'(?<![\d/])(\d+)/(\d+)(?![\d/])'


@dataclass(frozen=True)
class XumiTaskRow:
    mode: str
    kind: str
    current: int | None
    goal: int
    claim_point: tuple | None = None
    claimed: bool = False


def read_tasks(screen):
    """Return known rows from a visible panel; missed values remain unknown."""
    if not screen.hits('每日任务', _TITLE):
        return ()
    rows = []
    for kind, pattern, goal in _REQUIREMENTS:
        for label in screen.hits(pattern, _LABELS):
            mode = _MODES['方寸' if '方寸' in label.text else '大千']
            _, y = label.point
            x1, y1, x2, y2 = _CONTROLS
            bounds = (x1, max(y1, y-48), x2, min(y2, y+48))
            text = screen.text(bounds).replace('／', '/')
            counts = re.findall(_PROGRESS, text)
            current = None
            if len(counts) == 1:
                count, limit = map(int, counts[0])
                if limit == goal and count <= goal:
                    current = count
            status = re.sub(_PROGRESS, '', text)
            claim_status = not counts or current == goal
            claimed = status == '已领取' and claim_status
            claim = screen.hits(r'(?<!已)领取(?:奖励)?', bounds) if (
                status in ('领取', '领取奖励') and claim_status) else []
            rows.append(XumiTaskRow(mode, kind, current, goal, claim[0].point if claim else None, claimed))
    return tuple(row for row in rows if sum((other.mode, other.kind) == (row.mode, row.kind) for other in rows) == 1)


def read_task_panel(screen, ocr, stop=None):
    """Read small progress digits and unresolved claim labels at a larger scale."""
    rows = read_tasks(screen)
    def needs_crop(row):
        return _unresolved(row) or (row.kind == 'score' and not row.claimed and row.claim_point is None)

    if not any(needs_crop(row) for row in rows):
        return rows
    x1, y1, x2, y2 = _CONTROLS
    crop = cv2.resize(screen.frame[y1:y2, x1:x2], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    result = ocr.read(crop, stop=stop)
    if result is None:
        return rows
    words = [word for word in screen.ocr.get('words', ())
             if not (x1 <= word['bbox'][0]+word['bbox'][2]/2 < x2 and
                     y1 <= word['bbox'][1]+word['bbox'][3]/2 < y2)]
    for word in result.get('words', ()):
        x, y, w, h = word['bbox']
        words.append({**word, 'bbox': (x1+x/3, y1+y/3, w/3, h/3)})
    found = read_tasks(DailyScreen(screen.frame, {'words': words}, screen.page))
    by_key = {(row.mode, row.kind): row for row in found if not _unresolved(row)}
    return tuple(by_key.get((row.mode, row.kind), row) if needs_crop(row) else row for row in rows)


def _unresolved(row):
    return not row.claimed and row.claim_point is None and row.current in (None, row.goal)


def tasks_claimed(rows):
    return len(rows) == len(_EXPECTED) and {(row.mode, row.kind) for row in rows if row.claimed} == _EXPECTED
