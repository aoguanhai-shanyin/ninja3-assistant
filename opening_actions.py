"""Ground acceleration and repeated ninjutsu input during battle."""
from dataclasses import dataclass
import os

import cv2
import numpy as np

from vision import ASSETS


@dataclass(frozen=True)
class OpeningAction:
    kind: str
    point: tuple[int, int] | None = None


class OpeningActions:
    def __init__(self):
        self.images = {name: cv2.imread(str(ASSETS/(name+'.png')), cv2.IMREAD_GRAYSCALE)
                       for name in ('battle-slide', 'battle-jump', 'ninjutsu-hand', 'ninjutsu-icon')}
        if any(image is None for image in self.images.values()):
            raise RuntimeError('开局动作所需模板不完整')
        self.reset()

    def reset(self):
        self.ground_frames = 0
        self.last_skill_action = float('-inf')
        self.last_completed_kind = None

    def _match(self, gray, name, bounds):
        x, y, w, h = bounds
        area = gray[y:y+h, x:x+w]
        template = self.images[name]
        _, score, _, at = cv2.minMaxLoc(cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED))
        height, width = template.shape
        patch = area[at[1]:at[1]+height, at[0]:at[0]+width]
        error = float(np.abs(patch.astype(np.float32)-template).mean())
        return score, error, (x+at[0], y+at[1])

    def _ninjutsu_point(self, gray):
        # The same character's button has been observed in both positions
        # during boss combat; choose the visible icon rather than the scene.
        top = self._match(gray, 'ninjutsu-icon', (0, 0, 180, 140))[0]
        lower = self._match(gray, 'ninjutsu-icon', (0, 260, 180, 160))[0]
        if max(top, lower) < .4:
            return None
        return (55, 60) if top >= lower else (70, 330)

    def decide(self, frame, now, *, boss_active=False, battle_allowed=True):
        if not battle_allowed or not isinstance(frame, np.ndarray) or frame.shape != (720, 1280, 3):
            self.ground_frames = 0
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        ground = False
        if not boss_active:
            slide = self._match(gray, 'battle-slide', (80,550,230,165))
            jump = self._match(gray, 'battle-jump', (1010,550,180,165))
            ground = slide[0] >= .93 and slide[1] < 25 and jump[0] >= .93 and jump[1] < 25
        self.ground_frames = self.ground_frames + 1 if ground else 0
        active = boss_active or ground
        if not active:
            return None
        ground_ready = ground and self.ground_frames >= 2
        skill_due = now-self.last_skill_action >= .25
        if skill_due and not (ground_ready and self.last_completed_kind == 'ninjutsu'):
            point = self._ninjutsu_point(gray)
            if point is not None:
                return OpeningAction('ninjutsu', point)
        if ground_ready:
            return OpeningAction('ground_jump_slide')
        return None

    def scheduled(self, action, now):
        """Rate-limit a queued action without claiming the game accepted it."""
        self.last_completed_kind = action.kind
        if action.kind == 'ninjutsu':
            self.last_skill_action = now

    def completed(self, action, now):
        self.scheduled(action, now)


def execute_opening(action, device, control, emit):
    """Execute only an already-gated opening action; report SDK presses, not effects."""
    def pressed(kind):
        emit('opening_action', {'kind': kind, 'stage': 'sdk_pressed', 'time': control.now()})

    if action.kind == 'ninjutsu':
        def skill_pressed():
            emit('opening_action', {'kind': 'ninjutsu', 'stage': 'sdk_pressed',
                                    'time': control.now(), 'point': action.point})
        return device.tap(action.point, control.stop, hold_seconds=.015,
                          on_pressed=skill_pressed)
    if action.kind != 'ground_jump_slide':
        raise ValueError('unknown opening action')
    if os.environ.get('NINJA_SKATE_MULTITOUCH', '1') == '1':
        from mumu_multitouch import ExperimentalMultiTouch
        # Set NINJA_SKATE_MULTITOUCH=0 to use the single-finger fallback.
        return ExperimentalMultiTouch(device).run(
            control.stop, lambda event: emit('opening_action', {'kind': 'multitouch', **event}))
    # The tutorial shows slide -> jump -> slide cancellation. The SDK exposes
    # one contact, so this sequential candidate is not a proven two-finger overlap.
    if not device.swipe((190,620), (190,700), control.stop, step_seconds=.005, steps=4,
                        on_pressed=lambda: pressed('ground_slide_start')):
        return False
    if control.stop.is_set():
        return False
    if not device.tap((1090,630), control.stop, hold_seconds=.015,
                      on_pressed=lambda: pressed('ground_jump')):
        return False
    if control.stop.is_set():
        return False
    return device.swipe((190,620), (190,700), control.stop, step_seconds=.005, steps=4,
                        on_pressed=lambda: pressed('ground_slide_cancel'))
