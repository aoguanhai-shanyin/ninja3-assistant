"""Recognize the observed Boss, rope position and combat controls."""
from dataclasses import dataclass

import cv2
import numpy as np

from ninja3_assistant.vision.vision import ASSETS


@dataclass(frozen=True)
class BossScreen:
    target_boss: bool = False
    rope_y: int | None = None
    side: str = 'unknown'
    player_x: int | None = None
    weapon_ready: bool = False


class BossScreenDetector:
    def __init__(self):
        names = ('boss-zen-heart', 'battle-jump', 'battle-up', 'weapon-ready',
                 'avatar-above-1', 'avatar-above-2', 'avatar-above-3',
                 'avatar-below', 'avatar-below-1', 'avatar-below-2')
        self.images = {name: cv2.imread(str(ASSETS / (name + '.png')), cv2.IMREAD_GRAYSCALE) for name in names}
        if any(image is None for image in self.images.values()):
            raise RuntimeError('Boss 实验所需的识别模板不完整')

    def match(self, gray, name, bounds):
        x, y, w, h = bounds
        template = self.images[name]
        area = gray[y:y+h, x:x+w]
        if area.shape[0] < template.shape[0] or area.shape[1] < template.shape[1]:
            return 0, None
        _, score, _, pos = cv2.minMaxLoc(cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED))
        return score, (x + pos[0], y + pos[1])

    def detect(self, frame):
        if frame.shape != (720, 1280, 3):
            return BossScreen()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if gray.mean() < 35:
            return BossScreen()
        boss_score, boss_at = self.match(gray, 'boss-zen-heart', (370, 20, 90, 80))
        target = boss_score >= .88
        if target:
            x, y = boss_at
            patch = gray[y:y+53, x:x+55].astype(np.int16)
            target = np.abs(patch - self.images['boss-zen-heart'].astype(np.int16)).mean() < 30

        dark = (gray[200:530, 100:1180] < 30).mean(axis=1)
        smooth = np.convolve(dark, np.ones(5) / 5, mode='same')
        contrast = smooth - (np.roll(smooth, 9) + np.roll(smooth, -9)) / 2
        index = int(np.argmax(contrast[10:-10])) + 10
        rope_y = index + 200 if smooth[index] >= .7 and contrast[index] >= .45 else None

        side = 'unknown'
        player_x = None
        if rope_y is not None:
            up, _ = self.match(gray, 'battle-up', (1010, 550, 180, 150))
            jump, _ = self.match(gray, 'battle-jump', (1010, 550, 180, 150))
            if up >= .9:
                side = 'below'
                names = ('avatar-below', 'avatar-below-1', 'avatar-below-2')
                bounds = (120, max(100, rope_y - 20), 430, min(690, rope_y + 220) - max(100, rope_y - 20))
            elif jump >= .85:
                side = 'above'
                names = ('avatar-above-1', 'avatar-above-2', 'avatar-above-3')
                bounds = (120, max(100, rope_y - 250), 430, min(690, rope_y + 80) - max(100, rope_y - 250))
            else:
                names = ()
            best_score = 0
            for name in names:
                score, pos = self.match(gray, name, bounds)
                if score > best_score:
                    best_score = score
                    player_x = pos[0] + self.images[name].shape[1] // 2
                if score >= .9:
                    break
            if best_score < .74:
                player_x = None

        weapon, _ = self.match(gray, 'weapon-ready', (310, 560, 160, 150))
        return BossScreen(target, rope_y, side, player_x, weapon >= .93)
