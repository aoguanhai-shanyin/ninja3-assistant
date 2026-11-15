"""Experimental coarse fire setup cue. Not a numbered-board classifier."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class WindupCue:
    family: str
    response_class: str | None
    observed_at: float
    first_observed_at: float
    center: tuple[int, int]
    warm_pixels: int
    experimental: bool = True


class FireSetupCue:
    """Observe persistent warm effects opposite the player during a battle.

    This experimental signal does not identify fire1/2, nor guarantee an
    unobstructed boss pose. Callers must separately gate any trial actions.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.previous_at = None
        self.fired = False

    def update(self, rgb, screen, now):
        if not screen.active or screen.result:
            self.reset()
            return None
        if screen.player_x is None:
            self.previous = self.previous_at = None
            return None
        if self.fired:
            return None
        # The warm mask is limited to rows 155..489.  The 70-pixel halo on
        # either side preserves the original 140-pixel box-filter results.
        top, bottom = 85, 560
        hsv = cv2.cvtColor(rgb[top:bottom], cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (2, 65, 170), (28, 255, 255))
        mask[:155-top] = 0
        mask[490-top:] = 0
        mask[:, :130] = 0
        mask[:, 1250:] = 0
        # Remove the fixed left-hand skill control, then inspect both sides.
        mask[245-top:490-top, :165] = 0
        mask[:230-top, 220:] = 0
        x = float(screen.player_x)
        mask[:, max(0, int(x - 300)):min(1280, int(x + 300))] = 0
        crop = hsv[155-top:490-top]
        red_low = cv2.inRange(crop, (0, 161, 101), (6, 255, 255))
        red_high = cv2.inRange(crop, (173, 161, 101), (179, 255, 255))
        red = cv2.countNonZero(cv2.bitwise_or(red_low, red_high))
        if red > .12 * crop.shape[0] * crop.shape[1]:
            self.previous = self.previous_at = None
            return None
        density = cv2.boxFilter((mask > 0).astype(np.float32), -1, (180, 140), normalize=False)
        _, value, _, center = cv2.minMaxLoc(density)
        center = (center[0], center[1] + top)
        if value < 60:
            self.previous = self.previous_at = None
            return None
        if (self.previous is None or self.previous_at is None or
                not 0 < now - self.previous_at <= .45 or
                abs(center[0]-self.previous[0]) > 100 or
                abs(center[1]-self.previous[1]) > 90):
            self.previous, self.previous_at = center, now
            return None
        first = self.previous_at
        self.fired = True
        return WindupCue('fire', None, now, first, center, int(value))
