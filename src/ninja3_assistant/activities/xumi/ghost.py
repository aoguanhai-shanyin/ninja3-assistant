"""Avoid the opening mirrored blades of Ghost in Xumi rope combat."""

from dataclasses import dataclass

import cv2
import numpy as np


_GHOST_PORTRAIT_HASH = 0xe6ada751d91556d1
_FLIP_UP = (1088, 630)
_FLIP_DOWN = (190, 630)
_CLOSE_GAPS = np.ones((5, 5), np.uint8)
_CLOSE_SPRITE_GAPS = np.ones((3, 3), np.uint8)


def _ghost_portrait(frame):
    # A compact contrast descriptor of the fixed inner portrait excludes its
    # animated rim, level, and health. Reference: the frame before the first
    # collision in the 20261119-172722 Fangcun replay (battle-0089.png).
    gray = cv2.cvtColor(frame[32:78, 384:436], cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    fingerprint = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), 'big')
    return (fingerprint ^ _GHOST_PORTRAIT_HASH).bit_count() <= 4


def _rope_profile(frame, left, right):
    region = frame[280:510, left:right:3]
    dark = (region.max(axis=2) < 38).mean(axis=1)
    smooth = np.convolve(dark, np.full(5, .2), mode='same')
    contrast = smooth - (np.roll(smooth, 10) + np.roll(smooth, -10)) / 2
    y = int(contrast[12:-12].argmax()) + 12
    return y+280, smooth[y], contrast[y]


def _rope_height(frame):
    y, dark, contrast = _rope_profile(frame, 130, 1150)
    if dark >= .65 and contrast >= .36:
        return y
    # The dark-red cloud can cover the left segment. Two agreeing exposed
    # segments still locate the rope before the upper blade reaches the lead.
    segments = [_rope_profile(frame, left, right) for left, right in ((130, 440), (440, 760), (760, 1150))]
    strongest = max(segments, key=lambda segment: segment[2])
    if strongest[2] < .8:
        return None
    if sum(abs(height-strongest[0]) <= 3 and edge >= .34 for height, _, edge in segments) >= 2:
        return strongest[0]
    # A largely unobstructed segment can still agree with the weaker full
    # horizontal profile while the Boss effect covers both other segments.
    if strongest[2] >= .9 and dark >= .45 and contrast >= .42 and abs(y-strongest[0]) <= 3:
        return strongest[0]
    return None


def _player_below(frame, rope_y):
    # The local health bar only appears after damage. On the opening frames
    # the lead's rope pose provides position until then.
    left, right = 180, 945
    bars = frame[rope_y-44:rope_y-16, left:right]
    fill = cv2.inRange(bars, (200, 95, 0), (255, 235, 120))
    _, _, components, _ = cv2.connectedComponentsWithStats(fill)
    centers = []
    for x, y, w, h, area in components[1:]:
        if 8 <= w <= 32 and 3 <= h <= 5 and area >= w*h*.7 and 0 < y < len(bars)-h:
            border = bars[[y-1, y+h], x:x+w]
            if (border.max(axis=2) < 145).mean() >= .8:
                # The gold fill starts at the fixed left edge of the bar.
                centers.append(int(left+x+16))
    if centers:
        return centers[0] if len(centers) == 1 else None

    region = frame[rope_y-5:rope_y+65, left:right]
    dimmest = region.min(axis=2)
    white = ((dimmest >= 190) & (region.max(axis=2)-dimmest <= 42)).astype(np.uint8)
    _, _, components, _ = cv2.connectedComponentsWithStats(white)
    hooks = [int(left+x+w//2) for x, y, w, h, area in components[1:]
             if 1 <= y <= 12 and 10 <= w <= 28 and 21 <= h <= 43 and .24 <= area/(w*h) <= .7]
    if hooks:
        return hooks[0] if len(hooks) == 1 else None
    blue_lead = _cyan_player_below(frame[rope_y-52:rope_y+102, left:right], left)
    if blue_lead is not None:
        return blue_lead
    return _tucked_player_below(frame, rope_y, left, components[1:])


def _cyan_silhouette(region, *, marker_rows, min_cyan):
    red, green, blue = region.astype(np.int16).transpose(2, 0, 1)
    cyan = (blue >= 140) & (green >= 135) & (green-red >= 20) & (blue-red >= 24) & (blue-green <= 80)
    if np.count_nonzero(cyan[:marker_rows]) < min_cyan:
        return None
    silver = ((red >= 155) & (green >= 170) & (blue >= 175)
              & (blue-red <= 94) & (green-red <= 68) & (red-green <= 45))
    mask = cv2.morphologyEx((cyan | silver).astype(np.uint8), cv2.MORPH_CLOSE, _CLOSE_SPRITE_GAPS)
    _, _, components, _ = cv2.connectedComponentsWithStats(mask)
    return cyan, components[1:]


def _cyan_player_below(region, left):
    # Daqian's blue lead has a cyan shape at the rope and distinct silver
    # face/armor below it. The detached cyan effect alone gives no player x.
    silhouette = _cyan_silhouette(region, marker_rows=55, min_cyan=100)
    if silhouette is None:
        return None
    cyan, components = silhouette
    rope = []
    armor = []
    heads = []
    for x, y, w, h, area in components:
        center = int(x + w//2)
        if (y <= 29 and 35 <= w <= 65 and 24 <= h <= 55 and 340 <= area <= 1100
                and 44 <= y+h <= 68 and np.count_nonzero(cyan[y:y+h, x:x+w]) >= 100):
            rope.append(center)
        if 72 <= y <= 92 and 14 <= w <= 34 and 16 <= h <= 37 and 145 <= area <= 380:
            armor.append(center)
        if 97 <= y <= 118 and 25 <= w <= 42 and 25 <= h <= 46 and 380 <= area <= 980:
            heads.append(center)
    leads = {(body+head)//2 for head in heads for body in armor if 7 <= head-body <= 30
             and any(10 <= marker-head <= 48 for marker in rope)}
    return left+leads.pop() if len(leads) == 1 else None


def _tucked_player_below(frame, rope_y, left, components):
    # Fangcun's flame lead momentarily tucks its white boot below the rope.
    # The boot swings past the clothing. Both need to align with the face.
    boots = [int(left+x+w//2) for x, y, w, h, area in components
             if 16 <= y <= 28 and 20 <= w <= 32 and 18 <= h <= 31 and 165 <= area <= 280]
    clothing = [int(left+x+w//2) for x, y, w, h, area in components
                if 55 <= y <= 63 and 9 <= w <= 21 and 5 <= h <= 11 and 24 <= area <= 85]
    leads = set()
    for boot in boots:
        if not any(-6 <= boot-body <= 25 for body in clothing):
            continue
        face = frame[rope_y+55:rope_y+79, boot:boot+40].astype(np.int16)
        red, green, blue = face.transpose(2, 0, 1)
        warm = ((red >= 150) & (green >= 105) & (blue >= 65) & (red-green >= 10)
                & (green-blue >= 9) & (red-blue <= 85)).astype(np.uint8)
        _, _, patches, _ = cv2.connectedComponentsWithStats(warm)
        for x, _, w, h, area in patches[1:]:
            face_offset = int(x+w//2)
            if 8 <= face_offset <= 33 and 13 <= w <= 21 and 10 <= h <= 17 and 85 <= area <= 160:
                leads.add(boot+face_offset-12)
    return leads.pop() if len(leads) == 1 else None


def _player_above(frame, rope_y, recent_x):
    # The standing lead has a small pale head and a second, lower clothing
    # component. Once tracked, the lower component can outlast a covered head.
    left = 180 if recent_x is None else max(180, recent_x-75)
    right = 945 if recent_x is None else min(945, recent_x+75)
    if right-left < 30:
        return None, False
    region = frame[rope_y-82:rope_y-2, left:right]
    dimmest = region.min(axis=2)
    white = ((dimmest >= 180) & (region.max(axis=2)-dimmest <= 55)).astype(np.uint8)
    _, _, components, _ = cv2.connectedComponentsWithStats(white)
    heads = [int(left+x+w//2) for x, y, w, h, area in components[1:]
             if 8 <= y <= 22 and 12 <= w <= 22 and 8 <= h <= 13 and 65 <= area <= 135]
    bodies = [int(left+x+w//2) for x, y, w, h, area in components[1:]
              if 37 <= y <= 52 and 10 <= w <= 38 and 14 <= h <= 38 and 90 <= area <= 360]
    players = [head for head in heads if any(head-48 <= body <= head+5 for body in bodies)]
    if len(players) == 1:
        return players[0], True
    if not players and recent_x is not None and any(abs(recent_x-body) <= 42 for body in bodies):
        return recent_x, False
    return None, False


def _cyan_player_above(frame, rope_y):
    left = 180
    silhouette = _cyan_silhouette(frame[rope_y-128:rope_y-12, left:945], marker_rows=78, min_cyan=250)
    if silhouette is None:
        return None
    cyan, components = silhouette
    heads = []
    armor = []
    for x, y, w, h, area in components:
        center = int(x+w//2)
        if (9 <= y <= 27 and 54 <= w <= 74 and 53 <= h <= 72 and 950 <= area <= 1580
                and np.count_nonzero(cyan[y:y+h, x:x+w]) >= 290):
            heads.append(center)
        if (64 <= y <= 86 and 16 <= w <= 77 and 12 <= h <= 38 and 150 <= area <= 560
                and np.count_nonzero(cyan[y:y+h, x:x+w]) >= 45):
            armor.append(center)
    leads = [head for head in heads if any(17 <= head-body <= 68 for body in armor)]
    if leads:
        # The connected upper shape includes the companion to the right.
        return left+leads[0]-27 if len(leads) == 1 else None

    # During an arm swing the armor and head can touch. Isolate their two
    # bands and still require both shapes and their relative x positions.
    head_band = _cyan_silhouette(frame[rope_y-120:rope_y-57, left:945], marker_rows=63, min_cyan=350)
    armor_band = _cyan_silhouette(frame[rope_y-65:rope_y-24, left:945], marker_rows=41, min_cyan=140)
    if head_band is None or armor_band is None:
        return None
    cyan, components = head_band
    heads = [int(x+w//2) for x, y, w, h, area in components
             if 6 <= y <= 17 and 54 <= w <= 74 and 45 <= h <= 58 and 950 <= area <= 1520
             and np.count_nonzero(cyan[y:y+h, x:x+w]) >= 350]
    cyan, components = armor_band
    armor = [int(x+w//2) for x, y, w, h, area in components
             if y <= 9 and 62 <= w <= 94 and 25 <= h <= 41 and 460 <= area <= 930
             and np.count_nonzero(cyan[y:y+h, x:x+w]) >= 140]
    leads = [head for head in heads if any(23 <= head-body <= 65 for body in armor)]
    return left+leads[0]-27 if len(leads) == 1 else None


def _red(frame):
    red, green, blue = frame.astype(np.int16).transpose(2, 0, 1)
    return ((red >= 145) & (red-green >= 90) & (red-blue >= 63)).astype(np.uint8)


def _blade_boxes(mask, top):
    _, _, components, _ = cv2.connectedComponentsWithStats(mask)
    return [(int(x), int(y+top), int(w), int(h)) for x, y, w, h, area in components[1:]
            if 85 <= w <= 190 and 72 <= h <= 175 and area >= 2500]


def _mirrored_blade(frame, rope_y, player_x):
    # The opening has two broad blade orbs in matching columns far apart in
    # height. Individual straight swords and the player's flames are smaller.
    left, right = max(180, player_x+70), min(1210, player_x+690)
    if right-left < 190:
        return None
    bottom = min(565, rope_y+185)
    red = cv2.morphologyEx(_red(frame[:bottom, left:right]), cv2.MORPH_CLOSE, _CLOSE_GAPS)
    upper = _blade_boxes(red[:rope_y-160], 0)
    lower = _blade_boxes(red[rope_y-4:], rope_y-4)
    candidates = []
    for x, y, w, h in lower:
        if x < 8 or x+w > right-left-8 or not rope_y+30 <= y+h/2 <= rope_y+135:
            continue
        if any(abs((x+w/2)-(xx+ww/2)) <= 65 for xx, _, ww, _ in upper):
            candidates.append((left+x, y+h//2))
    return min(candidates, default=None)


def _upper_blade(frame, rope_y, player_x):
    left = player_x-25
    region = frame[rope_y-144:rope_y+2, left:min(1210, player_x+630)]
    if region.shape[1] < 160:
        return None
    mask = cv2.morphologyEx(_red(region), cv2.MORPH_CLOSE, _CLOSE_GAPS)
    _, _, components, _ = cv2.connectedComponentsWithStats(mask)
    # Ignore the distant high orb and the cloud spreading behind the player.
    return min(((int(left+x), int(rope_y-144+y+h//2)) for x, y, w, h, area in components[1:]
                if 95 <= w <= 220 and h >= 62 and area >= 5000 and y+h >= 95 and left+x+w >= player_x+75),
               default=None)


@dataclass(frozen=True, slots=True)
class _Approach:
    captured_at: float
    player_x: int
    blade_x: int
    blade_y: int
    point: tuple[int, int]


class GhostAvoidance:
    """Propose one verified flip away from an approaching opening blade."""

    def __init__(self, down_template, jump_template=None):
        self.down = down_template
        self.up = cv2.flip(down_template, 0)
        self.jump = jump_template
        self.reset()

    def reset(self):
        self._previous = None
        self._above_position = None
        self._last_frame = float('-inf')
        self._proposed_at = None
        self._accepted_at = None

    def accepted(self):
        """Acknowledge the caller's native input; wait before proposing a retry."""
        if self._proposed_at is not None:
            self._accepted_at = self._proposed_at
            self._proposed_at = None
            self._previous = None
            self._above_position = None

    @staticmethod
    def _arrow(frame, x, template):
        region = cv2.cvtColor(frame[570:690, x-65:x+65], cv2.COLOR_RGB2GRAY)
        glyph = cv2.inRange(region, 200, 255)
        return cv2.minMaxLoc(cv2.matchTemplate(glyph, template, cv2.TM_CCOEFF_NORMED))[1] >= .9

    def update(self, frame, captured_at):
        """Return the confirmed control point or None. Dispatch belongs to the caller."""
        if captured_at <= self._last_frame:
            return None
        self._last_frame = captured_at
        self._proposed_at = None
        if (not isinstance(frame, np.ndarray) or frame.shape != (720, 1280, 3)
                or not _ghost_portrait(frame) or not self._arrow(frame, _FLIP_DOWN[0], self.down)):
            self._previous = None
            self._above_position = None
            return None
        below = self._arrow(frame, _FLIP_UP[0], self.up)
        if not below and (self.jump is None or not self._arrow(frame, _FLIP_UP[0], self.jump)):
            self._previous = None
            self._above_position = None
            return None
        if self._accepted_at is not None and captured_at-self._accepted_at < .8:
            return None

        rope_y = _rope_height(frame)
        if rope_y is None:
            self._previous = None
            return None
        point = _FLIP_UP if below else _FLIP_DOWN
        if below:
            self._above_position = None
            player_x = _player_below(frame, rope_y)
            blade = _mirrored_blade(frame, rope_y, player_x) if player_x is not None else None
        else:
            recent = self._above_position
            recent_x = recent[0] if recent is not None and captured_at-recent[1] <= .36 else None
            player_x, confirmed = _player_above(frame, rope_y, recent_x)
            if confirmed:
                self._above_position = player_x, captured_at
            elif player_x is None:
                player_x = _cyan_player_above(frame, rope_y)
            blade = _upper_blade(frame, rope_y, player_x) if player_x is not None else None
        if blade is None:
            self._previous = None
            return None

        sample = _Approach(captured_at, player_x, *blade, point)
        previous = self._previous
        if previous is not None and previous.point != point:
            previous = None
        elapsed = captured_at-previous.captured_at if previous else 0
        if previous is None or elapsed >= .06:
            self._previous = sample
        if previous is None or not .06 <= elapsed <= .36:
            return None
        if abs(player_x-previous.player_x) > 65 or abs(sample.blade_y-previous.blade_y) > 65:
            return None

        gap = sample.blade_x-player_x
        closing = (previous.blade_x-previous.player_x-gap) / elapsed
        reach = 300 if below else 220
        if not (-25 <= gap <= reach and closing >= 130 and (gap-45)/closing <= .65):
            return None

        # Check the crossing and opposite landing corridor for red effects.
        path = slice(rope_y-122, rope_y+10) if below else slice(rope_y+6, min(566, rope_y+132))
        landing = _red(frame[path, player_x-90:player_x+90])
        if landing.mean() > .035:
            return None
        self._proposed_at = captured_at
        return point
