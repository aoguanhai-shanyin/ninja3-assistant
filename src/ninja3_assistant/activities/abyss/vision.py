"""Recognize Abyss preparation menus and the observed 1280×720 battle HUD."""

import cv2
import numpy as np


TABS = ((290, 605), (655, 605), (1027, 605))
REWARD_ROWS = (305, 436)
REWARDS_CLOSE = (984, 201)
SKILLS = ((380, 628, 55), (522, 618, 47), (635, 618, 49), (900, 628, 55))
NINJUTSU = ((55, 62, 44), (74, 374, 57))
ANGLES = np.linspace(0, 2*np.pi, 48, endpoint=False)


def gold(pixels):
    r, g, b = np.moveaxis(pixels.astype(np.int16), -1, 0)
    return (r >= 216) & (g >= 135) & (r-g >= 14) & (r-b >= 75)


def selected_dungeon(frame):
    # The tab footer stays visible underneath the reward dialog.
    selected = [index for index, (x, _) in enumerate(TABS)
                if amber_footer(frame[700:716, x-110:x-75])]
    return selected[0] if len(selected) == 1 else None


def amber_footer(patch):
    r, g, b = patch.astype(np.int16).transpose(2, 0, 1)
    return ((r > 85) & (r-g > 18) & (g-b > 35)).mean() > .8


def preparation_page(frame):
    # The full-bright border, selected tab and dark progress bar identify
    # the list even when OCR drops a character in the progress label.
    return (frame is not None and selected_dungeon(frame) is not None and
            gold(frame[487:491, 130:1182]).mean() > .45 and
            dark_hud(frame, (165, 420, 430, 460)))


def reward_dungeon(frame):
    if frame is None or (selected := selected_dungeon(frame)) is None:
        return None
    # The two reward rows and the dark header retain their colors across
    # all three tabs and all claim states. No title OCR is needed for exit.
    for y, color in ((209, (88, 73, 104)), (365, (165, 149, 199)), (495, (165, 149, 199))):
        patch = frame[y-2:y+3, 511:570]
        if not np.all(np.abs(patch.mean(axis=(0, 1))-color) < 10):
            return None
    if (frame[180:219, 957:1009].min(axis=2) > 220).mean() < .23:
        return None
    return selected


def dark_hud(frame, box):
    x1, y1, x2, y2 = box
    return (frame[y1:y2, x1:x2].max(axis=2) < 65).mean() > .65


def relay_prompt(frame):
    patch = frame[48:95, 506:773].astype(np.int16)
    r, g, b = patch.transpose(2, 0, 1)
    return ((g > 210) & (g-r > 22) & (g-b > 100)).mean() > .15


def free_relay(frame, x):
    # Price figures are white with a dark outline. A single tall oval with
    # a single central hole distinguishes 0 from other costs.
    crop = frame[439:473, x-8:x+39]
    mask = cv2.inRange(crop, (213, 213, 213), (255, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 10]
    if len(digits) != 1:
        return False
    left, top, width, height = digits[0]
    if not (11 <= width <= 22 and 18 <= height <= 26):
        return False
    center = mask[top+height//3:top+2*height//3, left:left+width]
    rows = (center[:, :width//4].max(axis=1) > 0) & (center[:, -width//4:].max(axis=1) > 0)
    return rows.mean() > .8 and (center[:, width//3:2*width//3] == 0).mean() > .8


def result_panel(frame):
    return (dark_hud(frame, (270, 253, 1000, 527)) and
            (frame[534:537, 80:1200].min(axis=2) > 185).mean() > .2)


def read_reward(screen, y):
    label = screen.text((862, y-20, 982, y+21))
    if '已领取' in label or '已领' in label:
        return 'claimed'
    patch = screen.frame[y-12:y+12, 873:971]
    if gold(patch).mean() > .25:
        return 'claim'
    # Claimed rows replace the button with brown text on the purple panel.
    # The full progress bar distinguishes the stamp from an empty panel.
    r, g, b = patch.astype(np.int16).transpose(2, 0, 1)
    purple = (b-r > 20) & (r-g > 8)
    brown = (r < 180) & (r-g > 40) & (g-b > 25)
    if (purple.mean() > .4 and brown.mean() > .12 and
            gold(screen.frame[y-8:y+8, 823:845]).mean() > .85):
        return 'claimed'
    neutral = (np.ptp(patch, axis=2) < 16) & (patch.min(axis=2) > 120)
    if neutral.mean() > .6:
        return 'pending'
    return None


def ring(frame, x, y, radius):
    radii = np.arange(radius-4, radius+5)[:, None]
    columns = np.rint(x+radii*np.cos(ANGLES)).astype(np.intp)
    rows = np.rint(y+radii*np.sin(ANGLES)).astype(np.intp)
    valid = (columns >= 0) & (rows >= 0)
    return np.where(valid[..., None], frame[rows.clip(min=0), columns.clip(min=0)], 0)


def ready_ninjutsu(frame):
    # A charged ninjutsu has a filled golden ring. The relay control is lower
    # and larger. Its unfilled part is white even while it is still charging.
    for x, y, radius in NINJUTSU:
        if gold(ring(frame, x, y, radius)).any(axis=0).mean() >= .88:
            return (x, y)
    return None


def visible_ninjutsu(frame):
    # A brief tap on the confirmed control also covers characters whose
    # ready animation overlays the gold ring.
    for x, y, radius in reversed(NINJUTSU):
        samples = ring(frame, x, y, radius)
        light = gold(samples) | (samples.min(axis=2) >= 223)
        edge = ring(frame, x, y, radius-9).max(axis=2) < 65
        if light.any(axis=0).mean() >= .70 and edge.any(axis=0).mean() >= .6:
            return (x, y)
    return None


def ready_weapons(frame):
    for x, y, radius in SKILLS:
        samples = ring(frame, x, y, radius)
        if (samples.min(axis=2) >= 221).any(axis=0).mean() >= .88:
            # A contrasting inner edge distinguishes the glow from scenery.
            inner = ring(frame, x, y, radius-12)
            r, g, b = np.moveaxis(samples.astype(np.int16), -1, 0)
            disabled = ((r >= 160) & (r-g >= 65) & (r-b >= 65)).any(axis=0).mean() > .6
            if not disabled and (inner.max(axis=2) < 110).any(axis=0).mean() >= .55:
                yield (x, y)


def nearby_ground_hazard(frame):
    """Red eyes of fish and dark enemies in the approach to the runner."""
    r, g, b = frame[350:490, 407:537].astype(np.int16).transpose(2, 0, 1)
    return np.count_nonzero((r > 195) & (r-g > 95) & (r-b > 100)) >= 20


def same_region(frame, fresh, box, threshold=8):
    x1, y1, x2, y2 = box
    return fresh is not None and max(cv2.mean(cv2.absdiff(
        frame[y1:y2, x1:x2], fresh[y1:y2, x1:x2]))[:3]) <= threshold
