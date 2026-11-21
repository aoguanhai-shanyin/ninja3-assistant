"""The arena ranking reward entry and its confirmation page."""

from dataclasses import dataclass
import re

import cv2
import numpy as np

from ninja3_assistant.game_ui import GamePage, PageAction
from ninja3_assistant.vision.control_match import same_text
from ninja3_assistant.vision.game_update import game_update_banner


RANK_STRIPE_EDGE = (150, 417, 870, 427)
RANK_CLAIM_LABEL = (710, 427, 870, 462)
TITLE = (650, 82, 816, 140)
RANK_ROW = (302, 155, 615, 220)
CLAIM_BUTTON = (518, 546, 805, 611)
CLAIM_LABEL = (535, 562, 603, 600)
CLAIM_AMOUNT = (677, 550, 800, 612)
CLAIM_POINT = (657, 579)


def _amber_pixels(region):
    red, green, blue = region.astype(np.int16).transpose(2, 0, 1)
    return (red >= 200) & (green >= 100) & (green <= 205) & (blue <= 85) & (red - green >= 35)


def rank_reward_entry(frame):
    """Find the lobby's active reward entry within the ranking stripe."""
    if frame.shape != (720, 1280, 3):
        return None
    left, top, right, bottom = RANK_STRIPE_EDGE
    if _amber_pixels(frame[top:bottom, left:right]).mean() < .8:
        return None
    left, top, right, bottom = RANK_CLAIM_LABEL
    label = frame[top:bottom, left:right]
    # The countdown puts a dark timer capsule where the active label appears.
    if _amber_pixels(label).mean() < .45:
        return None
    glyphs = (label.min(axis=2) >= 195) & (np.ptp(label, axis=2) <= 45)
    columns = np.flatnonzero(glyphs.any(axis=0))
    rows = np.flatnonzero(glyphs.any(axis=1))
    if len(columns) < 88 or not 13 <= len(rows) <= 26 or glyphs.mean() > .28:
        return None
    return left + int((columns[0] + columns[-1]) // 2), top + int((rows[0] + rows[-1]) // 2)


def _claim_button_visible(frame):
    if frame.shape != (720, 1280, 3):
        return False
    left, top, right, bottom = CLAIM_BUTTON
    red, green, blue = frame[top:bottom, left:right].astype(np.int16).transpose(2, 0, 1)
    yellow = (red >= 215) & (green >= 153) & (blue <= 145) & (red - blue >= 75)
    left, top, right, bottom = CLAIM_LABEL
    glyphs = frame[top:bottom, left:right].max(axis=2) < 155
    return yellow.mean() >= .55 and .06 <= glyphs.mean() <= .45


def _same_dark_text(frame, fresh, box):
    x, y, width, height = box
    left, top, right, bottom = int(x), int(y), round(x+width), round(y+height)
    before = cv2.bitwise_not(frame[top:bottom, left:right])
    after = cv2.bitwise_not(fresh[top:bottom, left:right])
    local = (0, 0, right-left, bottom-top)
    # Compare both foregrounds so additional dark strokes also reject a tap.
    return same_text(before, after, local, max_error=5) and same_text(after, before, local, max_error=5)


@dataclass(frozen=True)
class ArenaRewardView:
    rank: int
    amount: int | None
    claim_point: tuple[int, int] | None
    title_box: tuple
    rank_box: tuple
    amount_box: tuple | None

    def claim_action(self, frame):
        if self.claim_point is None:
            return None

        def guard(fresh):
            if game_update_banner(fresh) or not _claim_button_visible(fresh):
                return False
            left, top, right, bottom = CLAIM_LABEL
            return (same_text(frame, fresh, self.title_box)
                    and _same_dark_text(frame, fresh, self.rank_box)
                    and _same_dark_text(frame, fresh, (left, top, right-left, bottom-top))
                    and _same_dark_text(frame, fresh, self.amount_box))

        return PageAction('claim_arena_reward', self.claim_point, guard=guard,
                          hold_seconds=.07, max_age=.25)


class ArenaRewardPage(GamePage):
    page_id = 'daily.arena_reward'

    @property
    def rank(self):
        return self.observation.detail.rank if self.observation.detail else None

    @property
    def amount(self):
        return self.observation.detail.amount if self.observation.detail else None

    @property
    def can_claim_reward(self):
        return 'claim_arena_reward' in self.observation.controls

    def claim_reward(self):
        return self._perform('claim_arena_reward')

    @staticmethod
    def observe(screen):
        title = screen.hits(r'^排名奖励$', TITLE)
        row = screen.hits(r'^第(\d{1,3})名排名奖励[!！]?$', RANK_ROW)
        if len(title) != 1 or len(row) != 1:
            return None
        rank = int(re.search(r'\d+', row[0].text)[0])
        if rank < 1:
            return None
        amount = screen.hits(r'^[xX×]\d{1,7}$', CLAIM_AMOUNT)
        value = int(amount[0].text[1:]) if len(amount) == 1 else None
        point = CLAIM_POINT if value and _claim_button_visible(screen.frame) else None
        return ArenaRewardView(rank, value, point, title[0].box, row[0].box,
                               amount[0].box if amount else None)
