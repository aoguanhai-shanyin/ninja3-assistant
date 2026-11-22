"""Weekend race driven by the project-wide WinML matcher and battle kernel.

One graph (``assets/template-match.onnx``) answers every question this module
asks: which page is on screen, where the obstacles are, whether the weapon is
armed. There is no per-rotation model, no OpenCV template path and no
"legacy vs current" branch: a new rotation adds a template entry and nothing
else. Movement, timing and input bounds stay in ``RealtimeBattleKernel``.

Scene judgement is measured, not guessed
----------------------------------------
The counts and thresholds below come from the 2026-11-21 live evidence and the
2026-11-15 recordings:

* ``race-distance`` scores 0.98..1.00 on every race frame and 0.00..0.52 on
  every menu page, so 0.93 confirms a race without a per-map switch. When a
  smoke card covers the board, ``race-effect`` (0.93..1.00 on race frames)
  confirms the older rotation instead.
* The ability cluster's bright-pixel ratio is 0.57..0.79 when a slot is armed
  and 0.00..0.32 while cooling, on every observed character skin. The icon art
  changes with the character, so a template match cannot do this job; the glow
  can.
* A dark-line test cannot separate ropes from water: the water map scores 72
  dark rows against 8 on the rope map. Ropes, boardwalks and ice are therefore
  treated as the same terrain, and obstacles are what the runner reacts to.
"""
from ninja3_assistant.paths import resource_root
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import cv2
import numpy as np

from ninja3_assistant.activities.abyss.vision import gold, ring
from ninja3_assistant.activities.intelligence.ocr import WindowsOcr
from ninja3_assistant.activities.intelligence.vision import ActivityVision
from ninja3_assistant.device.mumu import MuMu, prepare_game
from ninja3_assistant.device.native_touch import create_native_touch, executable_path, TimedTap
from ninja3_assistant.combat.opening_actions import OpeningActions
from ninja3_assistant.game_ui import GameUI, PageAction, PageObservation
from ninja3_assistant.runtime.realtime_battle import RealtimeBattleKernel, FrameDiagnostics
from ninja3_assistant.scheduling.task_outcome import TaskOutcome
from ninja3_assistant.scheduling.invitation_handoff import bounty_handoff, clear_handled_invitation
from ninja3_assistant.vision.invitation import InvitationVision
from ninja3_assistant.vision.template_match import matcher_session, scores_for

ROOT = resource_root()
HALL_ENTRY = (1130, 390)
HALL_ENTRY_REGION = (900, 110, 380, 120)
HALL_SEARCH = ((1150, 410), (300, 410))
OPEN_TASKS = (885, 47)
CLOSE_TASKS = (1190, 400)
MATCH_REGION = (1030, 655, 230, 55)
PAGE_IDS = {'home': 'weekend.lobby', 'hall': 'game.hall', 'invite': 'game.invite'}
# The 2026-11-21 rotation shows an announcement page, a leaderboard layout and
# a five-row daily-task panel. These points come from
# data/evidence/2026-11-21/weekend-activity.
ENTER_ACTIVITY = (1065, 635)
SUMMARY_RETURN = (640, 667)
OPEN_TASKS_CURRENT = (990, 50)
CLOSE_TASKS_CURRENT = (1160, 105)
START_MATCH_CURRENT = (990, 675)
SPECTATE_ENTRY = (860, 668)
SPECTATE_BACK = (540, 570)
CLAIM_ALL_CURRENT = (1110, 680)
# "跳过" on the relay panel, measured on two captured maps: (615,691,50,23).
RELAY_SKIP = (640, 702)
# The relay panel's title is bright green on both captured maps and absent
# everywhere else: 0.229..0.303 of this band against 0.0000 on nine pages and
# every race frame. Template matching cannot do this job -- a flat band scores
# 1.00 under NCC -- so the colour identifies it instead.
RELAY_TITLE_BOX = (500, 40, 290, 62)
RELAY_GREEN_RATIO = .06

# This week's rotation has no 25-match cap, so a match goal is a user choice:
# the dailies need 4 races, and anything up to 25 may be asked for. ``daily``
# stops once the tasks are claimed and ``unlimited`` races until stopped.
WEEKEND_MODES = ('daily', 'unlimited', *(str(n) for n in range(4, 26)))
DEFAULT_WEEKEND_MODE = 'daily'

# Ability cluster: the four slots left of the jump button. A slot that is armed
# glows far brighter than a cooling one, and the glow survives skin changes.
ABILITY_CLUSTER = (250, 440, 210, 180)
ABILITY_READY_RATIO = .45
WEAPON_POINT = (380, 630)
JUMP_POINT = (1090, 630)
OVERHEAD_POINT = (190, 630)
CARD_POINTS = ((900, 630), (520, 630), (635, 630), (740, 630))

# There are two weapon icons on the bottom bar: the main weapon at x~380 and
# the side weapon at x~900 (the first card point). Both show a big countdown
# number while cooling -- green when a reduction is active, white otherwise --
# and show the weapon art with a bright cyan ready ring once the cooldown ends.
# Readiness is therefore "no countdown digit", never a brightness guess: the
# water map's highlights and a ready weapon are equally bright. Measured on the
# 2026-11-22 user crops: ready centre white 0.245 / ring 0.208; a green or
# white digit has centre strokes 0.21..0.33 but no cyan ring (0.00..0.02).
WEAPON_ICONS = ((380, 613), (900, 613))  # (cx, cy) of main & side weapon glyphs
WEAPON_DIGIT_BOX = 12      # half-size of the centre box holding the countdown
# A countdown digit (green or white) only fills part of this centre box, while
# the ready weapon art plus its bright ring fills most of it. Measured on the
# 2026-11-22 user crops: ready bright-share 0.458; white digit 0.160; green
# digits 0.203..0.248. Colour is deliberately ignored -- a reduction turns the
# digit green, so brightness of the strokes is the only stable signal.
WEAPON_READY_BRIGHT = .30  # centre bright-share at or above which no digit remains

# Ninjutsu (ultimate): the top-left icon is a blue orb ringed by a golden arc
# that acts as the energy meter. Live race frames show a 0.375 golden arc while
# charging; the ring is fully gold once the ultimate is ready. The (55,62)
# centre and radius 44 match the magnified live crop, and the arc/ring
# measurement reuses the abyss rotation's verified gold() and ring() rules.
NINJUTSU_RING = (55, 62, 44)
NINJUTSU_POINT = (55, 60)
NINJUTSU_READY_RATIO = .88

# A race frame is confirmed by the distance board, never by a version flag.
#
# Death is a relay countdown, and what it takes away is the controls: the jump
# glyph is a connected white silhouette while a race is live and it vanishes
# when the countdown opens. Measured on the 2026-11-14 death frame and eight
# live 2026-11-21 frames: death scores 0 px of silhouette, every live frame
# 4189..6256 px. Darkness cannot do this job -- the death frame is a bright
# night rooftop at 0.68 deep-dark share, and the spectator trace page scores
# 0.70 without being a death at all.
JUMP_BUTTON_BOX = (1045, 579, 95, 85)

# What each rotation is called, so one runner serves both without a switch.
WEEKEND_LABELS = {'multiplayer': '多人竞速', 'samescreen': '同屏竞速'}


@dataclass
class WeekendProgress:
    day: str = ''
    matches: int = 0

    def refresh(self, day):
        if self.day != day:
            self.day, self.matches = day, 0

    def record(self):
        self.matches += 1


@dataclass(frozen=True, slots=True)
class Obstacle:
    kind: str
    box: tuple
    score: float


def remaining_matches(text):
    match = re.search(r'剩余\s*(\d{1,2})\s*次', ''.join(text.split()))
    if not match and '积分场次' in ''.join(text.split()):
        match = re.search(r'(\d{1,2})次', ''.join(text.split()))
    return int(match[1]) if match and int(match[1]) <= 99 else None


def movement_button(frame, gray=None):
    """Return the jump glyph's footprint, or 0 when the controls are gone.

    A death takes the glyph away; a live race keeps a large connected white
    silhouette. The threshold band rejects both an almost-empty button area and
    a texture-heavy one, so no page outside a race reports a footprint either.
    """
    if gray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    x, y, w, h = JUMP_BUTTON_BOX
    mask = cv2.inRange(gray[y:y + h, x:x + w], 200, 255)
    if not 250 < cv2.countNonZero(mask) < 4500:
        return 0
    best = 0
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        bw, bh = cv2.boundingRect(contour)[2:]
        if bw > 20 and bh > 32:
            best = max(best, bw * bh)
    return best


def relay_dialog(frame):
    """True when the multi-player rotation's relay panel is on screen.

    Measured, not guessed: the panel's bright green title covers 0.229..0.303
    of this band on the two captured maps (bamboo, cave) and exactly 0.0000 on
    nine menu pages and every race frame, so 0.06 separates them with room to
    spare. The panel opens over the race HUD, which is why it has to be judged
    before the race confirmation.
    """
    x, y, w, h = RELAY_TITLE_BOX
    patch = frame[y:y + h, x:x + w].astype(np.int16)
    if patch.size == 0:
        return False
    r, g, b = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
    return float(np.mean((g > 150) & (g - r > 40) & (g - b > 40))) >= RELAY_GREEN_RATIO


def ninjutsu_ready(frame):
    """True when the ninjutsu ring is fully charged (golden all the way round).

    The ring is the energy meter: captured race frames show a 0.375 golden arc
    while charging, and the same ring is fully golden once the ultimate can be
    fired. 0.88 separates a charging arc from a full ring -- no charging frame
    in the evidence reaches it. The full-ring (ready) sample has not been
    captured yet, so the threshold follows the abyss rotation's verified rule
    for the same in-game mechanic and should be confirmed on a live full ring.
    """
    x, y, radius = NINJUTSU_RING
    return bool(gold(ring(frame, x, y, radius)).any(axis=0).mean() >= NINJUTSU_READY_RATIO)


def _weapon_icon_state(frame, cx, cy):
    """Return True when this weapon icon has no countdown digit (i.e. is armed).

    A cooling weapon paints a countdown number in the middle of the icon. The
    digit turns green while a cooldown reduction is active and white otherwise,
    so colour is deliberately ignored. Instead the bright-pixel share of the
    centre box is read: the digit covers only part of it (0.16..0.25 on the
    2026-11-22 user crops) while the ready weapon art and its bright ring fill
    most of it (0.46). 0.30 separates the two on both green and white digits.
    """
    half = WEAPON_DIGIT_BOX
    box = frame[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    if box.size == 0:
        return False
    bright = (box.max(axis=2) > 185).mean()
    return bool(bright >= WEAPON_READY_BRIGHT)


def ready_weapon_points(frame):
    """Click points of the main and side weapons with no countdown digit."""
    return tuple((cx, cy + 17) for cx, cy in WEAPON_ICONS
                 if _weapon_icon_state(frame, cx, cy))


class RaceVision:
    """Answer every scene question from one WinML inference per frame."""
    def __init__(self, session=None):
        self.invitations = InvitationVision()
        self.opening = OpeningActions()
        self.layout = None
        self.intro = False
        self.confirmed_race = False
        self.obscured = False
        self.weapon_ready = False
        self.weapon_points = ()
        self.ninjutsu_ready = False
        self.controls_hidden = False
        self.hidden_frames = 0
        self._session = session
        self._frame = None
        self._scores = None

    def observe(self, frame):
        """Score this frame once; every later question reuses the result."""
        if frame is not self._frame:
            if self._session is None:
                self._session, _ = matcher_session()
            self._frame = frame
            self._scores = scores_for(frame, self._session)
        return self._scores

    def match(self, frame, name, box=None):
        return self.observe(frame).matched(name)

    def match_any(self, frame, names, box=None):
        scores = self.observe(frame)
        return any(scores.matched(name) for name in names)

    def count(self, frame, name, box=None):
        return self.observe(frame).count(name)

    def current_tasks(self, frame):
        return self.match(frame, 'tasks-current')

    def current_claimed_count(self, frame):
        return self.count(frame, 'claimed-current')

    def claimed(self, frame):
        if self.current_tasks(frame):
            return self.current_claimed_count(frame) >= 5
        scores = self.observe(frame)
        return scores.count('claimed') >= 5

    def claimable(self, frame):
        if self.current_tasks(frame):
            return self.match(frame, 'claim-all-current')
        scores = self.observe(frame)
        return scores.count('claim') > scores.count('claimed')

    @staticmethod
    def armed_slots(frame):
        """Click points of weapons whose countdown digit has disappeared.

        Kept as the single per-frame question so one inference serves page,
        obstacle and weapon decisions together.
        """
        return ready_weapon_points(frame)

    def detect(self, frame, gray=None):
        self.confirmed_race = False
        self.obscured = False
        self.layout = None
        self.intro = False
        self.weapon_ready = False
        self.weapon_points = ()
        self.ninjutsu_ready = False
        self.controls_hidden = False
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.invitations.detect(gray):
            return 'invite'
        scores = self.observe(frame)
        if scores.matched('finish'):
            return 'finish'
        if scores.matched('result'):
            return 'result'
        if scores.matched('standings'):
            return 'standings'
        # Check the spectator trace page first: the post-match summary
        # template also fires on it at 1.00, but its back button sits at a
        # different place, so the two must not share one state.
        if scores.matched('spectate-back'):
            return 'spectate'
        if scores.matched('summary-return'):
            return 'summary'
        # A death on the multi-player rotation opens the relay panel while the
        # race HUD is still on screen, so it must be judged before the race
        # confirmation below -- otherwise the distance board wins and the
        # runner keeps sending movement into the panel.
        if relay_dialog(frame):
            self.confirmed_race = True
            return 'relay'
        # A race is confirmed by the distance board or, when a smoke card or
        # the standings list covers that board, by the top-centre effect
        # marker. Both score under 0.18 on every menu page.
        if (scores.matched('race-marker-current') or scores.matched('race-distance') or
                scores.matched('race-effect')):
            self.confirmed_race = True
            self.weapon_points = self.armed_slots(frame)
            self.weapon_ready = bool(self.weapon_points)
            self.ninjutsu_ready = ninjutsu_ready(frame)
            region = frame[200:580, 300:1000]
            neutral = region.max(axis=2) - region.min(axis=2) < 18
            brightness = gray[200:580, 300:1000]
            self.obscured = float(np.mean(neutral & (brightness > 45) & (brightness < 230))) > .45
            # A death takes the jump glyph away. Darkness cannot decide this:
            # the death frame is a lit night rooftop. Two consecutive hidden
            # frames keep a camera pass behind a roof from looking like one,
            # and the state holds until the controls return.
            self.controls_hidden = movement_button(frame, gray) == 0
            self.hidden_frames = self.hidden_frames + 1 if self.controls_hidden and not self.obscured else 0
            if self.hidden_frames >= 2:
                return 'respawn'
            return 'race'
        if scores.matched('reward'):
            return 'reward'
        # The activity home swaps its primary button to "enter spectate"
        # while other racers are still on the course. That is its own state,
        # not an unknown frame and not a dead end: the runner waits for the
        # button to flip back to "start match".
        if scores.matched('ranking-spectate'):
            self.layout = 'current'
            return 'ranking'
        if scores.matched('intro-go'):
            self.layout, self.intro = 'intro', True
            return 'home'
        if scores.matched('tasks-current'):
            self.layout = 'current'
            return 'tasks'
        if scores.matched('ready'):
            return 'ready'
        if scores.matched('start-current'):
            self.layout = 'current'
            return 'home'
        if scores.matched('tasks'):
            return 'tasks'
        if scores.matched('home'):
            return 'home'
        if scores.matched('hall'):
            return 'hall'
        return 'unknown'


class RaceAvoider:
    """React to the obstacles WinML located on the current road.

    ``scores`` is a callable returning :class:`FrameScores` for a frame. The
    runner passes ``RaceVision.observe`` so one inference answers the page,
    obstacle and weapon questions together. Called without one, the avoider
    scores the frame itself rather than silently ignoring every obstacle.
    """
    def __init__(self, scores=None):
        self.last_jump = float('-inf')
        self.air_until = 0.
        self.hazard = None
        self.ground = None
        self.road_confirmed = False
        self.candidate = None
        self.confirmed_frames = 0
        self.player_x, self.player_y = 600, 350
        self.player_tracked = False
        self.blocked = False
        self.detected = ()
        self.last_player = float('-inf')
        self.previous_obstacle = None
        self.approach_speed = 0.
        self._scores = scores

    def track_player(self, frame, now):
        # The local player's small health bar moves horizontally when blocked
        # or overtaken. Obstacles must be measured from that player, rather
        # than from a fixed screen coordinate occupied by another racer.
        region = frame[160:590, 220:800]
        mask = cv2.inRange(region, (150, 40, 0), (255, 240, 125))
        r, g, b = region.astype(np.int16).transpose(2, 0, 1)
        mask[(r * 20 <= g * 21) | (g <= b)] = 0
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        bars = []
        for x, y, w, h, a in stats[1:]:
            if not (6 <= w <= 30 and 2 <= h <= 5 and w >= h * 1.5 and 0 < y and y + h < region.shape[0]):
                continue
            # The health fill is bounded by two dark horizontal borders.
            # Gold scenery and particles can have the same fill colour.
            border = region[[y - 1, y + h], x:x + w]
            if np.mean(border.max(axis=2) < 145) >= .8:
                bars.append((int(x) + 234, int(y) + 160, int(w)))
        if bars:
            self.player_x, self.player_y, _ = min(bars, key=lambda p:
                abs(p[0] - self.player_x) + abs(p[1] - self.player_y) * .3 - p[2] * 2)
            self.player_tracked = True
            self.last_player = now
        elif now - self.last_player > .4:
            self.player_tracked = False

    def _obstacles(self, scores, ground):
        if scores is None:
            return ()
        return tuple(Obstacle(item['kind'], item['box'], item['score'])
                     for item in scores.obstacles(ground))

    def decide(self, frame, now, gray=None, *, scores=None):
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if scores is None:
            scores = self._scores(frame) if self._scores is not None else scores_for(frame)
        self.track_player(frame, now)
        # Find the broad road/cliff boundary. A narrow reference patch can be
        # covered by another player or a skill effect and jump to background
        # edges, repeatedly inventing pits on an otherwise continuous road.
        sums = np.vstack((np.zeros((1, 320), np.float32),
                          np.cumsum(gray[:, ::4], axis=0, dtype=np.float32)))
        rows = np.arange(300, 580, 2)
        contrast = ((sums[rows - 5] - sums[rows - 20]) / 15 -
                    (sums[rows + 40] - sums[rows + 15]) / 25)
        behind = max(38, (self.player_x - 240) // 4)
        profile = np.median(contrast[:, behind:max(behind + 12, (self.player_x - 70) // 4)], axis=1)
        # Feet sit near the road; a strong edge far above the player is the
        # horizon or an overhead platform and cannot be the floor.
        profile[rows - 15 < self.player_y + 45] = -1
        best = int(profile.argmax())
        reference = float(profile[best])
        previous_ground = self.ground
        candidate = int(rows[best]) - 15 if reference > 15 else None
        # Mountain edges can out-contrast the road. Re-taking a global peak
        # every frame makes the floor jump 300..560 and mis-crops obstacles.
        if (candidate is not None and previous_ground is not None and
                abs(candidate - previous_ground) > 90):
            near = np.abs(rows - 15 - previous_ground) <= 90
            if near.any():
                local = np.where(near, profile, -1.)
                local_best = int(local.argmax())
                candidate = (int(rows[local_best]) - 15
                             if float(profile[local_best]) > max(12., reference * .5)
                             else previous_ground)
            else:
                candidate = previous_ground
        ground = candidate if candidate is not None else previous_ground or 540
        self.ground = ground
        self.hazard = None
        self.road_confirmed = reference > 15
        self.blocked = False
        self.detected = self._obstacles(scores, ground)
        # Missing terrain under a full-screen effect is not evidence of a pit.
        # A gap requires the current road plus a missing continuation ahead.
        if reference > 15:
            missing = 0
            ahead = min(1000, self.player_x + 100)
            for x in range(ahead, ahead + 211, 30):
                # A sloping/stepped road is still road: search nearby heights
                # before treating disappearance of one horizontal row as a gap.
                nearby = contrast[max(0, best - 35):best + 36, x // 4:(x + 30) // 4]
                edge = float(np.max(np.median(nearby, axis=1)))
                missing = missing + 1 if edge < max(15, reference * .25) else 0
                if missing >= 2:
                    self.hazard = 'pit'
                    break
            left = self.player_x + 35
            area = gray[max(150, ground - 150):ground + 12, left:min(1280, ahead + 240)]
            lines = cv2.HoughLinesP(cv2.Canny(area, 70, 150), 1, np.pi / 180, 30,
                                    minLineLength=30, maxLineGap=7)
            if lines is not None:
                for line in lines.reshape(-1, 4):
                    x1, y1, x2, y2 = map(int, line)
                    near = min(x1, x2) < ahead - left
                    if (abs(x2 - x1) < 8 and min(x1, x2) < ahead - left + 180 and
                            max(y1, y2) > area.shape[0] - 35 and (not near or abs(y2 - y1) >= 60)):
                        edge = left + min(x1, x2)
                        solid = max(float(np.mean(gray[max(0, ground - 65):ground + 5,
                                                        max(0, edge + dx):min(1280, edge + dx + 24)] < 85))
                                    for dx in (-12, 0))
                        if solid < .65 or self.hazard == 'pit':
                            continue
                        self.hazard = 'wall'
                        self.blocked = near
                        break
        action = JUMP_POINT
        upcoming = [item for item in self.detected
                    if item.box[0] + item.box[2] > self.player_x + 15 and
                    (item.kind != 'bomb' or item.box[1] + item.box[3] > ground - 170)]
        if upcoming:
            obstacle = min(upcoming, key=lambda item: item.box[0])
            distance = obstacle.box[0] - self.player_x
            previous = self.previous_obstacle
            if previous and previous[0] == obstacle.kind and 0 < now - previous[2] < .25:
                speed = (previous[1] - distance) / (now - previous[2])
                if 0 <= speed < 2500:
                    self.approach_speed = self.approach_speed * .5 + speed * .5
            self.previous_obstacle = (obstacle.kind, distance, now)
            lead = min(360, max(200, self.approach_speed * .28 + 60))
            if distance < lead and self.hazard != 'pit':
                if obstacle.kind == 'bomb':
                    # Do not jump into a hanging trap. An overhead trap above
                    # the play corridor needs a duck instead.
                    if obstacle.box[1] + obstacle.box[3] > ground - 170:
                        self.hazard, action = 'overhead', OVERHEAD_POINT
                else:
                    self.hazard = obstacle.kind
                    self.blocked = distance < 70
        else:
            self.previous_obstacle = None
            self.approach_speed = 0.
        candidate = self.hazard
        self.confirmed_frames = self.confirmed_frames + 1 if candidate == self.candidate else 1
        self.candidate = candidate
        if self.confirmed_frames < 2:
            self.hazard = None
        # Let at least one complete skate cycle run after a wall hop. A wall
        # visible for many frames must not keep extending the native air hold.
        retry = .55 if self.hazard != 'pit' else .32
        if self.hazard and now - self.last_jump >= retry:
            self.last_jump = now
            self.air_until = now + .42
            return action
        return None


class WeaponRelease:
    """Fire the weapon once per arm cycle, re-armed after the icon disappears."""
    def __init__(self):
        self.used = False
        self.missing = None
        self.last = float('-inf')

    def due(self, ready, now):
        if ready:
            self.missing = None
            return not self.used and now - self.last >= .35
        if self.missing is None:
            self.missing = now
        elif now - self.missing >= 1:
            self.used = False
        return False

    def scheduled(self, now):
        self.used = True
        self.last = now


class RaceSkills:
    """Tap only weapons with no countdown digit; the cards keep their cadence."""
    def __init__(self):
        self.last = float('-inf')

    def due(self, now):
        return now - self.last >= .6

    def scheduled(self, now):
        self.last = now

    @staticmethod
    def actions(weapon_points):
        if weapon_points is True:
            weapon_points = (WEAPON_POINT,)
        elif not isinstance(weapon_points, (tuple, list)):
            weapon_points = ()
        points = tuple(weapon_points) + CARD_POINTS
        # The first tap used to be offset 20ms with the default 20ms slack, so
        # its 40ms deadline expired while the PLAN command travelled down the
        # pipe: NinjaTouch dropped index 0 -- always the weapon -- and only the
        # later cards were pressed. Give every tap an 80ms slack so the first
        # point survives the hand-off, and keep 6 taps inside the finger-4
        # horizon: last offset 20+5*30=170ms + 15ms hold = 185ms < 200ms.
        return tuple(TimedTap(20 + i * 30, point, slack_ms=80)
                     for i, point in enumerate(points))


def run_weekend(folder, index, control, emit, mode=DEFAULT_WEEKEND_MODE, record=False, practice=False,
                *, checkpoint=False, progress=None, activity='multiplayer', goal=None):
    """Run a weekend race rotation until the configured goal is met.

    ``mode`` is the active phase: ``daily`` stops once the five tasks are
    claimed, ``unlimited`` keeps racing until stopped by hand, and ``4``..``25``
    stops after that many matches. ``goal`` is the user's chosen match count and
    is passed separately so the daily phase also stops at that count -- the
    planner runs the daily phase first, and without it a "complete 4 races"
    choice would race on until every daily happened to be claimed.

    ``activity`` picks the rotation. Both share this runner, the one WinML
    catalog and the battle kernel; only the hall card and the death behaviour
    differ, because the game rules differ -- the multi-player rotation offers a
    relay panel, the same-screen rotation revives by itself.
    """
    label = WEEKEND_LABELS.get(activity, WEEKEND_LABELS['multiplayer'])
    if mode not in WEEKEND_MODES:
        raise ValueError('weekend mode must be a match count 4..25, unlimited, or daily')
    target = goal if goal is not None else (int(mode) if mode.isdigit() else None)
    claimed_stop = mode == 'daily'
    device = native = recorder = None
    kernel = RealtimeBattleKernel(control, movement_horizon=3.)
    diagnostics = FrameDiagnostics()
    output = (Path(sys.executable).parent if getattr(sys, 'frozen', False) else ROOT / 'data' / 'runtime') / 'weekend-race' / 'runs' / (time.strftime('%Y%m%d-%H%M%S') + f'-{time.time_ns() % 1000000:06d}')
    output.mkdir(parents=True, exist_ok=False)
    last = None
    checked = False
    last_action = float('-inf')
    skills = RaceSkills()
    weapon = WeaponRelease()
    # The ultimate reuses the same once-per-arm-cycle cadence as the weapon:
    # fire on a full ring, then wait for the ring to drain before re-arming.
    ninjutsu = WeaponRelease()
    armed = False
    armed_at = None
    hall_swipes = 0
    last_race = float('-inf')
    sampled_start = float('-inf')
    observed_race = False
    recovering = False
    frames = 0
    measured_at = control.now()
    result_confirmed = False
    result_pending = False
    safe_invite = False
    unknown_since = None
    ranking_since = None
    try:
        if prepare_game(folder, index, control, lambda s: emit('status', s)) is None:
            return TaskOutcome.STOPPED
        vision = RaceVision()
        navigation, ocr = ActivityVision(), WindowsOcr()
        avoider = RaceAvoider(vision.observe)
        native = create_native_touch(folder, index, control.stop, emit, observation_timeout=.5, lease_ms=500, movement='jump-two-slide')
        if native is None:
            raise RuntimeError(f'{label}需要 NinjaTouch.exe C++ 触控组件')
        native = kernel.attach(native)
        control.set_release_input(native.stop_all)
        device = MuMu(folder, index)

        def swipe(start, end):
            subprocess.run([str(executable_path()), '--folder', str(folder), '--index', str(index),
                            '--swipe', *map(str, (*start, *end))], capture_output=True, timeout=8,
                           check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            return not control.stop.is_set()

        ui = GameUI(control, lambda: kernel.capture(device), native, swipe=swipe, clock=kernel.clock)

        def settle_match():
            """Record the finished match once, from whichever page proves it.

            The settlement board closes itself after about a second, so a
            confirm tap can miss its window. A match can also end with no
            result page at all and slide straight back to a startable home.
            Both paths funnel through here, so the 25-match counter cannot
            drift the way it did while only the result page was counted.
            """
            nonlocal result_pending, result_confirmed
            if not result_pending:
                return
            result_pending = False
            result_confirmed = True
            if progress is not None:
                progress.record()
                # Keep the WinUI settlement counter in step with the worker's
                # own match bookkeeping; the kill event is already the
                # "confirmed settlement" channel for the other activities.
                emit('kill', 1)

        def reached_target():
            """True only when a match-count goal was asked for and is met."""
            return target is not None and progress is not None and progress.matches >= target

        def weekend_entry(fresh):
            # Both rotations occupy the same hall card slot. The selected
            # rotation's own crop decides first, and the other crop is still
            # accepted so a rotation change never blocks entry.
            primary = 'entry-current' if activity == 'multiplayer' else 'entry'
            return vision.match_any(fresh, (primary, 'entry', 'entry-current'), HALL_ENTRY_REGION)

        emit('backend', 'WinML 统一模板匹配 / 起跳2次→下滑 / Windows OCR')
        if record:
            recorder = cv2.VideoWriter(str(output / 'race.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (1280, 720))
        emit('state', 'running')
        generation = control.generation()
        with (output / 'events.jsonl').open('w', encoding='utf-8') as log:
            while control.wait_active():
                captured = time.perf_counter()
                frame = kernel.capture(device)
                if armed and last is not None and last[0] == 'race' and not recovering:
                    if not kernel.recognition_due(urgent=avoider.hazard is not None):
                        kernel.renew()
                        control.stop.wait(max(0, 1 / 60 - (time.perf_counter() - captured)))
                        continue
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                kind = vision.detect(frame, gray)
                if kind != 'ranking':
                    ranking_since = None
                if kind == 'invite':
                    # The shared passive-invite policy owns this decision. The
                    # scheduler builds bounty_invite_targets from the enabled
                    # bounty chores and the BountyPassiveInvite switch, so a
                    # matching grade hands off to the bounty task (agree) while
                    # a disabled switch, a mismatched grade or a live race
                    # falls through to the decline below (disagree). No new
                    # per-activity invite rules belong here.
                    handoff = bounty_handoff(frame, vision.invitations, ocr, control,
                                             safe=safe_invite and not armed)
                    if handoff:
                        return handoff
                else:
                    safe_invite = not armed and kind in ('home', 'hall', 'tasks')
                    if safe_invite:
                        clear_handled_invitation(control)
                now = control.now()
                if not armed and kind in ('ready', 'race', 'respawn', 'relay') and vision.confirmed_race:
                    armed = True
                    armed_at = now
                    recovering = kind == 'respawn' and activity == 'samescreen'
                    result_confirmed = False
                    result_pending = True
                if armed and not observed_race and armed_at is not None and now - armed_at >= 60:
                    emit('error', '竞速开局等待超时，交回计划重试')
                    return TaskOutcome.ERROR
                if kind == 'unknown':
                    if unknown_since is None:
                        unknown_since = now
                    elif now - unknown_since >= (60 if armed and not observed_race else 15):
                        emit('error', f'{label}画面等待超时，交回计划重试')
                        return TaskOutcome.ERROR
                else:
                    unknown_since = None
                if armed and not observed_race and kind == 'unknown' and now - sampled_start >= .25:
                    diagnostics.save(output / f'{time.time_ns()}-startup.png', frame)
                    sampled_start = now
                if kind == 'race':
                    observed_race = True
                    last_race = now
                elif kind == 'unknown' and armed and now - last_race < .35:
                    kind = 'race'
                if generation != control.generation():
                    checked = False
                    avoider = RaceAvoider(vision.observe)
                    last_race = float('-inf')
                    generation = control.generation()
                point, message = None, f'等待识别{label}页面'
                if kind == 'race' and armed:
                    point = avoider.decide(frame, now, gray)
                    kernel.recognized(stable=point is None and avoider.hazard is None and not recovering)
                    if recovering and not vision.obscured:
                        point = JUMP_POINT
                        avoider.last_jump, avoider.air_until = now, now + .42
                        recovering = False
                        emit('recovery', {'phase': 'jump', 'player_x': avoider.player_x,
                                          'hazard': avoider.hazard})
                    frames += 1
                    if now - measured_at >= 1:
                        emit('metrics', {'fps': round(frames / (now - measured_at), 2),
                                         'player_x': avoider.player_x, 'ground': avoider.ground,
                                         'hazard': avoider.hazard, 'blocked': avoider.blocked,
                                         'weapon_ready': vision.weapon_ready, **kernel.metrics()})
                        measured_at, frames = now, 0
                    if recorder:
                        recorder.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if point:
                        control.run_if_active(lambda: native.priority_tap(
                            point, captured_at=captured, deadline=captured + .3,
                            hold_seconds=.1, preserve_slide=True))
                    else:
                        def skate():
                            native.note_frame(captured)
                            native.renew()
                        control.run_if_active(skate)
                    # Fire each weapon only while its countdown digit is gone;
                    # the cards keep their own cadence so a ready slot is never
                    # skipped. weapon_ready is the bool gate, weapon_points the
                    # actual main/side click points from the digit check.
                    if skills.due(now):
                        armed_now = weapon.due(vision.weapon_ready, now)
                        points = vision.weapon_points if armed_now else ()
                        if kernel.plan(4, RaceSkills.actions(points)) is not None:
                            skills.scheduled(now)
                            if armed_now:
                                weapon.scheduled(now)
                    # The ultimate fires the moment its ring is full, on its own
                    # cadence so it never waits for the weapon/card plan. The
                    # icon sits top-left, clear of the skating controls.
                    if vision.ninjutsu_ready and ninjutsu.due(True, now):
                        if kernel.priority_tap(NINJUTSU_POINT, captured_at=captured,
                                               deadline=captured + .3, hold_seconds=.05) is not None:
                            ninjutsu.scheduled(now)
                    message = '起跳2次→下滑，WinML 障碍识别，武器/忍术就绪即释放'
                elif kind == 'respawn' and armed:
                    # Only the same-screen rotation revives by itself. Hold
                    # still and let the revival land, then hop out of the
                    # hazard; the multi-player rotation never reaches here
                    # because its death shows the relay panel instead.
                    native.quiesce(reason='respawn')
                    recovering = activity == 'samescreen' and observed_race
                    message = ('等待自动复活，复活后先跳出危险区' if recovering
                               else f'{label}等待比赛恢复')
                elif kind == 'respawn':
                    message = ('竞速画面未确认，等待比赛开始' if not vision.confirmed_race
                               else '竞速已确认，等待起跑')
                else:
                    # While a race is live, an unrecognised frame (a new map or
                    # a full-screen effect) must not freeze the runner: reuse
                    # the native jump-two-slide movement to skate out, with the
                    # unknown_since timeout above as the safety net. A genuine
                    # menu/loading unknown (not armed, no race seen) quiesces.
                    recovering_unknown = kind == 'unknown' and armed and observed_race
                    if recovering_unknown:
                        def recover_skate():
                            native.note_frame(captured)
                            native.renew()
                        control.run_if_active(recover_skate)
                        message = '画面未识别，按上跳2次+下滑继续脱困'
                    else:
                        native.quiesce()
                    # The settlement board closes itself after about a second,
                    # and a match can also slide straight back to a startable
                    # home with no result page at all. Every settlement page
                    # funnels into one idempotent record.
                    if kind in ('finish', 'result', 'ranking'):
                        settle_match()
                    if now - last_action < 1:
                        control.stop.wait(.05)
                        continue
                    action = navigation_screen = None
                    if recovering_unknown:
                        # Recovery frames only move; there is no page button to
                        # tap and navigation.detect must not fire on a race.
                        pass
                    elif kind == 'relay':
                        # The multi-player rotation answers a death with a
                        # "选择忍者接力" panel. Skip it instead of joining: the
                        # match then settles and a fresh one starts.
                        action = PageAction('skip_relay', RELAY_SKIP, guard=relay_dialog)
                        message = '接力弹框出现，不接接力，跳过并等待本局结算'
                    elif kind == 'finish':
                        action, message = PageAction('open_result', (640, 680)), '比赛结束，继续查看结算'
                    elif kind == 'standings':
                        message = '等待成绩展示完成，不发送战斗动作'
                    elif kind == 'summary':
                        action = PageAction('confirm_summary', SUMMARY_RETURN,
                                            guard=lambda fresh: vision.match(fresh, 'summary-return'))
                        message = '确认赛后轨迹，继续查看结算'
                    elif kind == 'spectate':
                        # Spectator replay trace page: its back button sits at
                        # (540,570) and leads to the finish page, which then
                        # settles normally. Tapping the post-match point here
                        # lands outside the button and loops the replay.
                        action = PageAction('spectate_back', SPECTATE_BACK,
                                            guard=lambda fresh: vision.match(fresh, 'spectate-back'))
                        message = '退出观战轨迹，继续查看结算'
                    elif kind == 'invite':
                        # Reaching here means the shared policy did not hand
                        # off: passive invites are off, no enabled bounty chore
                        # matches this grade/source, or a race is live. Decline
                        # is the configured answer for those cases; the "agree"
                        # path above returns an InvitationHandoff instead.
                        message = '按配置拒绝悬赏邀请，继续周末竞速'
                        if decline := vision.invitations.detect(gray):
                            def same_invitation(fresh):
                                return vision.invitations.detect(
                                    cv2.cvtColor(fresh, cv2.COLOR_RGB2GRAY)) == decline
                            action = PageAction('decline_invitation', decline, guard=same_invitation)
                    elif kind == 'result':
                        action, message = PageAction('confirm_result', (1145, 681)), f'确认{label}结算'
                        checked, armed = False, False
                        recovering = False
                    elif kind == 'ranking':
                        # Same page as the activity home, with the primary
                        # button swapped to "enter spectate" while a finished
                        # match is still watchable. It is a real entry point,
                        # not a dead end: tapping it opens the result board,
                        # and confirming that returns to a startable home in
                        # about five seconds. Waiting it out took 2 to 8
                        # minutes and sometimes never recovered, so drive it.
                        armed, observed_race = False, False
                        if ranking_since is None:
                            ranking_since = now
                        if now - ranking_since >= 60:
                            action = PageAction('leave_ranking', (45, 45),
                                                guard=lambda fresh: vision.match(fresh, 'ranking-spectate'))
                            message = '观战入口无法进入，退出活动重试'
                        else:
                            action = PageAction('enter_spectate', SPECTATE_ENTRY,
                                                guard=lambda fresh: vision.match(fresh, 'ranking-spectate'))
                            message = '进入观战消费结算，随后返回开始比赛'
                    elif kind == 'reward':
                        action, message = PageAction('confirm_reward', (640, 644)), '确认周末任务奖励'
                    elif kind == 'tasks':
                        current = vision.current_tasks(frame)
                        if vision.claimable(frame):
                            claim_point = CLAIM_ALL_CURRENT if current else (1055, 679)
                            action = PageAction('claim_tasks', claim_point, guard=vision.claimable)
                            message = '一键领取周末日常奖励'
                        elif vision.claimed(frame) and claimed_stop:
                            emit('status', f'{label}五项日常已全部领取，任务结束')
                            return TaskOutcome.DAILY_DONE
                        else:
                            close_point = CLOSE_TASKS_CURRENT if current else CLOSE_TASKS
                            action, message = PageAction('close_tasks', close_point), '关闭任务页，准备继续竞速'
                            checked = True
                    elif kind == 'home':
                        if checkpoint and result_confirmed:
                            if reached_target():
                                emit('status', f'今日已完成 {target} 场积分比赛，任务结束')
                                return TaskOutcome.LIMIT_REACHED
                            return TaskOutcome.MATCH_DONE
                        if armed and observed_race:
                            # Returning to a startable home is the normal end
                            # of a match: the board auto-closes, so no result
                            # page is ever seen. Settle here rather than
                            # treating the return as a lost settlement.
                            settle_match()
                            armed, observed_race = False, False
                        if checkpoint and checked:
                            from datetime import datetime
                            from ninja3_assistant.activities.intelligence.flow import CHINA
                            if not 9 <= datetime.now(CHINA).hour < 23:
                                return TaskOutcome.WAITING
                        if armed:
                            message = '等待竞速比赛加载'
                        elif vision.intro is True:
                            action = PageAction('enter_weekend', ENTER_ACTIVITY,
                                                guard=lambda fresh: vision.match(fresh, 'intro-go'))
                            message = f'从活动介绍进入{label}主页'
                        elif not checked:
                            open_point = OPEN_TASKS_CURRENT if vision.layout == 'current' else OPEN_TASKS
                            action, message = PageAction('open_tasks', open_point), '检查周末每日任务'
                        elif reached_target():
                            emit('status', f'今日已完成 {target} 场积分比赛，任务结束')
                            return TaskOutcome.LIMIT_REACHED
                        elif vision.layout == 'current':
                            if vision.match(frame, 'start-current'):
                                action = PageAction('start_match', START_MATCH_CURRENT,
                                                    guard=lambda fresh: vision.match(fresh, 'start-current'))
                                completed = progress.matches if progress is not None else 0
                                goal = f'{completed}/{target}' if target is not None else f'{completed}'
                                message = f'开始{label}，今日已完成 {goal} 场'
                        elif vision.match(frame, 'match', MATCH_REGION):
                            result = ocr.read(cv2.resize(frame[638:718, 995:1270], None, fx=3, fy=3), stop=control.stop)
                            left = remaining_matches((result or {}).get('text', ''))
                            if left == 0 and not practice:
                                emit('status', '今日积分场次已用完，任务结束')
                                return TaskOutcome.LIMIT_REACHED
                            if left is not None:
                                action = PageAction('start_match', (1140, 677),
                                                    guard=lambda fresh: vision.match(fresh, 'match'))
                                message = (f'开始{label}，今日剩余 {left} 场积分比赛'
                                           if left else '积分次数已用完，仅实测避障动作')
                            else:
                                message = '等待读取今日剩余积分场次'
                    elif kind == 'ready':
                        action, message = PageAction('ready', (640, 579)), f'准备{label}'
                        if not armed:
                            # Arm the settlement recorder here too: the loop
                            # head only arms on a confirmed race, and a ready
                            # page never confirms one, so without this a match
                            # started from the matchmaking panel would settle
                            # without ever being counted.
                            armed = True
                            result_confirmed = False
                            result_pending = True
                        observed_race = False
                        recovering = False
                    elif kind == 'hall':
                        if weekend_entry(frame):
                            action = PageAction('enter_weekend', HALL_ENTRY, guard=weekend_entry)
                            message = f'进入已确认的{label}'
                        else:
                            if hall_swipes >= 3:
                                emit('status', f'当前周末入口未确认{label}，任务结束')
                                return TaskOutcome.UNAVAILABLE
                            action = PageAction('search_weekend', *HALL_SEARCH,
                                                guard=lambda fresh: not weekend_entry(fresh))
                            message = f'向右查找周末活动；仅进入{label}'
                    else:
                        nav = navigation.detect(frame)
                        if nav.kind in ('login', 'home') and nav.point:
                            navigation_screen = nav
                            name = 'login' if nav.kind == 'login' else 'open_hall'
                            def same_navigation_control(fresh):
                                current = navigation.detect(fresh)
                                return (current.kind, current.point) == (nav.kind, nav.point)
                            action, message = PageAction(name, nav.point, guard=same_navigation_control), '前往集会所'
                    if action:
                        check_control = action.guard
                        def same_page(fresh):
                            return vision.detect(fresh) == kind and (check_control is None or check_control(fresh))
                        page = (f'game.{navigation_screen.kind}' if navigation_screen is not None else
                                PAGE_IDS.get(kind, f'weekend.{kind}'))
                        action = replace(action, guard=same_page)
                        observed = PageObservation(page, frame, kernel.generation, kernel.captured_at,
                                                   controls={action.name: action}, detail=kind)
                        if action.end is None:
                            point = action.point
                        if ui.perform(observed, action.name):
                            last_action = now
                            if action.name == 'search_weekend':
                                hall_swipes += 1
                state = (kind, message)
                if state != last and (kind not in ('race', 'unknown') or now - getattr(avoider, 'logged_at', -1e9) >= 1):
                    avoider.logged_at = now
                    emit('status', message)
                    log.write(json.dumps({'time': time.time(), 'kind': kind, 'message': message,
                                          'point': point}, ensure_ascii=False) + '\n')
                    log.flush()
                    diagnostics.save(output / f'{time.time_ns()}-{kind}.png', frame)
                    last = state
                if kind in ('race', 'respawn'):
                    delay = max(0, 1 / 60 - (time.perf_counter() - captured))
                elif kind == 'ranking':
                    # The spectate button can stay for minutes; poll slowly.
                    delay = .5
                else:
                    delay = .15
                control.stop.wait(delay)
    except Exception as error:
        emit('error', str(error))
        return TaskOutcome.ERROR
    finally:
        control.set_release_input(None)
        if native:
            native.close()
        if recorder:
            recorder.release()
        if device:
            device.close()
        diagnostics.close()
        emit('finished', '')
    return TaskOutcome.STOPPED
