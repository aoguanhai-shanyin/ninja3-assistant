"""Recognize the opening of Xumi's eight-crow Boss fight."""

from dataclasses import dataclass, replace

import cv2
import numpy as np

from ninja3_assistant.vision.vision import ASSETS


@dataclass(frozen=True, slots=True)
class CrowAction:
    kind: str
    point: tuple[int, int]
    feather_wave: int | None = None
    return_delay_ms: int = 360


_DOWN = CrowAction('down', (190, 630))
_CHARGE_JUMP = CrowAction('jump', (1090, 630))
# Separate the two replayed left-side fans, with time to finish the first roll.
_FEATHER_WAVES = (
    (3., 4.9, CrowAction('up', (1090, 630), feather_wave=1)),
    (6., 6.76, CrowAction('up', (1090, 630), feather_wave=3)),
)
_LEFT_PLAYER = (205, 270)
_ROLL_DELAYS = (360, 300, 420, 260, 480, 540)


def _crow_portrait(frame):
    # The Xumi HUD uses the same portrait in Fangcun and Daqian. Use only
    # interior landmarks so the bar, level, and background can change.
    red, green, blue = frame[30:90, 381:445].astype(np.int16).transpose(2, 0, 1)
    eye = (red >= 150) & (red - green >= 65) & (red - blue >= 40)
    crown = (red >= 90) & (red - green >= 45) & (red - blue >= 32)
    low = np.minimum(np.minimum(red, green), blue)
    high = np.maximum(np.maximum(red, green), blue)
    feather = (low >= 105) & (high - low <= 55)
    purple = (blue >= 135) & (blue - red >= 35) & (red - green >= 18)
    return (crown[:12, 9:19].mean() >= .55
            and eye[20:33, 12:21].mean() >= .34
            and feather[23:34, 22:33].mean() >= .75
            and feather[42:55, :13].mean() >= .36
            and purple[47:, 41:].mean() >= .45)


def _rope_y(frame):
    gray = cv2.cvtColor(frame[200:530, 100:1180], cv2.COLOR_RGB2GRAY)
    darkness = (gray < 30).mean(axis=1)
    line = np.convolve(darkness, np.ones(5) / 5, mode='same')
    contrast = line[9:-9] - (line[:-18] + line[18:]) / 2
    row = int(np.argmax(contrast))
    if line[row + 9] < .78 or contrast[row] < .60:
        return None
    return row + 209


def _charge_telegraph(frame):
    rope_y = _rope_y(frame)
    if rope_y is None:
        return False

    # The first charge is preceded by two small crows summoned to the right.
    # Their eyes are close to level, about 86 px apart and 45–50 px above
    # the rope. The dark silhouettes are also visible in the red summon.
    top, left = rope_y - 54, 500
    red, green, blue = frame[top:rope_y - 39, left:1240].astype(np.int16).transpose(2, 0, 1)
    bright = (red >= 190) & (green >= 50) & (blue >= 50) & (red - green >= 38) & (red - blue >= 30)
    _, _, stats, centers = cv2.connectedComponentsWithStats(bright.astype(np.uint8))
    eyes = []
    for (_, _, width, height, area), (x, y) in zip(stats[1:], centers[1:]):
        if not (3 <= width <= 8 and 2 <= height <= 7 and 7 <= area <= 28):
            continue
        x, y = int(x) + left, int(y) + top
        silhouette = frame[y + 8:y + 32, x - 5:x + 6, 1:]
        if (silhouette.max(axis=2) < 66).mean() >= .78:
            eyes.append((x, y))
    return any(81 <= bx - ax <= 93 and abs(by - ay) <= 5
               for ax, ay in eyes for bx, by in eyes)


def _feather_contours(frame, rope):
    left, top = 125, rope - 220
    red, green, blue = frame[top:rope + 125, left:945].astype(np.int16).transpose(2, 0, 1)
    rim = ((red >= 125) & (red - green >= 80) & (red - blue >= 60)).astype(np.uint8)
    rim = cv2.morphologyEx(rim, cv2.MORPH_CLOSE, np.ones((5, 9), dtype=np.uint8))
    feathers = []
    for contour in cv2.findContours(rim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, width, height = cv2.boundingRect(contour)
        if not (47 <= width <= 83 and 12 <= height <= 33 and 290 <= cv2.contourArea(contour) <= 730):
            continue
        _, axes, _ = cv2.minAreaRect(contour)
        if max(axes) < 2.6 * min(axes):
            continue
        vx, vy, _, _ = cv2.fitLine(contour, cv2.DIST_L2, 0, .01, .01).ravel()
        if abs(vx) < .7 or not -.46 <= vy / vx <= .02:
            continue
        feathers.append((x + left, y + top - rope, width, height))
    return feathers


def _feather_tracks(previous, current, dt):
    pairs = []
    for index, (x, y, width, height) in enumerate(current):
        for old_index, (px, py, pw, ph) in enumerate(previous):
            vx, vy = (px - x) / dt, (y - py) / dt
            if (650 <= vx <= 1400 and .04 <= vy / vx <= .49
                    and abs(width - pw) <= 9 and abs(height - ph) <= 6):
                error = abs(vx - 950.) + 22 * (abs(width - pw) + abs(height - ph))
                pairs.append((error,index,old_index,vx,vy))
    tracks = []
    matched, used_previous = set(), set()
    for _,index,old_index,vx,vy in sorted(pairs):
        if index in matched or old_index in used_previous:
            continue
        matched.add(index)
        used_previous.add(old_index)
        tracks.append((*current[index],vx,vy))
    return tracks


def _crosses_lane(track, lane, window):
    x, y, width, height, vx, vy = track
    player_left, player_right = _LEFT_PLAYER
    first = max(window[0], (x - player_right) / vx)
    last = min(window[1], (x + width - player_left) / vx)
    return first <= last and y + vy * first <= lane[1] and y + height + vy * last >= lane[0]


def _choose_roll_delay(tracks):
    # Screen replays show the avatar remains exposed above the rope after the
    # down press. Test the entire stay above and the landing below, with overlap
    # while the avatar crosses the rope.
    for delay in _ROLL_DELAYS:
        seconds = delay / 1000
        if not any(_crosses_lane(track, (-57, -12), (.14, seconds+.15))
                   or _crosses_lane(track, (43, 89), (seconds+.12, seconds+.32)) for track in tracks):
            return delay
    return None


class CrowOpening:
    """Propose bounded opening, feather, and first-charge dodges per Boss."""

    def __init__(self, *, enable_feather=False):
        self.enable_feather = enable_feather
        self.controls = {}
        for name in ('down', 'jump', 'up'):
            path = ASSETS / f'battle-{name}.png'
            template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise RuntimeError(f'缺少须弥绳索控制识别模板 {path.name}')
            self.controls[name] = cv2.inRange(template, 200, 255)
        self.reset()

    def reset(self):
        self.started = False
        self.charge_handled = False
        self.handled_feather_waves = set()
        self.feather_sample = None

    @staticmethod
    def _score(region, template):
        return cv2.minMaxLoc(cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED))[1]

    def _rope_side(self, frame):
        left = cv2.inRange(cv2.cvtColor(frame[575:685, 135:245], cv2.COLOR_RGB2GRAY), 200, 255)
        if self._score(left, self.controls['down']) < .92:
            return None
        right = cv2.inRange(cv2.cvtColor(frame[575:685, 1035:1145], cv2.COLOR_RGB2GRAY), 200, 255)
        if self._score(right, self.controls['up']) >= .92:
            return 'below'
        # The jump icon has a different pose in Xumi than in the training
        # template. With Down present its observed match is about .71.
        if self._score(right, self.controls['jump']) >= .68:
            return 'above'
        return None

    def _feather(self, frame, elapsed, side, action):
        rope = _rope_y(frame)
        if rope is None or side != 'below':
            self.feather_sample = None
            return None
        contours = _feather_contours(frame, rope)
        previous = self.feather_sample
        self.feather_sample = (action.feather_wave, elapsed, contours)
        if previous is None or previous[0] != action.feather_wave or not .010 <= elapsed - previous[1] <= .30:
            return None
        tracks = _feather_tracks(previous[2], contours, elapsed - previous[1])
        # Each window handles just its leading low, falling feather.
        low = [track for track in tracks if .28 <= track[5] / track[4] <= .49
               and _crosses_lane(track, (43, 89), (0., .25))]
        if any(track[0] <= _LEFT_PLAYER[1] for track in low):
            self.handled_feather_waves.add(action.feather_wave)
            return None
        if not any(_crosses_lane(track, (43, 89), (.08, .25)) for track in low):
            return None
        delay = _choose_roll_delay(tracks)
        return None if delay is None else replace(action,return_delay_ms=delay)

    def propose(self, frame, *, boss_active, boss_elapsed=None):
        down = not self.started and (boss_elapsed is None or 0 <= boss_elapsed <= 3.)
        charge = not self.charge_handled and boss_elapsed is not None and 8.5 <= boss_elapsed <= 11.
        feather = None
        if self.enable_feather and boss_elapsed is not None:
            feather = next((action for start, end, action in _FEATHER_WAVES
                            if start <= boss_elapsed <= end and action.feather_wave not in self.handled_feather_waves),
                           None)
        if not (down or charge or feather) or not boss_active or not isinstance(frame, np.ndarray) or frame.shape != (720, 1280, 3):
            return None
        if not _crow_portrait(frame):
            return None
        side = self._rope_side(frame)
        if down:
            if side == 'below':
                self.started = True
            if side == 'above':
                return _DOWN
        if charge and side is not None and _charge_telegraph(frame):
            if side == 'above':
                return _CHARGE_JUMP
            # The below-rope player survived the live charge without moving.
            self.charge_handled = True
        if feather is not None:
            return self._feather(frame, boss_elapsed, side, feather)
        return None

    def accepted(self, action):
        if action == _DOWN:
            self.started = True
        if action == _CHARGE_JUMP:
            self.charge_handled = True
        if action.feather_wave is not None:
            self.handled_feather_waves.add(action.feather_wave)
