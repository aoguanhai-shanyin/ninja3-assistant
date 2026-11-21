"""Observable readiness of the two teams on a Xumi trial setup screen."""

from dataclasses import dataclass


TEAM_LEADS = ((561, 564), (562, 647))
DIFFICULTY_LABELS = ((924,1014), (1038,1126), (1144,1241))


def gold_control(patch):
    red, green, blue = patch.transpose(2, 0, 1)
    return ((red > 215) & (green > 153) & (blue < 140)).mean() > .55


def lead_state(frame, point):
    x, y = point
    patch = frame[y-22:y+19, x-23:x+28]
    red, green, blue = patch.astype('int16').transpose(2, 0, 1)
    white = (patch.min(axis=2) > 222).mean()
    pale_blue = ((blue > 170) & (blue-red > 40) & (green > 108)).mean()
    if white > .2 and pale_blue > .35:
        return 'empty'
    dark = (patch.max(axis=2) < 85).mean()
    neutral = ((patch.max(axis=2) - patch.min(axis=2)) < 42).mean()
    portrait = dark > .10 or (dark > .04 and neutral > .50)
    if white < .19 and pale_blue < .36 and dark < .75 and portrait and patch.mean(axis=2).std() > 25:
        return 'filled'
    return 'unknown'


@dataclass(frozen=True)
class TrialSetup:
    leads: tuple[str, str]
    selected_difficulty: int | None
    start: tuple[int, int] | None

    @property
    def first_difficulty(self):
        return self.selected_difficulty == 1

    @property
    def can_start(self):
        return self.start is not None


def _difficulty(frame):
    highlighted = []
    for number, (left, right) in enumerate(DIFFICULTY_LABELS, 1):
        red, green, blue = frame[171:201,left:right].astype('int16').transpose(2, 0, 1)
        if ((red > 178) & (green > 150) & (green > blue+20)).mean() > .12:
            highlighted.append(number)
    return highlighted[0] if len(highlighted) == 1 else None


def trial_setup(frame, start, notice=None):
    leads = tuple(lead_state(frame, point) for point in TEAM_LEADS)
    current = _difficulty(frame)
    if (leads != ('filled', 'filled') or current is None or notice is not None
            or not gold_control(frame[615:654, 970:997])):
        start = None
    return TrialSetup(leads, current, start)
