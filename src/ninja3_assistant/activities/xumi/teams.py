"""Guarded Xumi team selection for the two observed trial layouts."""
import cv2

from ninja3_assistant.activities.daily.flow import DailyDecision


_LEADS = ((561, 564), (562, 647))
_TABS = ((120, 188), (120, 277))
_CARDS = ((399, 288), (632, 288), (864, 288), (1104, 288))
_FIXED = tuple((1045, y) for y in (214, 310, 406, 503, 598))
_SLOTS = ((136, 289), (265, 289), (115, 432), (250, 432))
_PORTRAITS = ((765, 377), (535, 254), (394, 377), (890, 377), (410, 254),
              (518, 377), (658, 254), (783, 254), (907, 254), (641, 377))
_PICK_FIXED = '选择方寸试炼固定角色'
_PICK_OWN = '加入大千试炼自有角色'
_CLOSE_FIXED = '关闭固定角色名单'
_SAVE_OWN = '保存大千试炼队伍'
_DISCARD_OWN = '返回当前大千队伍'
_KEEP_ONE = '保留当前可出战成员'


def _tap(point, status, guard=None):
    x, y = point
    return DailyDecision('tap', point, status, guard=guard or (x-23, y-21, x+24, y+22))


def _gold(patch):
    red, green, blue = patch.transpose(2, 0, 1)
    return ((red > 215)&(green > 153)&(blue < 140)).mean() > .55


def _blue(patch):
    red, green, blue = patch.astype('int16').transpose(2, 0, 1)
    return ((blue > 165)&(green > 125)&(blue-red > 55)).mean() > .4


def _lead(frame, point):
    x, y = point
    patch = frame[y-22:y+19, x-23:x+28]
    red, green, blue = patch.astype('int16').transpose(2, 0, 1)
    white = (patch.min(axis=2) > 222).mean()
    pale_blue = ((blue > 170)&(blue-red > 40)&(green > 108)).mean()
    if white > .2 and pale_blue > .35:
        return 'empty'
    if white < .19 and pale_blue < .36 and (patch.max(axis=2) < 85).mean() > .10:
        return 'filled'
    return 'unknown'


def _artwork(patch):
    hue, saturation, light = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV).transpose(2, 0, 1)
    colored = ((hue < 90)|(hue > 123)|(saturation < 50)).mean()
    return colored > .27 and light.std() > 26


def _card(frame, point):
    x, y = point
    patch = frame[y-24:y+24, x-25:x+25]
    red, green, blue = patch.transpose(2, 0, 1)
    cross = (red > 70)&(red < 135)&(green > 125)&(green < 190)&(blue > 167)&(blue < 231)
    if cross[20:28].mean() > .75 and cross[:, 19:31].mean() > .65:
        return 'empty'
    return 'filled' if _artwork(frame[150:422, x-48:x+49]) else 'unknown'


def _bubble(frame, point):
    x, y = point
    patch = frame[y-28:y+29, x-28:x+29]
    hue, sat, light = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV).transpose(2, 0, 1)
    blue = ((hue >= 90)&(hue <= 120)&(sat >= 80)&(light >= 80)).mean()
    if blue > .92 and light.std() < 20:
        return 'empty'
    return 'filled' if _artwork(patch) else 'unknown'


def _selectable_portrait(frame, point):
    x, y = point
    if _blue(frame[y+39:y+53, x-24:x+24]):
        return False
    patch = frame[y-30:y+31, x-30:x+31]
    _, sat, light = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV).transpose(2, 0, 1)
    return (light > 160).mean() >= .17 and ((sat > 64)&(light > 160)).mean() >= .04


def _fixed_owner(frame, x, y):
    red, green, blue = frame[y-50:y-24, x-33:x+37].astype('int16').transpose(2, 0, 1)
    return ((red >= 80)&(red < 185)&(green < 145)&(blue >= 110)&(blue > red)&(red-green > 20)).mean() > .12


class TeamSetup:
    def __init__(self):
        self._reset(None)

    def _reset(self, mode):
        self._mode = mode
        self._wanted = set()
        self._ready = set()
        self._short = set()
        self._team = None
        self._team_members = 0
        self._fixed_pending = None
        self._fixed_used = set()
        self._own_seen = 0
        self._own_pending = None
        self._own_used = set()
        self._own_changed = False

    def step(self, frame, kind, mode):
        if getattr(frame, 'shape', None) != (720, 1280, 3) or mode not in ('fangcun', 'daqian'):
            return DailyDecision(status='等待确认须弥幻境配队画面')
        if self._mode != mode:
            self._reset(mode)
        if kind == mode+'_setup':
            leads = tuple(_lead(frame, point) for point in _LEADS)
            # Editing a configured team can reset its existing trial score.
            self._wanted = {index for index, lead in enumerate(leads) if lead == 'empty'}
            if self._wanted:
                self._ready.difference_update(self._wanted)
                self._short.difference_update(self._wanted)
                team = min(self._wanted)
                return _tap(_LEADS[team], f'配置队伍{team+1}')
            red, green, blue = frame[167:204, 924:1014].astype('int16').transpose(2, 0, 1)
            first_difficulty = ((red > 178)&(green > 150)&(green > blue+20)).mean() > .12
            if leads == ('filled', 'filled') and first_difficulty and _gold(frame[615:654, 970:997]):
                return DailyDecision('start', (1090, 638), '开始须弥幻境试炼', guard=(526, 528, 1225, 687))
            return DailyDecision(status='等待确认两队作战角色与第一重难度')
        if not self._wanted:
            return DailyDecision(status='等待确认本次试炼需要配置的队伍')
        if kind == 'team':
            return self._team_step(frame, mode)
        if kind == 'roster' and mode == 'fangcun' and self._team in self._wanted:
            return self._fixed_step(frame)
        if kind == 'own_roster' and mode == 'daqian' and self._team in self._wanted:
            return self._own_step(frame)
        return DailyDecision(status='等待确认须弥幻境队伍界面')

    def _team_step(self, frame, mode):
        own_layout = _blue(frame[224:242, 1238:1249])
        tabs = [index for index, (_, y) in enumerate(_TABS) if _gold(frame[y-16:y-10, 157:193])]
        if own_layout != (mode == 'daqian') or len(tabs) != 1:
            return DailyDecision(status='等待确认当前试炼队伍页签')
        self._team = tabs[0]
        remaining = self._wanted-self._ready
        if self._team in remaining:
            cards = tuple(_card(frame, point) for point in _CARDS)
            self._team_members = cards.count('filled')
            if cards[0] == 'filled' and (self._team_members >= 2 or self._team in self._short):
                self._ready.add(self._team)
                remaining.discard(self._team)
            elif self._team_members < 2 and 'unknown' not in cards and 'empty' in cards:
                return _tap(_CARDS[cards.index('empty')], '打开当前队伍的空位')
            else:
                return DailyDecision(status='等待确认已有成员与队伍空位')
        if remaining:
            team = min(remaining)
            return _tap(_TABS[team], f'切换队伍{team+1}', guard=(80, 155, 195, 302))
        return _tap((46, 37), '返回试炼核对两个队伍')

    def _fixed_step(self, frame):
        if self._fixed_pending:
            x, y = self._fixed_pending
            if _blue(frame[y-7:y+7, x-42:x-25]):
                self._fixed_used.add(self._fixed_pending)
                return _tap((624, 76), _CLOSE_FIXED, guard=(525, 104, 756, 143))
            return DailyDecision(status='等待该固定角色确认出战')
        available = [(x, y) for x, y in _FIXED if (x, y) not in self._fixed_used and not _fixed_owner(frame, x, y)]
        for x, y in available:
            if _gold(frame[y-7:y+7, x-42:x-25]):
                return _tap((x, y), _PICK_FIXED)
        if self._team_members and all(_blue(frame[y-7:y+7, x-42:x-25]) for x, y in available):
            return _tap((624, 76), _KEEP_ONE, guard=(525, 104, 756, 143))
        return DailyDecision(status='等待可出战的固定角色')

    def _own_step(self, frame):
        slots = tuple(_bubble(frame, point) for point in _SLOTS)
        if 'unknown' in slots:
            return DailyDecision(status='等待确认快速编辑中的四个队伍位置')
        self._own_seen = slots.count('filled')
        if self._own_pending:
            point, before = self._own_pending
            if self._own_seen != before+1:
                return DailyDecision(status='等待确认刚选角色已加入当前队伍')
            self._own_used.add(point)
            self._own_pending = None
            self._own_changed = True
        if self._own_seen < 2:
            for point in _PORTRAITS:
                if point not in self._own_used and _selectable_portrait(frame, point):
                    return _tap(point, _PICK_OWN)
        if self._own_changed:
            if _gold(frame[577:600, 625:654]):
                return _tap((720, 597), _SAVE_OWN)
            return DailyDecision(status='等待确认保存修改按钮')
        if self._own_seen:
            return _tap((372, 597), _DISCARD_OWN)
        return DailyDecision(status='等待可加入队伍的自有角色')

    def completed(self, decision):
        if decision.status == _PICK_FIXED:
            self._fixed_pending = decision.point
        elif decision.status == _CLOSE_FIXED:
            self._fixed_pending = None
        elif decision.status == _KEEP_ONE:
            self._short.add(self._team)
        elif decision.status == _PICK_OWN:
            self._own_pending = decision.point, self._own_seen
        elif decision.status in (_SAVE_OWN, _DISCARD_OWN):
            if self._own_seen == 1:
                self._short.add(self._team)
            self._own_changed = False
