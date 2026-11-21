"""Equip only positively recognized empty weapon slots in Daqian."""
import cv2

from ninja3_assistant.activities.daily.flow import DailyDecision


_ROWS = (345, 211)
_WEAPON_COLUMNS = (253, 378)
_SLOTS = tuple((x, y) for x in _WEAPON_COLUMNS for y in _ROWS)
_INVENTORY = (1010, 215)
_SAVE = (1115, 678)
_SELECT = '选择当前最高位可用武器'
_EQUIP = '打开本次选中武器的装备位置'
_PLACE = '装备到已确认的空武器位置'


def _navy(patch):
    red, green, blue = patch.transpose(2, 0, 1)
    return ((red <= 12)&(green >= 44)&(green <= 65)&(blue >= 67)&(blue <= 94)).mean()


def _two_member_panel(frame):
    # The current team's blue panel ends above the rows for other characters.
    blue = frame[405:421, 95:260, 2].mean(axis=1)
    return blue[:4].max() > 195 and blue[4:7].min() < 100 and blue[8:].mean() > 125


def _slots(frame, rows):
    slots = {}
    for y in rows:
        portrait = frame[y-27:y+14, 92:151].max(axis=2)
        if portrait.std() < 39 or (portrait > 145).mean() < .14:
            continue
        for x in _WEAPON_COLUMNS:
            label = frame[y-29:y+14, x-70:x-47]
            if (label[:,:,2] > 151).mean() < .45:
                continue
            icon = frame[y-22:y+14, x-31:x+30]
            if _navy(icon) > .68 and _navy(frame[y-29:y-12, x-31:x-16]) > .92:
                slots[x,y] = 'empty'
            elif (icon.max(axis=2) > 145).mean() > .33 and icon.max(axis=2).std() > 26:
                slots[x,y] = 'filled'
    return slots


def _same_icon(frame, template, point):
    x, y = point
    area = frame[y-36:y+12, x-36:x+37]
    _, score, _, (left, top) = cv2.minMaxLoc(cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED))
    height, width = template.shape[:2]
    patch = area[top:top+height, left:left+width]
    return score >= .90 and cv2.absdiff(patch, template).mean() <= 23


def _weapons_open(frame):
    return frame[82:99, 1023:1035].mean() - frame[82:99, 1170:1186].mean() > 26


class EquipmentSetup:
    def __init__(self, expected_team=None):
        self.expected_team = expected_team
        self._stage = 'inventory'
        self._target = None
        self._icon = None
        self._prior_inventory = None
        self._changed = False
        self._support_rows = None

    def step(self, frame, scene):
        if getattr(frame, 'shape', None) != (720, 1280, 3):
            return DailyDecision(status='等待确认大千试炼装备画面')
        if self.expected_team is not None and getattr(scene,'team_number',None) != self.expected_team:
            return DailyDecision(status='等待确认装备页属于当前大千试炼队伍')
        if not _two_member_panel(frame):
            return DailyDecision(status='等待确认当前小队的两名成员范围')
        if scene.kind == 'equipment':
            support = tuple(getattr(scene,'support_rows',()))
            if support not in ((), (495,)) or (self._support_rows is not None and support != self._support_rows):
                return DailyDecision(status='等待确认当前小队标记的支援成员')
            self._support_rows = support
        support = self._support_rows or ()
        rows = _ROWS+support
        points = _SLOTS+tuple((x,y) for y in support for x in _WEAPON_COLUMNS)
        if scene.kind == 'equipment_detail' and self._stage == 'detail':
            if scene.point == (1134, 573) and _same_icon(frame, self._icon, _INVENTORY):
                return DailyDecision.tap('equip_weapon', scene.point, _EQUIP, guard=(1046, 543, 1223, 600))
        if scene.kind == 'equipment_target' and self._stage == 'target':
            x, y = self._target
            if (self._target in scene.equipment_targets and _slots(frame,rows).get(self._target) == 'empty' and
                    _same_icon(frame, self._icon, _INVENTORY)):
                return DailyDecision.tap('place_weapon', self._target, _PLACE, guard=(x-45, y-45, x+45, y+45))
        if scene.kind != 'equipment':
            return DailyDecision(status='等待确认选中武器及可装备位置')
        slots = _slots(frame,rows)
        if self._stage == 'verify':
            # Match the small size changes seen between the inventory and slot sprites.
            icons = (cv2.resize(self._icon,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
                     for scale in (.94,.96,.98))
            if slots.get(self._target) != 'filled' or not any(
                    _same_icon(frame,icon,self._target) for icon in icons):
                return DailyDecision(status='等待确认选中武器已出现在指定武器位置')
            self._changed = True
            self._stage = 'inventory'
        if self._stage != 'inventory' or len(slots) != len(points) or not _weapons_open(frame):
            return DailyDecision(status='等待确认当前小队的主副武器位置和武器列表')
        empties = [point for point in points if slots[point] == 'empty']
        if not empties:
            if self._changed:
                return DailyDecision('gear_done', _SAVE, '保存已确认的大千试炼装备',
                                     guard=(1000, 647, 1223, 711), operation='save_equipment')
            if scene.back:
                return DailyDecision('gear_done', scene.back, '当前小队主副武器均已装备，返回配队',
                                     guard=(15, 0, 83, 73), operation='back')
            return DailyDecision(status='等待确认已装备队伍的返回入口')
        inventory = frame[177:466, 972:1256]
        if self._prior_inventory is not None and cv2.absdiff(inventory, self._prior_inventory).mean() <= 6:
            return DailyDecision(status='等待确认使用后的武器列表已刷新')
        icon = frame[193:219, 987:1034]
        red, green, blue = frame[174:185, 969:982].transpose(2, 0, 1)
        stars = ((red > 225)&(green > 188)&(blue < 190)).mean()
        if stars < .15 or icon.std() < 26:
            return DailyDecision(status='等待确认可用的自有武器')
        self._target = empties[0]
        self._icon = icon.copy()
        self._prior_inventory = inventory.copy()
        return DailyDecision.tap('select_weapon', _INVENTORY, _SELECT, guard=(966, 167, 1059, 263))

    def completed(self, decision):
        if decision.operation == 'select_weapon':
            self._stage = 'detail'
        elif decision.operation == 'equip_weapon':
            self._stage = 'target'
        elif decision.operation == 'place_weapon':
            self._stage = 'verify'
        elif decision.action == 'gear_done':
            self._stage = 'done'
