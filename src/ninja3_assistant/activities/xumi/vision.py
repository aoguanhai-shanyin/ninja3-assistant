"""Observed 1280×720 menus and battle controls for 须弥幻境."""
from dataclasses import dataclass

import cv2

from ninja3_assistant.activities.daily.vision import DailyVision
from ninja3_assistant.activities.xumi.teams import team_mode
from ninja3_assistant.game_ui import PageAction, PageObservation
from ninja3_assistant.game_ui.pages import page_id
from ninja3_assistant.game_ui.xumi import TrialSetup, trial_setup
from ninja3_assistant.paths import resource_root
from ninja3_assistant.vision.game_update import game_update_banner


BACK = (47, 40)
ROSTER_CLOSE = (624, 76)
TITLE = (82, 0, 320, 88)
DAILY_BUTTON = (1050, 68, 1160, 111)
WEAPON_SLOTS = tuple((x, y) for y in (211, 345, 495, 628) for x in (253, 378))


@dataclass(frozen=True)
class XumiScreen:
    kind: str = 'unknown'
    point: tuple[int, int] | None = None
    back: tuple[int, int] | None = None
    fangcun: tuple[int, int] | None = None
    daqian: tuple[int, int] | None = None
    notice: str | None = None
    equipment_targets: tuple[tuple[int, int], ...] = ()
    team_number: int | None = None
    support_rows: tuple[int, ...] = ()
    team_mode: str | None = None
    setup: TrialSetup | None = None

    @property
    def navigation_kind(self):
        return self.team_mode+'_team' if self.kind == 'team' and self.team_mode else self.kind


class XumiVision:
    def __init__(self, daily=None):
        self.daily = daily if daily is not None else DailyVision()
        for path in (resource_root()/'assets'/'xumi').glob('*.png'):
            self.daily.templates['xumi-'+path.stem] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    def match(self, frame, name, area, threshold=.94):
        return self.daily.match(frame, 'xumi-'+name, area, threshold=threshold)

    def menu_observation(self, frame, scene, decision, invitation, generation, captured_at):
        controls = {}
        if decision.point is not None and decision.operation:
            controls[decision.operation] = PageAction(
                decision.operation, decision.point, decision.end, hold_seconds=.055,
                guard=lambda fresh: self.current_action(frame, scene, decision, invitation, fresh),
            )
        kind = ('invite' if invitation else 'reward' if decision.operation == 'dismiss_reward' else scene.navigation_kind)
        area = 'daily' if kind == 'reward' else 'xumi'
        return PageObservation(page_id(area,kind),frame,generation,captured_at,controls,detail=scene)

    def current_action(self, frame, scene, decision, invitation, fresh):
        if game_update_banner(fresh):
            return False
        fresh_invite = self.daily.invitations.detect(cv2.cvtColor(fresh,cv2.COLOR_RGB2GRAY))
        if fresh_invite != invitation:
            return False
        fresh_scene = self.scene(fresh)
        if fresh_scene.kind != scene.kind or (scene.team_number is not None and
                fresh_scene.team_number != scene.team_number):
            return False
        if fresh_scene.team_mode != scene.team_mode:
            return False
        if decision.operation == 'start_trial':
            return (scene.setup is not None and fresh_scene.setup is not None
                    and scene.setup.can_start and fresh_scene.setup.can_start
                    and scene.setup.start == decision.point == fresh_scene.setup.start
                    and scene.setup.selected_difficulty == fresh_scene.setup.selected_difficulty)
        if scene.kind in ('home','hall','guide') and fresh_scene.point != scene.point:
            return False
        if scene.kind == 'equipment_target' and decision.point not in fresh_scene.equipment_targets:
            return False
        if decision.operation == 'back' and fresh_scene.back != scene.back:
            return False
        if decision.operation == 'open_tasks' and fresh_scene.point != decision.point:
            return False
        if decision.operation in ('open_fangcun','open_daqian') and (
                getattr(fresh_scene, decision.operation.removeprefix('open_')) != decision.point):
            return False
        if decision.operation == 'dismiss_reward':
            reward, point = self.daily.navigation(fresh)
            if reward not in ('reward','reward_continue') or point != decision.point:
                return False
        if decision.guard:
            x1,y1,x2,y2 = decision.guard
            if cv2.absdiff(frame[y1:y2,x1:x2],fresh[y1:y2,x1:x2]).mean() > 14:
                return False
        return True

    def equipment_team(self, frame):
        heading = cv2.cvtColor(frame[70:105,814:940], cv2.COLOR_RGB2GRAY)

        def matches(name, region):
            template = self.daily.templates.get('xumi-equipment-team-'+name)
            return (template is not None and
                    cv2.minMaxLoc(cv2.matchTemplate(region,template,cv2.TM_CCOEFF_NORMED))[1] >= .94)

        # The save confirmation darkens the heading without changing either glyph shape.
        if not matches('label', heading[:,18:]):
            return None
        numbers = [number for number in (1, 2) if matches(str(number), heading[:,:28])]
        return numbers[0] if len(numbers) == 1 else None

    @staticmethod
    def relay(frame):
        red, green, blue = frame[44:101,500:780].astype('int16').transpose(2, 0, 1)
        lime = (red >= 150) & (green >= 205) & (blue < 140) & (green-red >= 15) & (red-blue >= 58)
        # Six separate glyphs of 选择忍者接力, independent of countdown and battle art.
        glyphs = ((8,49),(53,94),(99,137),(143,182),(186,227),(232,268))
        if not all(.17 < lime[:,left:right].mean() < .43 for left,right in glyphs):
            return False
        if lime[:,[51,97,140,184,229]].mean() > .03:
            return False
        skip_label = frame[689:717,610:670]
        skip_background = frame[688:719,586:607]
        return bool((skip_label.min(axis=2) > 215).mean() > .17
                    and (skip_background.max(axis=2) < 50).mean() > .9)

    @staticmethod
    def selected_team(frame):
        numbers = []
        for number,y in ((1,172),(2,258)):
            red,green,blue = frame[y:y+32,56:79].astype('int16').transpose(2, 0, 1)
            orange = (red >= 210) & (green >= 140) & (blue <= 110) & (red-green >= 25)
            if orange.mean() >= .85:
                numbers.append(number)
        return numbers[0] if len(numbers) == 1 else None

    @staticmethod
    def weapon_targets(frame):
        def border_visible(edge, y):
            # Even the lowest row has two visible vertical edges above the footer.
            border = frame[y-25:min(y+26,640),edge-5:edge+6].astype('int16')
            red, green, blue = border.transpose(2, 0, 1)
            glow = ((red >= 200) & (green >= 185) & (blue >= 105) &
                    (red-blue >= 12) & (green-blue >= 6))
            return glow.mean(axis=0).max() >= .8

        return tuple((x, y) for x, y in WEAPON_SLOTS
                     if border_visible(x-46, y) and border_visible(x+46, y))

    def scene(self, frame):
        if frame.shape != (720, 1280, 3):
            return XumiScreen()

        if self.relay(frame):
            return XumiScreen('relay')

        if (self.match(frame, 'team-score-confirm-message', (518, 337, 842, 405)) and
                self.match(frame, 'equipment-confirm-title', (585, 252, 694, 314))):
            return XumiScreen('score_confirm')

        if (self.match(frame, 'result-complete', (205, 148, 435, 307)) and
                self.match(frame, 'result-total', (82, 393, 206, 445))):
            if self.match(frame, 'result-return', (765, 601, 1015, 668)):
                return XumiScreen('result_done', (890, 635), back=(890, 635))
            if (self.match(frame, 'result-unplayed', (437, 354, 532, 401)) and
                    self.match(frame, 'result-next', (878, 603, 1061, 669))):
                return XumiScreen('result_next', (976, 635))

        lobby = self.match(frame, 'lobby-title', TITLE)
        overlay = lobby or self.match(frame, 'lobby-overlay', TITLE)
        if overlay:
            if self.daily.guide_chrome(frame):
                return XumiScreen('guide', self.daily.guide_point(frame))
            if self.match(frame, 'tasks-title', (550, 85, 728, 150)):
                close = self.match(frame, 'tasks-close', (1040, 86, 1119, 150))
                if close:
                    return XumiScreen('tasks', close, close)
            if lobby and self.match(frame, 'lobby-daily', DAILY_BUTTON):
                # Only the brush-stroke labels stay fixed while the portals glow.
                fangcun = (330, 355) if self.match(frame, 'fangcun-entry', (232, 507, 316, 559)) else None
                daqian = (965, 355) if self.match(frame, 'daqian-entry', (965, 508, 1049, 560)) else None
                return XumiScreen('lobby', (1104, 52), BACK, fangcun, daqian)

        team_area = (91, 0, 325, 78)
        team = self.match(frame, 'team-title', team_area)
        if team or self.match(frame, 'team-overlay', team_area):
            if self.match(frame, 'roster-fixed', (519, 105, 761, 148)):
                return XumiScreen('roster', back=ROSTER_CLOSE)
            if (self.match(frame, 'roster-own', (463, 107, 643, 152)) and
                    self.match(frame, 'roster-own-discard', (298, 566, 453, 628))):
                return XumiScreen('own_roster', back=(370, 597))
            if team and self.match(frame, 'team-skills', (291, 578, 443, 642)):
                return XumiScreen('team', back=BACK, team_number=self.selected_team(frame), team_mode=team_mode(frame))

        if (self.match(frame, 'equipment-title-dim', TITLE) and
                self.match(frame, 'equipment-confirm-title', (585, 252, 694, 314)) and
                self.match(frame, 'equipment-confirm-message', (450, 336, 825, 407)) and
                self.match(frame, 'equipment-confirm-save', (700, 489, 922, 564))):
            return XumiScreen('equipment_confirm', (810, 529), team_number=self.equipment_team(frame))

        if self.match(frame, 'equipment-title', TITLE):
            if (self.match(frame, 'equipment-equip', (1085, 542, 1184, 601)) and
                    self.match(frame, 'equipment-cancel', (1085, 623, 1184, 678))):
                return XumiScreen('equipment_detail', (1134, 573), (1134, 650), team_number=self.equipment_team(frame))
            if self.match(frame, 'equipment-save', (1017, 645, 1203, 711)):
                support_rows = (495,) if self.match(frame, 'equipment-support', (87, 513, 172, 557)) else ()
                return XumiScreen('equipment', back=BACK, team_number=self.equipment_team(frame),
                                  support_rows=support_rows)
            if targets := self.weapon_targets(frame):
                return XumiScreen('equipment_target', equipment_targets=targets, team_number=self.equipment_team(frame))

        if start := self.match(frame, 'setup-start', (989, 600, 1197, 668)):
            for mode in ('fangcun', 'daqian'):
                if self.match(frame, mode+'-setup', TITLE):
                    notice = None
                    if (mode == 'daqian' and
                            self.match(frame, 'start-failed', (285, 234, 542, 284)) and
                            self.match(frame, 'start-no-weapon', (615, 234, 1007, 284))):
                        notice = 'missing_weapon'
                    return XumiScreen(mode+'_setup', back=BACK, notice=notice,
                                      setup=trial_setup(frame,start,notice))

        if self.daily.match(frame, 'hall', (0, 0, 400, 120)):
            entry = self.match(frame, 'hall-entry', (110, 108, 1280, 236))
            return XumiScreen('hall', (entry[0], 400) if entry else None, (77, 54))
        if self.daily.home(frame):
            return XumiScreen('home', (865, 600))
        return XumiScreen()
