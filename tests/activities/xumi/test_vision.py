"""Replay the Xumi navigation screens captured on instance 152."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ninja3_assistant.activities.bounty.combat import BountyCombatVision
from ninja3_assistant.activities.xumi.vision import XumiVision
from ninja3_assistant.paths import project_root


EVIDENCE = project_root()/'data/evidence/2026-11-19/xumi-152'


def frame(name):
    return cv2.cvtColor(cv2.imread(str(EVIDENCE/name)), cv2.COLOR_BGR2RGB)


@unittest.skipUnless((EVIDENCE/'020-own-roster.png').is_file(), 'local Xumi screenshots unavailable')
class XumiVisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = XumiVision()

    def test_hall_guide_lobby_and_daily_opening(self):
        cases = (
            ('000-home.png', 'home', (865, 600)),
            ('001-entry.png', 'hall', (725, 400)),
            ('002-event.png', 'guide', (1083, 595)),
            ('003-guide-2.png', 'guide', (1083, 595)),
            ('004-guide-3.png', 'guide', (1083, 595)),
            ('005-guide-4.png', 'guide', (1083, 595)),
            ('006-guide-5.png', 'guide', (640, 100)),
            ('008-lobby.png', 'lobby', (1104, 52)),
            ('009-daily.png', 'tasks', (1078, 118)),
            ('010-closed.png', 'lobby', (1104, 52)),
            ('017-lobby-again.png', 'lobby', (1104, 52)),
        )
        for name, kind, point in cases:
            with self.subTest(name=name):
                screen = self.vision.scene(frame(name))
                self.assertEqual((screen.kind, screen.point), (kind, point))
                if kind == 'lobby':
                    self.assertEqual((screen.fangcun, screen.daqian), ((330, 355), (965, 355)))

    def test_trial_team_and_roster_disambiguate_foreground(self):
        cases = (
            ('011-system.png', 'fangcun_setup', (47, 40)),
            ('012-system-team.png', 'team', (47, 40)),
            ('013-system-roster.png', 'roster', (624, 76)),
            ('014-system-selected.png', 'roster', (624, 76)),
            ('015-system-team-filled.png', 'team', (47, 40)),
            ('016-system-after.png', 'fangcun_setup', (47, 40)),
            ('018-own.png', 'daqian_setup', (47, 40)),
            ('019-own-team.png', 'team', (47, 40)),
            ('020-own-roster.png', 'own_roster', (370, 597)),
        )
        for name, kind, back in cases:
            with self.subTest(name=name):
                screen = self.vision.scene(frame(name))
                self.assertEqual((screen.kind, screen.back), (kind, back))
                self.assertIsNone(screen.point)

    def test_missing_portal_or_modal_anchor_is_not_actionable(self):
        lobby = frame('008-lobby.png')
        lobby[507:559, 232:316] = 0
        screen = self.vision.scene(lobby)
        self.assertIsNone(screen.fangcun)
        self.assertEqual(screen.daqian, (965, 355))

        tasks = frame('009-daily.png')
        tasks[86:150, 1040:1119] = 0
        screen = self.vision.scene(tasks)
        self.assertEqual(screen.kind, 'unknown')
        self.assertIsNone(screen.point)

        resized = np.zeros((360, 640, 3), dtype=np.uint8)
        self.assertEqual(self.vision.scene(resized).kind, 'unknown')

    @unittest.skipUnless((EVIDENCE/'024-fangcun-transition.png').is_file(), 'Xumi result screenshot unavailable')
    def test_first_boss_result_continues_to_unplayed_boss_only(self):
        result = frame('024-fangcun-transition.png')
        screen = self.vision.scene(result)
        self.assertEqual((screen.kind, screen.point, screen.back), ('result_next', (976, 635), None))
        for x1,y1,x2,y2 in ((205,148,435,307),(82,393,206,445),(437,354,532,401),(878,603,1061,669)):
            changed = result.copy()
            changed[y1:y2, x1:x2] = 0
            self.assertNotEqual(self.vision.scene(changed).kind, 'result_next')

    @unittest.skipUnless((EVIDENCE/'025-fangcun-final.png').is_file(), 'Xumi final result screenshot unavailable')
    def test_full_result_returns_after_both_bosses(self):
        result = frame('025-fangcun-final.png')
        screen = self.vision.scene(result)
        self.assertEqual((screen.kind, screen.point, screen.back), ('result_done', (890, 635), (890, 635)))
        self.assertNotEqual(self.vision.scene(frame('024-fangcun-transition.png')).kind, 'result_done')
        for x1,y1,x2,y2 in ((205,148,435,307),(82,393,206,445),(765,601,1015,668)):
            changed = result.copy()
            changed[y1:y2, x1:x2] = 0
            self.assertNotEqual(self.vision.scene(changed).kind, 'result_done')

    @unittest.skipUnless((EVIDENCE/'029-own-equipment.png').is_file(), 'Xumi equipment screenshot unavailable')
    def test_equipment_screen_and_daqian_missing_weapon_notice_are_distinct(self):
        equipment = frame('029-own-equipment.png')
        screen = self.vision.scene(equipment)
        self.assertEqual((screen.kind, screen.back, screen.point), ('equipment', (47, 40), None))
        equipment[645:711, 1017:1203] = 0
        self.assertNotEqual(self.vision.scene(equipment).kind, 'equipment')

        setup = frame('027-own-start-response.png')
        screen = self.vision.scene(setup)
        self.assertEqual((screen.kind, screen.notice, screen.point), ('daqian_setup', 'missing_weapon', None))
        self.assertIsNone(self.vision.scene(frame('018-own.png')).notice)
        setup[234:284, 615:1007] = 0
        self.assertIsNone(self.vision.scene(setup).notice)

    @unittest.skipUnless((EVIDENCE/'032-own-weapon-targets.png').is_file(), 'Xumi target screenshot unavailable')
    def test_equipment_detail_and_weapon_targets_require_visible_controls(self):
        detail = frame('031-own-weapon-selected.png')
        screen = self.vision.scene(detail)
        self.assertEqual((screen.kind, screen.point, screen.back),
                         ('equipment_detail', (1134, 573), (1134, 650)))
        detail[623:678, 1085:1184] = 0
        self.assertNotEqual(self.vision.scene(detail).kind, 'equipment_detail')

        target = frame('032-own-weapon-targets.png')
        screen = self.vision.scene(target)
        self.assertEqual((screen.kind, screen.point, screen.back), ('equipment_target', None, None))
        expected = tuple((x, y) for y in (211, 345, 495, 628) for x in (253, 378))
        self.assertEqual(screen.equipment_targets, expected)
        target[186:237, 202:213] = 0  # One missing side invalidates that particular slot.
        self.assertEqual(self.vision.scene(target).equipment_targets, expected[1:])
        target[151:641, 201:437] = 0
        self.assertNotEqual(self.vision.scene(target).kind, 'equipment_target')

    @unittest.skipUnless((EVIDENCE/'035-own-gear-progress.png').is_file(), 'Xumi equipment confirm unavailable')
    def test_equipment_confirm_exposes_only_the_verified_save(self):
        confirm = frame('035-own-gear-progress.png')
        screen = self.vision.scene(confirm)
        self.assertEqual((screen.kind, screen.point, screen.back), ('equipment_confirm', (810, 529), None))
        for x1,y1,x2,y2 in ((82,0,320,88),(585,252,694,314),(450,336,825,407),(700,489,922,564)):
            changed = confirm.copy()
            changed[y1:y2,x1:x2] = 0
            self.assertNotEqual(self.vision.scene(changed).kind, 'equipment_confirm')

    @unittest.skipUnless((EVIDENCE/'053-own-team2-gear.png').is_file(), 'Xumi second team screenshot unavailable')
    def test_equipment_heading_and_active_tabs_identify_team(self):
        cases = (
            ('029-own-equipment.png', 'equipment', 1),
            ('031-own-weapon-selected.png', 'equipment_detail', 1),
            ('032-own-weapon-targets.png', 'equipment_target', 1),
            ('035-own-gear-progress.png', 'equipment_confirm', 1),
            ('034-own-equipment-saved.png', 'team', 1),
            ('052-own-team2.png', 'team', 2),
            ('053-own-team2-gear.png', 'equipment', 2),
        )
        for name,kind,number in cases:
            with self.subTest(name=name):
                screen = self.vision.scene(frame(name))
                self.assertEqual((screen.kind,screen.team_number), (kind,number))

        gear = frame('053-own-team2-gear.png')
        gear[70:105,814:842] = 0
        self.assertIsNone(self.vision.scene(gear).team_number)
        team = frame('052-own-team2.png')
        team[172:204,56:79] = team[258:290,56:79]
        self.assertIsNone(self.vision.scene(team).team_number)

    @unittest.skipUnless((EVIDENCE/'053-own-team2-gear.png').is_file(), 'Xumi support row screenshot unavailable')
    def test_equipment_support_row_requires_its_team_badge(self):
        gear = frame('053-own-team2-gear.png')
        screen = self.vision.scene(gear)
        self.assertEqual((screen.kind, screen.team_number, screen.support_rows), ('equipment', 2, (495,)))
        self.assertEqual(self.vision.scene(frame('029-own-equipment.png')).support_rows, ())
        self.assertEqual(self.vision.scene(frame('032-own-weapon-targets.png')).support_rows, ())
        gear[513:557,87:172] = 0
        self.assertEqual(self.vision.scene(gear).support_rows, ())

    @unittest.skipUnless((EVIDENCE/'daqian-red-ninjutsu-boss2.png').is_file(), 'Xumi battle screenshot unavailable')
    def test_red_ninjutsu_requires_the_red_glyph_shape(self):
        red = frame('daqian-red-ninjutsu-boss2.png')
        combat = BountyCombatVision()
        with patch.object(combat.opening,'_ninjutsu_point',return_value=(70,330)):
            state = combat.detect(red)
        self.assertTrue(state.boss_active)
        self.assertEqual(state.ninjutsu_point,(55,60))
        self.assertIsNone(combat._red_ninjutsu_point(frame('daqian-blue-ninjutsu-boss1.png')))
        self.assertIsNone(combat._red_ninjutsu_point(frame('025-fangcun-final.png')))
        gray = cv2.cvtColor(cv2.cvtColor(red, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        self.assertIsNone(combat._red_ninjutsu_point(gray))
        red[39:85,28:84] = (255, 0, 0)
        self.assertIsNone(combat._red_ninjutsu_point(red))

    @unittest.skipUnless((EVIDENCE/'054-fangcun-score-confirm.png').is_file(), 'Xumi score confirmation unavailable')
    def test_team_score_confirmation_exposes_no_action(self):
        confirm = frame('054-fangcun-score-confirm.png')
        screen = self.vision.scene(confirm)
        self.assertEqual((screen.kind, screen.point, screen.back), ('score_confirm', None, None))
        self.assertEqual(self.vision.scene(frame('035-own-gear-progress.png')).kind, 'equipment_confirm')
        self.assertNotEqual(self.vision.scene(frame('055-cancel-score.png')).kind, 'score_confirm')
        confirm[349:391,535:823] = 0
        self.assertNotEqual(self.vision.scene(confirm).kind, 'score_confirm')
