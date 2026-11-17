"""Private battle-frame regression; never connects to a device."""
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ninja3_assistant.activities.bounty.combat import BountyCombatVision
from ninja3_assistant.activities.bounty.daily_runner import _recognize_frame
from ninja3_assistant.activities.bounty.daily_vision import DailyBountyVision
from ninja3_assistant.paths import project_root


@unittest.skipUnless((project_root()/'data/skate-video-study/live-easy-02/progress2.png').is_file(),
                     'local battle screenshots unavailable')
class BountyCombatReplayTest(unittest.TestCase):
    def frame(self, name):
        path = project_root()/name
        if not path.is_file():
            self.skipTest('local battle screenshot unavailable')
        return cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)

    def test_coin_running_does_not_release_ninjutsu_before_boss_health_appears(self):
        vision = BountyCombatVision()
        state = vision.detect(self.frame('data/hayabusa-windup-v1/live-calibration-first/frame-00280.jpg'))
        self.assertTrue(state.active)
        self.assertTrue(state.ground)
        self.assertFalse(state.on_rope)
        self.assertFalse(state.boss_active)
        self.assertIsNone(state.ninjutsu_point)

    def test_recorded_ground_controls_survive_all_background_brightness_changes(self):
        root=project_root()/'data/evidence/2026-11-17/bounty-combat-recognition'
        cases=json.loads((root/'ground-cases.json').read_text(encoding='utf-8'))
        vision=BountyCombatVision()
        self.assertEqual(len(cases),74)
        for case in cases:
            with self.subTest(frame=case['path']):
                frame=cv2.cvtColor(cv2.imread(str(root/case['path'])),cv2.COLOR_BGR2RGB)
                state=vision.detect(frame)
                self.assertTrue(state.active)
                self.assertTrue(state.ground)
                self.assertFalse(state.on_rope)
                self.assertFalse(state.boss_active)

    def portrait_frame(self, number=3804):
        return self.frame('data/evidence/2026-11-17/bounty-combat-recognition/'
                          f'boss-first-{number}.png')

    def test_first_portrait_frame_enables_ninjutsu_before_coloured_health_fill(self):
        vision=BountyCombatVision()
        self.assertFalse(vision.detect(self.portrait_frame(3803)).boss_active)
        first=self.portrait_frame()
        self.assertFalse(vision.boss_bar(first))
        state=vision.detect(first)
        self.assertTrue(state.boss_active)
        self.assertEqual(state.ninjutsu_point,(70,330))

    def test_portrait_is_not_gated_by_temporarily_hidden_movement_controls(self):
        frame=self.portrait_frame()
        frame[550:715,80:1190]=0
        state=BountyCombatVision().detect(frame)
        self.assertFalse(state.ground)
        self.assertFalse(state.on_rope)
        self.assertTrue(state.active)
        self.assertTrue(state.boss_active)
        self.assertEqual(state.ninjutsu_point,(70,330))

    def test_portrait_matching_uses_border_and_level_instead_of_this_boss_art(self):
        source=self.portrait_frame()
        y,x=np.indices(source.shape[:2])
        centre=(x-410)**2+(y-59)**2<24**2
        for colour in ((235,40,40),(40,80,235)):
            with self.subTest(colour=colour):
                frame=source.copy();frame[centre]=colour
                self.assertTrue(BountyCombatVision().detect(frame).boss_active)

    def test_portrait_needs_both_border_and_level_and_a_visible_skill_hud(self):
        source=self.portrait_frame()
        for box in ((444,32,490,70),(365,15,444,105),(0,260,180,420)):
            with self.subTest(box=box):
                frame=source.copy();x1,y1,x2,y2=box;frame[y1:y2,x1:x2]=0
                if box[0]==0:
                    frame[550:715,80:1190]=0
                    self.assertFalse(BountyCombatVision().detect(frame).active)
                else:
                    self.assertFalse(BountyCombatVision().detect(frame).boss_active)
        self.assertFalse(BountyCombatVision().detect((source*.35).astype('uint8')).active)

    def test_player_skill_avatar_and_actual_menus_do_not_become_a_boss(self):
        files=(
            'data/evidence/2026-11-17/bounty-combat-recognition/102058.png',
            'data/experiments/2026-11-17/bounty-validation/team-prep/095948-122600/0009-bounty_room-unavailable.png',
            'data/runtime/daily-cleanup/runs/20261117-101325-011400-bounty-A/0007-result-tap.png',
            'data/runtime/daily-cleanup/runs/20261117-095019-052000-bounty_shop/0005-unknown-wait.png')
        for path in files:
            with self.subTest(path=path):
                frame=self.frame(path)
                state=BountyCombatVision().detect(frame)
                self.assertFalse(state.boss_active)
                self.assertIsNone(state.ninjutsu_point)
        frame=self.portrait_frame()
        # Even with a genuine Lv label, the player's skill portrait is not
        # interchangeable with the Boss's white circular border.
        frame[24:97,374:447]=cv2.resize(frame[0:140,0:125],(73,73))
        self.assertFalse(BountyCombatVision().detect(frame).boss_active)

    def test_health_bar_is_still_a_supplement_when_portrait_is_unavailable(self):
        frame=self.portrait_frame()
        frame[:110,340:850]=0
        frame[40:50,450:530]=240
        frame[65:80,430:810]=(245,165,0)
        frame[550:715,80:1190]=0
        state=BountyCombatVision().detect(frame)
        self.assertTrue(state.active)
        self.assertTrue(state.boss_active)
        self.assertEqual(state.ninjutsu_point,(70,330))

    def test_rope_without_boss_stays_idle_and_does_not_become_ground(self):
        vision = BountyCombatVision()
        state = vision.detect(self.frame('data/experiments/weekend-avoidance/2026-11-15/rope-replay/00400.jpg'))
        self.assertTrue(state.active)
        self.assertTrue(state.on_rope)
        self.assertFalse(state.ground)
        self.assertFalse(state.boss_active)

    def test_dark_modal_overlay_cannot_reuse_running_controls_behind_it(self):
        vision = BountyCombatVision()
        frame = self.frame('data/hayabusa-windup-v1/live-calibration-first/frame-00280.jpg')
        self.assertFalse(vision.detect((frame*.35).astype('uint8')).active)

    def test_brown_scenery_and_warning_transition_are_not_boss_health(self):
        folder='data/runtime/daily-cleanup/runs/20261116-135300-314700-bounty-D/'
        vision = BountyCombatVision()
        for name in ('0023-battle.png','0025-battle.png'):
            with self.subTest(name=name):
                frame = self.frame(folder+name)
                self.assertFalse(vision.boss_bar(frame))
                state = vision.detect(frame)
                self.assertTrue(state.active)
                self.assertFalse(state.boss_active)
                self.assertIsNone(state.ninjutsu_point)

    def test_confirmed_boss_phase_survives_health_flash_until_explicit_scene_exit(self):
        vision = BountyCombatVision()
        frame = self.frame('data/skate-video-study/live-easy-02/progress2.png')
        self.assertTrue(vision.detect(frame).boss_active)
        frame[55:95,400:840] = 240
        self.assertTrue(vision.detect(frame).boss_active)
        self.assertFalse(vision.detect(frame,battle_allowed=False).boss_active)

    def test_active_battle_path_does_not_scan_all_menu_templates(self):
        menus = DailyBountyVision()
        battle = BountyCombatVision()
        frame = self.frame('data/runtime/daily-cleanup/runs/20261116-135300-314700-bounty-D/0023-battle.png')
        with patch.object(menus,'navigation',wraps=menus.navigation) as full_navigation:
            page,state = _recognize_frame(frame,menus,battle)
        self.assertIsNone(page)
        self.assertTrue(state.ground)
        self.assertFalse(state.boss_active)
        full_navigation.assert_not_called()

    def test_bright_grade_modal_blocks_unchanged_ground_hud_on_the_fast_path(self):
        frame = self.frame('data/hayabusa-windup-v1/live-calibration-first/frame-00280.jpg')
        modal = self.frame('data/evidence/2026-11-16/daily-exploration/134030-bounty-grade-dialog.png')
        frame[140:550,430:920] = modal[140:550,430:920]
        battle = BountyCombatVision()
        self.assertTrue(battle.detect(frame).ground)
        page,state = _recognize_frame(frame,DailyBountyVision(),battle)
        self.assertEqual(page,'bounty_grade')
        self.assertFalse(state.active)

    def test_boss_health_on_both_sides_of_rope_enables_only_attack_inputs(self):
        for path in ('data/skate-video-study/live-easy-02/progress2.png',
                     'data/skate-video-study/live-cpp-boss-02/current.png'):
            with self.subTest(path=path):
                vision = BountyCombatVision()
                image = self.frame(path)
                state = vision.detect(image)
                self.assertTrue(state.on_rope)
                self.assertTrue(state.boss_active)
                self.assertEqual(state.ninjutsu_point,(55,60))
                self.assertFalse(vision.detect(image,battle_allowed=False).active)
                self.assertFalse(vision.boss_seen)


if __name__ == '__main__':
    unittest.main()
