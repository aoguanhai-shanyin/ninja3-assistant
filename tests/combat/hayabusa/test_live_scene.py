from ninja3_assistant.paths import project_root
import unittest

import cv2

from ninja3_assistant.combat.hayabusa.scene import HayabusaScene
from ninja3_assistant.combat.hayabusa.vision import HayabusaVision


FIXTURES = project_root()/'tests/fixtures/hayabusa'


class HayabusaLiveSceneTest(unittest.TestCase):
    def setUp(self):
        self.vision = HayabusaVision(use_gpu=False, recognize_boards=False, detector=False)

    def frame(self, name):
        return cv2.cvtColor(cv2.imread(str(FIXTURES/name)), cv2.COLOR_BGR2RGB)

    def test_shadow_defeat_stops_combat_without_claiming_active_battle(self):
        frame = self.frame('live-shadow-defeat.png')
        self.assertTrue(HayabusaScene()._eligible(frame))
        state, tracks = self.vision.detect(frame, 1.)
        self.assertEqual(state.result, 'defeat')
        self.assertFalse(state.active)
        self.assertEqual(tracks, [])

    def test_training_menu_with_shadow_name_is_not_a_battle_or_result(self):
        frame = self.frame('live-shadow-menu.png')
        self.assertFalse(HayabusaScene()._eligible(frame))
        state, tracks = self.vision.detect(frame, 1.)
        self.assertFalse(state.active)
        self.assertIsNone(state.result)
        self.assertEqual(tracks, [])

    def test_defeat_requires_both_result_title_and_matching_boss_name(self):
        frame = self.frame('live-shadow-defeat.png')
        for bounds in ((500, 0, 800, 200), (240, 240, 660, 400)):
            with self.subTest(bounds=bounds):
                image = frame.copy()
                x1, y1, x2, y2 = bounds
                image[y1:y2, x1:x2] = 0
                state, _ = self.vision.detect(image, 1.)
                self.assertIsNone(state.result)
                self.assertFalse(state.active)

    def test_shadow_battle_hud_requires_both_live_name_and_portrait(self):
        frame = self.frame('live-shadow-battle-072.jpg')
        scene = HayabusaScene()
        self.assertTrue(scene._eligible(frame))
        state, _ = self.vision.detect(frame, 1.)
        self.assertTrue(state.active)
        self.assertAlmostEqual(state.player_head_x, 406, delta=8)
        self.assertAlmostEqual(state.player_head_y, 379, delta=8)
        self.assertAlmostEqual(state.player_x, 381, delta=8)
        self.assertAlmostEqual(state.player_y, 404, delta=8)
        self.assertEqual(state.right_control, 'jump')
        self.assertTrue(state.grounded)
        self.assertEqual(state.side, 'above')
        for bounds in ((448, 35, 550, 68), (375, 22, 448, 96)):
            with self.subTest(bounds=bounds):
                image = frame.copy()
                x1, y1, x2, y2 = bounds
                image[y1:y2, x1:x2] = 0
                self.assertFalse(scene._eligible(image))
                self.assertFalse(self.vision.detect(image, 2.)[0].active)

    def test_intro_warning_relay_and_dimmed_battle_cannot_enable_combat(self):
        scene = HayabusaScene()
        for name in ('live-shadow-intro.jpg', 'live-shadow-warning.jpg', 'live-shadow-relay.jpg'):
            with self.subTest(name=name):
                frame = self.frame(name)
                self.assertFalse(scene._eligible(frame))
                state, tracks = self.vision.detect(frame, 1.)
                self.assertFalse(state.active)
                self.assertIsNone(state.result)
                self.assertEqual(tracks, [])
        frame = (self.frame('live-shadow-battle-072.jpg')*.35).astype('uint8')
        self.assertFalse(scene._eligible(frame))
        self.assertFalse(self.vision.detect(frame, 2.)[0].active)

    def test_tracked_face_motion_bridges_unseen_animation_but_cannot_outlive_visual_identity(self):
        first = self.frame('live-shadow-battle-075.jpg')
        unseen = self.frame('live-shadow-battle-076.jpg')
        self.assertIsNotNone(self.vision.detect(first, 1.)[0].player_x)
        for timestamp in (1.5, 1.9):
            state, _ = self.vision.detect(unseen, timestamp)
            self.assertAlmostEqual(state.player_head_x, 406, delta=8)
            self.assertAlmostEqual(state.player_x, 381, delta=8)
            self.assertTrue(state.grounded)
        state, _ = self.vision.detect(unseen, 2.2)
        self.assertIsNone(state.player_x)
        self.assertIsNone(state.player_y)

    def test_hud_remains_visible_after_player_is_fallen_without_locking_onto_boss(self):
        state, _ = self.vision.detect(self.frame('live-shadow-battle-080.jpg'), 1.)
        self.assertTrue(state.active)
        self.assertIsNone(state.player_x)
        self.assertIsNone(state.player_y)
        self.assertEqual(state.side, 'unknown')


if __name__ == '__main__':
    unittest.main()
