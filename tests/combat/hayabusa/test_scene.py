from ninja3_assistant.paths import project_root
from concurrent.futures import Future
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ninja3_assistant.combat.hayabusa.scene import HayabusaScene
from ninja3_assistant.combat.hayabusa.vision import HayabusaScreen


class HayabusaSceneTest(unittest.TestCase):
    def test_other_scene_does_not_initialize_combat_engine_or_create_logs(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(HayabusaScene, '_initialize') as engine:
            scene = HayabusaScene(record_root=directory)
            self.assertIsNone(scene.step(np.zeros((720, 1280, 3), np.uint8), 1))
            self.assertIsNone(scene.step(object(), 2))
            engine.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_gate_recognizes_reference_hud_without_loading_gpu(self):
        scene = HayabusaScene()
        source = project_root()/'tests/fixtures'/'hayabusa'/'frame-0019.000.jpg'
        frame = cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB)
        self.assertTrue(scene._eligible(frame))
        self.assertFalse(scene._eligible(np.zeros_like(frame)))
        self.assertIsNone(scene.vision)

    def test_confirmed_scene_keeps_ownership_while_hud_is_hidden(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('ninja3_assistant.combat.hayabusa.scene.HayabusaVision') as engine:
            engine.return_value.detect.return_value = (HayabusaScreen(), [])
            engine.return_value.backend = 'test'
            scene = HayabusaScene(record_root=directory)
            frame = np.zeros((720, 1280, 3), np.uint8)
            with patch.object(scene, '_eligible', return_value=True):
                self.assertTrue(scene.step(frame, 1).claimed)
            with patch.object(scene, '_eligible', side_effect=AssertionError('re-gated owned scene')):
                self.assertTrue(scene.step(frame, 1.1).claimed)
            self.assertTrue(scene.owned)
            self.assertEqual(scene.clicks, 0)
            scene.close()

    def test_pending_gpu_holds_actions_until_a_fresh_frame(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('ninja3_assistant.combat.hayabusa.scene.HayabusaVision') as engine:
            vision = engine.return_value
            vision.backend = 'test'
            vision.detect.return_value = (HayabusaScreen(), [])
            scene = HayabusaScene(record_root=directory)
            scene._vision_ready = Future()
            frame = np.zeros((720, 1280, 3), np.uint8)
            with patch.object(scene, '_eligible', return_value=True):
                waiting = scene.step(frame, 1)
            self.assertTrue(waiting.claimed)
            self.assertTrue(waiting.active)
            self.assertTrue(waiting.stalled)
            self.assertIsNone(waiting.action)
            vision.detect.assert_not_called()
            scene._vision_ready.set_result(vision)
            fresh = np.ones_like(frame)
            self.assertFalse(scene.step(fresh, 1.1).stalled)
            vision.detect.assert_called_once_with(fresh, 1.1)
            scene.close()

    def test_pause_preserves_but_leaving_battle_clears_input_and_recipe_state(self):
        with patch('ninja3_assistant.combat.hayabusa.scene.HayabusaVision'):
            scene = HayabusaScene(debug=True)
        scene.policy.weapon_used = True
        scene.policy.jumps = 4
        scene.policy.recipe.blocked_epoch = 7
        scene.policy.recipe.blocked_board = 'p1_fire_1'
        scene.reset()
        self.assertTrue(scene.policy.weapon_used)
        self.assertEqual(scene.policy.jumps, 4)
        scene.release()
        self.assertFalse(scene.policy.weapon_used)
        self.assertEqual(scene.policy.jumps, 0)
        self.assertIsNone(scene.policy.recipe.blocked_epoch)
        self.assertFalse(scene.owned)


if __name__ == '__main__':
    unittest.main()
