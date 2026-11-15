import json
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np

from boss_policy import BossAction
from hayabusa_scene import HayabusaScene
from hayabusa_vision import HayabusaScreen
from hayabusa_windup import WindupCue


class FireProbeSceneTest(unittest.TestCase):
    def test_probe_is_opt_in_and_default_policy_is_preserved(self):
        with patch.dict('os.environ', {'NINJA_FIRE_PROBE':'0'}):
            scene = HayabusaScene()
        self.assertIsNone(scene.fire_probe)
        self.assertIsNone(scene.fire_cue)

    def test_one_real_callback_and_ascent_are_separately_recorded(self):
        state = HayabusaScreen(active=True, player_x=300, player_y=400, rope_y=430,
                               side='above', grounded=True, right_control='jump', left_control='down')
        cue = WindupCue('fire', None, 1.2, 1.1, (900, 340), 200)
        legacy = Mock()
        legacy.recipe = None
        legacy.last_reason = 'legacy'
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {'NINJA_FIRE_PROBE':'1', 'NINJA_PROBE_ONLY':'1'}), \
                patch('hayabusa_scene.HayabusaVision') as vision, patch('hayabusa_scene.FireSetupCue') as detector:
            vision.return_value.backend = 'test'
            vision.return_value.detect.return_value = (state, [])
            detector.return_value.update.return_value = cue
            scene = HayabusaScene(debug=True, policy=legacy, record_root=directory)
            for i, now in enumerate((1., 1.1, 1.2)):
                scene.step(np.full((720,1280,3), i, np.uint8), now)
            decision = scene.step(np.full((720,1280,3), 3, np.uint8), 1.8)
            self.assertEqual(decision.action.kind, 'jump')
            self.assertEqual(scene.fire_probe.state, 'proposed')
            self.assertIsNone(scene.pending_record['fire_probe']['on_pressed_at'])
            scene.completed(decision.action, 1.81)
            self.assertEqual(scene.fire_probe.state, 'waiting_ack')
            self.assertEqual(scene.pending_record['fire_probe']['on_pressed_at'], 1.81)
            vision.return_value.detect.return_value = (replace(state, player_y=380, grounded=False), [])
            self.assertIsNone(scene.step(np.full((720,1280,3), 4, np.uint8), 1.9).action)
            self.assertEqual(scene.fire_probe.state, 'complete')
            scene.step(np.full((720,1280,3), 5, np.uint8), 2.0)
            legacy.decide.assert_not_called()
            scene.close()
            records = [json.loads(x) for x in scene.log.read_text(encoding='utf-8').splitlines()]
            touched = next(r for r in records if r.get('sent'))
            self.assertEqual(touched['fire_probe']['cue']['family'], 'fire')
            self.assertIsNone(touched['fire_probe']['cue']['response_class'])
            self.assertEqual(touched['fire_probe']['on_pressed_at'], 1.81)
            self.assertTrue(any(r.get('fire_probe',{}).get('reason') == 'jump ascent confirmed' for r in records))

    def test_reset_prevents_repeat_of_consumed_event(self):
        with patch.dict('os.environ', {'NINJA_FIRE_PROBE':'1'}), patch('hayabusa_scene.HayabusaVision'):
            scene = HayabusaScene(debug=True)
            scene.fire_probe_attempted = True
            scene.fire_probe.start(1, 'fire_probe', 1., 1.)
            scene.reset()
            self.assertTrue(scene.fire_probe_attempted)
            self.assertEqual(scene.fire_probe.state, 'idle')
            scene.release()
            self.assertFalse(scene.fire_probe_attempted)

    def test_pair_waits_for_actual_first_press_and_airborne_confirmation(self):
        state = HayabusaScreen(active=True, player_x=300, player_y=400, rope_y=430,
                               side='above', grounded=True, right_control='jump', left_control='down')
        cue = WindupCue('fire', None, 1.2, 1.1, (900, 340), 200)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict('os.environ', {'NINJA_FIRE_PROBE':'1', 'NINJA_FIRE_PAIR':'1'}), \
                patch('hayabusa_scene.HayabusaVision') as vision, patch('hayabusa_scene.FireSetupCue') as detector:
            vision.return_value.backend = 'test'
            vision.return_value.detect.return_value = (state, [])
            detector.return_value.update.return_value = cue
            scene = HayabusaScene(debug=True, record_root=directory)
            for i, now in enumerate((1., 1.1, 1.2)):
                scene.step(np.full((720,1280,3), i, np.uint8), now)
            first = scene.step(np.full((720,1280,3), 3, np.uint8), 1.8).action
            self.assertIsNotNone(first)
            scene.completed(first, 1.81)
            vision.return_value.detect.return_value = (replace(state, player_y=380, grounded=False, airborne=True), [])
            self.assertIsNone(scene.step(np.full((720,1280,3), 4, np.uint8), 1.9).action)
            second = scene.step(np.full((720,1280,3), 5, np.uint8), 2.12).action
            self.assertIsNotNone(second)
            self.assertEqual(second.kind, 'jump')
            scene.completed(second, 2.13)
            self.assertEqual(scene.fire_probe.state, 'complete')
            self.assertEqual(scene.fire_probe.reason, 'air jump pressed; movement effect unverified')
            scene.close()

    def test_opt_in_shuriken_candidate_requests_one_down(self):
        state = HayabusaScreen(active=True, player_x=300, player_y=400, rope_y=430,
            side='above', grounded=True, right_control='jump', left_control='down',
            board_id='p1_shuriken_2', board_locked=True, board_confidence=.82,
            board_margin=.12, board_progress=.22, board_epoch=3)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict('os.environ', {'NINJA_FIRE_PROBE':'1', 'NINJA_SHURI_PROBE':'1'}), \
                patch('hayabusa_scene.HayabusaVision') as vision, patch('hayabusa_scene.FireSetupCue') as detector:
            vision.return_value.backend = 'test'
            vision.return_value.detect.return_value = (state, [])
            detector.return_value.update.return_value = None
            scene = HayabusaScene(debug=True, record_root=directory)
            for i, now in enumerate((1., 1.1)):
                self.assertIsNone(scene.step(np.full((720,1280,3), i, np.uint8), now).action)
            action = scene.step(np.full((720,1280,3), 2, np.uint8), 1.2).action
            self.assertEqual(action.kind, 'down')
            self.assertEqual(scene.pending_record['fire_probe']['cue']['response_class'], 'p1_shuriken_2')
            scene.completed(action, 1.21)
            vision.return_value.detect.return_value = (replace(state, right_control='up'), [])
            self.assertIsNone(scene.step(np.full((720,1280,3), 3, np.uint8), 1.3).action)
            self.assertEqual(scene.fire_probe.state, 'complete')
            scene.close()

    def test_unknown_attack_falls_back_and_fixed_candidate_owns_its_window(self):
        state = HayabusaScreen(active=True, player_x=300, player_y=400, rope_y=430,
                               side='above', grounded=True, right_control='jump', left_control='down')
        fallback = BossAction('down', (190, 630))
        policy = Mock()
        policy.recipe = None
        policy.last_reason = 'generic dodge'
        policy.decide.return_value = fallback
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict('os.environ', {'NINJA_FIRE_PROBE':'1', 'NINJA_PROBE_ONLY':'0'}), \
                patch('hayabusa_scene.HayabusaVision') as vision, patch('hayabusa_scene.FireSetupCue') as detector:
            vision.return_value.backend = 'test'
            vision.return_value.detect.return_value = (state, [])
            detector.return_value.update.return_value = None
            scene = HayabusaScene(debug=True, policy=policy, record_root=directory)
            for i, now in enumerate((1., 1.1)):
                scene.step(np.full((720,1280,3), i, np.uint8), now)
            self.assertIs(scene.step(np.full((720,1280,3), 2, np.uint8), 1.2).action, fallback)
            self.assertEqual(scene.pending_record['action_source'], 'generic')
            scene.frame_age(.21)
            self.assertEqual(scene.fire_probe.state, 'idle')
            detector.return_value.update.return_value = WindupCue('fire', None, 1.3, 1.2, (900, 340), 200)
            self.assertIsNone(scene.step(np.full((720,1280,3), 3, np.uint8), 1.3).action)
            self.assertEqual(policy.decide.call_count, 1)
            scene.close()


if __name__ == '__main__':
    unittest.main()
