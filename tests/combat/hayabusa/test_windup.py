import unittest
from types import SimpleNamespace

import numpy as np

from ninja3_assistant.combat.hayabusa.windup import FireSetupCue


class FireSetupCueTest(unittest.TestCase):
    def setUp(self):
        self.cue = FireSetupCue()
        self.screen = SimpleNamespace(active=True, result=False, player_x=350)
        self.fire = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.fire[330:345, 1050:1070] = (255, 155, 40)

    def test_two_observations_one_event_and_reacquisition(self):
        self.assertIsNone(self.cue.update(self.fire, self.screen, 10.0))
        event = self.cue.update(self.fire, self.screen, 10.2)
        self.assertEqual(event.family, 'fire')
        self.assertIsNone(event.response_class)
        self.assertTrue(event.experimental)
        self.assertEqual(event.first_observed_at, 10.0)
        self.assertIsNone(self.cue.update(self.fire, self.screen, 10.4))
        self.screen.player_x = None
        self.assertIsNone(self.cue.update(self.fire, self.screen, 10.6))
        self.screen.player_x = 350
        self.assertIsNone(self.cue.update(self.fire, self.screen, 10.8))
        self.cue.reset()
        self.assertIsNone(self.cue.update(self.fire, self.screen, 11.0))
        self.assertIsNotNone(self.cue.update(self.fire, self.screen, 11.2))

    def test_warning_and_own_side_effect_do_not_trigger(self):
        warning = self.fire.copy()
        warning[230:490, :] = (225, 8, 24)
        warning[330:345, 1050:1070] = (255, 155, 40)
        own = np.zeros_like(self.fire)
        own[330:345, 390:410] = (255, 155, 40)
        for index in range(3):
            self.assertIsNone(self.cue.update(warning, self.screen, 10+index*.2))
            self.assertIsNone(self.cue.update(own, self.screen, 11+index*.2))

    def test_left_edge_fire_with_player_near_screen_center(self):
        self.screen.player_x = 606
        left = np.zeros_like(self.fire)
        left[190:205, 165:185] = (255, 155, 40)
        self.assertIsNone(self.cue.update(left, self.screen, 10.0))
        event = self.cue.update(left, self.screen, 10.2)
        self.assertEqual(event.family, 'fire')
        self.assertLess(event.center[0], self.screen.player_x)


if __name__ == '__main__':
    unittest.main()
