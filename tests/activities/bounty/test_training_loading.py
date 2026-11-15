from ninja3_assistant.paths import project_root
import unittest

import cv2
import numpy as np

from ninja3_assistant.activities.bounty.flow import Flow
from ninja3_assistant.vision.vision import Screen, Vision


FIXTURES = project_root() / 'tests/fixtures' / 'hayabusa'


class TrainingLoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = Vision(use_gpu=False)

    def detect(self, name):
        frame = cv2.imread(str(FIXTURES / name))
        return self.vision.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def test_actual_training_menu_and_loading_are_distinct(self):
        menu = self.detect('live-shadow-menu.png')
        loading = self.detect('live-shadow-loading.png')
        self.assertTrue(menu.training)
        self.assertFalse(menu.game_loading)
        self.assertTrue(loading.game_loading)
        self.assertFalse(loading.training)
        self.assertFalse(self.vision.detect(np.zeros((720, 1280, 3), dtype=np.uint8)).game_loading)
        self.assertFalse(self.detect('live-shadow-defeat.png').game_loading)

    def test_loading_breaks_unknown_timer_without_claiming_battle(self):
        flow = Flow()
        self.assertEqual(flow.step(Screen(), 0).action, 'wait')
        self.assertEqual(flow.step(Screen(game_loading=True), 10).action, 'wait')
        self.assertEqual(flow.phase, 'start')
        self.assertEqual(flow.step(Screen(game_loading=True), 16).action, 'wait')
        self.assertEqual(flow.step(Screen(), 17).action, 'wait')
        self.assertEqual(flow.step(Screen(), 25).action, 'wait')
        self.assertEqual(flow.step(Screen(battle_active=True), 26).action, 'wait')
        self.assertEqual(flow.phase, 'battle')

    def test_training_page_waits_and_transition_has_a_fixed_deadline(self):
        flow = Flow()
        self.assertEqual(flow.step(Screen(training=True), 0).action, 'wait')
        self.assertEqual(flow.step(Screen(training=True), 60).action, 'wait')
        self.assertEqual(flow.step(Screen(), 61).action, 'wait')
        self.assertEqual(flow.step(Screen(), 76).action, 'wait')
        self.assertEqual(flow.step(Screen(), 104.9).action, 'wait')
        self.assertEqual(flow.step(Screen(), 105).action, 'reopen')
        self.assertEqual(flow.step(Screen(), 120).action, 'stop')

    def test_unknown_without_training_reopens_after_fifteen_seconds(self):
        flow = Flow()
        self.assertEqual(flow.step(Screen(), 0).action, 'wait')
        self.assertEqual(flow.step(Screen(), 14.9).action, 'wait')
        self.assertEqual(flow.step(Screen(), 15).action, 'reopen')

    def test_stuck_loading_remains_bounded_when_detection_flickers(self):
        flow = Flow()
        self.assertEqual(flow.step(Screen(game_loading=True), 0).action, 'wait')
        self.assertEqual(flow.step(Screen(), 10).action, 'wait')
        self.assertEqual(flow.step(Screen(game_loading=True), 20).action, 'wait')
        self.assertEqual(flow.step(Screen(game_loading=True), 30).action, 'wait')
        self.assertEqual(flow.step(Screen(game_loading=True), 45).action, 'reopen')


if __name__ == '__main__':
    unittest.main()
