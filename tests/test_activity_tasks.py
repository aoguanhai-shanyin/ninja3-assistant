from pathlib import Path
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from activity_tasks import TaskVision

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT/'data/intelligence-simulation/live'


def load(name):
    return cv2.cvtColor(cv2.imread(str(LIVE/name)), cv2.COLOR_BGR2RGB)


def word(text, x, y, w=65, h=20):
    return {'text': text, 'bbox': [x, y, w, h]}


class ActivityTasksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = TaskVision()
        cls.tasks = load('20261113-182048-after-match1-tasks.png')
        cls.reward = load('20261113-182110-achievement-claim-sample.png')

    def test_real_task_page_uses_only_daily_rows_not_claimable_achievement(self):
        result = self.vision.detect(self.tasks, {'words': [
            word('1/2', 1038, 182, 30), word('23/200', 1001, 282),
            word('领取奖励', 930, 443, 112),
        ]})
        self.assertEqual(result.kind, 'tasks')
        self.assertEqual(result.present, ('games', 'damage'))
        self.assertEqual(result.daily_states, {'games': 'incomplete', 'damage': 'incomplete'})
        self.assertEqual(result.progress, {'games': (1, 2), 'damage': (23, 200)})
        self.assertEqual(result.claim_points, {'games': None, 'damage': None})

    def test_actual_medal_reward_requires_title_icon_and_confirmation(self):
        result = self.vision.detect(self.reward)
        self.assertEqual((result.kind, result.confirm_point), ('reward', (640, 644)))
        for region in ((slice(90, 132), slice(557, 722)),
                       (slice(279, 359), slice(592, 692)),
                       (slice(627, 664), slice(609, 670))):
            modified = self.reward.copy()
            modified[region] = 0
            self.assertNotEqual(self.vision.detect(modified).kind, 'reward')

    def test_current_character_daily_and_daily_reward(self):
        path = ROOT/'data/intelligence-simulation/runs/20261114-132033-551800/000023-tasks-wait.png'
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        result = self.vision.detect(image)
        self.assertEqual(result.present, ('games', 'characters'))
        self.assertEqual(result.daily_states['characters'], 'claimable')
        self.assertIsNotNone(result.claim_points['characters'])
        self.assertEqual(self.vision.detect(load('20261114-characters-reward.png')).kind, 'reward')

    def test_reward_amount_is_not_part_of_icon_recognition(self):
        # Synthetic amount removal: the real sample has x100; daily uses x50.
        modified = self.reward.copy()
        modified[359:394, 590:710] = 0
        self.assertEqual(self.vision.detect(modified).kind, 'reward')

    def test_actual_post_achievement_claim_page_keeps_daily_tasks(self):
        after = load('20261113-182145-reward-confirm.png')
        result = self.vision.detect(after)
        self.assertEqual(result.kind, 'tasks')
        self.assertEqual(result.present, ('games', 'damage'))
        self.assertEqual(result.daily_states, {'games': 'unknown', 'damage': 'unknown'})

    def test_synthetic_removed_first_row_moves_damage_and_achievement_up(self):
        # This is layout testing from real crop pixels, not a claimed-state sample.
        shifted = self.tasks.copy()
        shifted[161:478, 180:1090] = self.tasks[263:580, 180:1090]
        shifted[478:580, 180:1090] = 0
        result = self.vision.detect(shifted, {'words': [word('23/200', 1001, 180)]})
        self.assertEqual(result.kind, 'tasks')
        self.assertEqual(result.present, ('damage',))
        self.assertEqual(result.daily_states, {'games': 'unknown', 'damage': 'incomplete'})
        self.assertEqual(result.progress['damage'], (23, 200))
        self.assertLess(result.header_bounds[1][1], 300)

    def test_synthetic_empty_daily_section_never_marks_missing_rows_complete(self):
        shifted = self.tasks.copy()
        shifted[160:194, 570:702] = self.tasks[361:395, 570:702]
        result = self.vision.detect(shifted)
        self.assertEqual(result.kind, 'tasks')
        self.assertEqual(result.present, ())
        self.assertEqual(result.daily_states, {'games': 'unknown', 'damage': 'unknown'})

    def test_synthetic_daily_claim_button_must_belong_to_a_known_daily_label(self):
        # Copy the observed achievement claim text into the first daily row.
        changed = self.tasks.copy()
        changed[190:222, 928:1043] = self.tasks[441:473, 928:1043]
        result = self.vision.detect(changed)
        self.assertEqual(result.daily_states['games'], 'claimable')
        self.assertIsNotNone(result.claim_points['games'])
        self.assertIsNone(result.claim_points['damage'])

    def test_wrong_or_finished_ratio_without_claim_button_remains_unknown(self):
        result = self.vision.detect(self.tasks, {'words': [
            word('2/2', 1038, 182, 30), word('23/20', 1001, 282),
        ]})
        self.assertEqual(result.daily_states, {'games': 'unknown', 'damage': 'unknown'})
        self.assertIsNone(result.progress['damage'])

    def test_read_scales_dynamic_crop_and_maps_ocr_back_to_source(self):
        engine = Mock()
        # Actual detected crop is x800,y157; these are its 3x coordinates.
        engine.read.return_value = {'words': [word('1/2', 714, 75, 90, 60),
                                              word('23/200', 603, 375, 195, 60)]}
        result = self.vision.read(self.tasks, engine)
        self.assertEqual(engine.read.call_args.args[0].shape, (594, 870, 3))
        self.assertEqual(result.progress, {'games': (1, 2), 'damage': (23, 200)})
        self.assertEqual(result.daily_states, {'games': 'incomplete', 'damage': 'incomplete'})

    def test_cancelled_ocr_does_not_create_task_completion(self):
        engine = Mock()
        engine.read.return_value = None
        result = self.vision.read(self.tasks, engine)
        self.assertEqual(result.daily_states, {'games': 'unknown', 'damage': 'unknown'})

    def test_other_or_obscured_pages_do_not_establish_daily_presence(self):
        for image in (np.zeros((720, 1280, 3), np.uint8),
                      load('20261113-140318-guide-close.png'),
                      load('20261113-135959-trial-back.png')):
            self.assertEqual(self.vision.detect(image).kind, 'unknown')


if __name__ == '__main__':
    unittest.main()
