from pathlib import Path
import unittest

import cv2
import numpy as np

from activity_vision import ActivityVision

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT/'data/intelligence-simulation/live'


def frame(name):
    return cv2.cvtColor(cv2.imread(str(LIVE/name)), cv2.COLOR_BGR2RGB)


class ActivityVisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = ActivityVision()

    def test_actual_hall_and_home_frames(self):
        hall = self.vision.detect(frame('20261113-140049-menu-leave.png'))
        self.assertEqual((hall.kind, hall.point), ('hall', (373, 389)))
        self.assertEqual(self.vision.detect(frame('20261113-140318-guide-close.png')).kind, 'event_home')

    def test_current_home_entry_requires_both_navigation_labels(self):
        for name in ('20261114-131650-resume.png', '20261114-live-home-unknown.png'):
            image = frame(name)
            self.assertEqual(self.vision.detect(image).kind, 'home')
            image[659:714, 625:735] = 0
            self.assertNotEqual(self.vision.detect(image).kind, 'home')

    def test_guide_overlay_and_other_modes_are_not_home(self):
        for name in ('20261113-140100-event-entry.png', '20261113-140113-guide-2.png',
                     '20261113-135959-trial-back.png', '20261113-140356-cards-01.png'):
            with self.subTest(name=name):
                self.assertEqual(self.vision.detect(frame(name)).kind, 'unknown')
        self.assertEqual(self.vision.detect(np.zeros((720, 1280, 3), np.uint8)).kind, 'unknown')

    def test_task_page_requires_both_daily_labels_and_achievement_boundary(self):
        image = frame('20261113-140331-tasks-first.png')
        result = self.vision.detect(image, {'words': [
            {'text': '0/2', 'bbox': [1030, 180, 40, 25]},
            {'text': '0/200', 'bbox': [1010, 282, 60, 25]},
        ]})
        self.assertEqual((result.kind, result.daily), ('tasks', ('incomplete', 'incomplete')))
        image[350:405, 520:750] = 0
        self.assertNotEqual(self.vision.detect(image).kind, 'tasks')

    def test_missing_or_complete_ratios_without_claim_state_are_unknown(self):
        image = frame('20261113-140331-tasks-first.png')
        self.assertEqual(self.vision.detect(image).daily, ('unknown', 'unknown'))
        result = self.vision.detect(image, {'words': [
            {'text': '2/2', 'bbox': [1030, 180, 40, 25]},
            {'text': '200/200', 'bbox': [990, 282, 80, 25]},
        ]})
        self.assertEqual(result.daily, ('unknown', 'unknown'))


if __name__ == '__main__':
    unittest.main()
