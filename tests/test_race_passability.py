from pathlib import Path
import unittest

import cv2
import numpy as np

from race_obstacles import RaceObstacleVision
from weekend_race import RaceAvoider


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'data/evidence/2026-11-15/weekend-passability'


def image(name):
    return cv2.cvtColor(cv2.imread(str(EVIDENCE/name)), cv2.COLOR_BGR2RGB)


class RacePassabilityTest(unittest.TestCase):
    def terrain(self):
        frame = np.full((720,1280,3),180,np.uint8)
        frame[540:600] = 50
        return frame

    def test_unknown_foreground_is_not_classified_as_an_obstacle(self):
        detector = RaceObstacleVision()
        for colour in ((200,0,200),(220,60,20),(10,10,10)):
            with self.subTest(colour=colour):
                frame = self.terrain()
                frame[440:540,730:790] = colour
                self.assertEqual(detector.detect(frame,600,540),())

    def test_unknown_hanging_object_does_not_request_down(self):
        frame = self.terrain()
        frame[395:445,730:780] = (200,0,200)
        avoider = RaceAvoider()
        avoider.decide(frame,0)
        self.assertIsNone(avoider.decide(frame,.02))
        self.assertIsNone(avoider.hazard)

    def test_real_wooden_gate_is_positively_identified(self):
        avoider = RaceAvoider()
        frame = image('frame-6.png')
        avoider.decide(frame,0)
        self.assertEqual(avoider.decide(frame,.02),(1090,630))
        self.assertTrue(any(item.kind == 'gate' and 690 <= item.box[0] <= 730 for item in avoider.detected))

    def test_real_plant_is_visible_before_it_reaches_the_action_window(self):
        found = RaceObstacleVision().detect(image('frame-18.png'),600,555)
        self.assertTrue(any(item.kind == 'plant' and 940 <= item.box[0] <= 980 and item.box[3] > 100 for item in found))

    def test_real_thin_spike_is_not_required_to_be_a_solid_wall(self):
        avoider = RaceAvoider()
        frame = image('frame-32.png')
        avoider.decide(frame,0)
        self.assertEqual(avoider.decide(frame,.02),(1090,630))
        self.assertTrue(any(item.kind == 'spike' and 750 <= item.box[0] <= 800 and item.box[2] < 45
                            for item in avoider.detected))

    def test_real_pit_remains_a_jump_not_a_down_action(self):
        avoider = RaceAvoider()
        frame = image('frame-24.png')
        avoider.decide(frame,0)
        self.assertEqual(avoider.decide(frame,.02),(1090,630))
        self.assertEqual(avoider.hazard,'pit')

    def test_player_location_expires_instead_of_becoming_permanent(self):
        avoider = RaceAvoider()
        avoider.decide(image('frame-32.png'),0)
        self.assertTrue(avoider.player_tracked)
        avoider.decide(self.terrain(),.5)
        self.assertFalse(avoider.player_tracked)

    def test_partial_health_bar_at_tracking_crop_bottom_does_not_crash(self):
        frame = self.terrain()
        frame[587:590,400:425] = (238,198,52)
        avoider = RaceAvoider()
        avoider.decide(frame,0)
        self.assertFalse(avoider.player_tracked)

    def test_a_lower_continuing_road_is_not_automatically_a_pit(self):
        frame = self.terrain()
        frame[530:600,700:1000] = 180
        frame[575:640,700:1000] = 50
        avoider = RaceAvoider()
        avoider.decide(frame,0)
        avoider.decide(frame,.02)
        self.assertNotEqual(avoider.hazard,'pit')
