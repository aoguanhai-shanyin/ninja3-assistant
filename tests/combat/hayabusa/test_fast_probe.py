from ninja3_assistant.paths import project_root
import unittest

import cv2
import numpy as np

from ninja3_assistant.combat.hayabusa.fast_probe import FastProbe


ROOT = project_root()
FRAMES = ROOT/'data/hayabusa-player-v1/qa/live-body-retry-no-screenrecord'


def frame(index):
    bgr = cv2.imread(str(FRAMES/f'frame-{index:04d}.jpg'))
    if bgr is None:
        raise unittest.SkipTest('live evidence frames unavailable')
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)


class FastProbeTest(unittest.TestCase):
    def test_unknown_and_obstructed_hud_do_not_confirm_battle(self):
        probe = FastProbe(max_player_age=.1)
        self.assertFalse(probe.update(np.zeros((720,1280,3),np.uint8),1).active)
        image = frame(202)
        self.assertTrue(probe.update(image,2).active)
        image[:115] = 0
        obs = probe.update(image,2.1,player_x=400,player_captured_at=2.1)
        self.assertFalse(obs.active)
        self.assertIsNone(obs.cue)
        self.assertEqual(obs.right_control,'unknown')

    def test_fire_requires_fresh_position_and_two_distinct_frames(self):
        probe = FastProbe(max_player_age=.1)
        first,second = frame(202),frame(203)
        self.assertIsNone(probe.update(first,1,player_x=400,player_captured_at=1).cue)
        self.assertIsNone(probe.update(second,1.2,player_x=400,player_captured_at=1).cue)
        self.assertIsNone(probe.update(first,2,player_x=400,player_captured_at=2).cue)
        cue = probe.update(second,2.2,player_x=400,player_captured_at=2.2).cue
        self.assertIsNotNone(cue)
        self.assertEqual(cue.family,'fire')
        self.assertIsNone(cue.response_class)
        self.assertIsNone(probe.update(second,2.3,player_x=400,player_captured_at=2.3).cue)

    def test_bad_input_and_unknown_left_control_fail_closed(self):
        probe = FastProbe(max_player_age=.1)
        self.assertFalse(probe.update(None,1).active)
        image=frame(202)
        image[550:720,:320] = 0
        self.assertFalse(probe.update(image,2).active)


if __name__ == '__main__':
    unittest.main()
