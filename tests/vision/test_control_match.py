import unittest

import numpy as np

from ninja3_assistant.vision.control_match import match_control, text_foreground


class ControlMatchTest(unittest.TestCase):
    def test_matches_foreground_after_the_rest_of_the_control_changes(self):
        rng = np.random.default_rng(12)
        template = rng.integers(0, 256, size=(18, 16), dtype=np.uint8)
        mask = np.zeros_like(template)
        mask[5:13, 4:12] = 255
        changed = np.where(mask, template, 255-template)
        screen = np.zeros((60, 80, 3), dtype=np.uint8)
        screen[20:38, 30:46] = changed[..., None]

        self.assertIsNone(match_control(screen, template, (30,20,46,38), max_error=8))
        self.assertEqual(match_control(screen, template, (0,0,80,60), mask=mask,
                                       threshold=.99, max_error=8), (38,29))

        screen[25:33, 34:42] = 0
        self.assertIsNone(match_control(screen, template, (30,20,46,38), mask=mask, threshold=.99))

    def test_blank_template_has_no_foreground_to_match(self):
        blank = np.full((9, 12), 35, dtype=np.uint8)
        screen = np.full((20, 30, 3), 35, dtype=np.uint8)

        self.assertIsNone(match_control(screen, blank, (0,0,30,20), mask=text_foreground(blank)))
