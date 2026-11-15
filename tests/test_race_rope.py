from pathlib import Path
from unittest.mock import patch
import unittest

import cv2
import numpy as np

from race_rope import RaceRopeTraversal


ROOT = Path(__file__).resolve().parents[1]


class RaceRopeTest(unittest.TestCase):
    def test_empty_rope_jumps_at_bounded_cadence_without_down(self):
        rope = RaceRopeTraversal()
        image = np.full((720,1280),180,np.uint8)
        image[550:556] = 0
        actions = rope.decide(image,0,600,458)
        self.assertIsNone(rope.platform)
        self.assertEqual([action.point for action in actions],[(1090,630)])
        rope.scheduled(0)
        self.assertFalse(rope.decide(image,.3,600,458))
        self.assertTrue(rope.decide(image,.48,600,458))

    def test_floor_above_requests_two_jumps_without_midair_down(self):
        rope = RaceRopeTraversal()
        image = np.full((720,1280),180,np.uint8)
        image[375:397,560:850] = 40
        with patch.object(rope,'upper_platform',return_value=(560,375,290,22)):
            actions = rope.decide(image,0,600,458)
        self.assertIsNotNone(rope.platform)
        self.assertEqual([action.point for action in actions],[(1090,630)]*2)
        self.assertGreater(actions[1].offset_ms,actions[0].offset_ms+actions[0].hold_ms)
        self.assertLess(actions[1].offset_ms+actions[1].hold_ms,200)

    def test_floor_below_and_thin_rope_do_not_trigger_upper_floor_mode(self):
        image = np.full((720,1280),180,np.uint8)
        image[400:406] = 0
        image[560:590] = 40
        self.assertIsNone(RaceRopeTraversal.upper_platform(image,600,458))

    def test_actual_empty_rope_and_portal_do_not_become_a_floor(self):
        for seconds in (40,42,44):
            path = ROOT/f'data/evidence/2026-11-15/rope-traversal/rope-{seconds}.png'
            with self.subTest(seconds=seconds):
                self.assertIsNone(RaceRopeTraversal.upper_platform(cv2.imread(str(path),0),600,458))

    def test_actual_grass_and_snow_platforms_are_found(self):
        for name in ('sample-104.png','20261114-170712-recovery-kernel.png'):
            with self.subTest(name=name):
                image = cv2.imread(str(ROOT/'data/weekend-race/live'/name),0)
                floor = RaceRopeTraversal.upper_platform(image,600,458)
                self.assertIsNotNone(floor)
                self.assertLess(floor[1],400)

    def test_uncommitted_jump_can_be_retried_on_next_fresh_frame(self):
        rope = RaceRopeTraversal()
        image = np.zeros((720,1280),np.uint8)
        self.assertTrue(rope.decide(image,0,600,458))
        self.assertTrue(rope.decide(image,.03,600,458))
