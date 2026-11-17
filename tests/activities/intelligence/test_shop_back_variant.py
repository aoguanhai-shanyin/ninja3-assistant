"""Replay shop backs captured in separate intelligence matches."""
import unittest

import cv2
import numpy as np

from ninja3_assistant.activities.intelligence.battle_vision import BattleVision
from ninja3_assistant.paths import project_root


class ShopBackVariantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = BattleVision()
        cls.runs = project_root()/'data/runtime/intelligence-simulation/runs'

    def read(self, run, crop):
        image = cv2.imread(str(self.runs/run/'unknown-cards'/crop))
        self.assertIsNotNone(image)
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    def empty(self, gray):
        height,width = gray.shape
        return self.vision._empty_pattern(gray,'shop',(25,40,width-20,height-40))

    def test_separate_recordings_of_blue_empty_shop_back(self):
        cases = (
            ('20261117-120524-949600','0003-round2-shop2.png'),
            ('20261117-121723-171700','0001-round4-shop2.png'),
            ('20261117-121945-198500','0002-round2-shop2.png'),
            ('20261117-124615-994600','0000-round2-shop2.png'),
            ('20261117-130118-863700','0002-round2-shop2.png'),
            ('20261117-131343-151200','0006-round2-shop2.png'),
        )
        for run,crop in cases:
            with self.subTest(run=run):
                self.assertTrue(self.empty(self.read(run,crop)))

    def test_incomplete_outline_and_unrelated_noise_are_not_empty(self):
        gray = self.read('20261117-130827-119700','0000-round2-shop2.png')
        left = gray.copy(); left[:,:gray.shape[1]//2] = 0
        right = gray.copy(); right[:,gray.shape[1]//2:] = 0
        noise = np.random.default_rng(15).integers(0,256,gray.shape,dtype=np.uint8)
        for image in (left,right,noise,np.full_like(gray,255)):
            self.assertFalse(self.empty(image))


if __name__ == '__main__':
    unittest.main()
