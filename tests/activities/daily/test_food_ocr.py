"""Counter fallback checks without local game assets or Windows OCR."""
import unittest

import numpy as np

from ninja3_assistant.activities.daily.vision import DailyScreen, DailyVision


class FoodCounterOcrTest(unittest.TestCase):
    def food(self, text, words=()):
        vision = DailyVision.__new__(DailyVision)
        vision.templates = {}
        screen = DailyScreen(np.zeros((720,1280,3),np.uint8),
            {'words':list(words),'food_counter_repeat':{'text':text}})
        return vision.food(screen)

    def test_three_identical_readings_preserve_the_complete_number(self):
        for value in (0,4,7,9,12,25,57,123):
            with self.subTest(value=value):
                self.assertEqual(self.food(f' {value} {value}\n{value} '),value)

    def test_incomplete_disagreeing_or_non_numeric_readings_remain_unknown(self):
        for text in ('','7','777','7 7','7 7 9','7 7 7 7','17 7 7',
                     '7 7 07','7 7 7金币','7a 7 7'):
            with self.subTest(text=text):
                self.assertIsNone(self.food(text))

    def test_regular_counter_reading_keeps_priority(self):
        self.assertEqual(self.food('7 7 7',
            [{'text':'12','bbox':[1183,40,23,16]}]),12)

    def test_food_icon_edge_is_not_a_counter_digit(self):
        self.assertEqual(self.food('8 》 8 》 8'),8)
        self.assertEqual(self.food('9 》 9 》 9'),9)
        self.assertIsNone(self.food('8 》 8 》 9'))


if __name__ == '__main__':
    unittest.main()
