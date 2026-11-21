"""Counter fallback checks without local game assets or Windows OCR."""
import unittest
from unittest.mock import Mock

import numpy as np

from ninja3_assistant.game_ui.trial import read_trial_counter, trial_counter_glyphs
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
                     '7 7 07','7 7 7金币','7a 7 7','0 0','0》'):
            with self.subTest(text=text):
                self.assertIsNone(self.food(text))

    def test_standalone_zero_is_safe_when_ocr_reads_only_one_copy(self):
        self.assertEqual(self.food(' 0\n'),0)

    def test_regular_counter_reading_keeps_priority(self):
        self.assertEqual(self.food('7 7 7',
            [{'text':'12','bbox':[1183,40,23,16]}]),12)

    def test_food_icon_edge_is_not_a_counter_digit(self):
        self.assertEqual(self.food('8 》 8 》 8'),8)
        self.assertEqual(self.food('9 》 9 》 9'),9)
        self.assertIsNone(self.food('8 》 8 》 9'))

    def test_isolated_foreground_ignores_the_icon_button_and_color_effect(self):
        frame = np.zeros((720,1280,3),np.uint8)
        frame[43:53,1190:1192] = 255
        frame[43:53,1200:1202] = 255
        clean = trial_counter_glyphs(frame)
        frame[40:55,1120:1168] = 255
        frame[40:55,1232:1248] = 255
        frame[39:42,1175:1190] = (225,185,95)
        np.testing.assert_array_equal(trial_counter_glyphs(frame),clean)
        ocr = Mock()
        ocr.read.side_effect = [{'text':''},{'text':'11 11 11'}]
        self.assertEqual(read_trial_counter(frame,ocr,None),{'text':'11 11 11'})
        self.assertEqual(ocr.read.call_count,2)
        for call in ocr.read.call_args_list:
            prepared = call.args[0]
            self.assertEqual(set(np.unique(prepared)),{0,255})
            self.assertTrue(np.array_equal(prepared[:,:,0],prepared[:,:,1]))

    def test_empty_foreground_does_not_use_unrelated_text_or_guess_counter(self):
        frame = np.zeros((720,1280,3),np.uint8)
        frame[31:61,1122:1160] = 255
        ocr = Mock()
        self.assertIsNone(self.food(read_trial_counter(frame,ocr,None)['text']))
        ocr.read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
