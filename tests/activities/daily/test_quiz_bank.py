from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ninja3_assistant.activities.daily.quiz_bank import QuizBank


class QuizBankTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'quiz.sqlite3'
        self.bank = QuizBank(self.path)

    def test_unknown_question_uses_one_of_the_visible_options(self):
        options = ('红色', '蓝色', '白色')
        self.assertIn(self.bank.choose('哪一种颜色？', options), options)
        self.assertFalse(self.path.exists())

    def test_wrong_options_are_skipped_and_known_correct_survives_reorder(self):
        options = ('红色', '蓝色', '白色')
        self.bank.record('哪一种颜色？', options, '红色', correct=False)
        self.bank.record('哪一种颜色？', options, '蓝色', correct=False)
        self.assertEqual(QuizBank(self.path).choose('哪 一 种 颜 色?', options[::-1]), '白色')
        self.bank.record('哪一种颜色？', options, '白色', correct=True)
        self.assertEqual(QuizBank(self.path).choose('哪一种颜色？', options[::-1]), '白色')

    def test_wrong_feedback_can_reveal_the_correct_answer(self):
        options = ('红色', '蓝色', '白色')
        self.bank.record('哪一种颜色？', options, '红色', correct=False, revealed='蓝色')
        self.assertEqual(self.bank.choose('哪一种颜色？', options), '蓝色')

    def test_question_variants_with_identical_stems_keep_separate_answers(self):
        for options in (('甲', '乙', '丙'), ('甲', '乙', '丁')):
            self.bank.record('选哪个？', options, options[-1], correct=True)
        self.assertEqual(self.bank.choose('选哪个？', ('乙', '甲', '丙')), '丙')
        self.assertEqual(self.bank.choose('选哪个？', ('乙', '甲', '丁')), '丁')

    def test_concurrent_workers_keep_all_confirmed_wrong_choices(self):
        options = ('甲', '乙', '丙')
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = [executor.submit(QuizBank(self.path).record, '选哪个？', options, option, correct=False)
                       for option in options]
            for result in results:
                result.result()
        self.assertIsNone(self.bank.choose('选哪个？', options))

    def test_baseline_uses_normalized_visible_text_across_restarts(self):
        count = self.bank.replace_baseline([('哪 一 种 颜 色？', '白 色'), ('哪一种颜色?', '白色')])
        self.assertEqual(count, 1)
        self.assertEqual(QuizBank(self.path).choose('哪一种颜色？', ('蓝色', '白色', '红色')), '白色')

    def test_game_feedback_can_disprove_or_override_a_baseline_hint(self):
        options = ('甲', '乙', '丙')
        self.bank.replace_baseline([('选哪个？', '甲')])
        self.bank.record('选哪个？', options, '甲', correct=False)
        with patch('ninja3_assistant.activities.daily.quiz_bank.random.choice', return_value='乙') as choose:
            self.assertEqual(self.bank.choose('选哪个？', options), '乙')
            choose.assert_called_once_with(['乙', '丙'])
        self.bank.replace_baseline([('选哪个？', '乙')])
        self.bank.record('选哪个？', options, '丙', correct=True)
        self.assertEqual(self.bank.choose('选哪个？', options), '丙')

    def test_ambiguous_or_absent_baseline_answers_fall_back_to_untried_options(self):
        self.bank.replace_baseline([('选哪个？', '甲'), ('选哪个？', '乙')])
        for options in (('甲', '乙', '丙'), ('庚', '辛', '丙')):
            with self.subTest(options=options), patch(
                    'ninja3_assistant.activities.daily.quiz_bank.random.choice', return_value='丙') as choose:
                self.assertEqual(self.bank.choose('选哪个？', options), '丙')
                choose.assert_called_once_with(list(options))

    def test_baseline_replacement_preserves_learned_answers_and_rejects_incomplete_rows(self):
        options = ('甲', '乙', '丙')
        self.bank.record('选择已有答案？', options, '乙', correct=True)
        self.bank.replace_baseline([('选择基准答案？', '甲')])
        with self.assertRaises(ValueError):
            self.bank.replace_baseline([('选择基准答案？', '丙'), ('', '乙')])
        self.assertEqual(self.bank.choose('选择基准答案？', options), '甲')
        self.bank.replace_baseline([('选择基准答案？', '丙')])
        self.assertEqual(self.bank.choose('选择基准答案？', options), '丙')
        self.assertEqual(self.bank.choose('选择已有答案？', options), '乙')


if __name__ == '__main__':
    unittest.main()
