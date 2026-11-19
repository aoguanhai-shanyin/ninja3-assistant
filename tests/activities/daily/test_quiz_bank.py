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

    def test_observed_ocr_variants_keep_the_original_visible_option(self):
        cases = (
            ('【雪舞·琳】的头发是什么颜色的？', '白色', '（雪舞．琳〗的头发是什么颜色的？', '白色'),
            ('炎魂·小黑怎么获得三段跳？', '装备【黑龙套装】', '炎魂·小黑怎么获得三段跳？', '装备(黑龙套装〗'),
            ('以下哪个忍术是属于神乐·琳的？', '冰天雪葬', '以下祁个忍木是属干神乐·琳的？', '冰天雪葬'),
            ('想要获得【诡兵尽藏】这一特殊忍阶要怎么办？', '去收集更多幻化武器',
             '想要获得〖兵尽藏〗这一特殊忍阶要怎么办？', '去收集更多幻化武器'),
            ('【雷狱光剑】的武器技能一次能放出几道闪电？', '3道',
             '〖雷狱先剑〗的武器技能一次能放出几遒闪电？', '3遒'),
        )
        self.bank.replace_baseline((question, answer) for question, answer, _, _ in cases)
        with patch('ninja3_assistant.activities.daily.quiz_bank.random.choice', side_effect=AssertionError):
            for _, _, read, option in cases:
                with self.subTest(read=read):
                    self.assertEqual(self.bank.choose(read, ('其他', option, '不确定')), option)

    def test_fuzzy_lookup_preserves_negation_and_numbers_and_requires_unique_answer(self):
        self.bank.replace_baseline([('【雷狱光剑】第2级技能能放出几道闪电？', '3道')])
        cases = (
            ('〖雷狱先剑〗第2级技能不能放出几遒闪电？', ('2道', '3遒', '4道')),
            ('〖雷狱先剑〗第3级技能能放出几遒闪电？', ('2道', '3遒', '4道')),
            ('〖雷狱先剑〗第2级技能能放出几遒闪电？', ('2道', '3道', '3遒')),
            ('〖雷狱先剑〗第2级技能能放出几遒闪电？', ('2道', '4道', '5道')),
        )
        for text, options in cases:
            with self.subTest(text=text, options=options), patch(
                    'ninja3_assistant.activities.daily.quiz_bank.random.choice', return_value='2道') as choose:
                self.assertEqual(self.bank.choose(text, options), '2道')
                choose.assert_called_once_with(list(options))

    def test_fuzzy_lookup_rejects_ambiguous_prompts_or_confirmed_wrong_hint(self):
        observed = '〖先先枪〗的技能能放出几遒闪电？'
        baseline = [('【先光枪】的技能能放出几道闪电？', '2道'),
                    ('【光先枪】的技能能放出几道闪电？', '3道')]
        options = ('2道', '3道', '4道')
        self.bank.replace_baseline(baseline)
        with patch('ninja3_assistant.activities.daily.quiz_bank.random.choice', return_value='4道') as choose:
            self.assertEqual(self.bank.choose(observed, options), '4道')
            choose.assert_called_once_with(list(options))
        self.bank.replace_baseline(baseline[:1])
        self.bank.record(observed, options, '2道', correct=False)
        with patch('ninja3_assistant.activities.daily.quiz_bank.random.choice', return_value='4道') as choose:
            self.assertEqual(self.bank.choose(observed, options), '4道')
            choose.assert_called_once_with(['3道', '4道'])

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
