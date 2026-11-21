from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from ninja3_assistant.activities.daily.arena import ArenaReward, observe as arena_page
from ninja3_assistant.activities.daily.landlady import (
    LandladyQuiz, ROW_CENTERS, observe as quiz_page, read_question, recover_question_ocr, same_prompt,
)
from ninja3_assistant.activities.daily.vision import DailyScreen
from ninja3_assistant.game_ui.arena import rank_reward_entry


def screen(*words, page='unknown', frame=None):
    ocr = {'words': [{'text': text, 'bbox': (x, y, width, 20)} for text, x, y, width in words]}
    frame = np.zeros((720, 1280, 3), np.uint8) if frame is None else frame
    result = DailyScreen(frame, ocr, page)
    result.point = (60, 50)
    return result


def question_frame():
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[95:109, 425:1215] = 255
    frame[322:335, 655:1165] = 255
    for y in ROW_CENTERS:
        frame[y - 30:y + 30, 644:1179] = (229, 213, 203)
    return frame


def question(prefix='第1/10向'):
    return screen(('极娘问答', 172, 24, 229), (f'{prefix}：哪一种颜色？', 450, 145, 500),
                  ('A．红色', 850, 356, 160), ('B．蓝色', 850, 437, 160), ('C·白色', 850, 518, 160),
                  page='landlady-quiz', frame=question_frame())


class ArenaRewardTests(unittest.TestCase):
    def test_previous_currency_requires_all_visual_anchors_without_ocr(self):
        overlay = screen()
        positions = {'arena-previous-coins-title': (640, 110), 'arena-previous-coins-label': (640, 439),
                     'confirm': (638, 646)}
        vision = Mock()
        vision.match.side_effect = lambda frame, name, *args, **kwargs: positions.get(name)
        overlay.page, overlay.point = arena_page(overlay, vision)
        self.assertEqual((overlay.page, overlay.point), ('arena-currency', (638, 646)))
        reward = ArenaReward()
        decision = reward.step(overlay, vision)
        self.assertEqual((decision.action, decision.point, decision.status),
                         ('tap', (638, 646), '确认上次未领取竞技币'))
        reward.completed(decision, overlay)
        self.assertFalse(reward.finished)
        self.assertFalse(reward.claimed)
        for name in tuple(positions):
            point = positions.pop(name)
            with self.subTest(missing=name):
                self.assertIsNone(arena_page(overlay, vision))
            positions[name] = point

    def test_previous_results_and_trophy_dismiss_without_completing_either_task(self):
        overlays = (
            ('arena-history', '关闭竞技场上次成绩', (639, 662),
             (('上次成绩', 221, 65, 120), ('白银组', 580, 62, 130), ('确定', 605, 652, 68))),
            ('arena-trophy', '关闭竞技场奖杯结算', (639, 618),
             (('奖杯结算', 605, 226, 121), ('白银组', 555, 418, 78),
              ('第29名', 712, 418, 81), ('确定', 605, 608, 68))),
        )
        with tempfile.TemporaryDirectory() as directory:
            reward = ArenaReward()
            quiz = LandladyQuiz(Path(directory) / 'quiz.sqlite3')
            for page, status, point, words in overlays:
                with self.subTest(page=page):
                    overlay = screen(*words)
                    self.assertEqual(arena_page(overlay), (page, point))
                    overlay.page, overlay.point = page, point
                    for flow in (reward, quiz):
                        decision = flow.step(overlay, None)
                        self.assertEqual((decision.action, decision.point, decision.status), ('tap', point, status))
                    for omitted in range(len(words)):
                        self.assertIsNone(arena_page(screen(*(word for i, word in enumerate(words) if i != omitted))))
            self.assertFalse(reward.finished)
            self.assertFalse(reward.claimed)
            self.assertFalse(quiz.finished)

    def test_unranked_lobby_exits_without_starting_a_match(self):
        lobby = screen(('老板娘i司答', 1044, 80, 92), ('竟技商店', 1180, 80, 92),
                       ('50名玩家共同参加小组赛，赢取荣誉和奖励！', 320, 436, 645),
                       ('经典模式', 1040, 613, 120), page='arena')
        self.assertEqual(arena_page(lobby), ('arena', (60, 50)))
        flow = ArenaReward()
        self.assertEqual(flow.step(lobby, None).point, (60, 50))
        self.assertEqual(flow.step(screen(page='home'), None).action, 'done')

    def test_countdown_returns_home_without_tapping_preview_or_start(self):
        lobby = screen(('老板娘i司答', 1044, 80, 92), ('竟技商店', 1180, 80, 92),
                       ('当前排名奖励', 169, 421, 131), ('09小时27分', 721, 431, 100),
                       ('经典模式', 1040, 613, 120), page='arena')
        self.assertEqual(arena_page(lobby), ('arena', (60, 50)))
        flow = ArenaReward()
        self.assertEqual(flow.step(lobby, None).point, (60, 50))
        self.assertEqual(flow.step(screen(page='home'), None).action, 'done')

    def test_confirmed_settlement_claim_waits_for_game_reward(self):
        panel = screen(('竞技场结算', 552, 135, 130), ('领取奖励', 590, 581, 100), page='arena-reward')
        self.assertEqual(arena_page(panel), ('arena-reward', None))
        flow = ArenaReward()
        claim = flow.step(panel, None)
        self.assertEqual(claim.point, (640, 591))
        flow.completed(claim, panel)
        self.assertEqual(flow.step(panel, None).action, 'wait')
        self.assertIsNone(flow.step(screen(page='reward_continue'), None))
        self.assertEqual(flow.step(screen(page='arena'), None).point, (60, 50))

    def test_rank_reward_requires_a_visible_control_and_confirms_only_after_feedback(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        frame[414:472, 140:876] = (242, 155, 18)
        for x in range(727, 855, 20):
            frame[435:451, x:x+3] = (246, 240, 231)
            frame[435:439, x:x+14] = (246, 240, 231)
            frame[447:451, x:x+14] = (246, 240, 231)
        lobby = screen(page='arena', frame=frame)
        reward = ArenaReward()
        action = reward.step(lobby, None)
        self.assertEqual(action.operation, 'open_arena_reward')
        self.assertEqual(action.point, rank_reward_entry(frame))
        reward.completed(action, lobby)
        self.assertEqual(reward.step(lobby, None).action, 'wait')
        self.assertFalse(reward.pending)
        self.assertFalse(reward.claimed)
        panel = screen(('竞技场结算', 552, 135, 130), ('领取奖励', 590, 581, 100), page='arena-reward')
        claim = reward.step(panel, None)
        self.assertEqual(claim.operation, 'claim_arena_reward')
        reward.completed(claim, panel)
        self.assertIsNone(reward.step(screen(page='reward_continue'), None))
        self.assertEqual(reward.step(lobby, None).operation, 'back')

        countdown = frame.copy()
        countdown[425:461, 683:849] = (71, 38, 7)
        self.assertIsNone(rank_reward_entry(countdown))
        self.assertEqual(ArenaReward().step(screen(('领取', 753, 433, 50), page='arena',
                                                   frame=countdown), None).action, 'wait')


class LandladyQuizTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'quiz.sqlite3'

    def test_wrong_row_and_revealed_check_are_both_recorded(self):
        flow = LandladyQuiz(self.path)
        before = question()
        self.assertEqual(quiz_page(before), ('landlady-quiz', (60, 50)))
        flow.bank.choose = lambda *_: '蓝色'
        decision = flow.step(before, None)
        self.assertEqual(decision.point, (910, 450))
        flow.completed(decision, before)

        after = question_frame()
        after[420:480, 644:1179] = (239, 99, 39)
        after[510:550, 690:718] = (75, 145, 25)
        self.assertTrue(flow.observe_feedback(after))
        parsed = read_question(before)
        self.assertEqual(LandladyQuiz(self.path).bank.choose(parsed.text, parsed.options), '白色')
        self.assertIsNone(flow.pending)

    def test_missing_short_option_gets_local_ocr_without_replacing_readable_prompt(self):
        before = question(prefix='第问')
        before.ocr['words'].pop(2)
        local = screen(('第7/10向：识别不全的题干', 5, 30, 500),
                       ('A.红', 418, 244, 130), ('B·蓝色', 418, 325, 140), ('c·白色', 418, 406, 140))
        ocr = Mock()
        ocr.read.return_value = {**local.ocr, 'text': '局部识别'}
        self.assertIsNone(read_question(before))
        fixed = recover_question_ocr(before.frame, before.ocr, ocr, None)
        parsed = read_question(DailyScreen(before.frame, fixed))
        self.assertEqual(parsed.text, '哪一种颜色？')
        self.assertEqual(tuple(parsed.options), ('红', '蓝色', '白色'))

    def test_mid_animation_prompt_is_rejected(self):
        before = question_frame()
        fresh = before.copy()
        self.assertTrue(same_prompt(before, fresh))
        fresh[155:172, 700:732] = 255
        self.assertFalse(same_prompt(before, fresh))

    def test_completed_quiz_has_a_lower_panel_and_returns_home(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        frame[212:228, 425:1215] = 255
        complete = screen(('极娘问答', 172, 24, 229), ('今夭的向题已经全部答完了，', 600, 244, 400),
                          ('记得明夭再来哦！', 666, 305, 270), page='landlady-quiz', frame=frame)
        flow = LandladyQuiz(self.path)
        self.assertEqual(quiz_page(complete), ('landlady-quiz', (60, 50)))
        self.assertEqual(flow.step(complete, None).point, (60, 50))
        self.assertEqual(flow.step(screen(page='home'), None).action, 'done')

    def test_unconfirmed_and_unrelated_feedback_do_not_change_the_bank(self):
        flow = LandladyQuiz(self.path)
        flow.bank.choose = lambda *_: '蓝色'
        before = question()
        decision = flow.step(before, None)
        flow.completed(decision, before)
        different_screen = np.zeros((720, 1280, 3), np.uint8)
        different_screen[420:480, 644:1179] = (163, 217, 96)
        self.assertFalse(flow.observe_feedback(different_screen))
        self.assertEqual(flow.step(before, None).action, 'wait')
        following = question(prefix='第i司')
        following.ocr['words'][1]['text'] = '第i司：什么颜色的天空？'
        following.frame[433:460, 810:1040] = (35, 20, 15)
        self.assertIsNone(read_question(following).number)
        self.assertEqual(flow.step(following, None).point, (910, 450))
        self.assertFalse(self.path.exists())

    def test_chinese_option_keeps_first_character_when_delimiter_is_missing(self):
        before = question(prefix='第3/10向')
        before.ocr['words'][2:3] = [
            {'text': 'A', 'bbox': (863, 358, 22, 24)},
            {'text': '凇茸', 'bbox': (886, 355, 72, 31)},
        ]
        parsed = read_question(before)
        self.assertEqual(parsed.number, 3)
        self.assertEqual(parsed.options['凇茸'], (910, 370))
        before.ocr['words'][3]['text'] = 'BOSS'
        self.assertIsNone(read_question(before))

    def test_unreadable_question_defers_only_when_option_panel_is_visible(self):
        flow = LandladyQuiz(self.path)
        before = question()
        before.ocr['words'].pop(2)
        decision = flow.step(before, None)
        self.assertEqual(decision.action, 'wait')
        self.assertTrue(decision.defer_on_timeout)
        before.frame[322:335, 655:1165] = 0
        self.assertFalse(flow.step(before, None).defer_on_timeout)


if __name__ == '__main__':
    unittest.main()
