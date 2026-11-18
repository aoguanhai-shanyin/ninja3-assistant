import unittest

import numpy as np

from ninja3_assistant.activities.daily.pet_feed import PetFeed, observe
from ninja3_assistant.activities.daily.vision import DailyScreen


def screen(page, *words):
    frame = np.zeros((720, 1280, 3), np.uint8)
    ocr = {'words': [{'text': text, 'bbox': box} for text, box in words]}
    return DailyScreen(frame, ocr, page)


def routes(route, *, selected=False, empty=True):
    words = [('通灵兽巡逻', (120, 30, 210, 42)), ('开始巡逻', (1053, 632, 140, 38)),
             (route, (10, 318, 112, 36))]
    if selected:
        words.append(('额外食物', (1164, 284, 88, 22)))
    current = screen('pet-routes', *words)
    if empty:
        current.frame[335:382, 1202:1211] = 255
        current.frame[354:363, 1184:1230] = 255
    return current


class PetFeedTest(unittest.TestCase):
    def test_home_recovery_uses_observed_pet_slot_on_recognized_home_only(self):
        flow = PetFeed('exp')
        current = screen('home', ('通', (114, 302, 16, 16)))
        self.assertEqual(flow.step(current, None).point, (143, 280))
        current.page = 'unknown'
        self.assertIsNone(flow.step(current, None))
        self.assertIsNone(flow.step(screen('home'), None).point)

    def test_home_countdown_confirms_existing_patrol_without_opening_it(self):
        flow = PetFeed('exp')
        active = screen('home', ('01小时40分', (114, 302, 64, 13)))
        self.assertEqual(flow.step(active, None).action, 'done')
        active.ocr['words'].clear()
        active.frame[302:315, 97:108] = (35, 115, 235)
        active.frame[305:311, 125:155] = 255
        decision = flow.step(active, None)
        self.assertEqual(decision.action, 'done')
        self.assertIsNone(decision.point)

    def test_claim_is_sent_at_most_once_and_running_patrol_is_preserved(self):
        flow = PetFeed('exp')
        status = ('今日已获得', (800, 28, 141, 25))
        current = screen('pet-main', status, ('领取巡逻奖励', (960, 658, 165, 27)))
        self.assertEqual(observe(current), ('pet-main', (66, 53)))
        claim = flow.step(current, None)
        self.assertEqual(claim.status, '领取巡逻奖励')
        flow.completed(claim, current)
        self.assertEqual(flow.step(current, None).action, 'wait')
        idle = screen('pet-main', status, ('巡逻', (1019, 658, 65, 29)))
        self.assertEqual(flow.step(idle, None).status, '打开通灵兽巡逻路线')
        active = screen('unknown', status)
        active.frame[665:676, 885:1015] = 255
        active.frame[650:692, 1220:1245] = (59, 158, 213)
        active.page, _ = observe(active)
        self.assertEqual(flow.step(active, None).action, 'done')

    def test_starts_only_selected_reward_and_only_with_empty_extra_food(self):
        for reward, name in (('exp', '雾之湖'), ('coins', '忍村'), ('hammers', '雷王山')):
            with self.subTest(reward=reward):
                flow = PetFeed(reward)
                current = routes(name)
                self.assertEqual(observe(current), ('pet-routes', (66, 53)))
                self.assertEqual(flow.step(current, None).status, f'选择巡逻路线 {name}')
                self.assertEqual(flow.step(routes(name, selected=True, empty=False), None).action, 'wait')
                selected = routes(name, selected=True)
                start = flow.step(selected, None)
                self.assertEqual(start.status, '开始通灵兽巡逻')
                self.assertGreater(start.point[0], 995)
                flow.completed(start, selected)
                self.assertEqual(flow.step(selected, None).action, 'wait')

    def test_missing_reward_never_defaults_to_a_different_route(self):
        flow = PetFeed('exp')
        current = routes('忍村', selected=True)
        for _ in range(2):
            action = flow.step(current, None)
            self.assertEqual(action.action, 'swipe')
            flow.completed(action, current)
        self.assertEqual(flow.step(current, None).action, 'unavailable')
