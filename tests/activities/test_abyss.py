import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ninja3_assistant.activities.abyss.runner import play_abyss
from ninja3_assistant.activities.abyss.vision import (
    NINJUTSU, SKILLS, TABS, free_relay, read_reward, ready_ninjutsu, ready_weapons,
)
from ninja3_assistant.runtime.runner import RunControl


class AbyssControlTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((720, 1280, 3), np.uint8)

    def test_full_ninjutsu_ring_at_either_position(self):
        for x, y, radius in NINJUTSU:
            with self.subTest(point=(x, y)):
                self.frame.fill(0)
                cv2.circle(self.frame, (x, y), radius, (255, 255, 255), 5)
                cv2.ellipse(self.frame, (x, y), (radius, radius), 0, 0, 269, (253, 179, 19), 5)
                self.assertIsNone(ready_ninjutsu(self.frame))
                cv2.circle(self.frame, (x, y), radius, (253, 179, 19), 5)
                self.assertEqual(ready_ninjutsu(self.frame), (x, y))

    def test_both_weapon_slots_fire_when_their_rims_are_ready(self):
        for x, y, radius in (SKILLS[0], SKILLS[-1]):
            cv2.circle(self.frame, (x, y), radius, (255, 255, 255), 5)
        self.assertEqual(list(ready_weapons(self.frame)), [SKILLS[0][:2], SKILLS[-1][:2]])

    def test_free_relay_rejects_nonzero_costs(self):
        cv2.ellipse(self.frame, (450, 455), (6, 10), 0, 0, 360, (255, 255, 255), 3)
        self.assertTrue(free_relay(self.frame, 445))
        for price in ('6', '9', '10', '20'):
            with self.subTest(price=price):
                self.frame.fill(0)
                cv2.putText(self.frame, price, (440, 468), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                self.assertFalse(free_relay(self.frame, 445))

    def test_confirmed_dungeon_list_opens_rewards_without_reading_progress_text(self):
        self.frame[487:491, 130:1182] = (251, 189, 80)
        self.frame[700:716, 180:215] = (251, 189, 80)
        device, native, ocr, emit = Mock(), Mock(), Mock(), Mock()
        device.capture.return_value = self.frame
        control = RunControl()
        native.priority_tap.side_effect = lambda *args, **kwargs: (control.request_stop() or True)
        with patch('ninja3_assistant.activities.abyss.runner.DailyVision') as daily, \
                patch('ninja3_assistant.activities.abyss.runner.BountyCombatVision'):
            daily.return_value.navigation.return_value = (None, None)
            play_abyss(device, native, ocr, control, emit)
        ocr.read.assert_not_called()
        self.assertEqual(native.priority_tap.call_args.args[0], (494, 441))

    def test_finishes_the_current_dungeon_before_advancing(self):
        self.check_order(2, [
            ('list', '切换至副本 1'),
            ('list', '检查副本 1 奖励'),
            ('pending', '关闭奖励信息'),
            ('list', '挑战副本 1'),
            ('result', '结算并返回副本'),
            ('list', '检查副本 1 奖励'),
            ('pending', '关闭奖励信息'),
            ('list', '挑战副本 1'),
            ('result', '结算并返回副本'),
            ('list', '检查副本 1 奖励'),
            ('claim', '领取副本 1 奖励'),
            ('claimed', '关闭奖励信息'),
            ('list', '切换至副本 2'),
            ('list', '检查副本 2 奖励'),
            ('claimed', '关闭奖励信息'),
            ('list', '切换至副本 3'),
        ])

    def test_claimed_stamp_needs_a_full_bar_if_ocr_misses_the_text(self):
        screen = Mock(frame=self.frame)
        screen.text.return_value = ''
        self.frame[293:317, 873:971] = (165, 149, 199)
        self.frame[296:310, 889:926] = (135, 67, 24)
        self.assertIsNone(read_reward(screen, 305))
        self.frame[297:313, 823:845] = (251, 189, 80)
        self.assertEqual(read_reward(screen, 305), 'claimed')
        self.frame[293:317, 873:971] = (179, 179, 179)
        self.assertEqual(read_reward(screen, 305), 'pending')

    def test_zero_contribution_does_not_skip_to_a_later_dungeon(self):
        self.check_order(0, [
            ('list', '检查副本 1 奖励'),
            ('pending', '关闭奖励信息'),
            ('list', '挑战副本 1'),
            ('zero', '结算并返回副本'),
            ('list', '检查副本 1 奖励'),
            ('pending', '关闭奖励信息'),
            ('list', '挑战副本 1'),
            ('zero', '结算并返回副本'),
            ('list', '检查副本 1 奖励'),
            ('pending', '关闭奖励信息'),
            ('list', '退出备战副本列表'),
        ])

    def check_order(self, selected, steps):
        control, device, native, ocr, emit = RunControl(), Mock(), Mock(), Mock(), Mock()
        index = 0

        def frame():
            kind, _ = steps[index]
            image = np.zeros_like(self.frame)
            if kind == 'list':
                image[487:491, 130:1182] = (251, 189, 80)
            if kind in ('result', 'zero'):
                image[534:537, 80:1200] = 255
            else:
                x, _ = TABS[selected]
                image[700:716, x-110:x-75] = (251, 189, 80)
            if kind in ('pending', 'claim', 'claimed'):
                image[207:212, 511:570] = (88, 73, 104)
                image[363:368, 511:570] = (165, 149, 199)
                image[493:498, 511:570] = (165, 149, 199)
                image[180:219, 975:994] = 255
            return image

        def text_at(box):
            kind, _ = steps[index]
            if kind in ('result', 'zero'):
                return '任意位置继续' if box[1] > 600 else ('0' if kind == 'zero' else '9')
            return '' if kind == 'list' else '奖励信息'

        def tap(observation, action):
            nonlocal index, selected
            self.assertEqual(action.name, steps[index][1])
            if action.name.startswith('切换至副本'):
                selected = TABS.index(action.point)
            index += 1
            if index == len(steps):
                control.request_stop()
            return True

        device.capture.side_effect = frame
        with patch.object(control, 'now', side_effect=range(len(steps)+1)), \
                patch.object(control.stop, 'wait'), \
                patch('ninja3_assistant.activities.abyss.runner.DailyVision') as daily, \
                patch('ninja3_assistant.activities.abyss.runner.BountyCombatVision') as battle, \
                patch('ninja3_assistant.activities.abyss.runner.DailyScreen') as screen, \
                patch('ninja3_assistant.activities.abyss.runner.read_reward',
                      side_effect=lambda *_: steps[index][0]), \
                patch('ninja3_assistant.activities.abyss.runner.GameUI') as ui:
            battle.return_value.detect.return_value.active = False
            daily.return_value.navigation.return_value = (None, None)
            screen.return_value.text.side_effect = text_at
            ui.return_value.perform.side_effect = tap
            play_abyss(device, native, ocr, control, emit)
        self.assertEqual(index, len(steps))
