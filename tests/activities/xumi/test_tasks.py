import unittest
from unittest.mock import Mock

import numpy as np

from ninja3_assistant.activities.daily.vision import DailyScreen
from ninja3_assistant.activities.xumi.tasks import read_task_panel, read_tasks, tasks_claimed


def task_screen(statuses):
    words = [('每日任务', (574, 101, 120, 30))]
    labels = ('完咸1次方寸试炼', '完成1次大千试炼', '方寸试炼累计获得50000分', '大千试炼累计获得50000分')
    for index, (label, status) in enumerate(zip(labels, statuses)):
        y = 205 + 106*index
        words.append((label, (376+index, y+4, 275, 26)))
        if status:
            words.append((status, (930-index, y-14+index, 103, 24)))
    return DailyScreen(np.zeros((720,1280,3), np.uint8),
                       {'words': [{'text': text, 'bbox': box} for text, box in words]})


class XumiTasksTest(unittest.TestCase):
    def test_rows_follow_labels_and_keep_missing_or_conflicting_counts_unknown(self):
        rows = read_tasks(task_screen(('', '1／1', '00019/50000', '1/100000领取')))
        self.assertEqual([(row.mode, row.kind, row.current, row.goal) for row in rows], [
            ('fangcun', 'match', None, 1), ('daqian', 'match', 1, 1),
            ('fangcun', 'score', 19, 50000), ('daqian', 'score', None, 50000),
        ])
        self.assertTrue(all(row.claim_point is None for row in rows))
        self.assertFalse(tasks_claimed(rows))

    def test_all_four_explicit_claims_are_required_to_finish(self):
        rows = read_tasks(task_screen(('已领取', '已领取', '已领取', '50000/50000领取奖励')))
        self.assertTrue(all(row.claimed for row in rows[:3]))
        self.assertIsNotNone(rows[3].claim_point)
        self.assertFalse(tasks_claimed(rows))
        rows = read_tasks(task_screen(('已领取', '已领取', '已领取', '已领取')))
        self.assertTrue(tasks_claimed(rows))
        self.assertFalse(tasks_claimed(rows[:-1]))

    def test_right_crop_recovers_missing_counts_and_preserves_other_confirmed_rewards(self):
        screen = task_screen(('', '', '领取', '0/50000'))
        engine = Mock()
        engine.read.return_value = {'words': [
            {'text': '0/1', 'bbox': (690, 68, 77, 47)},
            {'text': '0/1', 'bbox': (690, 386, 77, 47)},
        ]}
        rows = read_task_panel(screen, engine)
        self.assertEqual(engine.read.call_args.args[0].shape, (1257, 840, 3))
        self.assertEqual([row.current for row in rows], [0, 0, None, 0])
        self.assertIsNotNone(rows[2].claim_point)

    def test_right_crop_prevents_dropping_leading_digits_of_a_valid_score(self):
        screen = task_screen(('已领取','已领取','694/50000','34/50000'))
        engine = Mock()
        engine.read.return_value = {'words': [
            {'text': '15694/50000', 'bbox': (357, 704, 438, 54)},
            {'text': '34/50000', 'bbox': (471, 1022, 324, 54)},
        ]}
        rows = read_task_panel(screen,engine)
        self.assertEqual([row.current for row in rows], [None,None,15694,34])


if __name__ == '__main__':
    unittest.main()
