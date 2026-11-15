import unittest
from datetime import datetime

from activity_flow import CHINA, ActivityFlow, ActivityScreen, daily_key, matching_open


def local(hour, minute=0, day=13):
    return datetime(2026, 11, day, hour, minute, tzinfo=CHINA)


def step_after_scan(flow, screen, now, wall_time):
    decision = flow.step(screen, now, wall_time)
    while decision.action in ('reset_tasks_scroll', 'scroll_tasks'):
        flow.completed(decision, now)
        now += 1.01
        decision = flow.step(screen, now, wall_time)
    return decision


class ActivityFlowTest(unittest.TestCase):
    def test_schedule_boundaries_use_china_time(self):
        for hour, expected in ((9, False), (10, True), (13, True), (14, False),
                               (17, False), (18, True), (21, True), (22, False)):
            self.assertEqual(matching_open(local(hour)), expected)
        self.assertEqual(daily_key(local(4)), local(0, day=12).date())
        self.assertEqual(daily_key(local(5)), local(0).date())

    def test_two_matches_never_substitute_for_the_damage_task(self):
        flow = ActivityFlow()
        screen = ActivityScreen('tasks', daily=('claimed', 'incomplete'), claim_points=(None, None))
        self.assertEqual(step_after_scan(flow, screen, 0, local(13)).action, 'close_tasks')
        self.assertFalse(flow.daily_complete)

    def test_stop_only_when_three_daily_rewards_confirmed_claimed(self):
        for states, action in ((('claimed', 'claimed', 'claimed'), 'done'),
                               (('claimed', 'unknown'), 'wait'),
                               (('claimed', 'claimable'), 'claim')):
            flow = ActivityFlow()
            result = step_after_scan(flow, ActivityScreen('tasks', daily=states,
                                             claim_points=(None, (960, 312), None), task_ids=('games', 'damage', 'characters') if len(states) == 3 else ('games', 'damage')), 0, local(13))
            self.assertEqual(result.action, action)

    def test_continuous_mode_does_not_stop_after_daily(self):
        flow = ActivityFlow('continuous')
        self.assertEqual(step_after_scan(flow, ActivityScreen('tasks', daily=('claimed', 'claimed', 'claimed'), task_ids=('games', 'damage', 'characters')),
                                   0, local(13)).action, 'close_tasks')
        self.assertNotEqual(step_after_scan(flow, ActivityScreen('event_home'), 2, local(13)).action, 'done')

    def test_closed_window_waits_without_tapping_match(self):
        flow = ActivityFlow()
        flow.checked_day = daily_key(local(14))
        result = step_after_scan(flow, ActivityScreen('event_home'), 0, local(14))
        self.assertEqual(result.action, 'wait')
        self.assertIsNone(result.point)
        self.assertIn('开放', result.status)

    def test_daily_rollover_rechecks_tasks(self):
        flow = ActivityFlow()
        flow.checked_day, flow.daily_complete = daily_key(local(4)), True
        result = step_after_scan(flow, ActivityScreen('event_home'), 0, local(5))
        self.assertEqual(result.action, 'tasks')
        self.assertFalse(flow.daily_complete)

    def test_unknown_and_incomplete_task_reading_never_click(self):
        for screen in (ActivityScreen(), ActivityScreen('tasks'),
                       ActivityScreen('tasks', daily=('claimed', 'unknown'))):
            result = step_after_scan(ActivityFlow(), screen, 0, local(13))
            self.assertEqual(result.action, 'wait')
            self.assertIsNone(result.point)

    def test_two_claimed_tasks_cannot_finish_three_daily_tasks(self):
        flow = ActivityFlow()
        screen = ActivityScreen('tasks', daily=('claimed', 'claimed'))
        self.assertEqual(step_after_scan(flow, screen, 0, local(13)).action, 'wait')
        self.assertFalse(flow.daily_complete)

    def test_daily_scan_is_a_swipe_inside_the_daily_list(self):
        flow = ActivityFlow()
        screen = ActivityScreen('tasks', daily=('incomplete',)*3, task_ids=('games', 'damage', 'characters'))
        reset = flow.step(screen, 0, local(13))
        self.assertEqual(reset.action, 'reset_tasks_scroll')
        flow.completed(reset, 0)
        scan = flow.step(screen, 2, local(13))
        self.assertEqual((scan.action, scan.point, scan.end_point),
                         ('scroll_tasks', (760, 315), (760, 177)))
        flow.completed(scan, 2)
        self.assertEqual(flow.step(screen, 4, local(13)).action, 'close_tasks')

    def test_only_completed_touch_advances_cooldown(self):
        flow = ActivityFlow()
        screen = ActivityScreen('hall', (373, 389))
        decision = step_after_scan(flow, screen, 0, local(13))
        self.assertEqual(decision.action, 'hall')
        self.assertEqual(step_after_scan(flow, screen, .1, local(13)).action, 'hall')
        flow.completed(decision, .1)
        self.assertEqual(step_after_scan(flow, screen, .5, local(13)).action, 'wait')
        self.assertEqual(step_after_scan(flow, screen, 4, local(13)).action, 'wait')


if __name__ == '__main__':
    unittest.main()
