import unittest
from unittest.mock import Mock, patch

import numpy as np

from ninja3_assistant.activities.intelligence.flow import ActivityScreen
from ninja3_assistant.runtime.runner import RunControl
from ninja3_assistant.scheduling.task_navigation import return_to_home


class TaskNavigationTest(unittest.TestCase):
    def navigate(self, screens, kinds, control=None, target=None, daily_pages=None):
        control = control or RunControl(clock=Mock(side_effect=range(100)))
        activity, race, device, native = Mock(), Mock(), Mock(), Mock()
        daily, bounty = Mock(), Mock()
        activity.detect.side_effect = screens
        activity.navigation.match.return_value = None
        race.detect.side_effect = kinds
        daily.navigation.return_value = (None, None)
        daily.navigation.side_effect = daily_pages
        bounty.navigation.return_value = (None, None)
        device.capture.return_value = np.zeros((720, 1280, 3), np.uint8)
        with patch('ninja3_assistant.device.mumu.prepare_game', return_value=False), \
                patch('ninja3_assistant.device.mumu.MuMu', return_value=device), \
                patch('ninja3_assistant.activities.intelligence.vision.ActivityVision', return_value=activity), \
                patch('ninja3_assistant.activities.weekend.runner.RaceVision', return_value=race), \
                patch('ninja3_assistant.activities.daily.vision.DailyVision', return_value=daily), \
                patch('ninja3_assistant.activities.bounty.daily_vision.DailyBountyVision', return_value=bounty), \
                patch('ninja3_assistant.device.native_touch.create_native_touch', return_value=native), \
                patch.object(control.stop, 'wait', return_value=False):
            result = return_to_home('f', 1, control, target=target)
        native.close.assert_called_once()
        device.close.assert_called_once()
        return result, native

    def test_confirmed_menu_handoff_closes_modal_and_returns_via_hall(self):
        result, native = self.navigate(
            [ActivityScreen('tasks'), ActivityScreen('event_home'), ActivityScreen(), ActivityScreen('home')],
            ['unknown', 'unknown', 'hall', 'unknown'])
        self.assertTrue(result)
        self.assertEqual([c.args[0] for c in native.priority_tap.call_args_list],
                         [(1120, 111), (50, 45), (50, 45)])

    def test_weekend_modal_uses_its_own_close_point(self):
        result, native = self.navigate([ActivityScreen(), ActivityScreen('home')], ['tasks', 'unknown'])
        self.assertTrue(result)
        native.priority_tap.assert_called_once()
        self.assertEqual(native.priority_tap.call_args.args[0], (1190, 400))

    def test_unknown_or_combat_never_gets_a_back_tap(self):
        result, native = self.navigate([ActivityScreen()]*30, ['race']*30)
        self.assertFalse(result)
        native.priority_tap.assert_not_called()

    def test_intelligence_battle_can_resume_without_leaving_it(self):
        result, native = self.navigate([ActivityScreen('battle')], ['unknown'], target='intelligence')
        self.assertTrue(result)
        native.priority_tap.assert_not_called()

    def test_weekend_battle_can_resume_without_leaving_it(self):
        result, native = self.navigate([ActivityScreen()], ['race'], target='weekend')
        self.assertTrue(result)
        native.priority_tap.assert_not_called()

    def test_rank_page_returns_home_before_an_activity_handoff(self):
        result, native = self.navigate(
            [ActivityScreen(), ActivityScreen('home')], ['unknown']*2,
            target='intelligence', daily_pages=[('rank', (75, 50)), ('home', None)])
        self.assertTrue(result)
        self.assertEqual([c.args[0] for c in native.priority_tap.call_args_list], [(75, 50)])

    def test_rank_daily_task_resumes_on_the_rank_page(self):
        result, native = self.navigate(
            [ActivityScreen()], ['unknown'], target='daily_rank',
            daily_pages=[('rank', (75, 50))])
        self.assertTrue(result)
        native.priority_tap.assert_not_called()

    def test_home_from_daily_detector_finishes_handoff_when_activity_detector_misses_it(self):
        result, native = self.navigate(
            [ActivityScreen(), ActivityScreen()], ['unknown']*2,
            target='intelligence', daily_pages=[('rank', (75, 50)), ('home', None)])
        self.assertTrue(result)
        native.priority_tap.assert_called_once()
        self.assertEqual(native.priority_tap.call_args.args[0],(75,50))

    def test_rank_reward_is_closed_before_returning_home(self):
        result, native = self.navigate(
            [ActivityScreen(), ActivityScreen(), ActivityScreen('home')], ['unknown']*3,
            target='weekend', daily_pages=[('reward_continue', (1130, 590)),
                                          ('rank', (75, 50)), ('home', None)])
        self.assertTrue(result)
        self.assertEqual([c.args[0] for c in native.priority_tap.call_args_list],
                         [(1130, 590), (75, 50)])


if __name__ == '__main__':
    unittest.main()
