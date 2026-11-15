import unittest
from unittest.mock import Mock, patch

import numpy as np

from ninja3_assistant.activities.intelligence.flow import ActivityScreen
from ninja3_assistant.runtime.runner import RunControl
from ninja3_assistant.scheduling.task_navigation import return_to_home


class TaskNavigationTest(unittest.TestCase):
    def navigate(self, screens, kinds, control=None, target=None):
        control = control or RunControl(clock=Mock(side_effect=range(100)))
        activity, race, device, native = Mock(), Mock(), Mock(), Mock()
        activity.detect.side_effect = screens
        activity.navigation.match.return_value = None
        race.detect.side_effect = kinds
        device.capture.return_value = np.zeros((720, 1280, 3), np.uint8)
        with patch('ninja3_assistant.device.mumu.prepare_game', return_value=False), \
                patch('ninja3_assistant.device.mumu.MuMu', return_value=device), \
                patch('ninja3_assistant.activities.intelligence.vision.ActivityVision', return_value=activity), \
                patch('ninja3_assistant.activities.weekend.runner.RaceVision', return_value=race), \
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


if __name__ == '__main__':
    unittest.main()
