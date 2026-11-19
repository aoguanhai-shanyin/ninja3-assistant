import unittest
from unittest.mock import Mock, patch

import numpy as np

from ninja3_assistant.activities.intelligence.flow import ActivityScreen
from ninja3_assistant.activities.daily.vision import DailyScreen
from ninja3_assistant.activities.xumi.vision import XumiScreen
from ninja3_assistant.runtime.runner import RunControl
from ninja3_assistant.scheduling.task_navigation import return_to_home
from ninja3_assistant.scheduling.invitation_handoff import InvitationHandoff
from ninja3_assistant.vision.game_update import game_update_point


class TaskNavigationTest(unittest.TestCase):
    def navigate(self, screens, kinds, control=None, target=None, daily_pages=None, chore=None,
                 purchase_page=None, update=None, xumi_page=None):
        control = control or RunControl(clock=Mock(side_effect=range(100)))
        activity, race, device, native = Mock(), Mock(), Mock(), Mock()
        daily, bounty = Mock(), Mock()
        activity.detect.side_effect = screens
        activity.navigation.match.return_value = None
        race.detect.side_effect = kinds
        daily.navigation.return_value = (None, None)
        daily.navigation.side_effect = daily_pages
        daily.chore_navigation.return_value = chore
        if purchase_page is not None:
            daily.observe.return_value = DailyScreen(None,{},purchase_page)
            daily.current_action.return_value = True
        bounty.navigation.return_value = (None, None)
        device.capture.return_value = np.zeros((720, 1280, 3), np.uint8)
        if update is not None:
            device.capture.side_effect = [update.frame,update.frame,device.capture.return_value]
            daily.current_action.return_value = True
        with patch('ninja3_assistant.device.mumu.prepare_game', return_value=False), \
                patch('ninja3_assistant.device.mumu.MuMu', return_value=device), \
                patch('ninja3_assistant.activities.intelligence.vision.ActivityVision', return_value=activity), \
                patch('ninja3_assistant.activities.weekend.runner.RaceVision', return_value=race), \
                patch('ninja3_assistant.activities.daily.vision.DailyVision', return_value=daily), \
                patch('ninja3_assistant.activities.xumi.vision.XumiVision') as xumi, \
                patch('ninja3_assistant.activities.intelligence.ocr.WindowsOcr') as ocr, \
                patch('ninja3_assistant.activities.bounty.daily_vision.DailyBountyVision', return_value=bounty), \
                patch('ninja3_assistant.device.native_touch.create_native_touch', return_value=native), \
                patch.object(control.stop, 'wait', return_value=False):
            xumi.return_value.scene.return_value = xumi_page or XumiScreen()
            if update is not None:
                ocr.return_value.read.return_value = update.ocr
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

    def test_restart_dismisses_purchase_dialog_before_running_daily_task(self):
        for target in ('daily_daily_shop','daily_bounty'):
            with self.subTest(target=target):
                result, native = self.navigate([ActivityScreen(), ActivityScreen('home')],
                    ['unknown','unknown'], target=target, purchase_page='bounty_dialog')
                self.assertTrue(result)
                native.priority_tap.assert_called_once()
                self.assertEqual(native.priority_tap.call_args.args[0],(1100,90))

    def test_client_update_preempts_navigation_and_requires_all_prompt_fields(self):
        frame = np.zeros((720,1280,3),np.uint8)
        frame[261:455] = (7,13,28)
        frame[261] = frame[454] = 255
        words = [dict(text=text,bbox=box) for text,box in (
            ('的游蜮有更新',(554,295,170,36)), ('请重启客户端进行游蜮更新',(468,375,343,28)),
            ('重启客户端',(569,507,137,28)))]
        update = DailyScreen(frame,{'words':words})
        self.assertIsNone(game_update_point(DailyScreen(frame,{'words':words[1:]})))
        self.assertIsNone(game_update_point(DailyScreen(np.zeros_like(frame),update.ocr)))
        result,native = self.navigate([ActivityScreen('home')],['unknown'],target='daily_daily_shop',update=update)
        self.assertTrue(result)
        native.priority_tap.assert_called_once()
        self.assertEqual(native.priority_tap.call_args.args[0],(638,521))

    def test_xumi_task_can_resume_from_its_team_editor(self):
        result,native = self.navigate([],[],target='xumi',xumi_page=XumiScreen('team',back=(47,40)))
        self.assertTrue(result)
        native.priority_tap.assert_not_called()

    def test_unknown_or_combat_never_gets_a_back_tap(self):
        result, native = self.navigate([ActivityScreen()]*30, ['race']*30)
        self.assertFalse(result)
        native.priority_tap.assert_not_called()

    def test_foreign_battle_prevents_back_tap_from_background_menu_match(self):
        result, native = self.navigate(
            [ActivityScreen('battle')]*30,['unknown']*30,target='daily_bounty',
            daily_pages=[('rank',(75,50))]*30)
        self.assertFalse(result)
        native.priority_tap.assert_not_called()

    def test_invitation_during_safe_navigation_returns_handoff_without_touch(self):
        control = RunControl(clock=Mock(side_effect=range(100)))
        control.bounty_invite_targets = {'A':'daily_bounty'}
        control.bounty_navigation_safe = True
        with patch('ninja3_assistant.activities.intelligence.ocr.WindowsOcr'), \
             patch('ninja3_assistant.scheduling.task_navigation.bounty_handoff',
                   return_value=InvitationHandoff('daily_bounty')):
            result,native = self.navigate([ActivityScreen('invite',(470,510))],['unknown'],control=control)
        self.assertEqual(result,InvitationHandoff('daily_bounty'))
        native.priority_tap.assert_not_called()

    def test_invitation_with_unknown_initial_context_is_not_handed_off(self):
        control = RunControl(clock=Mock(side_effect=range(100)))
        control.bounty_invite_targets = {'A':'daily_bounty'}
        with patch('ninja3_assistant.scheduling.task_navigation.bounty_handoff') as handoff:
            result,native = self.navigate([ActivityScreen('invite',(470,510)),ActivityScreen('home')],
                                         ['unknown','unknown'],control=control)
        self.assertTrue(result)
        handoff.assert_not_called()
        native.priority_tap.assert_called_once()

    def test_intelligence_battle_can_resume_without_leaving_it(self):
        result, native = self.navigate([ActivityScreen('battle')], ['unknown'], target='intelligence')
        self.assertTrue(result)
        native.priority_tap.assert_not_called()

    def test_intelligence_tasks_panel_routes_only_to_its_owner(self):
        panel = ActivityScreen('tasks')
        result, native = self.navigate([panel],['unknown'],target='intelligence')
        self.assertTrue(result)
        native.priority_tap.assert_not_called()
        result, native = self.navigate(
            [panel,ActivityScreen('event_home'),ActivityScreen('home')],['unknown']*3,
            target='daily_bounty',daily_pages=[('rank',(75,50)),(None,None),('home',None)])
        self.assertTrue(result)
        self.assertEqual([c.args[0] for c in native.priority_tap.call_args_list],[(1120,111),(50,45)])

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

    def test_bounty_returns_home_from_a_recognized_quiz_page(self):
        result, native = self.navigate(
            [ActivityScreen(), ActivityScreen('home')], ['unknown']*2,
            target='daily_bounty', chore=('landlady-quiz', (70,50)))
        self.assertTrue(result)
        self.assertEqual([call.args[0] for call in native.priority_tap.call_args_list],[(70,50)])

    def test_login_is_completed_before_handing_off_any_daily_runner(self):
        for target in ('daily_monthly','daily_bounty','daily_bounty_s'):
            with self.subTest(target=target):
                result,native=self.navigate(
                    [ActivityScreen(),ActivityScreen('home')],['unknown']*2,target=target,
                    daily_pages=[('login',(640,515)),('home',None)])
                self.assertTrue(result)
                self.assertEqual([call.args[0] for call in native.priority_tap.call_args_list],
                                 [(640,515)])

    def test_activity_login_anchor_also_waits_for_home_before_handoff(self):
        result,native=self.navigate(
            [ActivityScreen('login',(640,515)),ActivityScreen('home')],['unknown']*2,
            target='intelligence')
        self.assertTrue(result)
        native.priority_tap.assert_called_once()
        self.assertEqual(native.priority_tap.call_args.args[0],(640,515))

    def test_network_retry_waits_for_home_before_handoff(self):
        retry = ActivityScreen('network_retry', (638, 538))
        result, native = self.navigate([retry, retry, ActivityScreen('home')], ['unknown']*3,
                                       target='daily_monthly')
        self.assertTrue(result)
        native.priority_tap.assert_called_once()
        self.assertEqual(native.priority_tap.call_args.args[0], (638, 538))

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
