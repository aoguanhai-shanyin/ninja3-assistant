"""Runner lifecycle checks with simulated frames and native inputs only."""
from contextlib import ExitStack
from itertools import count
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ninja3_assistant.activities.bounty import daily_runner
from ninja3_assistant.activities.bounty.combat import BountyBattleState
from ninja3_assistant.activities.daily.flow import DailyDecision
from ninja3_assistant.runtime.runner import RunControl
from ninja3_assistant.scheduling.invitation_handoff import InvitationHandoff
from ninja3_assistant.scheduling.task_outcome import TaskOutcome
from ninja3_assistant.vision.invitation import BountyInvitation


class DailyBountyRunnerTest(unittest.TestCase):
    def test_s_burst_uses_bounded_native_taps_and_cancels_old_target(self):
        kernel=Mock(captured_at=10.)
        kernel.clock.return_value=10.01
        kernel.plan.side_effect=(1,2)
        burst=daily_runner._OrderJoinBurst(kernel)
        self.assertEqual(burst.submit((401,307)),1)
        finger,actions=kernel.plan.call_args.args
        self.assertEqual(finger,3)
        self.assertEqual([action.point for action in actions],[(401,307)]*3)
        self.assertEqual([a.offset_ms-b.offset_ms for a,b in zip(actions[1:],actions)],[30,30])
        self.assertLess(kernel.plan.call_args.kwargs['deadline'],10.15)
        self.assertIsNone(burst.submit((401,307)))
        kernel.plan.assert_called_once()
        self.assertEqual(burst.submit((771,307)),2)
        kernel.cancel_plan.assert_called_once_with(3)
        kernel.clock.return_value=10.08
        self.assertIsNone(burst.submit((771,307)))
        self.assertEqual(kernel.cancel_plan.call_count,2)
        burst.stop()
        self.assertEqual(kernel.plan.call_count,2)

    def run_simulation(self, states, *, page=None, ocr_page=None, decision=None, fresh=True, grade='D',
                       cleanup_error=False, stop_on_capture=False, navigation_pages=None, stop_on_ocr=0,
                       invite_only=False, invite_resume=False, recover_in_place=False, tasks_overlay=False,
                       active_invite=True, passive_invite=True):
        ticks = [0.]
        control = RunControl(clock=lambda:ticks[0])
        device,native,vision,combat_vision,ocr,flow = (Mock() for _ in range(6))
        frame = np.zeros((720,1280,3),np.uint8)
        count = [0]
        def capture():
            ticks[0] += .3
            count[0] += 1
            if stop_on_capture:
                control.request_stop()
            if count[0] > len(states)+(1 if decision and decision.point else 0):
                raise RuntimeError('simulated capture error')
            return frame
        device.capture.side_effect = capture
        native.movement_remaining.side_effect = lambda:2. if native.schedule_movement.called else 0.
        native.schedule_movement.return_value = 1
        native.timeline = []
        native.schedule_movement.side_effect = lambda *args,**kwargs:(
            native.timeline.append(('schedule',count[0])),1)[1]
        native.release.side_effect = lambda:native.timeline.append(('cancel',count[0]))
        native.plan.return_value = 1
        if cleanup_error:
            native.close.side_effect = RuntimeError('simulated native close error')
        vision.navigation.return_value = (page,None)
        vision.navigation.side_effect = navigation_pages
        vision.match.return_value = vision.common.match.return_value = None
        vision.order_result_confirm.return_value = None
        vision.tasks_panel.return_value = tasks_overlay
        vision.common.invitations.detect.return_value = (470,510) if page == 'invite' else None
        vision.observe.return_value = Mock(page=ocr_page or page or 'unknown')
        if stop_on_ocr == 2:
            vision.observe.return_value.remaining = None
        vision.current_action.return_value = fresh
        combat_vision.detect.side_effect = states
        ocr.read.return_value = {'words':[]}
        if stop_on_ocr:
            reads = [0]
            def read(*args,**kwargs):
                reads[0] += 1
                if reads[0] == stop_on_ocr:
                    control.request_stop()
                return {'words':[]}
            ocr.read.side_effect = read
        flow.battle_seen = False
        flow.step.return_value = decision or DailyDecision(status='等待')
        events = []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for name,value in (('project_root',lambda:Path(directory)),('prepare_game',Mock(return_value=False)),
                               ('MuMu',Mock(return_value=device)),('create_native_touch',Mock(return_value=native)),
                               ('DailyBountyVision',Mock(return_value=vision)),('BountyCombatVision',Mock(return_value=combat_vision)),
                               ('WindowsOcr',Mock(return_value=ocr)),('DailyBountyFlow',Mock(return_value=flow)),
                               ('FrameDiagnostics',Mock())):
                stack.enter_context(patch.object(daily_runner,name,value))
            flow_factory = daily_runner.DailyBountyFlow
            stack.enter_context(patch.object(control.stop,'wait',return_value=False))
            outcome = daily_runner.run_daily_bounty('folder',1,control,
                lambda kind,value:events.append((kind,value)),grade=grade,
                active_invite=active_invite,passive_invite=passive_invite,invite_only=invite_only,
                invite_resume=invite_resume,recover_in_place=recover_in_place)
        native.close.assert_called_once()
        device.close.assert_called_once()
        self.assertIsNone(control._release_input)
        self.assertIn(('finished',''),events)
        return outcome,device,native,vision,combat_vision,ocr,flow,flow_factory

    def test_runner_forwards_both_invitation_switches_to_the_flow(self):
        for active,passive in ((False,True),(True,False)):
            with self.subTest(active=active,passive=passive):
                *_,factory = self.run_simulation([BountyBattleState()],page='bounty_daily',decision=DailyDecision('done'),
                                                active_invite=active,passive_invite=passive)
                self.assertIs(factory.call_args.kwargs['active_invite'],active)
                self.assertIs(factory.call_args.kwargs['passive_invite'],passive)

    def test_confirmed_combat_never_uses_ocr_and_capture_error_retains_match(self):
        active = BountyBattleState(active=True,ground=True)
        outcome,_,native,vision,_,ocr,flow,_ = self.run_simulation([active,active])
        self.assertIs(outcome,TaskOutcome.RECOVER_IN_PLACE)
        self.assertTrue(flow.battle_seen)
        ocr.read.assert_not_called()
        native.schedule_movement.assert_called_once()
        vision.navigation.assert_not_called()

    def test_recovered_battle_keeps_ownership_before_processing_invitation(self):
        _,_,_,_,_,_,flow,_ = self.run_simulation(
            [BountyBattleState()],page='invite',recover_in_place=True)
        self.assertTrue(flow.battle_seen)
        flow.step.assert_called_once()

    def test_task_panel_over_battle_hud_cancels_movement_and_reads_menu(self):
        _,_,native,_,_,ocr,_,_ = self.run_simulation(
            [BountyBattleState(active=True,ground=True)],tasks_overlay=True,
            ocr_page='bounty_tasks')
        native.schedule_movement.assert_not_called()
        ocr.read.assert_called_once()

    def test_unknown_frame_during_match_only_uses_ocr_after_half_second(self):
        active = BountyBattleState(active=True,ground=True)
        unknown = BountyBattleState()
        outcome,_,_,_,_,ocr,_,_ = self.run_simulation([active,unknown,unknown,unknown])
        self.assertIs(outcome,TaskOutcome.RECOVER_IN_PLACE)
        ocr.read.assert_called_once()

    def test_short_unknown_retains_existing_movement_without_renewing_or_ocr(self):
        active = BountyBattleState(active=True,ground=True)
        outcome,_,native,_,_,ocr,_,_ = self.run_simulation(
            [active,BountyBattleState(),active])
        self.assertIs(outcome,TaskOutcome.RECOVER_IN_PLACE)
        self.assertEqual(native.timeline,[('schedule',1),('cancel',4)])
        ocr.read.assert_not_called()

    def test_unknown_timeout_cancels_once_and_cannot_restore_the_old_plan(self):
        active = BountyBattleState(active=True,ground=True)
        _,_,native,_,_,ocr,_,_ = self.run_simulation([active]+[BountyBattleState()]*4)
        self.assertEqual(native.timeline,[('schedule',1),('cancel',4)])
        native.schedule_movement.assert_called_once()
        ocr.read.assert_called_once()

    def test_recognized_invite_or_result_cancels_without_unknown_grace(self):
        active = BountyBattleState(active=True,ground=True)
        for page in ('invite','result'):
            with self.subTest(page=page):
                _,_,native,_,_,ocr,_,_ = self.run_simulation(
                    [active,BountyBattleState()],ocr_page=page,
                    navigation_pages=[(page,None)])
                self.assertEqual(native.timeline,[('schedule',1),('cancel',2)])
                ocr.read.assert_called_once()

    def test_recognized_rope_cancels_without_unknown_grace(self):
        active = BountyBattleState(active=True,ground=True)
        rope = BountyBattleState(active=True,on_rope=True)
        _,_,native,_,_,ocr,_,_ = self.run_simulation([active,rope])
        self.assertEqual(native.timeline,[('schedule',1),('cancel',2)])
        ocr.read.assert_not_called()

    def test_result_found_by_fallback_is_a_safe_menu_checkpoint(self):
        active = BountyBattleState(active=True,ground=True)
        unknown = BountyBattleState()
        outcome,_,_,_,_,ocr,_,_ = self.run_simulation(
            [active,unknown,unknown,unknown],ocr_page='result')
        self.assertIs(outcome,TaskOutcome.ERROR)
        ocr.read.assert_called_once()

    def test_invitation_probe_disallows_hud_behind_it_before_ocr(self):
        outcome,_,native,_,combat_vision,ocr,_,_ = self.run_simulation(
            [BountyBattleState(active=True,ground=True)],page='invite')
        self.assertIs(outcome,TaskOutcome.ERROR)
        combat_vision.reset.assert_called()
        native.schedule_movement.assert_not_called()
        native.plan.assert_not_called()
        ocr.read.assert_called_once()

    def test_changed_free_button_is_rejected_before_touch_and_is_not_completed(self):
        decision = DailyDecision('tap',(640,480),'领取免费次数',free_kind='bounty',guard=(530,450,746,507))
        _,_,native,vision,_,_,flow,_ = self.run_simulation(
            [BountyBattleState()],page='bounty_free',decision=decision,fresh=False)
        vision.current_action.assert_called_once()
        native.priority_tap.assert_not_called()
        flow.completed.assert_not_called()

    def test_s_order_dispatch_and_retry_yield_waiting_without_finishing_daily(self):
        outcome,_,_,_,_,_,_,factory = self.run_simulation(
            [BountyBattleState()],page='bounty_order',decision=DailyDecision('retry'),grade='S')
        self.assertEqual(factory.call_args.kwargs['kind'],'bounty_s')
        self.assertEqual(factory.call_args.kwargs['grade'],'D')
        self.assertTrue(callable(factory.call_args.kwargs['clock']))
        self.assertIs(outcome,TaskOutcome.WAITING)

    def test_cleanup_failure_does_not_claim_daily_completed(self):
        outcome,*_ = self.run_simulation([BountyBattleState()],page='bounty_daily',
                                        decision=DailyDecision('done'),cleanup_error=True)
        self.assertIs(outcome,TaskOutcome.ERROR)

    def test_stop_during_capture_releases_native_input_without_dispatching_battle(self):
        outcome,_,native,_,_,ocr,_,_ = self.run_simulation(
            [BountyBattleState(active=True,ground=True)],stop_on_capture=True)
        self.assertIs(outcome,TaskOutcome.STOPPED)
        native.stop_all.assert_called_once()
        native.schedule_movement.assert_not_called()
        native.plan.assert_not_called()
        ocr.read.assert_not_called()

    def test_stop_during_quota_ocr_does_not_mark_daily_complete(self):
        for read_number in (1,2):
            with self.subTest(read_number=read_number):
                outcome,_,native,_,_,ocr,flow,_ = self.run_simulation(
                    [BountyBattleState()],page='bounty_order',grade='S',
                    decision=DailyDecision('done'),stop_on_ocr=read_number)
                self.assertIs(outcome,TaskOutcome.STOPPED)
                self.assertEqual(ocr.read.call_count,read_number)
                flow.step.assert_not_called()
                native.priority_tap.assert_not_called()

    def test_expired_invitation_does_not_start_an_unrelated_bounty(self):
        outcome,_,native,_,_,_,flow,_ = self.run_simulation(
            [BountyBattleState()],page='bounty_daily',invite_only=True,
            decision=DailyDecision('tap',(875,670),'创建A级每日悬赏队伍'))
        self.assertIs(outcome,TaskOutcome.WAITING)
        flow.step.assert_not_called()
        native.priority_tap.assert_not_called()

    def test_changed_invitation_is_not_accepted_or_declined_in_handoff(self):
        outcome,_,native,_,_,_,flow,_ = self.run_simulation(
            [BountyBattleState()],page='invite',invite_only=True,
            decision=DailyDecision('tap',(470,510),'拒绝队友或家族邀请'))
        self.assertIs(outcome,TaskOutcome.WAITING)
        flow.completed.assert_not_called()
        native.priority_tap.assert_not_called()

    def test_idle_board_hands_off_a_selected_other_grade_without_touching_the_invite(self):
        cases = (('S','bounty_order','A',True,False,False,False,'daily_bounty'),
                 ('A','bounty_daily','S',True,False,False,False,'daily_bounty_s'),
                 ('A','bounty_order','S',True,False,False,False,None),
                 ('S','bounty_room','A',True,False,False,False,None),
                 ('S','bounty_order','A',True,True,False,False,None),
                 ('S','bounty_order','A',True,False,True,False,None),
                 ('S','bounty_order','A',True,False,False,True,None),
                 ('S','bounty_order','S',True,False,False,False,None),
                 ('A','bounty_daily','S',False,False,False,False,None),
                 ('S','bounty_order','B',True,False,False,False,None))
        for active_grade,page,grade,passive,in_room,in_battle,pending,target in cases:
            with self.subTest(active_grade=active_grade,page=page,grade=grade,passive=passive,in_room=in_room,
                              in_battle=in_battle,pending=pending):
                control = RunControl()
                control.bounty_invite_targets = {'A':'daily_bounty','S':'daily_bounty_s'}
                control.bounty_invite_suppressed = True
                device,native,vision,combat_vision,ocr,flow,kernel = (Mock() for _ in range(7))
                kernel.capture.return_value = np.zeros((720,1280,3),np.uint8)
                kernel.capture_ms,kernel.captured_at,kernel.stale_drops = 1.,0.,0
                vision.navigation.side_effect = [(page,None),('invite',None)]
                invitation = BountyInvitation('friend',grade,(810,526))
                vision.common.invitations.bounty_invitation.return_value = invitation
                vision.observe.side_effect = [Mock(page=page,remaining=2,s_filter='on',ss_filter='off',
                                                   join_pending=pending),Mock(page='invite',invitation=invitation)]
                combat_vision.detect.return_value = BountyBattleState()
                ocr.read.return_value = {'words':[]}
                flow.in_room,flow.order_join_pending_at = in_room,None
                flow.step.side_effect = [DailyDecision(),DailyDecision('done')]
                with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                    for name,value in (('project_root',lambda:Path(directory)),('prepare_game',Mock(return_value=False)),
                        ('MuMu',Mock(return_value=device)),('create_native_touch',Mock(return_value=native)),
                        ('DailyBountyVision',Mock(return_value=vision)),('BountyCombatVision',Mock(return_value=combat_vision)),
                        ('WindowsOcr',Mock(return_value=ocr)),('DailyBountyFlow',Mock(return_value=flow)),
                        ('RealtimeBattleKernel',Mock(return_value=kernel)),('BountyCombat',Mock()),('FrameDiagnostics',Mock())):
                        stack.enter_context(patch.object(daily_runner,name,value))
                    stack.enter_context(patch.object(control.stop,'wait',return_value=False))
                    stack.enter_context(patch.object(daily_runner.time,'perf_counter',side_effect=count(step=1).__next__))
                    result = daily_runner.run_daily_bounty('folder',1,control,Mock(),grade=active_grade,passive_invite=passive,
                                                          recover_in_place=in_battle)
                self.assertEqual(result,InvitationHandoff(target) if target else TaskOutcome.DAILY_DONE)
                self.assertEqual(flow.step.call_count,1 if target else 2)
                kernel.priority_tap.assert_not_called()
                kernel.plan.assert_not_called()
                native.close.assert_called_once()

    def test_handoff_accepts_one_invite_then_returns_after_settlement(self):
        control = RunControl()
        device,native,vision,combat_vision,ocr,flow,kernel = (Mock() for _ in range(7))
        frame = np.zeros((720,1280,3),np.uint8)
        kernel.capture.return_value = frame
        kernel.capture_ms,kernel.captured_at,kernel.stale_drops = 1.,0.,0
        kernel.priority_tap.return_value = True
        vision.navigation.side_effect = [(page,None) for page in ('invite','bounty_room','result','bounty_daily')]
        screens = [Mock(page=page) for page in ('invite','bounty_room','result','bounty_daily')]
        screens[0].invitation.accept = (810,526)
        screens[1].grade = 'A'
        vision.observe.side_effect = screens
        vision.current_action.return_value = True
        combat_vision.detect.return_value = BountyBattleState()
        ocr.read.return_value = {'words':[]}
        flow.step.side_effect = [DailyDecision('tap',(810,526),'接受好友A级悬赏邀请'),
                                 DailyDecision('tap',(1100,660),'悬赏队伍准备'),
                                 DailyDecision('tap',(640,680),'确认悬赏结算')]
        events = []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for name,value in (('project_root',lambda:Path(directory)),('prepare_game',Mock(return_value=False)),
                               ('MuMu',Mock(return_value=device)),('create_native_touch',Mock(return_value=native)),
                               ('DailyBountyVision',Mock(return_value=vision)),('BountyCombatVision',Mock(return_value=combat_vision)),
                               ('WindowsOcr',Mock(return_value=ocr)),('DailyBountyFlow',Mock(return_value=flow)),
                               ('RealtimeBattleKernel',Mock(return_value=kernel)),('BountyCombat',Mock()),('FrameDiagnostics',Mock())):
                stack.enter_context(patch.object(daily_runner,name,value))
            stack.enter_context(patch.object(control.stop,'wait',return_value=False))
            stack.enter_context(patch.object(daily_runner.time,'perf_counter',side_effect=count(step=1).__next__))
            outcome = daily_runner.run_daily_bounty('folder',1,control,
                lambda kind,value:events.append((kind,value)),grade='A',invite_only=True)
        self.assertIs(outcome,TaskOutcome.MATCH_DONE)
        self.assertEqual(flow.completed.call_count,3)
        self.assertEqual([call.args[0] for call in kernel.priority_tap.call_args_list],
                         [(810,526),(1100,660),(640,680)])

    def assert_invitation_join_does_not_start_daily_navigation(self, page, expected):
        control = RunControl(clock=count().__next__)
        device,native,vision,combat_vision,ocr,flow,kernel = (Mock() for _ in range(7))
        frame = np.zeros((720,1280,3),np.uint8)
        kernel.capture.return_value = frame
        kernel.capture_ms,kernel.captured_at,kernel.stale_drops = 1.,0.,0
        kernel.priority_tap.return_value = True
        vision.navigation.side_effect = [('invite',None)]+[(page,None)]*10
        invitation = Mock(page='invite')
        invitation.invitation.accept = (810,526)
        vision.observe.side_effect = [invitation]+[Mock(page=page)]*10
        vision.current_action.return_value = True
        combat_vision.detect.return_value = BountyBattleState()
        ocr.read.return_value = {'words':[]}
        flow.step.return_value = DailyDecision('tap',(810,526),'接受好友A级悬赏邀请')
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for name,value in (('project_root',lambda:Path(directory)),('prepare_game',Mock(return_value=False)),
                               ('MuMu',Mock(return_value=device)),('create_native_touch',Mock(return_value=native)),
                               ('DailyBountyVision',Mock(return_value=vision)),('BountyCombatVision',Mock(return_value=combat_vision)),
                               ('WindowsOcr',Mock(return_value=ocr)),('DailyBountyFlow',Mock(return_value=flow)),
                               ('RealtimeBattleKernel',Mock(return_value=kernel)),('BountyCombat',Mock()),('FrameDiagnostics',Mock())):
                stack.enter_context(patch.object(daily_runner,name,value))
            stack.enter_context(patch.object(control.stop,'wait',return_value=False))
            stack.enter_context(patch.object(daily_runner.time,'perf_counter',side_effect=count(step=1).__next__))
            outcome = daily_runner.run_daily_bounty('folder',1,control,Mock(),grade='A',invite_only=True)
        self.assertIs(outcome,expected)
        flow.step.assert_called_once()
        kernel.priority_tap.assert_called_once()

    def test_invitation_touch_alone_does_not_own_a_match(self):
        for page,expected in (('home',TaskOutcome.WAITING),('unknown',TaskOutcome.WAITING),
                              ('intelligence_tasks',TaskOutcome.SCENE_CHANGED)):
            with self.subTest(page=page):
                self.assert_invitation_join_does_not_start_daily_navigation(page,expected)

    def test_recognized_foreign_panel_ends_both_kinds_of_bounty_recovery(self):
        for invited in (False,True):
            with self.subTest(invited=invited):
                outcome,_,native,_,_,_,flow,_ = self.run_simulation(
                    [BountyBattleState()]*5,ocr_page='intelligence_tasks',invite_only=invited,
                    invite_resume=invited,recover_in_place=not invited)
                self.assertIs(outcome,TaskOutcome.SCENE_CHANGED)
                flow.step.assert_not_called()
                native.priority_tap.assert_not_called()

    def test_wrong_grade_invited_room_is_left_before_returning_to_original_task(self):
        control = RunControl()
        device,native,vision,combat_vision,ocr,flow,kernel = (Mock() for _ in range(7))
        frame = np.zeros((720,1280,3),np.uint8)
        kernel.capture.return_value = frame
        kernel.capture_ms,kernel.captured_at,kernel.stale_drops = 1.,0.,0
        kernel.priority_tap.return_value = True
        vision.navigation.side_effect = [(page,None) for page in ('bounty_room','bounty_leave','bounty_daily')]
        vision.observe.side_effect = [Mock(page='bounty_room',grade='B'),Mock(page='bounty_leave'),Mock(page='bounty_daily')]
        vision.current_action.return_value = True
        vision.match.return_value = True
        combat_vision.detect.return_value = BountyBattleState()
        ocr.read.return_value = {'words':[]}
        flow.step.return_value = DailyDecision('tap',(810,525),'确认离开等待中的悬赏队伍')
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for name,value in (('project_root',lambda:Path(directory)),('prepare_game',Mock(return_value=False)),
                               ('MuMu',Mock(return_value=device)),('create_native_touch',Mock(return_value=native)),
                               ('DailyBountyVision',Mock(return_value=vision)),('BountyCombatVision',Mock(return_value=combat_vision)),
                               ('WindowsOcr',Mock(return_value=ocr)),('DailyBountyFlow',Mock(return_value=flow)),
                               ('RealtimeBattleKernel',Mock(return_value=kernel)),('BountyCombat',Mock()),('FrameDiagnostics',Mock())):
                stack.enter_context(patch.object(daily_runner,name,value))
            stack.enter_context(patch.object(control.stop,'wait',return_value=False))
            stack.enter_context(patch.object(daily_runner.time,'perf_counter',side_effect=count(step=1).__next__))
            outcome = daily_runner.run_daily_bounty('folder',1,control,Mock(),grade='A',invite_only=True,invite_resume=True)
        self.assertIs(outcome,TaskOutcome.WAITING)
        self.assertEqual([call.args[0] for call in kernel.priority_tap.call_args_list],[(65,50),(810,525)])


if __name__ == '__main__':
    unittest.main()
