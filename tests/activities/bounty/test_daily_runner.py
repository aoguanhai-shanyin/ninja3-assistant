"""Runner lifecycle checks with simulated frames and native inputs only."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ninja3_assistant.activities.bounty import daily_runner
from ninja3_assistant.activities.bounty.combat import BountyBattleState
from ninja3_assistant.activities.daily.flow import DailyDecision
from ninja3_assistant.runtime.runner import RunControl
from ninja3_assistant.scheduling.task_outcome import TaskOutcome


class DailyBountyRunnerTest(unittest.TestCase):
    def run_simulation(self, states, *, page=None, ocr_page=None, decision=None, fresh=True, grade='D', cleanup_error=False, stop_on_capture=False):
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
        native.plan.return_value = 1
        if cleanup_error:
            native.close.side_effect = RuntimeError('simulated native close error')
        vision.navigation.return_value = (page,None)
        vision.match.return_value = vision.common.match.return_value = None
        vision.common.invitations.detect.return_value = (470,510) if page == 'invite' else None
        vision.observe.return_value = Mock(page=ocr_page or page or 'unknown')
        vision.current_action.return_value = fresh
        combat_vision.detect.side_effect = states
        ocr.read.return_value = {'words':[]}
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
                lambda kind,value:events.append((kind,value)),grade=grade)
        native.close.assert_called_once()
        device.close.assert_called_once()
        self.assertIsNone(control._release_input)
        self.assertIn(('finished',''),events)
        return outcome,device,native,vision,combat_vision,ocr,flow,flow_factory

    def test_confirmed_combat_never_uses_ocr_and_capture_error_retains_match(self):
        active = BountyBattleState(active=True,ground=True)
        outcome,_,native,vision,_,ocr,flow,_ = self.run_simulation([active,active])
        self.assertIs(outcome,TaskOutcome.RECOVER_IN_PLACE)
        self.assertTrue(flow.battle_seen)
        ocr.read.assert_not_called()
        native.schedule_movement.assert_called_once()
        vision.navigation.assert_not_called()

    def test_unknown_frame_during_match_only_uses_ocr_after_half_second(self):
        active = BountyBattleState(active=True,ground=True)
        unknown = BountyBattleState()
        outcome,_,_,_,_,ocr,_,_ = self.run_simulation([active,unknown,unknown,unknown])
        self.assertIs(outcome,TaskOutcome.RECOVER_IN_PLACE)
        ocr.read.assert_called_once()

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


if __name__ == '__main__':
    unittest.main()
