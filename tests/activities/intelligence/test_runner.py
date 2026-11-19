from ninja3_assistant.paths import project_root
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2

from ninja3_assistant.activities.intelligence.flow import ActivityDecision, ActivityScreen
from ninja3_assistant.activities.intelligence.battle_vision import BattleObservation, BattleVision
from ninja3_assistant.activities.intelligence.tasks import TaskObservation
from ninja3_assistant.activities.intelligence.runner import _coin_digit_changes, run_activity
from ninja3_assistant.runtime.runner import RunControl
from ninja3_assistant.scheduling.invitation_handoff import InvitationHandoff
from ninja3_assistant.scheduling.task_outcome import TaskOutcome


class ActivityRig:
    def __init__(self, screens=None):
        self.control = RunControl()
        self.frames = [np.full((720, 1280, 3), index, np.uint8)
                       for index in range(len(screens or [ActivityScreen('event_home')]))]
        for frame in self.frames:
            frame[164:188, 1195:1221] = 255
            frame[168:184, 1198:1205] = 30
        self.screens = dict(zip(map(id, self.frames), screens or [ActivityScreen('event_home')]))
        self.captured = 0
        self.events = []
        self.records = []
        self.capture_hook = self.detect_hook = self.emit_hook = None
        self.write_image = Mock(return_value=True)
        self.device = Mock()
        self.device.capture.side_effect = self.capture
        self.native = Mock()
        self.native.priority_tap.side_effect = lambda _, *, captured_at, **_kwargs: (captured_at, captured_at)
        self.swipe = Mock()
        self.vision = Mock(backend='test')
        self.vision.detect.side_effect = self.detect
        self.vision.battle.portrait_warmup_ms = None
        self.vision.battle.numbers_image.side_effect = lambda frame: frame[:20, :20]
        self.vision.battle.with_numbers.side_effect = BattleVision.with_numbers
        self.ocr = Mock()
        self.ocr.read.return_value = {'words': []}
        self.vision.tasks.detect.side_effect = self.detect_task
        self.vision.tasks.read.side_effect = self.read_task
        self.vision.tasks.ocr_region.return_value = (800, 160, 290, 181)
        self.vision.task_screen.side_effect = lambda task: ActivityScreen(
            'tasks', daily=tuple(task.daily_states[key] for key in ('games', 'damage')),
            claim_points=tuple(task.claim_points[key] for key in ('games', 'damage')), task=task)
        self.flow = SimpleNamespace(checked_day='previous', step=Mock(
            return_value=ActivityDecision('tasks', (866, 51), '检查任务')), completed=Mock(),
            completed_matches=0, reset_inputs=Mock(), last_action=None, last_action_at=float('-inf'))

    def capture(self):
        if self.capture_hook:
            self.capture_hook(self.captured)
        index = self.captured
        self.captured += 1
        if index >= len(self.frames):
            self.control.stop.set()
            return np.zeros((720, 1280, 3), np.uint8)
        return self.frames[index]

    def detect(self, frame, result=None):
        if self.detect_hook:
            self.detect_hook(frame, result)
        screen = self.screens.get(id(frame), ActivityScreen())
        if result is not None and screen.kind == 'tasks':
            return ActivityScreen('tasks', daily=('claimable', 'claimed'),
                                  claim_points=((970, 200), None))
        return screen

    def emit(self, kind, value):
        self.events.append((kind, value))
        if self.emit_hook:
            self.emit_hook(kind, value)

    def detect_task(self, frame):
        if self.screens.get(id(frame), ActivityScreen()).kind != 'tasks':
            return TaskObservation()
        return TaskObservation('tasks', {'games': 'unknown', 'damage': 'unknown'},
                               present=('games', 'damage'),
                               header_bounds=((570, 119, 130, 39), (570, 347, 132, 34)))

    def read_task(self, frame, engine, stop=None):
        result = engine.read(cv2.resize(frame[160:341, 800:1090], (870, 543)), stop=stop)
        task = self.detect_task(frame)
        if result is None:
            return task
        return TaskObservation('tasks', {'games': 'claimable', 'damage': 'claimed'},
                               {'games': (970, 200), 'damage': None}, task.present,
                               task.header_bounds)

    def run(self, *, prepare=False, mode='daily', checkpoint=False):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch('ninja3_assistant.activities.intelligence.runner.project_root', return_value=Path(directory)))
            stack.enter_context(patch('ninja3_assistant.activities.intelligence.runner.prepare_game', return_value=prepare))
            stack.enter_context(patch('ninja3_assistant.activities.intelligence.runner.ActivityVision', return_value=self.vision))
            self.ocr_factory = Mock(return_value=self.ocr)
            stack.enter_context(patch.dict(sys.modules, {'ninja3_assistant.activities.intelligence.ocr': SimpleNamespace(WindowsOcr=self.ocr_factory)}))
            self.flow_factory = stack.enter_context(patch('ninja3_assistant.activities.intelligence.runner.ActivityFlow', return_value=self.flow))
            self.sdk_factory = stack.enter_context(patch('ninja3_assistant.activities.intelligence.runner.MuMu', return_value=self.device))
            self.native_factory = stack.enter_context(patch('ninja3_assistant.activities.intelligence.runner.create_native_touch', return_value=self.native))
            stack.enter_context(patch('ninja3_assistant.activities.intelligence.runner.subprocess.run', self.swipe))
            self.manager = stack.enter_context(patch('ninja3_assistant.device.mumu.set_window_visibility'))
            stack.enter_context(patch('ninja3_assistant.activities.intelligence.runner.cv2.imwrite', self.write_image))
            stack.enter_context(patch.object(self.control.stop, 'wait',
                                            side_effect=lambda *_: self.control.stop.is_set()))
            self.outcome = run_activity('selected-folder', 1, self.control, self.emit,
                                        mode=mode, checkpoint=checkpoint)
            self.records = [json.loads(line) for path in Path(directory).rglob('events.jsonl')
                            for line in path.read_text(encoding='utf-8').splitlines()]


class ActivityRunnerTest(unittest.TestCase):
    def test_hall_maintenance_toast_yields_without_another_touch(self):
        rig = ActivityRig([ActivityScreen('hall', (373, 389))])
        rig.flow.last_action = 'hall'
        rig.flow.last_action_at = rig.control.now()
        rig.ocr.read.return_value = {'text': '当前服务器维护中，该括动暂时关闭，请维护结束后尝试'}
        rig.run(checkpoint=True)
        self.assertIs(rig.outcome, TaskOutcome.MAINTENANCE)
        self.assertEqual(rig.records[0]['event'], 'maintenance')
        rig.native.priority_tap.assert_not_called()
        rig.flow.step.assert_not_called()

    def test_unknown_card_samples_and_selection_reason_are_written_to_run_log(self):
        observation = BattleObservation('placement',board=(None,)*9,shop=(None,)*5,
            round_number=1,coins=5,unknown_shop=(2,),
            card_candidates={'shop:2':[{'card':'c036','score':.81}]})
        rig = ActivityRig([ActivityScreen('battle',battle=observation)]*2)
        rig.flow.step.return_value = ActivityDecision('prepare',(1118,654),'保留金币',
            selection={'budget':5,'plan':[],'estimated_gain':0})
        rig.run()
        samples = [r for r in rig.records if r.get('event') == 'unknown_cards']
        self.assertEqual(len(samples),1)
        self.assertEqual(samples[0]['slots'][0]['candidates'][0]['card'],'c036')
        actions = [r for r in rig.records if r.get('action') == 'prepare']
        self.assertEqual(actions[0]['selection']['budget'],5)

    def test_transient_unknown_shop_is_not_recorded_or_used_as_a_stable_state(self):
        transient = BattleObservation('placement',board=(None,)*9,shop=(None,)*5,
            round_number=1,coins=5,unknown_shop=(2,))
        settled = BattleObservation('placement',board=(None,)*9,shop=(None,)*5,
            round_number=1,coins=5)
        rig = ActivityRig([ActivityScreen('battle',battle=transient),
                           ActivityScreen('battle',battle=settled)])
        rig.run()
        self.assertFalse([r for r in rig.records if r.get('event') == 'unknown_cards'])
        self.assertFalse(rig.flow.step.called)

    def test_exception_during_match_retains_task_ownership_through_unknown_or_invite(self):
        for screen in (ActivityScreen(), ActivityScreen('invite', (10,10))):
            rig = ActivityRig([ActivityScreen('battle', battle=BattleObservation('combat')), screen])
            rig.flow.step.side_effect = [ActivityDecision(), RuntimeError('recognition failed')]
            rig.run(checkpoint=True)
            self.assertIs(rig.outcome, TaskOutcome.RECOVER_IN_PLACE)
            self.assert_finished(rig)

    def test_decline_uses_fresh_frame_after_slow_invite_ocr(self):
        point = (471, 526)
        invite = ActivityScreen('invite', point)
        rig = ActivityRig([ActivityScreen('event_home'), invite, invite])
        rig.frames[2][:] = rig.frames[1]
        rig.frames[2][300, 200] = 255  # Animation outside the sender and title.
        rig.control.bounty_invite_targets = {'A': 'daily_bounty'}
        rig.vision.navigation.invitations.bounty_invitation.return_value = None
        rig.flow.step.side_effect = [ActivityDecision(), ActivityDecision('invite', point)]
        clock = [100.0]

        def read_invite(*_args, **_kwargs):
            clock[0] += 1.8
            return {'words': []}

        rig.ocr.read.side_effect = read_invite
        with patch('ninja3_assistant.activities.intelligence.runner.time.perf_counter',
                   side_effect=lambda: clock[0]):
            rig.run()
        self.assert_priority_tap(rig, point)
        rig.ocr.read.assert_called_once()

    def test_replaced_invitation_is_rechecked_before_any_decline(self):
        point = (471, 526)
        invite = ActivityScreen('invite', point)
        rig = ActivityRig([ActivityScreen('event_home'), invite, invite, invite])
        rig.frames[2][:] = rig.frames[1]
        rig.frames[2][220:224, 200:220] = 255  # New sender, same button.
        rig.frames[3][:] = rig.frames[2]
        rig.control.bounty_invite_targets = {'A': 'daily_bounty'}
        rig.vision.navigation.invitations.bounty_invitation.side_effect = [None, SimpleNamespace(grade='A')]
        rig.flow.step.side_effect = [ActivityDecision(), ActivityDecision('invite', point)]

        rig.run()

        self.assertEqual(rig.outcome, InvitationHandoff('daily_bounty'))
        self.assertEqual(rig.ocr.read.call_count, 2)
        rig.native.priority_tap.assert_not_called()

    def test_error_after_confirmed_return_home_can_yield_to_other_tasks(self):
        rig = ActivityRig([ActivityScreen('battle', battle=BattleObservation('combat')), ActivityScreen('event_home')])
        rig.flow.step.side_effect = [ActivityDecision(), RuntimeError('failed')]
        rig.run(checkpoint=True)
        self.assertIs(rig.outcome, TaskOutcome.ERROR)

    def test_match_start_touch_retains_ownership_if_next_capture_fails(self):
        rig = ActivityRig()
        rig.flow.step.return_value = ActivityDecision('match', (640,563))
        def fail_after_first(index):
            if index:
                raise RuntimeError('capture failed')
        rig.capture_hook = fail_after_first
        rig.run(checkpoint=True)
        self.assertIs(rig.outcome, TaskOutcome.RECOVER_IN_PLACE)

    def test_checkpoint_returns_explicit_daily_and_error_outcomes(self):
        for action, expected in (('done', TaskOutcome.DAILY_DONE), ('stop', TaskOutcome.ERROR)):
            rig = ActivityRig()
            rig.flow.step.return_value = ActivityDecision(action)
            rig.run(checkpoint=True)
            self.assertIs(rig.outcome, expected)
            rig.native.priority_tap.assert_not_called()

    def test_checkpoint_yields_after_confirmed_match_without_new_input(self):
        for mode in ('daily', 'continuous'):
            rig = ActivityRig()
            rig.flow.completed_matches = 1
            rig.run(mode=mode, checkpoint=True)
            self.assertIs(rig.outcome, TaskOutcome.MATCH_DONE)
            rig.native.priority_tap.assert_not_called()
            self.assertFalse(rig.control.stop.is_set())
            self.assert_finished(rig)

    def test_only_scheduled_continuous_mode_reuses_confirmed_daily_status(self):
        for mode, checkpoint in (('daily', True), ('continuous', False), ('continuous', True)):
            with self.subTest(mode=mode, checkpoint=checkpoint):
                rig = ActivityRig()
                rig.flow.checked_day = None
                rig.flow.daily_complete = False
                rig.run(mode=mode, checkpoint=checkpoint)
                confirmed = mode == 'continuous' and checkpoint
                self.assertEqual(rig.flow.checked_day is not None, confirmed)
                self.assertEqual(rig.flow.daily_complete, confirmed)

    def assert_finished(self, rig, *, connected=True):
        if connected:
            rig.device.close.assert_called_once()
            rig.native.close.assert_called_once()
            rig.ocr.close.assert_called_once()
        rig.device.tap.assert_not_called()
        rig.device.swipe.assert_not_called()
        self.assertEqual(rig.events[-1], ('finished', ''))

    def assert_priority_tap(self, rig, point):
        rig.native.priority_tap.assert_called_once()
        args, kwargs = rig.native.priority_tap.call_args
        self.assertEqual(args, (point,))
        self.assertEqual(kwargs['deadline'], kwargs['captured_at']+.5)
        self.assertEqual(kwargs['hold_seconds'], .07)
        self.assertEqual(kwargs['wait_seconds'], .2)

    def test_selected_instance_mode_and_successful_touch_are_forwarded(self):
        rig = ActivityRig()
        rig.run(mode='continuous')
        rig.flow_factory.assert_called_once_with('continuous')
        rig.ocr_factory.assert_called_once_with(persistent=True)
        rig.sdk_factory.assert_called_once_with('selected-folder', 1)
        rig.native_factory.assert_called_once_with('selected-folder', 1, rig.control.stop, rig.emit)
        self.assert_priority_tap(rig, (866, 51))
        self.assertEqual(rig.flow.completed.call_count, 1)
        rig.manager.assert_not_called()
        self.assert_finished(rig)

    def test_failed_touch_is_not_completed(self):
        rig = ActivityRig()
        rig.native.priority_tap.side_effect = None
        rig.native.priority_tap.return_value = None
        rig.run()
        rig.native.priority_tap.assert_called_once()
        rig.flow.completed.assert_not_called()
        self.assert_finished(rig)

    def test_daily_list_scan_uses_swipe_and_can_be_stopped_before_input(self):
        for stopped in (False, True):
            rig = ActivityRig()
            rig.swipe.side_effect = lambda *args, **kwargs: rig.native.stop_all.assert_called_once()
            def decide(*_):
                if stopped:
                    rig.control.stop.set()
                return ActivityDecision('scroll_tasks', (760, 315), '检查第三项日常', (760, 177))
            rig.flow.step.side_effect = decide
            rig.run()
            rig.native.priority_tap.assert_not_called()
            if stopped:
                rig.swipe.assert_not_called()
                rig.native.stop_all.assert_not_called()
            else:
                rig.swipe.assert_called_once()
                self.assertEqual(rig.swipe.call_args.args[0][1:], [
                    '--folder', 'selected-folder', '--index', '1', '--swipe', '760', '315', '760', '177'])
                self.assertTrue(rig.swipe.call_args.kwargs['check'])
                rig.native.stop_all.assert_called_once()

    def test_swipe_rechecks_deadline_after_releasing_native_taps(self):
        rig, clock = ActivityRig(), [100.0]
        rig.flow.step.return_value = ActivityDecision('scroll_tasks', (760, 315), end_point=(760, 177))

        def stop_all():
            clock[0] += .501

        rig.native.stop_all.side_effect = stop_all
        with patch('ninja3_assistant.activities.intelligence.runner.time.perf_counter', side_effect=lambda: clock[0]):
            rig.run()
        rig.native.stop_all.assert_called_once()
        rig.swipe.assert_not_called()
        rig.flow.completed.assert_not_called()

    def test_result_score_is_logged_after_result_page_is_rechecked(self):
        result = ActivityScreen('battle', battle=BattleObservation('result'))
        rig = ActivityRig([result, result])
        rig.ocr.read.return_value = {'words': [{'text': '当前积分：12本局+2'}]}
        rig.flow.step.return_value = ActivityDecision('done', status='结算采样')
        rig.run()
        self.assertEqual(rig.records[0]['battle']['score'], 12)
        self.assertEqual(rig.records[0]['battle']['score_gain'], 2)
        self.assertEqual(rig.vision.detect.call_count, 2)

    def test_placement_applies_ocr_to_original_observation_and_checks_a_fresh_frame(self):
        placement = ActivityScreen('battle', battle=BattleObservation('placement',
            board=(None,)*9, shop=(None,)*5, round_number=2, refreshes=1, refresh_point=(1115, 535)))
        rig = ActivityRig([placement, placement])
        rig.ocr.read.return_value = {'words': [
            {'text': '金币：17', 'bbox': [30, 30, 90, 25]},
            {'text': '剩余刷新次数：0', 'bbox': [30, 150, 180, 25]},
        ]}
        rig.flow.step.return_value = ActivityDecision('done')
        rig.run()
        observation = rig.flow.step.call_args.args[0].battle
        self.assertEqual(observation.coins, 17)
        self.assertEqual(observation.round_number, 2)
        self.assertEqual(observation.refreshes, 0)
        self.assertIsNone(observation.refresh_point)
        self.assertEqual(rig.vision.detect.call_count, 2)

    def test_stop_during_recognition_discards_the_frame(self):
        rig = ActivityRig()
        rig.detect_hook = lambda *_: rig.control.stop.set()
        rig.run()
        rig.flow.step.assert_not_called()
        rig.native.priority_tap.assert_not_called()
        self.assert_finished(rig)

    def test_stop_after_decision_prevents_touch(self):
        rig = ActivityRig()
        def decision(*_):
            rig.control.stop.set()
            return ActivityDecision('claim', (970, 200), '领取')
        rig.flow.step.side_effect = decision
        rig.run()
        rig.native.priority_tap.assert_not_called()
        rig.flow.completed.assert_not_called()
        self.assert_finished(rig)

    def test_pause_after_decision_prevents_touch(self):
        rig = ActivityRig()
        def decision(*_):
            rig.control.pause()
            return ActivityDecision('claim', (970, 200), '领取')
        rig.flow.step.side_effect = decision
        with patch.object(rig.control, 'wait_active', side_effect=[True, True, False]):
            rig.run()
        rig.native.priority_tap.assert_not_called()
        rig.flow.completed.assert_not_called()
        self.assertTrue(rig.control.paused())
        rig.native.stop_all.assert_called_once()
        self.assert_finished(rig)

    def test_resume_discards_old_generation_and_rechecks_daily_on_fresh_capture(self):
        rig = ActivityRig([ActivityScreen('event_home')]*2)
        observations = []
        def capture(index):
            observations.append((index, rig.flow.checked_day))
        def detect(frame, _):
            if frame is rig.frames[0]:
                rig.control.pause()
                rig.control.resume()
        rig.capture_hook, rig.detect_hook = capture, detect
        rig.run()
        self.assertEqual(rig.native.priority_tap.call_count, 1)
        self.assertEqual(rig.flow.step.call_count, 1)
        self.assertIn((1, None), observations)
        rig.flow.reset_inputs.assert_called_once()
        self.assert_finished(rig)

    def test_generation_change_during_status_dispatch_prevents_old_touch(self):
        rig = ActivityRig()
        def emit(kind, value):
            if (kind, value) == ('status', '检查任务'):
                rig.control.pause()
                rig.control.resume()
        rig.emit_hook = emit
        rig.run()
        rig.native.priority_tap.assert_not_called()
        rig.flow.completed.assert_not_called()
        self.assert_finished(rig)

    def test_frame_older_than_half_a_second_is_not_touched(self):
        rig, clock = ActivityRig(), [100.]
        def decision(*_):
            clock[0] += .501
            return ActivityDecision('tasks', (866, 51), '检查任务')
        rig.flow.step.side_effect = decision
        with patch('ninja3_assistant.activities.intelligence.runner.time.perf_counter', side_effect=lambda: clock[0]):
            rig.run()
        rig.native.priority_tap.assert_not_called()
        rig.flow.completed.assert_not_called()
        self.assertGreater(rig.records[0]['frame_age_ms'], 500)
        self.assertNotIn('touch_completed', rig.records[0])
        self.assert_finished(rig)

    def test_touch_precedes_slow_diagnostic_image_write(self):
        rig, clock = ActivityRig(), [100.]
        def write_image(*_):
            clock[0] += .6
            return True
        rig.write_image.side_effect = write_image
        with patch('ninja3_assistant.activities.intelligence.runner.time.perf_counter', side_effect=lambda: clock[0]):
            rig.run()
        rig.native.priority_tap.assert_called_once()
        rig.write_image.assert_called_once()
        self.assert_finished(rig)

    def test_ocr_cancellation_does_not_use_unconfirmed_task_state(self):
        rig = ActivityRig([ActivityScreen('tasks')])
        rig.ocr.read.side_effect = lambda *_args, **_kwargs: (rig.control.stop.set(), None)[1]
        rig.run()
        self.assertEqual(rig.ocr.read.call_count, 1)
        self.assertEqual(rig.ocr.read.call_args.args[0].shape, (543, 870, 3))
        self.assertIs(rig.ocr.read.call_args.kwargs['stop'], rig.control.stop)
        rig.flow.step.assert_not_called()
        rig.native.priority_tap.assert_not_called()
        self.assert_finished(rig)

    def test_ocr_result_is_discarded_when_task_page_disappears(self):
        rig = ActivityRig([ActivityScreen('tasks'), ActivityScreen('event_home')])
        rig.run()
        rig.flow.step.assert_not_called()
        rig.native.priority_tap.assert_not_called()
        self.assertEqual(rig.ocr.read.call_count, 1)
        self.assert_finished(rig)

    def test_ocr_result_is_discarded_when_task_pixels_change(self):
        rig = ActivityRig([ActivityScreen('tasks')]*2)
        rig.frames[1][:] = 255
        rig.run()
        rig.flow.step.assert_not_called()
        rig.native.priority_tap.assert_not_called()
        self.assert_finished(rig)

    def test_ocr_result_is_discarded_when_only_the_claim_button_changes(self):
        rig = ActivityRig([ActivityScreen('tasks')]*2)
        rig.frames[1][:] = 0
        rig.frames[1][180:230, 930:1030] = 255
        rig.flow.step.return_value = ActivityDecision('claim', (970, 200), '领取')
        rig.run()
        rig.flow.step.assert_not_called()
        rig.native.priority_tap.assert_not_called()
        self.assert_finished(rig)

    def test_unchanged_task_page_can_use_the_confirmed_ocr_result(self):
        rig = ActivityRig([ActivityScreen('tasks')]*2)
        rig.frames[1][:] = 0
        rig.flow.step.return_value = ActivityDecision('claim', (970, 200), '领取')
        rig.run()
        screen = rig.flow.step.call_args.args[0]
        self.assertEqual(screen.daily, ('claimable', 'claimed'))
        self.assert_priority_tap(rig, (970, 200))
        self.assert_finished(rig)

    def test_capture_error_closes_handle_and_emits_failure_then_finished(self):
        rig = ActivityRig()
        rig.device.capture.side_effect = RuntimeError('capture failed')
        rig.run()
        rig.native.priority_tap.assert_not_called()
        self.assertIn(('error', 'capture failed'), rig.events)
        self.assert_finished(rig)

    def test_cancelled_preparation_finishes_without_opening_a_capture_handle(self):
        rig = ActivityRig()
        rig.run(prepare=None)
        rig.sdk_factory.assert_not_called()
        rig.native_factory.assert_not_called()
        rig.device.capture.assert_not_called()
        self.assert_finished(rig, connected=False)

    def test_battle_ocr_is_discarded_when_placement_changes_to_combat(self):
        placement = ActivityScreen('battle', battle=BattleObservation(
            'placement', board=(None,)*9, shop=(None,)*5, round_number=1, coins=5))
        combat = ActivityScreen('battle', battle=BattleObservation('combat', round_number=1))
        rig = ActivityRig([placement, combat])
        rig.run()
        rig.flow.step.assert_not_called()
        rig.native.priority_tap.assert_not_called()
        self.assert_finished(rig)

    def test_battle_ocr_is_discarded_when_round_changes(self):
        screens = [ActivityScreen('battle', battle=BattleObservation(
            'placement', board=(None,)*9, shop=(None,)*5, round_number=number, coins=5))
            for number in (1, 2)]
        rig = ActivityRig(screens)
        rig.run()
        rig.flow.step.assert_not_called()
        rig.native.priority_tap.assert_not_called()
        self.assert_finished(rig)

    def test_battle_ocr_is_discarded_when_empty_target_becomes_unrecognized(self):
        before = ActivityScreen('battle', battle=BattleObservation(
            'placement', board=(None,)*9, shop=(None,)*5, round_number=1, coins=5))
        after = ActivityScreen('battle', battle=BattleObservation(
            'placement', board=(None,)*9, shop=(None,)*5, round_number=1, coins=5,
            unknown_board=(0,)))
        rig = ActivityRig([before, after])
        rig.frames[1][:] = 0
        rig.flow.step.return_value = ActivityDecision('place', (140, 232), '落子')
        rig.run()
        rig.flow.step.assert_not_called()
        rig.native.priority_tap.assert_not_called()
        self.assert_finished(rig)

    def test_moving_coin_icon_does_not_discard_unchanged_digits(self):
        placement = ActivityScreen('battle', battle=BattleObservation(
            'placement', board=(None,)*9, shop=(None,)*5, round_number=1, coins=5))
        rig = ActivityRig([placement]*2)
        rig.frames[1][:] = rig.frames[0]
        rig.frames[1][154:190, 1095:1128] = 255
        rig.run()
        self.assertEqual(rig.flow.step.call_count, 1)
        rig.native.priority_tap.assert_called_once()
        self.assertTrue(rig.records[0]['touch_completed'])
        self.assertTrue({'capture_ms', 'vision_ms', 'ocr_ms', 'verify_ms', 'touch_ms'} <= rig.records[0].keys())
        self.assertFalse([record for record in rig.records if record.get('event') == 'discard'])
        self.assert_finished(rig)

    def test_real_coin_change_is_rejected_even_when_old_large_roi_mean_would_accept_it(self):
        root = project_root()/'data/intelligence-simulation'
        three = cv2.cvtColor(cv2.imread(str(root/'runs/20261113-183109-143500/000007-battle-select_card.png')),
                             cv2.COLOR_BGR2RGB)
        five = cv2.cvtColor(cv2.imread(str(root/'live/20261113-181458-match-search-01-end.png')),
                            cv2.COLOR_BGR2RGB)
        other_three = cv2.cvtColor(cv2.imread(str(root/'live/20261113-181543-match1-drop-purple.png')),
                                   cv2.COLOR_BGR2RGB)
        self.assertLessEqual(_coin_digit_changes(three, other_three), 9)
        self.assertGreater(_coin_digit_changes(three, five), 9)
        self.assertLess(cv2.absdiff(three[153:191, 1095:1220], five[153:191, 1095:1220]).mean(), 3)
        placement = ActivityScreen('battle', battle=BattleObservation(
            'placement', board=(None,)*9, shop=(None,)*5, round_number=1, coins=3))
        rig = ActivityRig([placement]*2)
        rig.frames[0][:] = three
        rig.frames[1][:] = three
        rig.frames[1][164:188, 1195:1221] = five[164:188, 1195:1221]
        rig.run()
        rig.flow.step.assert_not_called()
        rig.native.priority_tap.assert_not_called()
        self.assertEqual(rig.records[0]['reason'], 'coin_digits_changed')
        self.assertGreater(rig.records[0]['digit_changes'], 9)
        self.assert_finished(rig)

    def test_blank_coin_digits_remain_unconfirmed(self):
        frame = np.full((720, 1280, 3), 255, np.uint8)
        self.assertIsNone(_coin_digit_changes(frame, frame))
        placement = ActivityScreen('battle', battle=BattleObservation(
            'placement', board=(None,)*9, shop=(None,)*5, round_number=1, coins=3))
        rig = ActivityRig([placement]*2)
        for image in rig.frames:
            image[164:188, 1195:1221] = 255
        rig.run()
        rig.native.priority_tap.assert_not_called()
        self.assertIsNone(rig.records[0]['digit_changes'])
        self.assert_finished(rig)

    def test_discard_diagnostics_are_rate_limited_but_keep_total_counts(self):
        placement = ActivityScreen('battle', battle=BattleObservation(
            'placement', board=(None,)*9, shop=(None,)*5, round_number=1, coins=3))
        rig = ActivityRig([placement]*6)
        for image in rig.frames:
            image[164:188, 1195:1221] = 255
        with patch('ninja3_assistant.activities.intelligence.runner.time.perf_counter', return_value=100.):
            rig.run()
        rejected = [record for record in rig.records if record.get('event') == 'discard']
        self.assertEqual(len(rejected), 1)
        self.assertTrue({'capture_ms', 'vision_ms', 'ocr_ms', 'verify_ms'} <= rejected[0].keys())
        self.assertEqual(rig.records[-1]['event'], 'discard_summary')
        self.assertEqual(rig.records[-1]['counts'], {'coin_digits_changed': 3})
        rig.native.priority_tap.assert_not_called()
        self.assert_finished(rig)

    def test_match_statistics_are_emitted_after_flow_confirms_return_home(self):
        rig = ActivityRig()
        def decide(*_):
            rig.flow.completed_matches = 1
            return ActivityDecision('done', status='日常完成')
        rig.flow.step.side_effect = decide
        rig.run()
        self.assertEqual([event for event in rig.events if event[0] == 'kill'], [('kill', 1)])
        rig.native.priority_tap.assert_not_called()
        self.assert_finished(rig)


if __name__ == '__main__':
    unittest.main()
