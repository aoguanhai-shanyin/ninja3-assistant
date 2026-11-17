"""Daily runner recovery with simulated frames; no game or emulator input."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ninja3_assistant.activities.daily import runner
from ninja3_assistant.activities.daily.flow import DailyDecision
from ninja3_assistant.activities.daily.vision import DailyScreen
from ninja3_assistant.runtime.runner import RunControl
from ninja3_assistant.scheduling.task_outcome import TaskOutcome


class DailyRunnerTest(unittest.TestCase):
    def run_simulation(self, steps):
        ticks = [0.]
        control = RunControl(clock=lambda:ticks[0])
        device,native,vision,ocr,flow = (Mock() for _ in range(5))
        frame = np.zeros((720,1280,3),np.uint8)
        current = [None]
        reads = [0]
        pending = iter(steps)

        def capture():
            current[0] = next(pending)
            reads[0] = 0
            ticks[0] += current[0].get('seconds',1)
            if current[0].get('pause_seconds'):
                control.pause()
                ticks[0] += current[0]['pause_seconds']
                control.resume()
            if current[0].get('stop'):
                control.request_stop()
            return frame

        def read(*args,**kwargs):
            reads[0] += 1
            if reads[0] == current[0].get('stop_at_read'):
                control.request_stop()
            if reads[0] == current[0].get('pause_at_read'):
                control.pause()
                control.resume()
            if current[0].get('pause_stop'):
                control.pause()
                control.request_stop()
            return {'words':[],'page':current[0]['page'],'text':str(control.now())}

        device.capture.side_effect = capture
        ocr.read.side_effect = read
        vision.match.side_effect = lambda frame,name,*args,**kwargs: (
            (1,1) if name == current[0]['page'] and name in ('trial','house') else None)
        vision.food.return_value = 10
        vision.observe.side_effect = lambda frame,result:DailyScreen(frame,result,result['page'])
        flow.step.side_effect = lambda *args:current[0].get('decision',DailyDecision(status='等待页面'))
        events = []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            for name,value in (('project_root',lambda:root),
                               ('resource_path',lambda *args:root/'resources.json'),
                               ('prepare_game',Mock(return_value=False)),
                               ('MuMu',Mock(return_value=device)),
                               ('create_native_touch',Mock(return_value=native)),
                               ('DailyVision',Mock(return_value=vision)),
                               ('WindowsOcr',Mock(return_value=ocr)),
                               ('DailyFlow',Mock(return_value=flow))):
                stack.enter_context(patch.object(runner,name,value))
            stack.enter_context(patch.object(runner.cv2,'imwrite',return_value=True))
            stack.enter_context(patch.object(control.stop,'wait',side_effect=lambda *args:control.stop.is_set()))
            outcome = runner.run_daily('monthly','folder',1,control,
                lambda kind,value:events.append((kind,value)))
            path = next(root.rglob('events.jsonl'))
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        native.close.assert_called_once()
        device.close.assert_called_once()
        self.assertIsNone(control._release_input)
        self.assertIn(('finished',''),events)
        return outcome,records,events,device,native,vision,flow,control

    def test_continuous_unknown_yields_with_periodic_and_final_evidence(self):
        outcome,records,events,device,native,_,flow,_ = self.run_simulation(
            [{'page':'unknown'}]*20)
        self.assertIs(outcome,TaskOutcome.WAITING)
        self.assertEqual(device.capture.call_count,13)
        snapshots = [r for r in records if r.get('event')=='decision']
        self.assertEqual([r['ocr']['text'] for r in snapshots],['1.0','6.0','11.0','13.0'])
        self.assertIn('稍后重试',events[-2][1])
        native.priority_tap.assert_not_called()
        device.reopen_game.assert_not_called()
        flow.completed.assert_not_called()

    def test_short_unknown_recovers_without_yielding(self):
        outcome,_,_,device,native,_,_,_ = self.run_simulation([
            {'page':'unknown'},
            {'page':'monthly','decision':DailyDecision('done',status='已领取')},
        ])
        self.assertIs(outcome,TaskOutcome.DAILY_DONE)
        self.assertEqual(device.capture.call_count,2)
        native.priority_tap.assert_not_called()

    def test_known_screen_resets_the_continuous_unknown_budget(self):
        outcome,*_ = self.run_simulation([
            {'page':'unknown'}, {'page':'unknown','seconds':10},
            {'page':'home'},
            {'page':'unknown'}, {'page':'unknown','seconds':10},
            {'page':'monthly','decision':DailyDecision('done',status='已领取')},
        ])
        self.assertIs(outcome,TaskOutcome.DAILY_DONE)

    def test_recognized_home_missing_its_entry_yields_instead_of_waiting_four_minutes(self):
        outcome,records,_,device,native,_,flow,_ = self.run_simulation([
            {'page':'home','decision':DailyDecision(status='寻找月签到入口')},
        ]*20)
        self.assertIs(outcome,TaskOutcome.WAITING)
        self.assertEqual(device.capture.call_count,13)
        self.assertEqual(records[-1]['page'],'home')
        native.priority_tap.assert_not_called()
        flow.completed.assert_not_called()

    def test_changed_wait_reason_resets_the_no_progress_budget(self):
        outcome,*_ = self.run_simulation([
            {'page':'home','decision':DailyDecision(status='寻找入口')},
            {'page':'home','seconds':10,'decision':DailyDecision(status='寻找入口')},
            {'page':'home','decision':DailyDecision(status='等待入口展开')},
            {'page':'home','seconds':10,'decision':DailyDecision(status='等待入口展开')},
            {'page':'monthly','decision':DailyDecision('done',status='已领取')},
        ])
        self.assertIs(outcome,TaskOutcome.DAILY_DONE)

    def test_paused_wall_time_does_not_consume_unknown_budget(self):
        outcome,_,_,_,native,_,_,control = self.run_simulation([
            {'page':'unknown'}, {'page':'unknown','pause_seconds':120},
            {'page':'monthly','decision':DailyDecision('done',status='已领取')},
        ])
        self.assertIs(outcome,TaskOutcome.DAILY_DONE)
        self.assertEqual(control.now(),3)
        native.stop_all.assert_called_once()
        native.priority_tap.assert_not_called()

    def test_stop_after_capture_does_not_dispatch_or_claim_completion(self):
        outcome,_,_,_,native,vision,flow,_ = self.run_simulation([
            {'page':'monthly','stop':True,'decision':DailyDecision('done',status='已领取')},
        ])
        self.assertIs(outcome,TaskOutcome.STOPPED)
        native.stop_all.assert_called_once()
        native.priority_tap.assert_not_called()
        vision.observe.assert_not_called()
        flow.completed.assert_not_called()

    def test_stop_while_paused_during_ocr_releases_input_and_does_not_act(self):
        outcome,_,_,_,native,vision,_,_ = self.run_simulation([
            {'page':'home','pause_stop':True,
             'decision':DailyDecision('tap',(50,280),'进入每日清理项目')},
        ])
        self.assertIs(outcome,TaskOutcome.STOPPED)
        self.assertEqual(native.stop_all.call_count,2)
        native.priority_tap.assert_not_called()
        vision.observe.assert_not_called()

    def test_stop_during_secondary_counter_or_clock_ocr_does_not_claim_completion(self):
        for page in ('trial','house'):
            with self.subTest(page=page):
                outcome,records,_,_,native,vision,flow,_ = self.run_simulation([
                    {'page':page,'stop_at_read':2,
                     'decision':DailyDecision('done',status='已领取')},
                ])
                self.assertIs(outcome,TaskOutcome.STOPPED)
                self.assertFalse(records)
                vision.observe.assert_not_called()
                flow.step.assert_not_called()
                native.priority_tap.assert_not_called()

    def test_pause_and_resume_during_ocr_discards_old_frame_before_completion(self):
        for page,read_at in (('monthly',1),('trial',2),('house',2)):
            with self.subTest(page=page,read_at=read_at):
                outcome,records,_,device,native,vision,flow,_ = self.run_simulation([
                    {'page':page,'pause_at_read':read_at,
                     'decision':DailyDecision('done',status='旧帧已领取')},
                    {'page':'monthly','decision':DailyDecision('done',status='新帧已领取')},
                ])
                self.assertIs(outcome,TaskOutcome.DAILY_DONE)
                self.assertEqual(device.capture.call_count,2)
                self.assertEqual([r['status'] for r in records],['新帧已领取'])
                vision.observe.assert_called_once()
                flow.step.assert_called_once()
                native.priority_tap.assert_not_called()


if __name__ == '__main__':
    unittest.main()
