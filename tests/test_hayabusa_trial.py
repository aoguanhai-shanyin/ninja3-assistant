import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from boss_policy import BossAction
from hayabusa_trial import run_hayabusa
from hayabusa_vision import HayabusaScreen
from runner import RunControl


class HayabusaTrialTest(unittest.TestCase):
    def run_trial(self, *, observe=True, sequence=None, action=None, pause_on_detect=False,
                  stop_on_detect=False, late=False, static=False, stop_on_press=False,
                  fail_release=False, analysis_delay=0.):
        control = RunControl()
        events, policies = [], []
        clock = [100.]
        original_sequence = sequence or [HayabusaScreen(active=True, rope_y=430, player_x=300,
            player_y=394, side='above')]*4 + [HayabusaScreen(result='victory')]
        states = iter(original_sequence)

        class Device:
            def __init__(self, folder, index):
                self.args = (folder, index)
                self.taps = []
                self.closed = False
                self.frames = 0
            def capture(self):
                clock[0] += .04
                self.frames += 1
                return np.full((720, 1280, 3), 0 if static else self.frames, np.uint8)
            def tap(self, point, stop, hold_seconds, on_pressed=None):
                self.taps.append((point, hold_seconds))
                if on_pressed:
                    on_pressed()
                if stop_on_press:
                    stop.set()
                    return False
                if fail_release:
                    raise RuntimeError('release failed')
                return True
            def close(self):
                self.closed = True

        class Vision:
            backend = 'test GPU'
            def detect(self, frame, now):
                state = next(states)
                clock[0] += analysis_delay
                if pause_on_detect and state.active:
                    control.pause()
                    control.resume()
                if stop_on_detect and state.active:
                    control.stop.set()
                return state, []
            def reset(self, **kwargs):
                pass

        class Policy:
            last_reason = 'test'
            def __init__(self):
                self.calls, self.completed_actions, self.times = 0, [], []
                policies.append(self)
            def reset(self, **kwargs):
                pass
            def suspend(self, now):
                pass
            def decide(self, *args):
                self.calls += 1
                self.times.append(args[-1])
                return action or BossAction('down', (190, 630))
            def completed(self, chosen, now):
                self.completed_actions.append(chosen)

        device = Device('selected-folder', 3)
        with tempfile.TemporaryDirectory() as directory:
            with patch('hayabusa_trial.RECORDINGS', Path(directory)), \
                    patch('hayabusa_trial.create_native_touch', return_value=None), \
                    patch('hayabusa_scene.HayabusaVision', Vision), \
                    patch('hayabusa_trial.HayabusaPolicy', Policy), \
                    patch('opening_actions.OpeningActions._ninjutsu_point', return_value=(70, 330)), \
                    patch('hayabusa_trial.MuMu', return_value=device), \
                    patch('hayabusa_trial.time.perf_counter', side_effect=(lambda: 0.) if not late
                          else (lambda: clock[0]*10)), \
                    patch.object(control, 'now', side_effect=lambda: clock[0]):
                run_hayabusa('selected-folder', 3, control, lambda k, v: events.append((k, v)),
                             observe_only=observe)
            records = [json.loads(line) for file in Path(directory).glob('*.jsonl')
                       for line in file.read_text(encoding='utf-8').splitlines()]
        self.assertTrue(device.closed)
        self.assertEqual(events[-1][0], 'finished')
        return device, events, records, policies

    def test_observation_never_sends_touch_but_records_suggestions(self):
        device, events, records, policies = self.run_trial()
        self.assertEqual(device.taps, [])
        self.assertGreater(records[-1]['suggestions'], 0)
        self.assertEqual(records[-1]['clicks'], 0)
        self.assertTrue(all(not item.get('sent') for item in records))
        frame = next(item for item in records if item.get('suggested_action'))
        self.assertIsNotNone(frame['timing_ms']['capture'])
        self.assertIsNotNone(frame['timing_ms']['vision'])
        self.assertIsNotNone(frame['timing_ms']['policy'])
        self.assertIsNone(frame['timing_ms']['input'])
        self.assertEqual(records[-1]['timing_ms']['input']['count'], 0)

    def test_live_actions_only_after_three_confirmed_frames(self):
        device, events, records, _ = self.run_trial(observe=False)
        frame_records = [item for item in records if 'screen' in item]
        self.assertFalse(frame_records[0]['sent'])
        self.assertFalse(frame_records[1]['sent'])
        self.assertTrue(frame_records[2]['sent'])
        self.assertEqual(device.taps, [((70, 330), .015)] + [((190, 630), .025)]*2)
        skill = [value for kind, value in events if kind == 'opening_action' and value['kind'] == 'ninjutsu']
        self.assertEqual(len(skill), 1)
        self.assertEqual(skill[0]['point'], (70, 330))
        self.assertIsNotNone(frame_records[2]['timing_ms']['input'])
        self.assertEqual(records[-1]['timing_ms']['input']['count'], 2)
        self.assertIn('Python SDK includes hold/release', records[-1]['timing_scope'])
        self.assertIn('C++ ends at press acknowledgement or cancellation completion', records[-1]['timing_scope'])
        self.assertIn('neither measures visible game response', records[-1]['timing_scope'])

    def test_unknown_and_result_screens_never_click(self):
        device, _, _, policies = self.run_trial(observe=False,
            sequence=[HayabusaScreen()]*3 + [HayabusaScreen(result='victory')])
        self.assertEqual(device.taps, [])
        self.assertEqual(policies[0].calls, 0)

    def test_pause_and_resume_during_capture_invalidates_the_frame(self):
        device, _, _, policies = self.run_trial(observe=False, pause_on_detect=True)
        self.assertEqual(device.taps, [])
        self.assertEqual(policies[0].calls, 0)

    def test_stop_during_capture_never_sends_action(self):
        device, _, _, policies = self.run_trial(observe=False, stop_on_detect=True)
        self.assertEqual(device.taps, [])
        self.assertEqual(policies[0].calls, 0)

    def test_slow_analysis_is_recorded_but_not_executed(self):
        device, _, records, _ = self.run_trial(observe=False, late=True)
        self.assertEqual(device.taps, [])
        self.assertTrue(any(item.get('frame_age_ms', 0) > 200 for item in records))

    def test_stalled_battle_stops_sending_suggested_actions(self):
        state = HayabusaScreen(active=True, rope_y=430, player_x=300, player_y=394, side='above')
        _, events, records, _ = self.run_trial(observe=False, static=True,
            sequence=[state]*12+[HayabusaScreen(result='victory')])
        frozen = [item for item in records if item.get('stalled')]
        self.assertTrue(frozen)
        self.assertTrue(all(not item['sent'] for item in frozen))
        skills = [value for kind, value in events if kind == 'opening_action' and value['kind'] == 'ninjutsu']
        self.assertTrue(skills)
        self.assertTrue(all(item['time'] < frozen[0]['capture_time'] for item in skills))

    def test_initialization_failure_does_not_advertise_nonexistent_log(self):
        events = []
        with tempfile.TemporaryDirectory() as directory, \
                patch('hayabusa_trial.RECORDINGS', Path(directory)), \
                patch('hayabusa_scene.HayabusaVision', side_effect=RuntimeError('GPU unavailable')):
            run_hayabusa('folder', 0, RunControl(), lambda k, v: events.append((k, v)))
        self.assertEqual(events[-1][0], 'finished')
        statuses = [value for kind, value in events if kind == 'status']
        self.assertTrue(any('GPU unavailable' in value for value in statuses))
        self.assertIn(('error', 'GPU unavailable'), events)
        self.assertFalse(any('实验记录' in value for value in statuses))

    def test_inference_latency_is_included_in_decision_time(self):
        _, _, records, policies = self.run_trial(analysis_delay=.1)
        frame = next(item for item in records if item.get('suggested_action'))
        self.assertAlmostEqual(policies[0].times[0] - frame['capture_time'], .1)

    def test_stop_during_hold_still_records_the_sent_press(self):
        # Exercise the scene action rather than the earlier periodic skill tap.
        with patch('hayabusa_trial.OpeningActions.decide', return_value=None):
            device, _, records, _ = self.run_trial(observe=False, stop_on_press=True)
        self.assertEqual(len(device.taps), 1)
        self.assertEqual(records[-1]['clicks'], 1)
        self.assertTrue(next(item for item in records if item.get('suggested_action'))['sent'])

    def test_failed_release_is_logged_before_the_trial_ends(self):
        # Exercise release failure on a scene action; opening touches have their
        # own event path and can otherwise terminate this short synthetic trial.
        with patch('hayabusa_trial.OpeningActions.decide', return_value=None):
            _, events, records, _ = self.run_trial(observe=False, fail_release=True)
        frame = next(item for item in records if item.get('suggested_action'))
        self.assertTrue(frame['sent'])
        self.assertEqual(frame['touch_error'], 'release failed')
        self.assertIsNotNone(frame['timing_ms']['input'])
        self.assertTrue(any('release failed' in value for kind, value in events if kind == 'status'))


if __name__ == '__main__':
    unittest.main()
