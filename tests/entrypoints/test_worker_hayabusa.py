import io
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import ninja3_assistant.entrypoints.worker as worker


class WorkerHayabusaTest(unittest.TestCase):
    def invoke(self, args, run, commands=''):
        output = io.StringIO()
        thread = Mock()
        thread.start.side_effect = lambda: self.command_loop()
        with patch.object(sys, 'argv', ['worker.py', '--folder', 'selected-folder', '--index', '7', *args]), \
                patch.object(sys, 'stdin', io.StringIO(commands)), patch.object(sys, 'stdout', output), \
                patch.dict(sys.modules, {'ninja3_assistant.combat.hayabusa.trial': SimpleNamespace(run_hayabusa=run)}), \
                patch('ninja3_assistant.entrypoints.worker.threading.Thread', side_effect=lambda **kw: self.make_thread(thread, kw)), \
                patch('ninja3_assistant.entrypoints.worker.run_flow') as flow:
            worker.main()
        return flow, [json.loads(line) for line in output.getvalue().splitlines()]

    def make_thread(self, thread, arguments):
        self.command_loop = arguments['target']
        return thread

    def test_observation_dispatches_selected_instance_without_running_bounty_flow(self):
        run = Mock()
        flow, _ = self.invoke(['--mode', 'hayabusa', '--observe-only'], run)
        flow.assert_not_called()
        run.assert_called_once()
        folder, index, _, _ = run.call_args.args
        self.assertEqual((folder, index), ('selected-folder', 7))
        self.assertEqual(run.call_args.kwargs, {'observe_only': True})

    def test_scene_debug_is_always_observation_without_a_feature_switch(self):
        for extra in ([], ['--observe-only']):
            run = Mock()
            flow, _ = self.invoke(['--mode', 'hayabusa-debug', *extra], run)
            flow.assert_not_called()
            run.assert_called_once()
            self.assertEqual(run.call_args.args[:2], ('selected-folder', 7))
            self.assertEqual(run.call_args.kwargs, {'observe_only': True})

    def test_scene_debug_supports_pause_resume_stop_without_enabling_inputs(self):
        controls = []
        run = lambda folder, index, control, emit, **kw: controls.append((control, kw))
        commands = ''.join(json.dumps({'command': command}) + '\n' for command in ('pause', 'resume', 'stop'))
        flow, events = self.invoke(['--mode', 'hayabusa-debug'], run, commands)
        flow.assert_not_called()
        control, options = controls[0]
        self.assertTrue(control.stop.is_set())
        self.assertFalse(control.paused())
        self.assertEqual(options, {'observe_only': True})
        self.assertEqual(events, [{'event': 'state', 'value': 'paused'}, {'event': 'state', 'value': 'running'}])

    def test_execution_mode_and_pause_resume_stop_share_run_control(self):
        controls = []
        run = lambda folder, index, control, emit, **kw: controls.append((control, kw))
        commands = ''.join(json.dumps({'command': command}) + '\n' for command in ('pause', 'resume', 'stop'))
        _, events = self.invoke(['--mode', 'hayabusa'], run, commands)
        control, options = controls[0]
        self.assertTrue(control.stop.is_set())
        self.assertFalse(control.paused())
        self.assertEqual(options, {'observe_only': False})
        self.assertEqual(events, [{'event': 'state', 'value': 'paused'}, {'event': 'state', 'value': 'running'}])

    def test_existing_run_retains_join_interval(self):
        run = Mock()
        flow, _ = self.invoke(['--mode', 'run', '--join-interval-ms', '25'], run)
        run.assert_not_called()
        flow.assert_called_once()
        self.assertEqual(flow.call_args.args[:2], ('selected-folder', 7))
        self.assertEqual(flow.call_args.kwargs, {'join_interval_ms': 25})

    def test_probe_does_not_load_battle_controller_or_command_thread(self):
        with patch.object(sys, 'argv', ['worker.py', '--mode', 'probe', '--folder', 'selected-folder', '--index', '7']), \
                patch.object(sys, 'stdout', io.StringIO()), patch.dict(sys.modules, {'ninja3_assistant.combat.hayabusa.trial': None}), \
                patch('ninja3_assistant.entrypoints.worker.probe') as probe, patch('ninja3_assistant.entrypoints.worker.threading.Thread') as thread:
            worker.main()
        probe.assert_called_once()
        self.assertEqual(probe.call_args.args[:2], ('selected-folder', 7))
        thread.assert_not_called()

    def test_observe_only_cannot_silently_change_bounty_mode(self):
        with patch.object(sys, 'argv', ['worker.py', '--mode', 'run', '--folder', 'f', '--index', '0', '--observe-only']), \
                patch.object(sys, 'stderr', io.StringIO()), self.assertRaises(SystemExit) as error:
            worker.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
