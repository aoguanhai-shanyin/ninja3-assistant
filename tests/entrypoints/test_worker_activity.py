import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import ninja3_assistant.entrypoints.worker as worker


class WorkerActivityTest(unittest.TestCase):
    def test_daily_chores_share_the_same_task_planner(self):
        with patch.object(sys,'argv',['worker.py','--mode','run','--folder','selected','--index','1',
                         '--daily-tasks','monthly','house','village']),patch.object(sys,'stdout',io.StringIO()),\
             patch('ninja3_assistant.entrypoints.worker.threading.Thread'),\
             patch('ninja3_assistant.scheduling.task_scheduler.run_tasks') as run:
            worker.main()
        self.assertEqual([t.name for t in run.call_args.kwargs['tasks']],
                         ['daily_monthly','daily_house','daily_village'])

    def test_window_flag_reaches_task_control_and_defaults_to_visible(self):
        for extra,expected in (([],False),(['--hide-game-window'],True)):
            with self.subTest(hidden=expected),patch.object(sys,'argv',[
                 'worker.py','--mode','run','--folder','selected','--index','1',*extra]), \
                 patch.object(sys,'stdout',io.StringIO()),patch('ninja3_assistant.entrypoints.worker.threading.Thread'),patch('ninja3_assistant.entrypoints.worker.run_flow') as run:
                worker.main()
            self.assertIs(run.call_args.args[2].hide_game_window,expected)

    def test_enabled_activity_dispatches_both_stop_modes(self):
        for mode in ('daily', 'continuous'):
            with self.subTest(mode=mode):
                activity = Mock()
                with patch.object(sys, 'argv', ['worker.py', '--mode', 'run', '--folder', 'selected',
                                               '--index', '1', '--activity-enabled', '--activity-mode', mode]), \
                        patch.object(sys, 'stdout', io.StringIO()), \
                        patch('ninja3_assistant.entrypoints.worker.threading.Thread'), patch('ninja3_assistant.entrypoints.worker.run_flow') as flow, \
                        patch.dict(sys.modules, {'ninja3_assistant.scheduling.task_scheduler': SimpleNamespace(run_tasks=activity)}):
                    worker.main()
                activity.assert_called_once()
                self.assertEqual(activity.call_args.args[:2], ('selected', 1))
                tasks = activity.call_args.kwargs['tasks']
                self.assertEqual([(t.name, t.extra_mode) for t in tasks],
                                 [('intelligence', 'continuous' if mode == 'continuous' else None)])
                flow.assert_not_called()

    def test_activity_is_disabled_by_default_and_preserves_existing_run(self):
        with patch.object(sys, 'argv', ['worker.py', '--mode', 'run', '--folder', 'selected', '--index', '1',
                                       '--join-interval-ms', '25']), \
                patch.object(sys, 'stdout', io.StringIO()), patch('ninja3_assistant.entrypoints.worker.threading.Thread'), \
                patch('ninja3_assistant.entrypoints.worker.run_flow') as flow, patch.dict(sys.modules, {'ninja3_assistant.activities.intelligence.runner': None}):
            worker.main()
        flow.assert_called_once()
        self.assertEqual(flow.call_args.kwargs, {'join_interval_ms': 25})

    def test_enabled_activity_defaults_to_daily(self):
        activity = Mock()
        with patch.object(sys, 'argv', ['worker.py', '--mode', 'run', '--folder', 'selected',
                                       '--index', '1', '--activity-enabled']), \
                patch.object(sys, 'stdout', io.StringIO()), patch('ninja3_assistant.entrypoints.worker.threading.Thread'), \
                patch.dict(sys.modules, {'ninja3_assistant.scheduling.task_scheduler': SimpleNamespace(run_tasks=activity)}):
            worker.main()
        self.assertEqual([(t.name, t.extra_mode) for t in activity.call_args.kwargs['tasks']],
                         [('intelligence', None)])

    def test_multiple_activities_reach_one_planner(self):
        with patch.object(sys, 'argv', ['worker.py', '--mode', 'run', '--folder', 'selected',
                         '--index', '1', '--activity-enabled', '--activity-mode', 'continuous',
                         '--weekend-enabled', '--weekend-mode', '25']), \
                patch.object(sys, 'stdout', io.StringIO()), patch('ninja3_assistant.entrypoints.worker.threading.Thread'), \
                patch('ninja3_assistant.scheduling.task_scheduler.run_tasks') as planner, patch('ninja3_assistant.entrypoints.worker.run_flow') as flow:
            worker.main()
        self.assertEqual([(t.name, t.extra_mode) for t in planner.call_args.kwargs['tasks']],
                         [('intelligence', 'continuous'), ('weekend', '25')])
        flow.assert_not_called()

    def test_other_modes_reject_activity_before_creating_a_command_thread(self):
        for mode in ('probe', 'hayabusa', 'hayabusa-debug', 'hayabusa-video'):
            with self.subTest(mode=mode), \
                    patch.object(sys, 'argv', ['worker.py', '--mode', mode, '--folder', 'f', '--index', '1',
                                               '--video', 'v.mp4', '--activity-enabled']
                                 if mode == 'hayabusa-video' else
                                 ['worker.py', '--mode', mode, '--folder', 'f', '--index', '1', '--activity-enabled']), \
                    patch.object(sys, 'stderr', io.StringIO()), \
                    patch('ninja3_assistant.entrypoints.worker.threading.Thread') as thread, self.assertRaises(SystemExit) as error:
                worker.main()
            self.assertEqual(error.exception.code, 2)
            thread.assert_not_called()


if __name__ == '__main__':
    unittest.main()
