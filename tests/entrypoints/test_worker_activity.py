import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import ninja3_assistant.entrypoints.worker as worker


class WorkerActivityTest(unittest.TestCase):
    def test_bounty_uses_exact_selected_grade_and_s_is_a_separate_daily(self):
        for grade in ('D','C','B','A'):
            with self.subTest(grade=grade),patch.object(sys,'argv',[
                 'worker.py','--mode','run','--folder','selected','--index','1',
                 '--daily-tasks','bounty','bounty_s','--bounty-grade',grade]),\
                 patch.object(sys,'stdout',io.StringIO()),\
                 patch('ninja3_assistant.entrypoints.worker.threading.Thread'),\
                 patch('ninja3_assistant.scheduling.task_scheduler.run_tasks') as run:
                worker.main()
            tasks=run.call_args.kwargs['tasks']
            self.assertEqual([t.name for t in tasks],['daily_bounty','daily_bounty_s'])
            self.assertEqual(tasks[0].bounty_grade,grade)
            self.assertIn(grade+'级',tasks[0].label)
            self.assertTrue(all(t.extra_mode is None for t in tasks))

    def test_bounty_grade_cannot_be_silently_ignored_without_the_task(self):
        with patch.object(sys,'argv',['worker.py','--mode','run','--folder','f','--index','1',
                         '--bounty-grade','B']),patch.object(sys,'stderr',io.StringIO()),\
             patch('ninja3_assistant.entrypoints.worker.threading.Thread') as thread,\
             self.assertRaises(SystemExit) as error:
            worker.main()
        self.assertEqual(error.exception.code,2)
        thread.assert_not_called()

    def test_bounty_invitation_switches_reach_both_bounty_runners_independently(self):
        active_flag = '--no-bounty-active-invite'
        passive_flag = '--no-bounty-passive-invite'
        for flags in ([],[active_flag],[passive_flag],[active_flag,passive_flag]):
            with self.subTest(flags=flags),patch.object(sys,'argv',[
                 'worker.py','--mode','run','--folder','selected','--index','1',
                 '--daily-tasks','bounty','bounty_s',*flags]),\
                 patch.object(sys,'stdout',io.StringIO()),\
                 patch('ninja3_assistant.entrypoints.worker.threading.Thread'),\
                 patch('ninja3_assistant.scheduling.task_scheduler.run_tasks') as run:
                worker.main()
            with patch('ninja3_assistant.activities.bounty.daily_runner.run_daily_bounty') as bounty:
                for task in run.call_args.kwargs['tasks']:
                    task.run('selected',1,Mock(),Mock(),mode='daily')
                    self.assertIs(bounty.call_args.kwargs['active_invite'],active_flag not in flags)
                    self.assertIs(bounty.call_args.kwargs['passive_invite'],passive_flag not in flags)

    def test_daily_chores_share_the_same_task_planner(self):
        with patch.object(sys,'argv',['worker.py','--mode','run','--folder','selected','--index','1',
                         '--daily-tasks','monthly','house','village','rank']),patch.object(sys,'stdout',io.StringIO()),\
             patch('ninja3_assistant.entrypoints.worker.threading.Thread'),\
             patch('ninja3_assistant.scheduling.task_scheduler.run_tasks') as run:
            worker.main()
        self.assertEqual([t.name for t in run.call_args.kwargs['tasks']],
                         ['daily_monthly','daily_house','daily_village','daily_rank'])

    def test_new_chores_and_selected_feeding_reward_reach_the_same_planner(self):
        with patch.object(sys,'argv',['worker.py','--mode','run','--folder','selected','--index','1',
                         '--daily-tasks','pet_feed','arena_reward','landlady_quiz','--pet-feed-reward','hammers']),\
             patch.object(sys,'stdout',io.StringIO()),\
             patch('ninja3_assistant.entrypoints.worker.threading.Thread'),\
             patch('ninja3_assistant.scheduling.task_scheduler.run_tasks') as run:
            worker.main()
        tasks = run.call_args.kwargs['tasks']
        self.assertEqual([task.name for task in tasks],['daily_pet_feed','daily_arena_reward','daily_landlady_quiz'])
        self.assertEqual(tasks[0].pet_feed_reward,'hammers')
        self.assertIn('强化锤',tasks[0].label)

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
