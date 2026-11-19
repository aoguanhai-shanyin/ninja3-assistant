from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import Mock, patch

from ninja3_assistant.activities.intelligence.flow import CHINA, daily_key
from ninja3_assistant.scheduling.activity_task import ActivityTask, IntelligenceTask, WeekendTask, XumiTask, selected_tasks
from ninja3_assistant.runtime.runner import RunControl
from ninja3_assistant.scheduling.task_outcome import TaskOutcome as O
from ninja3_assistant.scheduling.task_scheduler import run_tasks
from ninja3_assistant.scheduling.task_planner import TaskPlan
from ninja3_assistant.scheduling.invitation_handoff import InvitationHandoff


class FakeTask(ActivityTask):
    def __init__(self, name, outcomes, calls, extra=None, windows=((0, 24),), hook=None):
        super().__init__(name, name, extra, windows)
        self.outcomes, self.calls, self.hook = iter(outcomes), calls, hook

    def run(self, folder, index, control, emit, *, mode):
        self.calls.append((self.name, mode))
        emit('finished', '')
        if self.hook:
            self.hook(control)
        return next(self.outcomes)


class PlannerTest(unittest.TestCase):
    def test_selected_s_waits_for_confirmed_a_quota_and_new_day_resets_order(self):
        now = datetime(2026,11,19,14,tzinfo=CHINA)
        tasks = selected_tasks(daily_tasks=('bounty','bounty_s'),bounty_grade='A')
        plan = TaskPlan(tasks,now)
        regular,summons = plan.tasks
        progress = regular.activity._bounty_progress
        progress.reset_day(daily_key(now))
        progress.regular_remaining = 2
        plan.complete(regular,O.WAITING,now)
        plan.refresh(now)
        self.assertFalse(summons.ready)

        progress.regular_remaining = 0
        plan.refresh(now)
        self.assertFalse(summons.ready)
        progress.free_checked = True
        plan.refresh(now)
        self.assertTrue(summons.ready)
        self.assertFalse(regular.daily_done)

        plan.refresh(now+timedelta(hours=15))
        self.assertFalse(summons.ready)

    def test_s_alone_can_start_without_a_task(self):
        now = datetime(2026,11,19,14,tzinfo=CHINA)
        plan = TaskPlan(selected_tasks(daily_tasks=('bounty_s',),bounty_grade='A'),now)
        plan.refresh(now)
        self.assertTrue(plan.tasks[0].ready)

    def test_s_invitation_can_handoff_while_s_active_cleanup_waits_for_a(self):
        now = datetime(2026,11,19,14,tzinfo=CHINA)
        plan = TaskPlan(selected_tasks(daily_tasks=('bounty','bounty_s'),bounty_grade='A'),now)
        control = RunControl()
        regular,summons = (item.activity for item in plan.tasks)
        with patch.object(regular,'run') as run_a, patch.object(summons,'run') as run_s:
            run_s.side_effect = lambda *args,**kwargs: control.request_stop() or O.STOPPED
            navigate = Mock(return_value=InvitationHandoff('daily_bounty_s'))
            run_tasks('f',1,control,Mock(),plan=plan,clock=lambda:now,navigate=navigate)
            run_a.assert_not_called()
            self.assertEqual(run_s.call_args.kwargs['mode'],'invite')
        self.assertFalse(plan.tasks[1].ready)

    def test_maintenance_keeps_the_task_pending_and_delays_retry(self):
        now = datetime(2026, 11, 19, 10, tzinfo=CHINA)
        plan = TaskPlan([FakeTask('prayer', [], []), FakeTask('other', [], [])], now)
        item, other = plan.tasks
        plan.complete(item, O.MAINTENANCE, now)
        self.assertFalse(item.daily_done)
        self.assertFalse(item.blocked)
        plan.refresh(now + timedelta(seconds=15))
        self.assertFalse(item.ready)
        self.assertTrue(other.ready)
        plan.refresh(now + timedelta(minutes=5))
        self.assertTrue(item.ready)

    def test_invite_interrupt_uses_enabled_bounty_and_resumes_without_changing_quota(self):
        calls = []
        control = RunControl()
        bounty = FakeTask('daily_bounty', [O.RECOVER_IN_PLACE,O.MATCH_DONE], calls)
        bounty.bounty_grade = 'A'
        activity = FakeTask('activity', [InvitationHandoff('daily_bounty'),O.DAILY_DONE], calls)
        now = datetime(2026,11,17,12,tzinfo=CHINA)
        plan = TaskPlan([activity,bounty],now)
        plan.tasks[1].daily_done = True
        navigate = Mock(return_value=True)
        with patch.object(control.stop,'wait',return_value=False):
            run_tasks('f',1,control,Mock(),plan=plan,clock=lambda:now,navigate=navigate)
        self.assertEqual(calls,[('activity','daily'),('daily_bounty','invite'),
                                ('daily_bounty','invite_resume'),('activity','daily')])
        self.assertTrue(plan.tasks[1].daily_done)
        self.assertFalse(hasattr(control,'bounty_invite_targets'))
        self.assertEqual([call.kwargs['target'] for call in navigate.call_args_list],['activity','activity'])

    def test_navigation_handoff_runs_before_the_next_task(self):
        calls = []
        bounty = FakeTask('daily_bounty',[O.MATCH_DONE],calls)
        bounty.bounty_grade = 'A'
        activity = FakeTask('activity',[O.DAILY_DONE],calls)
        now = datetime(2026,11,17,12,tzinfo=CHINA)
        plan = TaskPlan([activity,bounty],now)
        plan.tasks[1].daily_done = True
        navigate = Mock(side_effect=[InvitationHandoff('daily_bounty'),True])
        run_tasks('f',1,RunControl(),Mock(),plan=plan,clock=lambda:now,navigate=navigate)
        self.assertEqual(calls,[('daily_bounty','invite'),('activity','daily')])

    def test_invitation_targets_depend_on_passive_option_only(self):
        calls = []
        control = RunControl()
        bounty = FakeTask('daily_bounty',[O.DAILY_DONE],calls)
        bounty.bounty_grade = 'A'
        bounty.active_invite,bounty.passive_invite = True,False
        summons = FakeTask('daily_bounty_s',[O.DAILY_DONE],calls)
        summons.active_invite,summons.passive_invite = False,True
        def navigate(*_,**__):
            self.assertEqual(control.bounty_invite_targets,{'S':'daily_bounty_s'})
            return True
        self.run_plan([bounty,summons],control=control,navigate=navigate)
        self.assertEqual(calls,[('daily_bounty','daily'),('daily_bounty_s','daily')])

    def test_foreign_scene_ends_invite_recovery_and_routes_back_to_original_task(self):
        calls = []
        bounty = FakeTask('daily_bounty',[O.RECOVER_IN_PLACE,O.SCENE_CHANGED],calls)
        bounty.bounty_grade = 'A'
        activity = FakeTask('intelligence',[InvitationHandoff('daily_bounty'),O.DAILY_DONE],calls)
        now = datetime(2026,11,18,10,tzinfo=CHINA)
        plan = TaskPlan([activity,bounty],now)
        plan.tasks[1].daily_done = True
        control = RunControl()
        navigate = Mock(return_value=True)
        with patch.object(control.stop,'wait',return_value=False):
            run_tasks('f',1,control,Mock(),plan=plan,clock=lambda:now,navigate=navigate)
        self.assertEqual(calls,[('intelligence','daily'),('daily_bounty','invite'),
                                ('daily_bounty','invite_resume'),('intelligence','daily')])
        self.assertEqual([c.kwargs['target'] for c in navigate.call_args_list],['intelligence']*2)
        self.assertEqual(plan.tasks[1].last_run,0)

    def test_waiting_retry_routes_from_current_screen_even_for_same_task(self):
        calls = []
        wall = [datetime(2026,11,18,10,tzinfo=CHINA)]
        control = RunControl()
        navigate = Mock(side_effect=[True,False,True])
        task = FakeTask('intelligence',[O.WAITING,O.DAILY_DONE],calls)
        with patch.object(control.stop,'wait',side_effect=lambda delay:wall.__setitem__(0,wall[0]+timedelta(seconds=delay))):
            self.run_plan([task],control=control,clock=lambda:wall[0],navigate=navigate)
        self.assertEqual(navigate.call_count,3)
        self.assertEqual(calls,[('intelligence','daily')]*2)

    def test_scene_change_releases_match_ownership_without_crediting_completion(self):
        now = datetime(2026,11,18,10,tzinfo=CHINA)
        plan = TaskPlan([FakeTask('daily_bounty',[],[]),FakeTask('intelligence',[],[])],now)
        bounty, intelligence = plan.tasks
        plan.complete(bounty,O.RECOVER_IN_PLACE,now)
        plan.complete(bounty,O.SCENE_CHANGED,now+timedelta(seconds=1))
        plan.refresh(now+timedelta(seconds=1))
        self.assertFalse(bounty.in_progress)
        self.assertFalse(bounty.ready)
        self.assertFalse(bounty.daily_done)
        self.assertTrue(intelligence.ready)

    def test_bounty_recovery_runs_with_battle_ownership(self):
        calls = []
        control = RunControl()
        bounty = FakeTask('daily_bounty',[O.RECOVER_IN_PLACE,O.MATCH_DONE,O.DAILY_DONE],calls)
        bounty.bounty_grade = 'A'
        wall = [datetime(2026,11,17,12,tzinfo=CHINA)]
        with patch.object(control.stop,'wait',side_effect=lambda delay:wall.__setitem__(0,wall[0]+timedelta(seconds=delay))):
            self.run_plan([bounty],control=control,clock=lambda:wall[0])
        self.assertEqual(calls,[('daily_bounty','daily'),('daily_bounty','recover'),('daily_bounty','daily')])

    def test_in_match_recovery_keeps_ownership_until_safe_checkpoint(self):
        calls = []
        control = RunControl()
        wall = [datetime(2026, 11, 15, 11, 59, 59, tzinfo=CHINA)]
        navigate = Mock(return_value=True)
        with patch.object(control.stop, 'wait', side_effect=lambda delay:wall.__setitem__(0, wall[0]+timedelta(seconds=delay))):
            self.run_plan([FakeTask('a', [O.RECOVER_IN_PLACE, O.ERROR, O.MATCH_DONE, O.DAILY_DONE], calls),
                           FakeTask('b', [O.DAILY_DONE], calls, windows=((12,13),))],
                          control=control, clock=lambda:wall[0], navigate=navigate)
        self.assertEqual(calls, [('a','daily')]*3+[('b','daily'),('a','daily')])
        self.assertEqual([call.kwargs['target'] for call in navigate.call_args_list], ['a','b','a'])

    def test_recovery_preserves_extra_mode_even_after_daily_reset_and_closing(self):
        now = datetime(2026, 11, 15, 4, 59, 59, tzinfo=CHINA)
        plan = TaskPlan([FakeTask('a', [], [], 'continuous', windows=((10,14),)),
                         FakeTask('b', [], [])], now)
        item = plan.tasks[0]
        item.daily_done = True
        item.mode = 'continuous'
        plan.complete(item, O.RECOVER_IN_PLACE, now)
        plan.refresh(now+timedelta(seconds=1))
        self.assertTrue(item.ready)
        self.assertEqual(item.mode, 'continuous')
        self.assertFalse(plan.tasks[1].ready)
        self.assertFalse(item.daily_done)

    def test_plan_manages_add_remove_and_daily_to_extra_priority(self):
        now = datetime(2026, 11, 15, 12, tzinfo=CHINA)
        plan = TaskPlan([FakeTask('a', [], [], '25')], now)
        plan.add(FakeTask('b', [], []))
        with self.assertRaises(ValueError):
            plan.add(FakeTask('a', [], []))
        plan.refresh(now)
        self.assertEqual(plan.tasks[0].priority[0], 0)
        plan.complete(plan.tasks[0], O.DAILY_DONE, now)
        plan.refresh(now)
        self.assertEqual(plan.tasks[0].priority[0], 1)
        self.assertEqual(plan.tasks[0].mode, '25')
        self.assertEqual(plan.tasks[1].priority[0], 0)
        plan.remove('b')
        self.assertEqual([item.activity.name for item in plan.tasks], ['a'])

    def test_scheduler_follows_plan_priority_without_own_activity_policy(self):
        calls = []
        now = datetime(2026, 11, 15, 12, tzinfo=CHINA)
        class ReversePlan(TaskPlan):
            def refresh(self, wall):
                super().refresh(wall)
                for number, item in enumerate(self.tasks):
                    item.priority = (-number,)
        plan = ReversePlan([FakeTask('a', [O.DAILY_DONE], calls),
                            FakeTask('b', [O.DAILY_DONE], calls)], now)
        run_tasks('f', 1, RunControl(), Mock(), plan=plan, clock=lambda: now,
                  navigate=Mock(return_value=True))
        self.assertEqual(calls, [('b', 'daily'), ('a', 'daily')])

    def run_plan(self, tasks, control=None, clock=None, navigate=None):
        events = []
        run_tasks('folder', 1, control or RunControl(), lambda *v: events.append(v),
                  tasks=tasks,
                  clock=clock or (lambda: datetime(2026, 11, 15, 12, tzinfo=CHINA)),
                  navigate=navigate or Mock(return_value=True))
        self.assertEqual(events.count(('finished', '')), 1)
        return events

    def test_all_dailies_before_extras_and_extras_rotate_at_checkpoints(self):
        calls = []
        self.run_plan([
            FakeTask('a', [O.DAILY_DONE, O.MATCH_DONE, O.LIMIT_REACHED], calls, 'continuous'),
            FakeTask('b', [O.DAILY_DONE, O.MATCH_DONE, O.LIMIT_REACHED], calls, '25')])
        self.assertEqual(calls, [('a', 'daily'), ('b', 'daily'), ('a', 'continuous'),
                                 ('b', '25'), ('a', 'continuous'), ('b', '25')])

    def test_nearest_closing_daily_first(self):
        calls = []
        self.run_plan([FakeTask('late', [O.DAILY_DONE], calls),
                       FakeTask('early', [O.DAILY_DONE], calls, windows=((10, 14),))])
        self.assertEqual(calls, [('early', 'daily'), ('late', 'daily')])

    def test_error_rotation_and_cap_never_mark_daily_done(self):
        for outcome in (O.UNAVAILABLE, O.LIMIT_REACHED):
            calls = []
            with self.subTest(outcome=outcome):
                events = self.run_plan([FakeTask('failed', [outcome], calls, '25'),
                                        FakeTask('other', [O.DAILY_DONE], calls)])
                self.assertEqual(calls, [('failed', 'daily'), ('other', 'daily')])
                self.assertFalse(any('failed：已确认' in str(v) for v in events))

    def test_waiting_daily_yields_then_preempts_extras_after_opening(self):
        calls = []
        wall = [datetime(2026, 11, 15, 9, tzinfo=CHINA)]
        def open_window(control):
            if len(calls) == 3:
                wall[0] = wall[0].replace(hour=10)
        self.run_plan([
            FakeTask('closed', [O.WAITING, O.DAILY_DONE], calls, windows=((10, 14),)),
            FakeTask('open', [O.DAILY_DONE, O.MATCH_DONE, O.LIMIT_REACHED], calls, '25', hook=open_window)
        ], clock=lambda: wall[0])
        self.assertEqual(calls, [('open', 'daily'), ('closed', 'daily'), ('open', '25'),
                                 ('closed', 'daily'), ('open', '25')])

    def test_daily_reset_at_five_rechecks_before_next_extra(self):
        calls = []
        wall = [datetime(2026, 11, 15, 4, 59, tzinfo=CHINA)]
        def reset(control):
            if len(calls) == 2:
                wall[0] = wall[0].replace(hour=5)
        self.run_plan([FakeTask('a', [O.DAILY_DONE, O.MATCH_DONE, O.DAILY_DONE, O.LIMIT_REACHED],
                                      calls, '25', hook=reset)], clock=lambda: wall[0])
        self.assertEqual([m for _, m in calls], ['daily', '25', 'daily', '25'])

    def test_deferred_daily_yields_to_continuous_tasks_until_game_day_refresh(self):
        calls = []
        wall = [datetime(2026,11,18,4,59,30,tzinfo=CHINA)]
        def refresh_day(control):
            if len(calls) == 3:
                wall[0] += timedelta(seconds=30)
        events = self.run_plan([
            FakeTask('limited',[O.DEFERRED_TODAY,O.DAILY_DONE],calls),
            FakeTask('activity',[O.DAILY_DONE,O.MATCH_DONE,O.DAILY_DONE,O.LIMIT_REACHED],
                     calls,extra='continuous',hook=refresh_day),
        ],clock=lambda:wall[0])
        self.assertEqual(calls,[('limited','daily'),('activity','daily'),('activity','continuous'),
                                ('limited','daily'),('activity','daily'),('activity','continuous')])
        results = [message for _,message in events if message.startswith('limited：')]
        self.assertEqual(len(results),2)
        self.assertIn('未确认完成，今日暂缓',results[0])
        self.assertIn('已确认',results[1])

    def test_fresh_plan_can_check_a_deferred_daily_without_waiting_for_refresh(self):
        now = datetime(2026,11,18,12,tzinfo=CHINA)
        task = FakeTask('limited',[],[])
        plan = TaskPlan([task],now)
        item = plan.tasks[0]
        plan.complete(item,O.DEFERRED_TODAY,now)
        plan.refresh(now+timedelta(hours=1))
        self.assertFalse(item.daily_done)
        self.assertFalse(item.ready)
        self.assertEqual(plan.pending(),[])
        fresh = TaskPlan([task],now+timedelta(hours=1))
        fresh.refresh(now+timedelta(hours=1))
        self.assertTrue(fresh.tasks[0].ready)

    def test_newly_opened_earlier_deadline_preempts_at_daily_match_checkpoint(self):
        calls = []
        wall = [datetime(2026, 11, 15, 9, tzinfo=CHINA)]
        def open_window(control):
            wall[0] = wall[0].replace(hour=10)
        self.run_plan([
            FakeTask('early', [O.DAILY_DONE], calls, windows=((10, 14),)),
            FakeTask('late', [O.MATCH_DONE, O.DAILY_DONE], calls, hook=open_window)
        ], clock=lambda: wall[0])
        self.assertEqual(calls, [('late', 'daily'), ('early', 'daily'), ('late', 'daily')])

    def test_stop_does_not_start_next_task(self):
        calls = []
        self.run_plan([FakeTask('a', [O.DAILY_DONE], calls, hook=lambda c: c.request_stop()),
                       FakeTask('b', [O.DAILY_DONE], calls)])
        self.assertEqual(calls, [('a', 'daily')])

    def test_failed_navigation_does_not_run_task(self):
        calls = []
        navigate = Mock(side_effect=[False, True, True])
        control = RunControl()
        wall = [datetime(2026, 11, 15, 12, tzinfo=CHINA)]
        with patch.object(control.stop, 'wait', side_effect=lambda delay:wall.__setitem__(0,wall[0]+timedelta(seconds=delay))):
            self.run_plan([FakeTask('a', [O.DAILY_DONE], calls), FakeTask('b', [O.DAILY_DONE], calls)],
                          navigate=navigate,control=control,clock=lambda:wall[0])
        self.assertEqual(calls, [('b', 'daily'), ('a', 'daily')])

    def test_navigation_exception_keeps_other_tasks_runnable(self):
        calls = []
        control = RunControl()
        wall = [datetime(2026, 11, 15, 12, tzinfo=CHINA)]
        with patch.object(control.stop, 'wait', side_effect=lambda delay:wall.__setitem__(0,wall[0]+timedelta(seconds=delay))):
            events = self.run_plan([FakeTask('a', [O.DAILY_DONE], calls), FakeTask('b', [O.DAILY_DONE], calls)],
                                  control=control,clock=lambda:wall[0],
                                  navigate=Mock(side_effect=[RuntimeError('navigation failed'), True, True]))
        self.assertEqual(calls, [('b', 'daily'), ('a', 'daily')])
        self.assertTrue(any(k == 'error' and 'navigation failed' in v for k, v in events))

    def test_failed_task_retries_before_another_activity_opens(self):
        now = datetime(2026, 11, 15, 9, tzinfo=CHINA)
        plan = TaskPlan([FakeTask('weekend', [], [], windows=((9,23),)),
                         FakeTask('evening', [], [], windows=((18,22),))], now)
        weekend, evening = plan.tasks
        plan.complete(weekend, O.ERROR, now)
        plan.complete(evening, O.WAITING, now)
        plan.refresh(now)
        self.assertFalse(weekend.ready)
        self.assertFalse(weekend.blocked)
        self.assertEqual(plan.waiting(now)[1],15)
        plan.refresh(now+timedelta(seconds=15))
        self.assertTrue(weekend.ready)
        self.assertFalse(evening.ready)

    def test_waiting_for_a_bounty_team_yields_without_marking_done_or_error(self):
        now = datetime(2026,11,16,12,tzinfo=CHINA)
        plan = TaskPlan([FakeTask('bounty',[],[]),FakeTask('other',[],[])],now)
        bounty,other = plan.tasks
        kind,message=plan.complete(bounty,O.WAITING,now)
        self.assertEqual(kind,'status')
        self.assertIn('15秒后重试',message)
        self.assertFalse(bounty.daily_done)
        self.assertFalse(bounty.blocked)
        plan.refresh(now)
        self.assertFalse(bounty.ready)
        self.assertTrue(other.ready)
        plan.refresh(now+timedelta(seconds=15))
        self.assertTrue(bounty.ready)

    def test_repeated_runner_errors_do_not_finish_the_schedule(self):
        calls = []
        control = RunControl()
        wall = [datetime(2026, 11, 15, 12, tzinfo=CHINA)]
        with patch.object(control.stop, 'wait', side_effect=lambda delay:wall.__setitem__(0,wall[0]+timedelta(seconds=delay))) as wait:
            events = self.run_plan([FakeTask('activity', [O.ERROR, None, O.DAILY_DONE], calls)],
                                  control=control,clock=lambda:wall[0])
        self.assertEqual(calls,[('activity','daily')]*3)
        self.assertEqual(wait.call_count,2)
        self.assertEqual(events[-1],('finished',''))
        self.assertEqual(sum(k=='error' and '15秒后重试' in v for k,v in events),2)

    def test_slow_failing_dailies_do_not_starve_later_tasks_or_retries(self):
        for outcome in (O.WAITING,O.ERROR):
            with self.subTest(outcome=outcome):
                calls = []
                control = RunControl()
                wall = [datetime(2026,11,17,9,40,tzinfo=CHINA)]
                names = ('monthly','limited','blessing','weapon')
                def finish_attempt(current):
                    wall[0] += timedelta(seconds=20)
                    if len(calls) == 8:
                        current.request_stop()
                tasks = [FakeTask(name,[outcome]*8,calls,hook=finish_attempt) for name in names]
                for priority,task in enumerate(tasks):
                    task.daily_priority = priority
                self.run_plan(tasks,control=control,clock=lambda:wall[0])
                self.assertEqual([name for name,_ in calls],list(names)*2)

    def test_due_daily_retry_still_preempts_extra_matches(self):
        now = datetime(2026,11,17,12,tzinfo=CHINA)
        plan = TaskPlan([FakeTask('failed',[],[]),FakeTask('extra',[],[],'continuous')],now)
        daily,extra = plan.tasks
        plan.complete(extra,O.DAILY_DONE,now)
        plan.complete(daily,O.ERROR,now)
        plan.refresh(now+timedelta(seconds=15))
        ready = sorted((item for item in plan.tasks if item.ready),key=lambda item:item.priority)
        self.assertEqual([item.activity.name for item in ready],['failed','extra'])

    def test_xumi_round_gives_older_daily_retries_a_turn(self):
        now = datetime(2026,11,19,12,tzinfo=CHINA)
        plan = TaskPlan([FakeTask('daily_bounty',[],[]),XumiTask()],now)
        bounty,xumi = plan.tasks
        plan.complete(bounty,O.WAITING,now)
        plan.refresh(now)
        self.assertFalse(bounty.ready)
        self.assertTrue(xumi.ready)

        settled = now+timedelta(minutes=2)
        plan.complete(xumi,O.MATCH_DONE,settled)
        plan.refresh(settled)
        ready = sorted((item for item in plan.tasks if item.ready),key=lambda item:item.priority)
        self.assertEqual([item.activity.name for item in ready],['daily_bounty','xumi'])
        self.assertFalse(xumi.daily_done)

    def test_paused_scheduler_does_not_navigate_or_start_next_task(self):
        calls = []
        control = RunControl()
        control.pause()
        navigate = Mock(return_value=True)
        with patch.object(control, 'wait_active', return_value=False):
            self.run_plan([FakeTask('a', [], calls)], control=control, navigate=navigate)
        navigate.assert_not_called()
        self.assertEqual(calls, [])

    def test_closed_daily_waits_and_rechecks_when_window_opens(self):
        calls = []
        control = RunControl()
        wall = [datetime(2026, 11, 15, 9, tzinfo=CHINA)]
        def advance(_):
            wall[0] = wall[0].replace(hour=10)
        with patch.object(control.stop, 'wait', side_effect=advance) as wait:
            self.run_plan([FakeTask('a', [O.WAITING, O.DAILY_DONE], calls, windows=((10, 14),))],
                          control=control, clock=lambda: wall[0])
        self.assertEqual(calls, [('a', 'daily'), ('a', 'daily')])
        wait.assert_called_once_with(30)

    def test_task_modules_own_their_run_and_share_control(self):
        control, emit = RunControl(), Mock()
        for task, module in ((IntelligenceTask('continuous'), 'ninja3_assistant.activities.intelligence.runner.run_activity'),
                             (WeekendTask('25'), 'ninja3_assistant.activities.weekend.runner.run_weekend')):
            with self.subTest(task=task.name), patch(module, return_value=O.DAILY_DONE) as runner:
                self.assertIs(task.run('f', 1, control, emit, mode='daily'), O.DAILY_DONE)
                runner.assert_called_once_with('f', 1, control, emit, mode='daily', checkpoint=True)

    def test_xumi_uses_china_date_bounds_and_defers_opening_to_the_scene(self):
        task = XumiTask()
        begin = datetime(2026, 11, 19, tzinfo=CHINA)
        end = datetime(2026, 12, 10, tzinfo=CHINA)
        self.assertFalse(task.open(begin - timedelta(seconds=1)))
        self.assertTrue(task.open(begin.astimezone(timezone.utc)))
        self.assertTrue(task.open(end - timedelta(seconds=1)))
        self.assertFalse(task.open(end))

        control, emit = RunControl(), Mock()
        with patch('ninja3_assistant.activities.xumi.runner.run_xumi', return_value=O.WAITING) as runner, \
                patch('ninja3_assistant.scheduling.activity_task.datetime') as clock:
            clock.now.return_value = begin
            self.assertIs(task.run('f', 152, control, emit, mode='daily'), O.WAITING)
            clock.now.return_value = begin+timedelta(hours=4,minutes=59)
            self.assertIs(task.run('f', 152, control, emit, mode='daily'), O.WAITING)
            clock.now.return_value = begin+timedelta(hours=5)
            self.assertIs(task.run('f', 152, control, emit, mode='daily'), O.WAITING)
            clock.now.return_value = end
            self.assertIs(task.run('f', 152, control, emit, mode='daily'), O.UNAVAILABLE)
        first,resumed,reset = runner.call_args_list
        self.assertTrue(all(call.args == ('f', 152, control, emit) for call in (first,resumed,reset)))
        self.assertIs(first.kwargs['progress'],resumed.kwargs['progress'])
        self.assertIsNot(first.kwargs['progress'],reset.kwargs['progress'])


if __name__ == '__main__':
    unittest.main()
