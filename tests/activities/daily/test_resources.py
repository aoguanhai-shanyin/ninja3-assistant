from datetime import datetime,timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ninja3_assistant.activities.daily.resources import reserve_food, remaining_seconds, save_deadline, load_deadline
from ninja3_assistant.activities.intelligence.flow import CHINA
from ninja3_assistant.runtime.runner import RunControl
from ninja3_assistant.scheduling.activity_task import DailyTask,IntelligenceTask
from ninja3_assistant.scheduling.task_planner import TaskPlan
from ninja3_assistant.scheduling.task_outcome import TaskOutcome
from ninja3_assistant.scheduling.task_scheduler import run_tasks


class DailyResourceTest(unittest.TestCase):
    def test_food_reservation_counts_only_full_thirty_minute_periods(self):
        now = 10000
        self.assertEqual(reserve_food(None,now),5)
        self.assertEqual(reserve_food(now-1,now),5)
        self.assertEqual(reserve_food(now+1799,now),5)
        self.assertEqual(reserve_food(now+1800,now),4)
        self.assertEqual(reserve_food(now+8999,now),1)
        self.assertEqual(reserve_food(now+9000,now),0)
        self.assertEqual(reserve_food(now+86400,now),0)

    def test_counter_parses_hours_minutes_and_seconds(self):
        self.assertEqual(remaining_seconds('23小时59分01秒'),86341)
        self.assertEqual(remaining_seconds('30分钟'),1800)
        self.assertEqual(remaining_seconds('59秒'),59)
        self.assertIsNone(remaining_seconds('修炼完成'))

    def test_house_hour_ocr_variants_preserve_the_whole_countdown(self):
        for text in ('问17刁；时53分','17卜时53分','17小時53分','，17小时53分'):
            with self.subTest(text=text):
                self.assertEqual(remaining_seconds(text),17*3600+53*60)
        self.assertIsNone(remaining_seconds('17?时53分'))

    def test_scheduler_prefers_persisted_due_house_after_restart(self):
        now = datetime(2026,11,16,5,tzinfo=CHINA)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.json'
            self.assertIsNone(load_deadline(path))
            save_deadline(path,now.timestamp(),now.timestamp()-3600)
            self.assertEqual(load_deadline(path),now.timestamp())
            with (
                patch('ninja3_assistant.scheduling.task_planner.resource_path',return_value=path) as resource,
                patch.object(DailyTask,'run',autospec=True,return_value=TaskOutcome.DAILY_DONE) as run,
            ):
                run_tasks('mumu',1,RunControl(),Mock(),tasks=[DailyTask('monthly'),DailyTask('house')],
                          clock=lambda:now,navigate=Mock(return_value=True))
            resource.assert_called_once_with('mumu',1)
            self.assertEqual([call.args[0].kind for call in run.call_args_list],['house','monthly'])

    def test_house_runs_before_village_and_all_chores_before_activity_daily(self):
        now = datetime(2026,11,16,12,tzinfo=CHINA)
        plan = TaskPlan([DailyTask('village'),IntelligenceTask('continuous'),DailyTask('house')],now)
        plan.refresh(now)
        self.assertEqual([i.activity.name for i in sorted(plan.tasks,key=lambda i:i.priority)],
                         ['daily_house','daily_village','intelligence'])

    def test_house_countdown_becomes_ready_again_at_its_deadline(self):
        now = datetime(2026,11,16,4,59,55,tzinfo=CHINA)
        task = DailyTask('house')
        due = task.next_check_at = now+timedelta(seconds=5)
        plan = TaskPlan([DailyTask('monthly'),task],now)
        monthly, house = plan.tasks
        plan.complete(house,TaskOutcome.DAILY_DONE,now)
        plan.complete(monthly,TaskOutcome.WAITING,now)
        plan.refresh(now)
        self.assertFalse(house.ready)
        message, delay = plan.waiting(now)
        self.assertIn('倒计时',message)
        self.assertEqual(delay,5)
        plan.refresh(due)
        self.assertTrue(house.ready)
        self.assertFalse(house.daily_done)
        self.assertLess(house.priority,monthly.priority)

    def test_due_house_waits_for_match_then_prioritizes_eligible_retry(self):
        now = datetime(2026,11,16,12,tzinfo=CHINA)
        task = DailyTask('house')
        due = task.next_check_at = now+timedelta(seconds=1)
        plan = TaskPlan([DailyTask('monthly'),task,IntelligenceTask('continuous')],now)
        monthly, house, match = plan.tasks
        plan.complete(house,TaskOutcome.DAILY_DONE,now)
        plan.complete(match,TaskOutcome.RECOVER_IN_PLACE,now)
        plan.refresh(due)
        self.assertTrue(match.ready)
        self.assertFalse(house.ready)
        plan.complete(match,TaskOutcome.MATCH_DONE,due)
        plan.refresh(due)
        self.assertTrue(house.ready)
        self.assertLess(house.priority,monthly.priority)
        plan.complete(house,TaskOutcome.ERROR,due)
        plan.refresh(due)
        self.assertTrue(monthly.ready)
        self.assertFalse(house.ready)
        plan.refresh(due+timedelta(seconds=15))
        self.assertTrue(house.ready)
        self.assertLess(house.priority,monthly.priority)
        plan.complete(house,TaskOutcome.DAILY_DONE,due+timedelta(seconds=15))
        plan.refresh(due+timedelta(seconds=15))
        self.assertFalse(house.ready)
