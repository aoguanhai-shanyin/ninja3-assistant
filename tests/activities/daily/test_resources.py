from datetime import datetime,timedelta
from pathlib import Path
import tempfile
import unittest

from ninja3_assistant.activities.daily.resources import reserve_food, remaining_seconds, save_deadline, load_deadline
from ninja3_assistant.activities.intelligence.flow import CHINA
from ninja3_assistant.scheduling.activity_task import DailyTask,IntelligenceTask
from ninja3_assistant.scheduling.task_planner import TaskPlan
from ninja3_assistant.scheduling.task_outcome import TaskOutcome


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

    def test_house_deadline_survives_restart_and_crosses_daily_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.json'
            self.assertIsNone(load_deadline(path))
            save_deadline(path,10000,100)
            self.assertEqual(load_deadline(path),10000)

    def test_house_runs_before_village_and_all_chores_before_activity_daily(self):
        now = datetime(2026,11,16,12,tzinfo=CHINA)
        plan = TaskPlan([DailyTask('village'),IntelligenceTask('continuous'),DailyTask('house')],now)
        plan.refresh(now)
        self.assertEqual([i.activity.name for i in sorted(plan.tasks,key=lambda i:i.priority)],
                         ['daily_house','daily_village','intelligence'])

    def test_house_countdown_becomes_ready_again_at_its_deadline(self):
        now = datetime(2026,11,16,12,tzinfo=CHINA)
        task = DailyTask('house'); task.next_check_at = now+timedelta(hours=1)
        plan = TaskPlan([task],now);item=plan.tasks[0]
        plan.complete(item,TaskOutcome.DAILY_DONE,now)
        plan.refresh(now)
        self.assertFalse(item.ready)
        self.assertEqual(plan.pending(),[item])
        self.assertIn('倒计时',plan.waiting(now)[0])
        plan.refresh(now+timedelta(hours=1))
        self.assertTrue(item.ready)
        self.assertFalse(item.daily_done)
