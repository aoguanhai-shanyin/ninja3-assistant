"""Plan owns task management and policy; the scheduler only sorts priorities."""
from dataclasses import dataclass, field
from datetime import timedelta

from ninja3_assistant.activities.intelligence.flow import daily_key
from ninja3_assistant.scheduling.activity_task import ActivityTask
from ninja3_assistant.scheduling.task_outcome import TaskOutcome


@dataclass
class PlannedTask:
    activity: ActivityTask
    daily_done: bool = False
    blocked: bool = False
    checked: bool = False
    last_run: int = 0
    ready: bool = False
    mode: str = 'daily'
    phase: str = '日常优先'
    priority: tuple = field(default=(0, 0, 0))
    retry_at: object = None
    in_progress: bool = False
    recheck_at: object = None


class TaskPlan:
    description = '自动规划：先完成每日清理和活动日常，再打满场次或持续上分'

    def __init__(self, tasks, now):
        self.tasks = []
        self.day = daily_key(now)
        self.sequence = 0
        for task in tasks:
            self.add(task)

    def add(self, task):
        if any(item.activity.name == task.name for item in self.tasks):
            raise ValueError('重复任务：'+task.name)
        self.tasks.append(PlannedTask(task))

    def remove(self, name):
        self.tasks[:] = [item for item in self.tasks if item.activity.name != name]

    def refresh(self, now):
        if daily_key(now) != self.day:
            self.day = daily_key(now)
            for item in self.tasks:
                item.daily_done = item.blocked = item.checked = False
                item.retry_at = None
        owner = next((item for item in self.tasks if item.in_progress), None)
        for item in self.tasks:
            task = item.activity
            if item.recheck_at is not None and now >= item.recheck_at:
                item.daily_done = item.checked = False
                item.recheck_at = None
            item.ready = False
            if owner is not None:
                if item is owner:
                    item.ready = item.retry_at is None or now >= item.retry_at
                    item.priority = (-1, 0, 0)
                continue
            if item.blocked or (item.retry_at is not None and now < item.retry_at):
                continue
            if not item.daily_done:
                item.mode = 'daily'
                item.phase = '日常优先'
                item.ready = not item.checked or task.open(now)
                # Finish the first pass before retries, then rotate retries so
                # slow failures cannot repeatedly preempt later daily tasks.
                retry = item.retry_at is not None
                item.priority = (0, retry, item.last_run if retry else 0,
                                 task.daily_priority, not task.open(now), task.closes(now))
            elif task.extra_mode:
                item.mode = task.extra_mode
                item.phase = '打满场次 / 上分'
                item.ready = task.open(now)
                item.priority = (1, item.last_run, task.closes(now))

    def pending(self):
        return [item for item in self.tasks if not item.blocked and
                (not item.daily_done or item.activity.extra_mode or item.recheck_at is not None)]

    def waiting(self, now):
        pending = self.pending()
        owner = next((item for item in pending if item.in_progress), None)
        if owner is not None:
            delay = max(0, (owner.retry_at-now).total_seconds()) if owner.retry_at else 0
            return '等待恢复当前对局：'+owner.activity.label, min(30, delay)
        retries = [item for item in pending if item.retry_at is not None and item.retry_at > now]
        if retries:
            delay = min(30, min((item.retry_at-now).total_seconds() for item in retries))
            return '等待重试：'+'、'.join(item.activity.label for item in retries), delay
        scheduled = [item for item in pending if item.recheck_at is not None and item.recheck_at > now]
        if scheduled:
            due = min(item.recheck_at for item in scheduled)
            return '等待倒计时：'+'、'.join(item.activity.label for item in scheduled), min(30,(due-now).total_seconds())
        return '等待活动开放：'+'、'.join(item.activity.label for item in pending), 30

    def complete(self, item, outcome, now):
        self.sequence += 1
        item.last_run = self.sequence
        item.checked = True
        item.retry_at = None
        label = item.activity.label
        safe = outcome in (TaskOutcome.DAILY_DONE, TaskOutcome.MATCH_DONE, TaskOutcome.LIMIT_REACHED,
                           TaskOutcome.UNAVAILABLE) or (outcome == TaskOutcome.WAITING and not item.activity.open(now))
        if outcome == TaskOutcome.RECOVER_IN_PLACE or (item.in_progress and not safe):
            item.in_progress = True
            item.retry_at = now+timedelta(seconds=1)
            return 'error', label+'：对局尚未结束，1秒后在当前活动恢复'
        item.in_progress = False
        if outcome == TaskOutcome.DAILY_DONE:
            item.daily_done = True
            item.recheck_at = item.activity.next_check_at
            return 'status', label+'：已确认全部日常领取，进入后续计划'
        if outcome == TaskOutcome.MATCH_DONE:
            if not item.daily_done:
                item.checked = False
            return None
        if outcome == TaskOutcome.WAITING and not item.activity.open(now):
            return None
        if outcome == TaskOutcome.WAITING:
            item.retry_at = now+timedelta(seconds=15)
            item.checked = False
            return 'status', label+'：暂未就绪，15秒后重试，期间继续其他任务'
        if outcome == TaskOutcome.LIMIT_REACHED:
            item.blocked = True
            return 'status', label+'：今日积分场次已用完，不再匹配；未领取的日常不会记为完成'
        if outcome == TaskOutcome.UNAVAILABLE:
            item.blocked = True
            return 'status', label+'：当前活动不可用，继续其他任务'
        item.retry_at = now+timedelta(seconds=15)
        item.checked = False
        return 'error', label+'：执行异常，15秒后重试，期间继续其他任务'
