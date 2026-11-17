"""Activity task contract. Gameplay stays in each task's own runner."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ActivityTask(ABC):
    name: str
    label: str
    extra_mode: str | None
    windows: tuple
    daily_priority: int = 100
    next_check_at: object = None

    def open(self, now):
        return any(start <= now.hour < end for start, end in self.windows)

    def closes(self, now):
        return next((end for start, end in self.windows if start <= now.hour < end), 24)

    @abstractmethod
    def run(self, folder, index, control, emit, *, mode):
        """Return TaskOutcome at a safe checkpoint, with all task inputs released."""


class IntelligenceTask(ActivityTask):
    def __init__(self, mode):
        super().__init__('intelligence', '情报模拟战',
                         'continuous' if mode == 'continuous' else None, ((10, 14), (18, 22)))

    def run(self, folder, index, control, emit, *, mode):
        from ninja3_assistant.activities.intelligence.runner import run_activity
        return run_activity(folder, index, control, emit, mode=mode, checkpoint=True)


class WeekendTask(ActivityTask):
    def __init__(self, mode):
        super().__init__('weekend', '同屏竞速', '25' if mode == '25' else None, ((9, 23),))

    def run(self, folder, index, control, emit, *, mode):
        from ninja3_assistant.activities.weekend.runner import run_weekend
        return run_weekend(folder, index, control, emit, mode=mode, checkpoint=True)


class DailyTask(ActivityTask):
    def __init__(self, kind, *, bounty_grade='D'):
        from ninja3_assistant.activities.daily import CHORES
        if kind not in CHORES:
            raise ValueError('未知每日清理任务：'+kind)
        super().__init__('daily_'+kind,CHORES[kind],None,((0,24),),daily_priority=list(CHORES).index(kind))
        self.kind = kind
        self.bounty_grade = bounty_grade
        if kind == 'bounty':
            if bounty_grade not in ('D','C','B','A'):
                raise ValueError('每日悬赏等级应为D、C、B或A')
            self.label = CHORES[kind]+'（'+bounty_grade+'级）'

    def run(self, folder, index, control, emit, *, mode):
        if self.kind in ('bounty','bounty_s'):
            from ninja3_assistant.activities.bounty.daily_runner import run_daily_bounty
            return run_daily_bounty(folder,index,control,emit,
                                    grade='S' if self.kind == 'bounty_s' else self.bounty_grade,
                                    invite_only=mode in ('invite','invite_resume'),
                                    invite_resume=mode == 'invite_resume',
                                    recover_in_place=mode in ('recover','invite_resume'))
        from ninja3_assistant.activities.daily.runner import run_daily
        from datetime import datetime
        from ninja3_assistant.activities.intelligence.flow import CHINA
        self.next_check_at = None
        def forward(kind,value):
            if kind == 'daily_next_check':
                self.next_check_at = datetime.fromtimestamp(value,CHINA)
            else:
                emit(kind,value)
        return run_daily(self.kind,folder,index,control,forward)


def selected_tasks(*, activity_mode=None, weekend_mode=None, daily_tasks=(), bounty_grade='D'):
    tasks = [DailyTask(kind,bounty_grade=bounty_grade) for kind in dict.fromkeys(daily_tasks)]
    if activity_mode is not None:
        tasks.append(IntelligenceTask(activity_mode))
    if weekend_mode is not None:
        tasks.append(WeekendTask(weekend_mode))
    return tasks
