"""Activity task contract. Gameplay stays in each task's own runner."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ActivityTask(ABC):
    name: str
    label: str
    extra_mode: str | None
    windows: tuple

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


def selected_tasks(*, activity_mode=None, weekend_mode=None):
    tasks = []
    if activity_mode is not None:
        tasks.append(IntelligenceTask(activity_mode))
    if weekend_mode is not None:
        tasks.append(WeekendTask(weekend_mode))
    return tasks
