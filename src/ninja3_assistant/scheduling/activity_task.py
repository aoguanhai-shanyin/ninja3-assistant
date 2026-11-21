"""Activity task contract. Gameplay stays in each task's own runner."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime

from ninja3_assistant.activities.daily import CHORES, DEFAULT_PET_FEED_REWARD, PET_FEED_REWARDS
from ninja3_assistant.activities.intelligence.flow import CHINA, daily_key


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
        super().__init__('weekend', '多人竞速', '25' if mode == '25' else None, ((9, 23),))
        self._progress_day = None
        self._progress = None

    def run(self, folder, index, control, emit, *, mode):
        from ninja3_assistant.activities.weekend.runner import WeekendProgress, run_weekend

        day = daily_key(datetime.now(CHINA))
        if day != self._progress_day:
            self._progress_day = day
            self._progress = WeekendProgress()
        self._progress.refresh(day)
        return run_weekend(folder, index, control, emit, mode=mode, checkpoint=True,
                           progress=self._progress)


class XumiTask(ActivityTask):
    def __init__(self):
        super().__init__('xumi', '须弥幻境', None, ((0, 24),))
        self._progress_day = None
        self._progress = None

    def open(self, now):
        # The game's activity page determines when maintenance actually finishes.
        return date(2026, 11, 19) <= now.astimezone(CHINA).date() <= date(2026, 12, 9)

    def run(self, folder, index, control, emit, *, mode):
        from ninja3_assistant.scheduling.task_outcome import TaskOutcome

        now = datetime.now(CHINA)
        if not self.open(now):
            emit('status', '须弥幻境当前不在活动期')
            return TaskOutcome.UNAVAILABLE
        from ninja3_assistant.activities.xumi.runner import XumiProgress, run_xumi

        day = daily_key(now)
        if day != self._progress_day:
            self._progress_day = day
            self._progress = XumiProgress()
        return run_xumi(folder, index, control, emit, progress=self._progress)


class AbyssTask(ActivityTask):
    def __init__(self):
        super().__init__('abyss', '深渊备战', None, ((0, 24),))

    def run(self, folder, index, control, emit, *, mode):
        from ninja3_assistant.activities.abyss.runner import run_abyss
        return run_abyss(folder, index, control, emit)


class DailyTask(ActivityTask):
    def __init__(self, kind, *, bounty_grade='D', pet_feed_reward=DEFAULT_PET_FEED_REWARD,
                 active_invite=True, passive_invite=True, bounty_progress=None):
        if kind not in CHORES:
            raise ValueError('未知每日清理任务：'+kind)
        super().__init__('daily_'+kind,CHORES[kind],None,((0,24),),daily_priority=list(CHORES).index(kind))
        self.kind = kind
        self.bounty_grade = bounty_grade
        self.pet_feed_reward = pet_feed_reward
        self.active_invite = active_invite
        self.passive_invite = passive_invite
        if kind in ('bounty','bounty_s'):
            from ninja3_assistant.activities.bounty.daily_flow import BountyProgress
            self._bounty_progress = bounty_progress if bounty_progress is not None else BountyProgress()
        if kind == 'bounty':
            if bounty_grade not in ('D','C','B','A'):
                raise ValueError('每日悬赏等级应为D、C、B或A')
            self.label = CHORES[kind]+'（'+bounty_grade+'级）'
        if kind == 'pet_feed':
            if pet_feed_reward not in PET_FEED_REWARDS:
                raise ValueError('未知通灵兽喂养奖励：'+str(pet_feed_reward))
            self.label = CHORES[kind]+'（'+PET_FEED_REWARDS[pet_feed_reward]+'）'

    def bounty_quota_exhausted(self, now):
        if self.kind != 'bounty':
            return False
        progress = self._bounty_progress
        return progress.day == daily_key(now) and progress.free_checked and progress.regular_remaining == 0

    def run(self, folder, index, control, emit, *, mode):
        if self.kind in ('bounty','bounty_s'):
            from ninja3_assistant.activities.bounty.daily_runner import run_daily_bounty
            return run_daily_bounty(folder,index,control,emit,
                                    grade='S' if self.kind == 'bounty_s' else self.bounty_grade,
                                    invite_only=mode in ('invite','invite_resume'),
                                    invite_resume=mode == 'invite_resume',
                                    recover_in_place=mode in ('recover','invite_resume'),
                                    active_invite=self.active_invite,passive_invite=self.passive_invite,
                                    progress=self._bounty_progress)
        from ninja3_assistant.activities.daily.runner import run_daily
        self.next_check_at = None
        def forward(kind,value):
            if kind == 'daily_next_check':
                self.next_check_at = datetime.fromtimestamp(value,CHINA)
            else:
                emit(kind,value)
        if self.kind == 'pet_feed':
            return run_daily(self.kind,folder,index,control,forward,pet_feed_reward=self.pet_feed_reward)
        return run_daily(self.kind,folder,index,control,forward)


def selected_tasks(*, activity_mode=None, weekend_mode=None, xumi_enabled=False, daily_tasks=(), bounty_grade='D',
                   pet_feed_reward=DEFAULT_PET_FEED_REWARD, bounty_active_invite=True,
                   bounty_passive_invite=True, abyss_enabled=False):
    daily_tasks = tuple(dict.fromkeys(daily_tasks))
    bounty_progress = None
    if any(kind in ('bounty','bounty_s') for kind in daily_tasks):
        from ninja3_assistant.activities.bounty.daily_flow import BountyProgress
        bounty_progress = BountyProgress(order_after_regular=bounty_grade == 'A' and
                                         'bounty' in daily_tasks and 'bounty_s' in daily_tasks)
    tasks = [DailyTask(kind,bounty_grade=bounty_grade,pet_feed_reward=pet_feed_reward,
                       active_invite=bounty_active_invite,passive_invite=bounty_passive_invite,
                       bounty_progress=bounty_progress) for kind in daily_tasks]
    if activity_mode is not None:
        tasks.append(IntelligenceTask(activity_mode))
    if weekend_mode is not None:
        tasks.append(WeekendTask(weekend_mode))
    if xumi_enabled:
        tasks.append(XumiTask())
    if abyss_enabled:
        tasks.append(AbyssTask())
    return tasks
