"""Explicit checkpoints shared by activity runners and the task planner."""
from enum import Enum


class TaskOutcome(Enum):
    DAILY_DONE = 'daily_done'
    MATCH_DONE = 'match_done'
    LIMIT_REACHED = 'limit_reached'
    WAITING = 'waiting'
    DEFERRED_TODAY = 'deferred_today'
    UNAVAILABLE = 'unavailable'
    ERROR = 'error'
    SCENE_CHANGED = 'scene_changed'
    RECOVER_IN_PLACE = 'recover_in_place'
    STOPPED = 'stopped'
