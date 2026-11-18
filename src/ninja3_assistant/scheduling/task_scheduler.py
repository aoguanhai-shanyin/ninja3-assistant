"""Priority-only serial execution; task policy is supplied by TaskPlan."""
from datetime import datetime

from ninja3_assistant.activities.intelligence.flow import CHINA
from ninja3_assistant.scheduling.task_navigation import return_to_home
from ninja3_assistant.scheduling.task_outcome import TaskOutcome
from ninja3_assistant.scheduling.task_planner import TaskPlan
from ninja3_assistant.scheduling.invitation_handoff import InvitationHandoff


SAFE_CHECKPOINTS = (TaskOutcome.DAILY_DONE, TaskOutcome.MATCH_DONE, TaskOutcome.LIMIT_REACHED,
                    TaskOutcome.WAITING, TaskOutcome.UNAVAILABLE, TaskOutcome.SCENE_CHANGED)


def run_tasks(folder, index, control, emit, *, tasks=None, plan=None, clock=None, navigate=None):
    clock = clock or (lambda: datetime.now(CHINA))
    navigate = navigate or return_to_home
    plan = plan if plan is not None else TaskPlan(tasks or [], clock())
    prior_control = {name:getattr(control,name,None) for name in
                     ('bounty_invite_targets','bounty_invite_suppressed','bounty_navigation_safe')}
    control.bounty_invite_suppressed = False
    control.bounty_navigation_safe = False
    control.bounty_invite_targets = {
        ('S' if item.activity.name == 'daily_bounty_s' else item.activity.bounty_grade): item.activity.name
        for item in plan.tasks
        if item.activity.name in ('daily_bounty', 'daily_bounty_s')
    }

    def forward(kind, value):
        if kind != 'finished':
            emit(kind, value)

    def execute(task, mode, *, route):
        try:
            if route:
                navigation = navigate(folder, index, control, target=task.name)
                if control.stop.is_set():
                    return TaskOutcome.STOPPED
                if isinstance(navigation, InvitationHandoff):
                    return navigation
                if not navigation:
                    emit('error', task.label+'：未能确认可切换的游戏场景，保留任务未完成状态')
                    return TaskOutcome.ERROR
            return task.run(folder, index, control, forward, mode=mode)
        except Exception as error:
            emit('error', task.label+'：'+str(error))
            return TaskOutcome.RECOVER_IN_PLACE if mode == 'invite_resume' else TaskOutcome.ERROR

    invite_task = invite_mode = None
    try:
        emit('status', plan.description)
        while control.wait_active():
            if invite_task is not None:
                result = execute(invite_task, invite_mode, route=False)
                if control.stop.is_set() or result == TaskOutcome.STOPPED:
                    return
                if result == TaskOutcome.RECOVER_IN_PLACE:
                    invite_mode = 'invite_resume'
                    emit('status', '正在根据当前画面恢复悬赏对局')
                    control.stop.wait(1)
                else:
                    if result == TaskOutcome.SCENE_CHANGED:
                        emit('status', '当前画面已离开悬赏场景，继续原任务')
                    control.bounty_navigation_safe = result in SAFE_CHECKPOINTS
                    invite_task = None
                continue
            plan.refresh(clock().astimezone(CHINA))
            ready = sorted((item for item in plan.tasks if item.ready), key=lambda item: item.priority)
            if not ready:
                pending = plan.pending()
                if not pending:
                    emit('status', '所选任务处理结束；未完成或不可用的任务见前面的提示')
                    return
                message, delay = plan.waiting(clock().astimezone(CHINA))
                emit('status', message)
                control.stop.wait(delay)
                continue
            item = ready[0]
            task = item.activity
            emit('status', f'自动规划 · {item.phase} · {task.label}')
            mode = ('recover' if item.in_progress and task.name in ('daily_bounty','daily_bounty_s')
                    else item.mode)
            outcome = execute(task, mode, route=not item.in_progress)
            if control.stop.is_set() or outcome == TaskOutcome.STOPPED:
                return
            if isinstance(outcome, InvitationHandoff):
                invite_task = next((candidate.activity for candidate in plan.tasks
                               if candidate.activity.name == outcome.target and
                               outcome.target in control.bounty_invite_targets.values()), None)
                if invite_task is None:
                    continue
                emit('status', '安全间隙处理悬赏邀请，完成后继续原任务')
                control.bounty_invite_suppressed = True
                invite_mode = 'invite'
                continue
            message = plan.complete(item, outcome, clock().astimezone(CHINA))
            control.bounty_navigation_safe = outcome in SAFE_CHECKPOINTS and not item.in_progress
            if message:
                emit(*message)
    finally:
        for name,value in prior_control.items():
            if value is None:
                if hasattr(control,name):
                    delattr(control,name)
            else:
                setattr(control,name,value)
        emit('finished', '')
