"""Priority-only serial execution; task policy is supplied by TaskPlan."""
from datetime import datetime

from ninja3_assistant.activities.intelligence.flow import CHINA
from ninja3_assistant.scheduling.task_navigation import return_to_home
from ninja3_assistant.scheduling.task_outcome import TaskOutcome
from ninja3_assistant.scheduling.task_planner import TaskPlan


def run_tasks(folder, index, control, emit, *, tasks=None, plan=None, clock=None, navigate=None):
    clock = clock or (lambda: datetime.now(CHINA))
    navigate = navigate or return_to_home
    plan = plan if plan is not None else TaskPlan(tasks or [], clock())
    previous = None

    def forward(kind, value):
        if kind != 'finished':
            emit(kind, value)

    try:
        emit('status', plan.description)
        while control.wait_active():
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
            try:
                if previous != task.name and not navigate(folder, index, control, target=task.name):
                    if control.stop.is_set():
                        return
                    emit('error', task.label+'：未能返回游戏主页，保留任务未完成状态')
                    outcome = TaskOutcome.ERROR
                else:
                    previous = task.name
                    outcome = task.run(folder, index, control, forward, mode=item.mode)
            except Exception as error:
                emit('error', task.label+'：'+str(error))
                outcome = TaskOutcome.ERROR
            if control.stop.is_set() or outcome == TaskOutcome.STOPPED:
                return
            message = plan.complete(item, outcome, clock().astimezone(CHINA))
            if message:
                emit(*message)
    finally:
        emit('finished', '')
