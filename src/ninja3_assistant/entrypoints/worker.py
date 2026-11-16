"""JSON-lines worker used by the WinUI front end."""

import argparse
import json
import sys
import threading

from ninja3_assistant.runtime.runner import RunControl, probe, run_flow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('run', 'probe', 'hayabusa-debug', 'hayabusa', 'hayabusa-video'), required=True)
    parser.add_argument('--folder')
    parser.add_argument('--index', type=int)
    parser.add_argument('--video')
    parser.add_argument('--join-interval-ms', type=int, default=30)
    parser.add_argument('--observe-only', action='store_true')
    parser.add_argument('--hide-game-window', action='store_true')
    parser.add_argument('--activity-enabled', action='store_true')
    parser.add_argument('--activity-mode', choices=('daily', 'continuous'), default='daily')
    parser.add_argument('--weekend-enabled', action='store_true')
    parser.add_argument('--weekend-mode', choices=('daily', '25'), default='daily')
    from ninja3_assistant.activities.daily import CHORES
    parser.add_argument('--daily-tasks',nargs='+',choices=tuple(CHORES),default=[])
    parser.add_argument('--bounty-grade',choices=('D','C','B','A'),default=None)
    args = parser.parse_args()
    if args.mode == 'hayabusa-video':
        if not args.video:
            parser.error('video is required in hayabusa-video mode')
    elif args.folder is None or args.index is None:
        parser.error('folder and index are required for emulator modes')
    elif args.video:
        parser.error('video is available only in hayabusa-video mode')
    if args.index is not None and args.index < 0:
        parser.error('instance index must not be negative')
    if not 20 <= args.join_interval_ms <= 1000:
        parser.error('join interval must be between 20 and 1000 ms')
    if args.observe_only and args.mode not in ('hayabusa', 'hayabusa-debug'):
        parser.error('observe-only is available only in hayabusa modes')
    if args.activity_enabled and args.mode != 'run':
        parser.error('activity is available only in run mode')
    if args.weekend_enabled and args.mode != 'run':
        parser.error('weekend is available only in run mode')
    if args.daily_tasks and args.mode != 'run':
        parser.error('daily chores are available only in run mode')
    if args.bounty_grade is not None and (args.mode != 'run' or 'bounty' not in args.daily_tasks):
        parser.error('bounty-grade requires the bounty daily task in run mode')

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    output_lock = threading.Lock()

    def emit(kind, value):
        with output_lock:
            print(json.dumps({'event': kind, 'value': value}, ensure_ascii=False), flush=True)

    if args.mode == 'hayabusa-video':
        from ninja3_assistant.combat.hayabusa.video_debug import run_video_debug
        run_video_debug(args.video, emit, sys.stdin)
        return

    if args.mode == 'probe':
        probe(args.folder, args.index, emit)
        return

    control = RunControl(hide_game_window=args.hide_game_window)

    def commands():
        for line in sys.stdin:
            try:
                command = json.loads(line).get('command')
            except (ValueError, AttributeError):
                continue
            if command == 'pause' and control.pause():
                emit('state', 'paused')
            elif command == 'resume' and control.resume():
                emit('state', 'running')
            elif command == 'stop':
                control.request_stop()
                return
        control.request_stop()

    threading.Thread(target=commands, daemon=True).start()
    if args.mode in ('hayabusa', 'hayabusa-debug'):
        from ninja3_assistant.combat.hayabusa.trial import run_hayabusa
        run_hayabusa(args.folder, args.index, control, emit,
                     observe_only=args.mode == 'hayabusa-debug' or args.observe_only)
    elif args.weekend_enabled or args.activity_enabled or args.daily_tasks:
        from ninja3_assistant.scheduling.activity_task import selected_tasks
        from ninja3_assistant.scheduling.task_scheduler import run_tasks
        tasks = selected_tasks(activity_mode=args.activity_mode if args.activity_enabled else None,
                               weekend_mode=args.weekend_mode if args.weekend_enabled else None,
                               daily_tasks=args.daily_tasks,bounty_grade=args.bounty_grade or 'D')
        run_tasks(args.folder, args.index, control, emit, tasks=tasks)
    else:
        run_flow(args.folder, args.index, control, emit, join_interval_ms=args.join_interval_ms)


if __name__ == '__main__':
    main()
