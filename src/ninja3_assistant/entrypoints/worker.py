"""JSON-lines worker used by the WinUI front end."""

import argparse
import json
import sys
import threading

from ninja3_assistant.activities.daily import CHORES, DEFAULT_PET_FEED_REWARD, PET_FEED_REWARDS
from ninja3_assistant.device.input_owner import InputBusy, input_owner
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
    # This week's rotation has no 25-match cap, so the goal is a choice: the
    # dailies need 4 races, anything up to 25 may be asked for, and "unlimited"
    # races until stopped. Must match weekend.runner.WEEKEND_MODES.
    parser.add_argument('--weekend-mode', choices=('daily', 'unlimited',
                                                   *(str(n) for n in range(4, 26))), default='daily')
    parser.add_argument('--weekend-activity', choices=('multiplayer', 'samescreen'),
                        default='multiplayer')
    parser.add_argument('--xumi-enabled', action='store_true')
    parser.add_argument('--abyss-enabled', action='store_true')
    parser.add_argument('--daily-tasks',nargs='+',choices=tuple(CHORES),default=[])
    parser.add_argument('--bounty-grade',choices=('D','C','B','A'),default=None)
    parser.add_argument('--no-bounty-active-invite',dest='bounty_active_invite',action='store_false',
                        help='disable proactively inviting friends and clan members')
    parser.add_argument('--no-bounty-passive-invite',dest='bounty_passive_invite',action='store_false',
                        help='disable accepting bounty invitations from friends and clan members')
    parser.add_argument('--pet-feed-reward',choices=tuple(PET_FEED_REWARDS))
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
    if args.xumi_enabled and args.mode != 'run':
        parser.error('xumi is available only in run mode')
    if args.abyss_enabled and args.mode != 'run':
        parser.error('abyss is available only in run mode')
    if args.daily_tasks and args.mode != 'run':
        parser.error('daily chores are available only in run mode')
    if args.bounty_grade is not None and (args.mode != 'run' or 'bounty' not in args.daily_tasks):
        parser.error('bounty-grade requires the bounty daily task in run mode')
    if not (args.bounty_active_invite and args.bounty_passive_invite) and not (
            args.mode == 'run' and {'bounty','bounty_s'}.intersection(args.daily_tasks)):
        parser.error('bounty invitation options require a bounty daily task in run mode')
    if args.pet_feed_reward is not None and (args.mode != 'run' or 'pet_feed' not in args.daily_tasks):
        parser.error('pet-feed-reward requires the pet_feed daily task in run mode')

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

    try:
        with input_owner(args.folder, args.index):
            run_selected(args, emit)
    except InputBusy as error:
        emit('error', str(error))
        emit('finished', '')


def run_selected(args, emit):
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
    elif args.weekend_enabled or args.activity_enabled or args.xumi_enabled or args.abyss_enabled or args.daily_tasks:
        from ninja3_assistant.scheduling.activity_task import selected_tasks
        from ninja3_assistant.scheduling.task_scheduler import run_tasks
        tasks = selected_tasks(activity_mode=args.activity_mode if args.activity_enabled else None,
                               weekend_mode=args.weekend_mode if args.weekend_enabled else None,
                               weekend_activity=args.weekend_activity,
                               xumi_enabled=args.xumi_enabled,
                               abyss_enabled=args.abyss_enabled,
                               daily_tasks=args.daily_tasks,bounty_grade=args.bounty_grade or 'D',
                               pet_feed_reward=args.pet_feed_reward or DEFAULT_PET_FEED_REWARD,
                               bounty_active_invite=args.bounty_active_invite,
                               bounty_passive_invite=args.bounty_passive_invite)
        run_tasks(args.folder, args.index, control, emit, tasks=tasks)
    else:
        run_flow(args.folder, args.index, control, emit, join_interval_ms=args.join_interval_ms)


if __name__ == '__main__':
    main()
