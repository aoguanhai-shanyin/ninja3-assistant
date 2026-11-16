"""Recognized menu handoff shared by activity tasks."""
import time


def return_to_home(folder, index, control, *, target=None):
    """Leave only recognized activity menus; never press back during combat."""
    import cv2
    from ninja3_assistant.activities.intelligence.vision import ActivityVision
    from ninja3_assistant.activities.weekend.runner import RaceVision
    from ninja3_assistant.device.mumu import MuMu, prepare_game
    from ninja3_assistant.device.native_touch import create_native_touch
    from ninja3_assistant.activities.daily.vision import DailyVision
    from ninja3_assistant.activities.bounty.daily_vision import DailyBountyVision

    if prepare_game(folder, index, control, lambda s: None) is None:
        return False
    activity, race = ActivityVision(), RaceVision()
    daily = DailyVision()
    bounty = DailyBountyVision()
    native = None
    device = MuMu(folder, index)
    try:
        native = create_native_touch(folder, index, control.stop, lambda *a: None)
        if native is None:
            raise RuntimeError('多任务切换需要 NinjaTouch.exe')
        control.set_release_input(native.stop_all)
        started = control.now()
        while control.wait_active() and control.now()-started < 20:
            generation = control.generation()
            captured = time.perf_counter()
            frame = device.capture()
            screen, kind = activity.detect(frame), race.detect(frame)
            daily_page,daily_back = daily.navigation(frame)
            bounty_page,bounty_back = bounty.navigation(frame)
            if (target in ('daily_bounty','daily_bounty_s') and bounty_page and
                    bounty_page.startswith('bounty_')):
                return True
            if target and target.startswith('daily_') and daily_page is not None:
                return True
            if target == 'intelligence' and screen.kind == 'battle':
                return True
            if target == 'weekend' and (kind in ('ready','dash') or
                    (kind in ('race','respawn') and race.confirmed_race)):
                return True
            if daily_page == 'home' or screen.kind in ('home', 'login'):
                return True
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            hall = activity.navigation.match(gray, 'startup-hall-title', 80, 0, 300, 110,
                                             threshold=.93, max_error=30)
            if screen.kind == 'invite':
                point = screen.point
            elif screen.kind == 'tasks':
                point = (1120, 111)
            elif kind == 'tasks':
                point = (1190, 400)
            elif screen.kind == 'event_home':
                point = (50, 45)
            elif kind == 'home':
                point = (65, 45)
            elif hall or kind == 'hall':
                point = (50, 45)
            elif daily_page:
                point = daily_back
            elif bounty_page:
                point = bounty_back
            else:
                point = None
            if point:
                def tap():
                    if generation == control.generation():
                        return native.priority_tap(point, captured_at=captured,
                                                   deadline=captured+.2, wait_seconds=.2)
                control.run_if_active(tap)
            control.stop.wait(.75)
        return False
    finally:
        control.set_release_input(None)
        if native:
            native.close()
        device.close()
