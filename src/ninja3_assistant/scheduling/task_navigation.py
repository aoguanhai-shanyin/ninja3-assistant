"""Recognized menu handoff shared by activity tasks."""
import time

from ninja3_assistant.scheduling.invitation_handoff import bounty_handoff, clear_handled_invitation


def _ready_to_run(target, activity_page, race_page, daily_page, bounty_page, race_active):
    if activity_page in ('invite','login'):
        return False
    if activity_page in ('event_home','tasks','battle'):
        return target == 'intelligence'
    if race_page in ('tasks','home') or race_active:
        return target == 'weekend'
    if daily_page == 'login':
        return False
    if daily_page == 'home' or activity_page == 'home':
        return True
    if target == 'intelligence':
        return activity_page in ('hall','hall_search')
    if target == 'weekend':
        return race_page in ('hall','reward')
    if target in ('daily_bounty','daily_bounty_s') and bounty_page and bounty_page.startswith('bounty_'):
        return True
    return bool(target and target.startswith('daily_') and daily_page is not None)


def return_to_home(folder, index, control, *, target=None):
    """Find a frame the target can handle or leave a recognized menu."""
    import cv2
    from ninja3_assistant.activities.intelligence.vision import ActivityVision
    from ninja3_assistant.activities.weekend.runner import RaceVision
    from ninja3_assistant.device.mumu import MuMu, prepare_game
    from ninja3_assistant.device.native_touch import create_native_touch
    from ninja3_assistant.activities.daily.flow import DailyDecision
    from ninja3_assistant.activities.daily.vision import DailyScreen, DailyVision
    from ninja3_assistant.activities.bounty.daily_vision import DailyBountyVision

    if prepare_game(folder, index, control, lambda s: None) is None:
        return False
    activity, race = ActivityVision(), RaceVision()
    daily = DailyVision()
    bounty = DailyBountyVision()
    native = None
    ocr = None
    safe_invite = getattr(control,'bounty_navigation_safe',False)
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
            race_active = kind in ('ready','dash','finish','result','standings') or (
                kind in ('race','respawn') and race.confirmed_race)
            daily_page,daily_back = daily.navigation(frame)
            bounty_page,bounty_back = bounty.navigation(frame)
            chore_screen = None
            if (daily_page is None and bounty_page is None and screen.kind not in
                    ('login','invite','battle','home','hall','event_home','tasks','hall_search') and
                    not race_active and kind not in ('home','hall','tasks')):
                if ocr is None:
                    from ninja3_assistant.activities.intelligence.ocr import WindowsOcr
                    ocr = WindowsOcr()
                result = ocr.read(frame,stop=control.stop)
                if generation != control.generation() or control.stop.is_set():
                    continue
                if result is not None:
                    observed = DailyScreen(frame,result)
                    chore = daily.chore_navigation(observed)
                    if chore:
                        daily_page,daily_back = chore
                        observed.page = daily_page
                        chore_screen = observed
            if screen.kind == 'invite' and safe_invite and getattr(control,'bounty_invite_targets',{}):
                if ocr is None:
                    from ninja3_assistant.activities.intelligence.ocr import WindowsOcr
                    ocr = WindowsOcr()
                handoff = bounty_handoff(frame,activity.navigation.invitations,ocr,control,safe=True)
                if handoff:
                    return handoff
            elif screen.kind != 'invite':
                safe_invite = (screen.kind in ('home','hall','event_home','tasks','hall_search') or
                               kind in ('home','hall','tasks') or daily_page == 'home' or
                               bounty_page in ('bounty_daily','bounty_order','bounty_hub'))
                if safe_invite:
                    clear_handled_invitation(control)
            if (chore_screen is None or target in ('daily_pet_feed','daily_arena_reward','daily_landlady_quiz')) and \
                    _ready_to_run(target,screen.kind,kind,daily_page,bounty_page,race_active):
                return True
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            hall = activity.navigation.match(gray, 'startup-hall-title', 80, 0, 300, 110,
                                             threshold=.93, max_error=30)
            if screen.kind == 'invite':
                point = screen.point
            elif screen.kind == 'battle' or race_active:
                point = None
            elif daily_page == 'login' or screen.kind == 'login':
                point = daily_back if daily_page == 'login' else screen.point
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
                        current_capture = captured
                        if chore_screen is not None:
                            x,y = point
                            decision = DailyDecision('tap',point,'返回主页，继续切换任务',
                                guard=(max(0,x-25),max(0,y-25),min(1280,x+25),min(720,y+25)))
                            current_capture = time.perf_counter()
                            if not daily.current_action(chore_screen,decision,device.capture()):
                                return False
                        return native.priority_tap(point, captured_at=current_capture,
                                                   deadline=current_capture+.2, wait_seconds=.2)
                control.run_if_active(tap)
            control.stop.wait(.75)
        return False
    finally:
        control.set_release_input(None)
        if native:
            native.close()
        device.close()
