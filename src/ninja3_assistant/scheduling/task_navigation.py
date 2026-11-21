"""Recognized menu handoff shared by activity tasks."""
import time

from ninja3_assistant.game_ui import GameUI, PageAction, PageObservation
from ninja3_assistant.game_ui.pages import HALL, HOME, NAVIGABLE_PAGES, ROUTES, page_id
from ninja3_assistant.scheduling.invitation_handoff import bounty_handoff, clear_handled_invitation


def _editing_xumi(kind):
    return kind in ('team', 'roster', 'own_roster', 'score_confirm') or kind.startswith('equipment')


def _ready_to_run(target, activity_page, race_page, daily_page, bounty_page, race_active):
    if activity_page in ('invite','login','network_retry'):
        return False
    if activity_page in ('event_home','tasks','battle','guide'):
        return target == 'intelligence'
    if race_page in ('tasks','home') or race_active:
        return target == 'weekend'
    if daily_page in ('login','shop_dialog','bounty_dialog'):
        return False
    if daily_page == 'family':
        return target in ('daily_blessing', 'abyss')
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
    from ninja3_assistant.activities.abyss.vision import (
        REWARDS_CLOSE, preparation_page, relay_prompt, result_panel, reward_dungeon, same_region,
    )
    from ninja3_assistant.activities.intelligence.vision import ActivityVision
    from ninja3_assistant.activities.weekend.runner import RaceVision
    from ninja3_assistant.device.mumu import MuMu, prepare_game
    from ninja3_assistant.device.native_touch import create_native_touch
    from ninja3_assistant.activities.daily.flow import DailyDecision
    from ninja3_assistant.activities.daily.vision import DailyScreen, DailyVision
    from ninja3_assistant.activities.bounty.daily_vision import DailyBountyVision
    from ninja3_assistant.activities.xumi.vision import XumiVision
    from ninja3_assistant.vision.game_update import game_update_banner, game_update_point
    from ninja3_assistant.vision.promotion_popup import promotion_popup_close

    activity = None
    if target == 'intelligence' and not control.stop.is_set():
        activity = ActivityVision(use_winml=True)
        activity.battle.warmup()
    if prepare_game(folder, index, control, lambda s: None) is None:
        return False
    activity = activity or ActivityVision()
    race = RaceVision()
    daily = DailyVision()
    bounty = DailyBountyVision()
    xumi = XumiVision(daily)
    native = None
    ocr = None
    safe_invite = getattr(control,'bounty_navigation_safe',False)
    device = MuMu(folder, index)
    try:
        native = create_native_touch(folder, index, control.stop, lambda *a: None)
        if native is None:
            raise RuntimeError('多任务切换需要 NinjaTouch.exe')
        control.set_release_input(native.stop_all)
        ui = GameUI(control, device.capture, native, routes=ROUTES)

        def press(page, name, point, guard):
            action = PageAction(name, point, guard=lambda fresh: (
                not game_update_banner(fresh) and guard(fresh) and not _editing_xumi(xumi.scene(fresh).kind)))
            observation = PageObservation(page, frame, generation, captured, {name: action})
            if page in NAVIGABLE_PAGES and name == 'back':
                hall_targets = ('intelligence', 'weekend', 'xumi', 'daily_bounty', 'daily_bounty_s', 'daily_village')
                destination = HALL if target in hall_targets and page != HALL else HOME
                route = ui.plan_route(page, destination) or ui.plan_route(page, HOME)
                if not route:
                    return None
                name = route[0].action
            return ui.perform(observation, name)

        deadline = control.now()+20
        last_network_retry = float('-inf')
        restarted_for_update = False
        while control.wait_active() and control.now() < deadline:
            generation = control.generation()
            captured = time.perf_counter()
            frame = device.capture()
            if game_update_banner(frame):
                if ocr is None:
                    from ninja3_assistant.activities.intelligence.ocr import WindowsOcr
                    ocr = WindowsOcr()
                result = ocr.read(frame,stop=control.stop)
                update = DailyScreen(frame,result,'game_update') if result is not None else None
                point = game_update_point(update) if update is not None else None
                if point:
                    decision = DailyDecision('tap', point, '重启客户端安装游戏更新', guard=(510,484,775,552),
                                             operation='restart_for_update')
                    action = PageAction('restart_for_update', point, guard=lambda fresh: (
                        game_update_banner(fresh) and daily.current_action(update, decision, fresh)))
                    observation = PageObservation('game.update', frame, generation, captured, {action.name: action})
                    if not restarted_for_update and ui.perform(observation, action.name):
                        restarted_for_update = True
                        deadline = max(deadline,control.now()+60)
                    control.stop.wait(.75)
                    continue
            if (close := promotion_popup_close(frame)) is not None:
                press('game.promotion_popup', 'close_promotion_popup', close,
                      lambda fresh: promotion_popup_close(fresh) == close)
                control.stop.wait(.45)
                continue
            if (reward_tab := reward_dungeon(frame)) is not None:
                if target == 'abyss':
                    return True
                press('abyss.rewards', 'close_abyss_rewards', REWARDS_CLOSE,
                      lambda fresh: reward_dungeon(fresh) == reward_tab)
                control.stop.wait(.45)
                continue
            xumi_screen = xumi.scene(frame)
            if preparation_page(frame):
                if target == 'abyss':
                    return True
                press('abyss.preparation', 'back', (42, 43), preparation_page)
                control.stop.wait(.75)
                continue
            if target == 'xumi' and xumi_screen.kind != 'unknown':
                return True
            if _editing_xumi(xumi_screen.kind):
                return False
            screen, kind = activity.detect(frame), race.detect(frame)
            race_active = kind in ('ready','dash','finish','result','standings') or (
                kind in ('race','respawn') and race.confirmed_race)
            daily_page,daily_back = daily.navigation(frame)
            bounty_page,bounty_back = bounty.navigation(frame)
            chore_screen = purchase_screen = None
            if (daily_page is None and bounty_page is None and screen.kind not in
                    ('login','invite','battle','home','hall','event_home','tasks','guide','hall_search','network_retry') and
                    not race_active and kind not in ('home','hall','tasks')):
                if xumi_screen.back is not None:
                    press(page_id('xumi', xumi_screen.navigation_kind), 'back', xumi_screen.back,
                          lambda fresh: xumi.scene(fresh) == xumi_screen)
                    control.stop.wait(.75)
                    continue
                if ocr is None:
                    from ninja3_assistant.activities.intelligence.ocr import WindowsOcr
                    ocr = WindowsOcr()
                result = ocr.read(frame,stop=control.stop)
                if generation != control.generation() or control.stop.is_set():
                    continue
                if result is not None:
                    observed = DailyScreen(frame,result)
                    if '深渊冒险' in observed.text((100, 14, 353, 89)):
                        if target == 'abyss':
                            return True
                        press('abyss.lobby', 'back', (52, 56),
                              lambda fresh: same_region(frame, fresh, (16, 20, 343, 94)))
                        control.stop.wait(.75)
                        continue
                    if target == 'abyss' and (
                            (result_panel(frame) and '任意位置继续' in observed.text((505, 612, 780, 665))) or
                            (relay_prompt(frame) and '忍者接力' in observed.text((490, 40, 790, 101)))):
                        return True
                    chore = daily.chore_navigation(observed)
                    if chore:
                        daily_page,daily_back = chore
                        observed.page = daily_page
                        chore_screen = observed
                    else:
                        observed = daily.observe(frame,result)
                        if observed.page in ('shop_dialog','bounty_dialog'):
                            daily_page,daily_back = observed.page,(1100,90)
                            purchase_screen = observed
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
            page, point, operation = None, None, 'back'
            check = None
            if screen.kind == 'invite':
                page, point, operation = 'game.invite', screen.point, 'decline_invite'
                check = lambda fresh: activity.detect(fresh) == screen
            elif screen.kind == 'guide':
                page, point, operation = 'intelligence.guide', screen.point, 'advance_guide'
                check = lambda fresh: activity.detect(fresh) == screen
            elif screen.kind == 'network_retry':
                if control.now()-last_network_retry >= 8:
                    page, point, operation = 'game.network_retry', screen.point, 'retry_network'
                    check = lambda fresh: activity.detect(fresh) == screen
            elif screen.kind == 'battle' or race_active:
                point = None
            elif daily_page == 'login' or screen.kind == 'login':
                page, operation = 'game.login', 'login'
                point = daily_back if daily_page == 'login' else screen.point
                check = (lambda fresh: daily.navigation(fresh) == (daily_page, point)) if daily_page == 'login' else (
                    lambda fresh: activity.detect(fresh) == screen)
            elif screen.kind == 'tasks':
                page, point = 'intelligence.tasks', (1120, 111)
                check = lambda fresh: activity.detect(fresh).kind == screen.kind
            elif kind == 'tasks':
                from ninja3_assistant.activities.weekend.runner import CLOSE_TASKS_CURRENT
                page, point = 'weekend.tasks', (CLOSE_TASKS_CURRENT if race.layout == 'current'
                                               else (1190, 400))
                check = lambda fresh: race.detect(fresh) == kind
            elif screen.kind == 'event_home':
                page, point = 'intelligence.lobby', (50, 45)
                check = lambda fresh: activity.detect(fresh).kind == screen.kind
            elif kind == 'home':
                page, point = 'weekend.lobby', (65, 45)
                check = lambda fresh: race.detect(fresh) == kind
            elif hall or kind == 'hall':
                page, point = 'game.hall', (50, 45)
                check = (lambda fresh: race.detect(fresh) == kind) if kind == 'hall' else (
                    lambda fresh: bool(activity.navigation.match(cv2.cvtColor(fresh, cv2.COLOR_RGB2GRAY),
                        'startup-hall-title', 80, 0, 300, 110, threshold=.93, max_error=30)))
            elif daily_page:
                page, point = page_id('daily', daily_page), daily_back
                check = lambda fresh: daily.navigation(fresh) == (daily_page, point)
            elif bounty_page:
                page, point = page_id('bounty', bounty_page), bounty_back
                check = lambda fresh: bounty.navigation(fresh) == (bounty_page, point)
            if point:
                guarded_screen = purchase_screen or chore_screen
                if guarded_screen is not None:
                    if purchase_screen is not None:
                        guard = (510,632,780,704) if daily_page == 'bounty_dialog' else (535,505,800,570)
                    else:
                        x, y = point
                        guard = (max(0,x-25),max(0,y-25),min(1280,x+25),min(720,y+25))
                    decision = DailyDecision('tap', point, '返回主页，继续切换任务', guard=guard, operation=operation)
                    check = lambda fresh: daily.current_action(guarded_screen, decision, fresh)
                if press(page, operation, point, check) and screen.kind == 'network_retry':
                    last_network_retry = control.now()
            control.stop.wait(.75)
        return False
    finally:
        control.set_release_input(None)
        if native:
            native.close()
        device.close()
