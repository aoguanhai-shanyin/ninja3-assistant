"""Navigation controls backed by the game's existing menu recognizers."""

from functools import cached_property

import cv2
import numpy as np

from ninja3_assistant.game_ui import PageAction, PageObservation, Transition
from ninja3_assistant.vision.game_update import game_update_banner


HOME = 'game.home'
HALL = 'game.hall'

DAILY_PARENTS = {
    'activities': HOME, 'monthly': HOME, 'anniversary': HOME, 'ten_signin': 'daily.summon',
    'rank_preview': 'daily.rank', 'rank': HOME, 'trial': HALL, 'prayer': 'daily.family',
    'family': HOME, 'curse': 'daily.vault', 'weapon': 'daily.vault', 'vault': HOME,
    'summon': HOME, 'house': HOME, 'shop': HOME, 'limited_shop': HOME,
    'daily_shop': HOME, 'bounty_shop': HOME, 'no_family': HOME,
    'arena_menu': HOME, 'arena': 'daily.arena_menu', 'arena_rules': 'daily.arena',
    'arena_reward': 'daily.arena', 'arena_history': 'daily.arena', 'arena_trophy': 'daily.arena',
    'landlady_quiz': 'daily.arena', 'pet_main': HOME, 'pet_routes': 'daily.pet_main',
    'friend_list': HOME, 'friend_rice_inbox': 'daily.friend_list',
}

XUMI_PARENTS = {
    'lobby': HALL, 'tasks': 'xumi.lobby', 'fangcun_setup': 'xumi.lobby',
    'daqian_setup': 'xumi.lobby', 'roster': 'xumi.fangcun_team',
    'fangcun_team': 'xumi.fangcun_setup', 'daqian_team': 'xumi.daqian_setup',
    'equipment': 'xumi.daqian_team', 'equipment_detail': 'xumi.equipment',
}

ROUTES = (
    Transition(HOME, HALL, 'open_hall'),
    Transition(HALL, HOME, 'back'),
    Transition(HALL, 'intelligence.lobby', 'open_intelligence', 'find_intelligence', 3),
    Transition('intelligence.guide', 'intelligence.lobby', 'advance_guide', attempts=8),
    Transition('intelligence.lobby', HALL, 'back'),
    Transition('intelligence.lobby', 'intelligence.tasks', 'open_tasks'),
    Transition('intelligence.tasks', 'intelligence.lobby', 'back'),
    Transition(HALL, 'weekend.lobby', 'open_weekend', 'find_weekend', 3),
    Transition('weekend.lobby', HALL, 'back'),
    Transition('weekend.lobby', 'weekend.tasks', 'open_tasks'),
    Transition('weekend.tasks', 'weekend.lobby', 'back'),
    Transition(HALL, 'xumi.lobby', 'open_xumi', 'find_xumi', 6, reverse_search='find_xumi_left'),
    Transition('xumi.guide', 'xumi.lobby', 'advance_guide', attempts=8),
    Transition('xumi.lobby', 'xumi.tasks', 'open_tasks'),
    Transition('xumi.lobby', 'xumi.fangcun_setup', 'open_fangcun'),
    Transition('xumi.lobby', 'xumi.daqian_setup', 'open_daqian'),
    Transition(HALL, 'bounty.hub', 'open_bounty', 'find_bounty', 6, reverse_search='find_bounty_left'),
    Transition('bounty.hub', HALL, 'back'),
    Transition('bounty.hub', 'bounty.daily', 'open_daily'),
    Transition('bounty.hub', 'bounty.order', 'open_order'),
    Transition('bounty.hub', 'bounty.tasks', 'open_tasks'),
    Transition('bounty.hub', 'reverse_bounty.list', 'open_reverse'),
    Transition('bounty.daily', 'bounty.hub', 'back'),
    Transition('bounty.daily', 'bounty.tasks', 'open_tasks'),
    Transition('bounty.daily', 'bounty.grade', 'open_grade'),
    Transition('bounty.daily', 'bounty.free', 'open_quota'),
    Transition('bounty.order', 'bounty.hub', 'back'),
    Transition('bounty.tasks', 'bounty.hub', 'back'),
    Transition('bounty.grade', 'bounty.daily', 'back'),
    Transition('bounty.free', 'bounty.daily', 'back'),
    Transition('bounty.locked', 'bounty.hub', 'back'),
    Transition(HOME, 'daily.rank', 'open_rank'),
    Transition(HOME, 'daily.house', 'open_house'),
    Transition(HOME, 'daily.monthly', 'open_monthly'),
    Transition(HOME, 'daily.anniversary', 'open_anniversary'),
    Transition(HOME, 'daily.summon', 'open_summon'),
    Transition(HOME, 'daily.shop', 'open_shop'),
    Transition(HOME, 'daily.family', 'open_family'),
    Transition(HOME, 'daily.arena_menu', 'open_arena_menu'),
    Transition('daily.arena_menu', 'daily.arena', 'open_arena'),
    Transition('daily.arena', 'daily.landlady_quiz', 'open_quiz'),
    Transition('daily.arena', 'daily.arena_reward', 'open_arena_reward'),
    Transition(HOME, 'daily.pet_main', 'open_pet'),
    Transition('daily.pet_main', 'daily.pet_routes', 'open_patrol_routes'),
    Transition(HOME, 'daily.friend_list', 'open_friends'),
    Transition('daily.friend_guide', 'daily.friend_list', 'dismiss_guide'),
    Transition('daily.friend_list', 'daily.friend_rice_inbox', 'open_rice_inbox'),
    Transition(HALL, 'daily.trial', 'open_trial', 'find_trial', 6),
    Transition('daily.family', 'daily.prayer', 'open_prayer'),
    Transition('daily.activities', 'daily.monthly', 'open_monthly'),
    Transition('daily.summon', 'daily.ten_signin', 'open_ten_signin'),
    Transition('daily.summon', 'daily.weapon', 'open_weapon', 'scroll_summons', 3),
    Transition('daily.vault', 'daily.weapon', 'open_weapon', 'scroll_summons', 3),
    Transition('daily.summon', 'daily.curse', 'open_curse'),
    Transition('daily.vault', 'daily.curse', 'open_curse'),
    Transition('daily.shop', 'daily.limited_shop', 'open_limited_shop'),
    Transition('daily.limited_shop', 'daily.daily_shop', 'open_daily_shop'),
    Transition('daily.limited_shop', 'daily.bounty_shop', 'open_bounty_shop'),
    *(Transition('daily.' + page, parent, 'back') for page, parent in DAILY_PARENTS.items()),
    *(Transition('xumi.' + page, parent, 'back') for page, parent in XUMI_PARENTS.items()),
)
NAVIGABLE_PAGES = frozenset(page for edge in ROUTES for page in (edge.source, edge.target))


def page_id(area, kind):
    if not kind or kind == 'unknown':
        return None
    if kind in ('home', 'hall', 'hall_search', 'login', 'invite', 'network_retry', 'game_update'):
        return 'game.' + {'hall_search': 'hall', 'game_update': 'update'}.get(kind, kind)
    if area == 'bounty':
        kind = kind.removeprefix('bounty_')
    if area == 'intelligence' and kind == 'event_home':
        kind = 'lobby'
    return area + '.' + kind.replace('-', '_')


class MenuVision:
    def __init__(self, stop=None):
        self.stop = stop

    @cached_property
    def daily(self):
        from ninja3_assistant.activities.daily.vision import DailyVision
        return DailyVision()

    @cached_property
    def bounty(self):
        from ninja3_assistant.activities.bounty.daily_vision import DailyBountyVision
        return DailyBountyVision()

    @cached_property
    def intelligence(self):
        from ninja3_assistant.activities.intelligence.vision import ActivityVision
        return ActivityVision()

    @cached_property
    def weekend(self):
        from ninja3_assistant.activities.weekend.runner import RaceVision
        return RaceVision()

    @cached_property
    def xumi(self):
        from ninja3_assistant.activities.xumi.vision import XumiVision
        return XumiVision(self.daily)

    @cached_property
    def ocr(self):
        from ninja3_assistant.activities.intelligence.ocr import WindowsOcr
        return WindowsOcr()

    def __call__(self, frame, generation, captured_at):
        page, points, detail, blocking = self._inspect(frame)

        def guard(fresh, name, point, end, ocr_only):
            if fresh.shape != frame.shape:
                return False
            current, controls, _, blocked = self._inspect(fresh, use_ocr=False)
            if blocked or (current is not None and current != page):
                return False
            if name == 'open_arena_reward':
                from ninja3_assistant.game_ui.arena import rank_reward_entry

                return rank_reward_entry(fresh) == point
            if controls.get(name, ())[:2] == (point, end):
                return True
            if not ocr_only or name in controls:
                return False
            # OCR-only controls retain their location only while their local
            # pixels and the page's header area still agree with the observation.
            if end is not None:
                return False
            x, y = point
            for area in ((max(0, x-27), max(0, y-18), min(1280, x+28), min(720, y+19)),
                         (75, 14, 300, 75)):
                x1, y1, x2, y2 = area
                diff = cv2.absdiff(frame[y1:y2, x1:x2], fresh[y1:y2, x1:x2])
                if float(np.mean(diff)) >= 8:
                    return False
            return True

        controls = {
            name: PageAction(name, point, end,
                             guard=lambda fresh, name=name, point=point, end=end, ocr_only=ocr_only:
                             guard(fresh, name, point, end, ocr_only))
            for name, (point, end, ocr_only) in points.items()
        }
        if page in ('daily.house', 'daily.house_continue') and detail is not None:
            from ninja3_assistant.game_ui.house import house_view

            detail = house_view(self.daily, detail)
            controls.update(detail.controls)
        if page == 'daily.arena_reward' and detail is not None:
            from ninja3_assistant.game_ui.arena import ArenaRewardPage

            detail = ArenaRewardPage.observe(detail)
            if detail is not None:
                blocking = False
                if action := detail.claim_action(frame):
                    controls[action.name] = action
        return PageObservation(page, frame, generation, captured_at, controls, detail, blocking)

    def _inspect(self, frame, *, use_ocr=True):
        points = {}

        def add(name, point, end=None, *, ocr_only=False):
            if point is not None:
                points[name] = (point, end, ocr_only)

        def page(area, kind, detail=None, blocking=False):
            return page_id(area, kind), points, detail, blocking

        if frame.shape != (720, 1280, 3):
            return None, points, None, False
        if game_update_banner(frame):
            return page('game', 'game_update', blocking=True)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.daily.invitations.detect(gray):
            return page('game', 'invite', blocking=True)
        intelligence = self.intelligence.detect(frame)
        if intelligence.kind == 'battle':
            return page('intelligence', 'battle', intelligence, True)
        weekend = self.weekend.detect(frame, gray)
        if weekend in ('dash', 'race', 'respawn', 'ready', 'finish', 'result', 'standings', 'reward'):
            return page('weekend', weekend, blocking=True)
        xumi = self.xumi.scene(frame)
        if xumi.kind not in ('home', 'hall', 'unknown'):
            if xumi.kind == 'guide':
                add('advance_guide', xumi.point)
            elif xumi.kind == 'lobby':
                add('open_tasks', xumi.point)
                add('open_fangcun', xumi.fangcun)
                add('open_daqian', xumi.daqian)
            kind = xumi.navigation_kind
            if kind in XUMI_PARENTS:
                add('back', xumi.back)
            return page('xumi', kind, xumi, xumi.kind.startswith('result') or xumi.kind in (
                'relay','score_confirm','equipment_confirm','own_roster','equipment_target',
            ))
        if intelligence.kind in ('tasks', 'event_home', 'guide', 'reward'):
            from ninja3_assistant.activities.intelligence.flow import CLOSE_TASKS, OPEN_TASKS

            if intelligence.kind == 'tasks':
                add('back', CLOSE_TASKS)
            elif intelligence.kind == 'event_home':
                add('back', (50, 45))
                add('open_tasks', OPEN_TASKS)
            elif intelligence.kind == 'guide':
                add('advance_guide', intelligence.point)
            return page('intelligence', intelligence.kind, intelligence, intelligence.kind == 'reward')
        if weekend == 'tasks':
            from ninja3_assistant.activities.weekend.runner import CLOSE_TASKS, CLOSE_TASKS_CURRENT
            add('back', CLOSE_TASKS_CURRENT if self.weekend.layout == 'current' else CLOSE_TASKS)
            return page('weekend', 'tasks')
        if weekend == 'home':
            from ninja3_assistant.activities.weekend.runner import (
                ENTER_ACTIVITY, OPEN_TASKS, OPEN_TASKS_CURRENT,
            )
            add('back', (65, 45))
            if self.weekend.intro:
                add('enter_weekend', ENTER_ACTIVITY)
            else:
                add('open_tasks', OPEN_TASKS_CURRENT if self.weekend.layout == 'current' else OPEN_TASKS)
            return page('weekend', 'lobby')

        daily, back = self.daily.navigation(frame)
        bounty, bounty_back = self.bounty.navigation(frame)
        if bounty and bounty.startswith('bounty_'):
            if bounty == 'bounty_hub':
                add('open_daily', (360, 400))
                add('open_order', (930, 400))
                add('open_tasks', (155, 675))
                add('open_reverse', self.intelligence.navigation.match(gray, 'menu-reverse', 900, 410, 320, 130))
            elif bounty == 'bounty_daily':
                add('open_tasks', (330, 670))
                add('open_grade', (730, 670))
                add('open_quota', (380, 172))
            identifier = page_id('bounty', bounty)
            if identifier in NAVIGABLE_PAGES:
                add('back', bounty_back)
            return page('bounty', bounty, blocking=bounty in ('bounty_room', 'bounty_recruit', 'bounty_leave'))
        if bounty == 'result' or daily in ('reward', 'reward_continue'):
            return page('bounty' if bounty == 'result' else 'daily', bounty if bounty == 'result' else daily,
                        blocking=True)
        if daily == 'login' or intelligence.kind in ('login', 'network_retry'):
            return page('game', 'login' if daily == 'login' else intelligence.kind, blocking=True)
        if daily == 'hall' or intelligence.kind in ('hall', 'hall_search') or weekend == 'hall' or xumi.kind == 'hall':
            from ninja3_assistant.activities.bounty.daily_menu import HALL_SEARCH_LEFT, HALL_SEARCH_RIGHT
            from ninja3_assistant.activities.intelligence.flow import HALL_SEARCH as INTELLIGENCE_SEARCH
            from ninja3_assistant.activities.weekend.runner import HALL_ENTRY, HALL_ENTRY_REGION, HALL_SEARCH

            add('back', back if daily == 'hall' else xumi.back if xumi.kind == 'hall' else (50, 45))
            add('open_intelligence', intelligence.point if intelligence.kind == 'hall' else None)
            add('find_intelligence', *INTELLIGENCE_SEARCH)
            if self.weekend.match_any(frame, ('entry', 'entry-current'), HALL_ENTRY_REGION):
                add('open_weekend', HALL_ENTRY)
            add('find_weekend', *HALL_SEARCH)
            if xumi.kind == 'hall':
                add('open_xumi', xumi.point)
                add('find_xumi', (1080, 400), (280, 400))
                add('find_xumi_left', (280, 400), (1080, 400))
            add('open_bounty', self.bounty.hall_entry(frame))
            add('find_bounty', HALL_SEARCH_RIGHT, HALL_SEARCH_LEFT)
            add('find_bounty_left', HALL_SEARCH_LEFT, HALL_SEARCH_RIGHT)
            add('find_trial', (1120, 400), (320, 400))
            if use_ocr:
                from ninja3_assistant.activities.daily.vision import DailyScreen
                screen = DailyScreen(frame, self.ocr.read(frame, stop=self.stop) or {}, 'hall')
                add('open_trial', self.daily.trial_entry(screen), ocr_only=True)
            return page('game', 'hall')

        if daily == 'home' or intelligence.kind == 'home' or xumi.kind == 'home':
            from ninja3_assistant.activities.daily.friend_rice import FRIEND_HOME
            from ninja3_assistant.activities.daily.pet_feed import home_entry

            summon = self.daily.home(frame)
            add('open_hall', intelligence.point if intelligence.kind == 'home' else (865, 600))
            add('open_rank', (65, 50))
            add('open_family', (1010, 450))
            add('open_monthly', self.daily.home_monthly(frame))
            add('open_anniversary', self.daily.match(frame, 'home-thanks', (80, 130, 330, 600), threshold=.94))
            add('open_summon', summon)
            add('open_shop', (summon[0]-104, summon[1]) if summon else None)
            add('open_friends', self.daily.match(frame, 'friend-home', FRIEND_HOME))
            if use_ocr:
                screen = self.daily.observe(frame, self.ocr.read(frame, stop=self.stop) or {})
                if screen.page in ('shop_dialog', 'bounty_dialog', 'house_continue'):
                    points.clear()
                    return page('daily', screen.page, screen, True)
                if screen.page == 'home':
                    add('open_house', screen.home_points.get('house'), ocr_only=True)
                    add('open_arena_menu', screen.home_points.get('arena_reward'), ocr_only=True)
                    add('open_pet', home_entry(screen), ocr_only=True)
                    if 'open_anniversary' not in points:
                        add('open_anniversary', screen.home_points.get('limited'), ocr_only=True)
            return page('game', 'home')

        if daily in ('summon', 'shop', 'limited-shop'):
            if daily == 'summon':
                add('open_ten_signin', self.daily.match(frame, 'ten-signin-entry', (0, 600, 580, 715), threshold=.94))
            elif daily == 'shop':
                add('open_limited_shop', (1000, 690))
            else:
                add('open_daily_shop', (1000, 690))
                add('open_bounty_shop', (1190, 378))
        if daily == 'family':
            add('open_prayer', self.daily.family_entry(frame))
        if daily in ('summon','vault'):
            weapon = self.daily.weapon_entry(frame)
            add('open_weapon', weapon)
            if not weapon:
                add('scroll_summons', (150,545), (150,325))
            add('open_curse', (1215,365) if daily == 'vault' else (140,462))
        if daily == 'friend-guide':
            add('dismiss_guide', back)
        if daily == 'friend-list':
            from ninja3_assistant.activities.daily.friend_rice import INBOX_POINT
            add('open_rice_inbox', INBOX_POINT)
        if daily == 'house' and use_ocr:
            from ninja3_assistant.game_ui.house import observe_house

            screen = observe_house(self.daily, frame, self.ocr, self.stop)
            if screen is None:
                return None, points, None, False
            if screen.page == 'house_continue':
                return page('daily', screen.page, screen, True)
            if screen.page == 'house':
                add('back', back)
                return page('daily', screen.page, screen)
            return page('daily', screen.page, blocking=True)
        if daily is not None:
            if daily.replace('-', '_') in DAILY_PARENTS:
                add('back', back)
            return page('daily', daily, blocking=not points)

        nav = self.intelligence.navigation.detect(frame)
        if nav.board or nav.ready or nav.prepared or nav.result or nav.battle_active or nav.battle_loading or nav.training:
            kind = 'list' if nav.board else 'room' if nav.ready or nav.prepared else 'result' if nav.result else 'battle'
            return page('reverse_bounty', kind, nav, kind != 'list')
        if use_ocr:
            from ninja3_assistant.activities.daily.arena import lobby_entry, menu_entry
            from ninja3_assistant.activities.daily.pet_feed import patrol_entry
            screen = self.daily.observe(frame, self.ocr.read(frame, stop=self.stop) or {})
            if screen.page == 'activities':
                add('back', (1238, 48), ocr_only=True)
                add('open_monthly', (135, 180), ocr_only=True)
            elif screen.page == 'arena-menu':
                add('open_arena', menu_entry(screen, self.daily), ocr_only=True)
            elif screen.page == 'arena':
                from ninja3_assistant.game_ui.arena import rank_reward_entry

                add('open_quiz', lobby_entry(screen), ocr_only=True)
                add('open_arena_reward', rank_reward_entry(frame))
            elif screen.page == 'pet-main':
                add('open_patrol_routes', patrol_entry(screen), ocr_only=True)
            if screen.page.replace('-', '_') in DAILY_PARENTS:
                add('back', screen.point, ocr_only=True)
            identifier = page_id('daily', screen.page)
            return identifier, points, screen, bool(identifier and not points)
        return None, points, None, False
