"""Navigate to the arena and collect its available settlement rewards."""
import re


ARENA_TITLE = r'[竞竟][技枝]场'
QUIZ_ENTRY = r'老板娘(?:[问問向间]|[i丨]司)[答荅]|老板娘答题'
RANK_REWARD = r'(?:每日|昨日|排名|排行(?:榜)?|分组|组别|每周)奖励|(?:每日|排名|竞技场)结算'
NO_REWARD = r'(?:已经?|均已|全部已|已全部)领取|暂无(?:可领取的?)?奖励|没有可领取(?:的)?奖励|未参与竞技场|未获得排名|未达到领取条件'
TOP_LEFT = (90, 0, 430, 105)
MENU_CARD = (275, 105, 605, 250)
RANK_STRIPE = (140, 410, 875, 480)
HISTORY_TITLE = (210, 52, 360, 106)
HISTORY_BUTTON = (530, 632, 750, 699)
TROPHY_TITLE = (580, 208, 752, 275)
TROPHY_BUTTON = (535, 588, 755, 650)
CURRENCY_BUTTON = (525, 604, 756, 682)
BACK = (60, 50)
RULES_CLOSE = (1150, 510)


def previous_currency_point(frame, vision):
    # Full-frame OCR can be blank on this animated reward notice.
    if (vision is not None and
            vision.match(frame, 'arena-previous-coins-title', (451, 74, 829, 150), threshold=.94) and
            vision.match(frame, 'arena-previous-coins-label', (571, 406, 715, 473), threshold=.94)):
        return vision.match(frame, 'confirm', CURRENCY_BUTTON)
    return None


def observe(screen, vision=None):
    if point := previous_currency_point(screen.frame, vision):
        return 'arena-currency', point
    if (screen.hits('奖杯结算', TROPHY_TITLE) and
            screen.hits(r'[\u3400-\u9fff]{2,4}[组組]', (540, 399, 680, 454)) and
            screen.hits(r'第\d{1,3}名', (690, 399, 827, 454))):
        confirm = screen.hits('确定', TROPHY_BUTTON)
        if len(confirm) == 1:
            return 'arena-trophy', confirm[0].point
    if (screen.hits('上次成绩', HISTORY_TITLE) and
            screen.hits(r'[\u3400-\u9fff]{2,4}[组組]', (570, 49, 726, 108))):
        confirm = screen.hits('确定', HISTORY_BUTTON)
        if len(confirm) == 1:
            return 'arena-history', confirm[0].point
    if screen.hits('排名奖励规则', (645, 130, 1050, 240)):
        return 'arena-rules', RULES_CLOSE
    if (screen.hits(r'(?:竞技场|排名|每日|每周|分组)结算', (200, 25, 1080, 260)) and
            re.search(rf'{NO_REWARD}|竞技币|排名|领取', screen.text((180, 150, 1090, 650)))):
        return 'arena-reward', close_reward(screen)
    if ((lobby_entry(screen) and screen.hits(r'[竞竟]技商店', (1150, 63, 1280, 115))) or
            (screen.hits(r'观看本场最高分', (370, 105, 670, 165)) and
             screen.hits(r'百[强強]榜', (0, 320, 125, 420)))):
        return 'arena', BACK
    if menu_entry(screen, vision):
        return 'arena-menu', (75, 55)
    return None


def tap(point, status, guard=None):
    from ninja3_assistant.activities.daily.flow import DailyDecision

    if guard is None:
        x, y = point
        guard = (max(0, x - 50), max(0, y - 25), min(1280, x + 50), min(720, y + 25))
    return DailyDecision('tap', point, status, guard=guard)


def menu_entry(screen, vision):
    if screen.hits(r'决[斗鬥]场', TOP_LEFT):
        hits = screen.hits(ARENA_TITLE, MENU_CARD)
        if len(hits) == 1:
            return hits[0].point
    if vision is not None and vision.match(screen.frame, 'duel-arena-menu', MENU_CARD):
        return (430, 400)
    return None


def navigate(screen, vision):
    if screen.page == 'arena-history' and screen.point:
        return tap(screen.point, '关闭竞技场上次成绩', HISTORY_BUTTON)
    if screen.page == 'arena-trophy' and screen.point:
        return tap(screen.point, '关闭竞技场奖杯结算', TROPHY_BUTTON)
    if screen.page == 'arena-currency' and screen.point:
        return tap(screen.point, '确认上次未领取竞技币', CURRENCY_BUTTON)
    if screen.page == 'home':
        hits = screen.hits(rf'{ARENA_TITLE}|决[斗鬥]场', (970, 525, 1280, 700))
        point = hits[0].point if hits else (1110, 625)
        return tap(point, '从主页进入决斗场', (985, 550, 1245, 682))
    if screen.page == 'arena-menu':
        point = menu_entry(screen, vision)
        if point:
            return tap(point, '进入竞技场', MENU_CARD)
    return None


def lobby_entry(screen):
    hits = screen.hits(QUIZ_ENTRY, (995, 63, 1150, 115))
    return (1088, 48) if len(hits) == 1 else None


def close_reward(screen):
    hits = screen.hits(r'关闭|返回|确定', (160, 480, 1120, 720))
    return hits[0].point if len(hits) == 1 else None


class ArenaReward:
    def __init__(self):
        self.claimed = False
        self.pending = False
        self.finished = False

    def step(self, screen, vision):
        from ninja3_assistant.activities.daily.flow import DailyDecision

        if self.finished:
            if screen.page == 'home':
                return DailyDecision('done', status='竞技场结算奖励已检查')
            if screen.page in ('arena', 'arena-menu'):
                return tap(screen.point, '返回主页，继续每日清理')
        if screen.page in ('reward', 'reward_continue') and self.pending:
            self.claimed = True
            self.pending = False
            return None  # Use the shared reward dismissal.
        if screen.page == 'arena-reward':
            region = (180, 125, 1090, 670)
            if re.search(NO_REWARD, screen.text(region)) or self.claimed:
                self.pending = False
                self.claimed = True
                close = close_reward(screen)
                if close:
                    return tap(close, '关闭竞技场结算奖励')
            else:
                claim = screen.hits(r'(?<![已未])领取(?:奖励)?|确认领取', (200, 490, 1070, 680))
                if len(claim) == 1 and not self.pending:
                    return tap(claim[0].point, '领取竞技场结算奖励')
            return DailyDecision(status='等待确认竞技场结算奖励状态')
        if screen.page == 'arena-rules':
            return tap(RULES_CLOSE, '关闭竞技场排名奖励规则', (640, 135, 1050, 230))
        if screen.page == 'arena':
            stripe = screen.text(RANK_STRIPE)
            if re.search(RANK_REWARD, stripe):
                buttons = screen.hits(r'(?<![已未])领取(?:奖励)?', (620, 410, 875, 480))
                if len(buttons) == 1 and not self.pending and not re.search(NO_REWARD, stripe):
                    return tap(buttons[0].point, '领取竞技场结算奖励', RANK_STRIPE)
            countdown = (re.search(RANK_REWARD, stripe) and
                         re.search(r'\d+(?:小[时時]|分钟?|秒)', screen.text((670, 410, 855, 480))))
            unranked = screen.hits('50名玩家共同参加小组赛', (280, 415, 1000, 500))
            if self.claimed or countdown or unranked or re.search(NO_REWARD, stripe):
                self.finished = True
                return tap(BACK, '返回主页，继续每日清理')
            return DailyDecision(status='等待确认竞技场结算奖励入口', defer_on_timeout=True)
        if screen.page.startswith('landlady-'):
            return None  # Leave quiz screens through shared daily navigation.
        return navigate(screen, vision)

    def completed(self, decision, screen):
        if decision.status == '领取竞技场结算奖励':
            self.pending = True
