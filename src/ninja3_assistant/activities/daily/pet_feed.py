"""Recognize and feed one daily spirit-beast patrol using the selected reward."""


_ROUTES = {'exp': '雾之湖', 'coins': '忍村', 'hammers': '雷王山'}
_TITLE = (110, 15, 345, 90)
_MAIN_ACTION = (915, 625, 1215, 716)
_ROUTE_NAMES = (0, 135, 135, 555)
_EXTRA_FOOD = (1155, 130, 1265, 535)
_REWARD_PREVIEW = (239, 595, 347, 704)
_START_BUTTON = (995, 610, 1250, 690)
_CLAIM = '领取巡逻奖励'
_START = '开始通灵兽巡逻'
_SELECT = '选择巡逻路线 '
_SCROLL = '寻找通灵兽巡逻路线'


def observe(screen):
    if screen.hits('通灵兽巡逻', _TITLE):
        return 'pet-routes', (66, 53)
    if (screen.hits('今日已获得', (790, 15, 955, 70)) and
            (screen.hits('巡逻', _MAIN_ACTION) or _patrol_running(screen))):
        return 'pet-main', (66, 53)
    return None


def _patrol_running(screen):
    if screen.hits('巡逻结束|巡逻中', (820, 625, 1040, 716)):
        return True
    # Full-frame OCR can miss the stylized timer and 取消 text. The active
    # timer occupies two white lines and shifts the blue button to the right.
    timer = screen.frame[647:706, 885:1024]
    cancel = screen.frame[650:692, 1220:1245]
    return ((timer.min(axis=2) > 220).mean() > .05 and
            cancel[:, :, 1].mean() > 140 and cancel[:, :, 2].mean() > 200)


def _home_patrol_running(screen):
    # The countdown replaces the pet label while its patrol is running.
    if screen.hits(r'\d{1,2}小时\d{1,2}分|\d{1,2}分钟|\d{1,2}秒', (111, 296, 185, 323)):
        return True
    clock = screen.frame[302:315, 97:108].astype('int16')
    red, green, blue = clock.transpose(2, 0, 1)
    badge = screen.frame[301:318, 111:181]
    return (((blue > 170) & (green > 80) & (blue-red > 55)).mean() > .5 and
            (badge.max(axis=2) < 35).mean() > .5 and (badge.min(axis=2) > 200).mean() > .08)


def home_entry(screen):
    point = screen.home_points.get('pet_feed')
    # The stylized home label often reads only 通 at its fixed slot.
    return (143, 280) if point is None and screen.hits('通', (108, 297, 132, 325)) else point


def patrol_entry(screen):
    patrol = screen.hits('巡逻', _MAIN_ACTION)
    return patrol[0].point if patrol and screen.text(_MAIN_ACTION) == '巡逻' else None


def _empty_food_slot(frame, label):
    # The unfilled extra-food slot shows a solid white plus on grey.
    y = label.point[1] + 63
    white = frame[y-36:y+37, 1186:1227].min(axis=2) > 225
    columns, rows = white.sum(axis=0), white.sum(axis=1)
    x, y = columns.argmax(), rows.argmax()
    return columns[x] >= 33 and rows[y] >= 33 and white[y, x]


def extra_food_still_empty(screen, fresh):
    extra = screen.hits('[额額]外食物', _EXTRA_FOOD)
    return bool(extra and _empty_food_slot(fresh, extra[0]))


class PetFeed:
    def __init__(self, reward):
        self.route = _ROUTES[reward]
        self.claim_sent = False
        self.route_sent = False
        self.start_sent = False
        self.scrolls = 0

    def step(self, screen, vision):
        from ninja3_assistant.activities.daily.flow import DailyDecision

        if screen.page == 'home':
            if _home_patrol_running(screen):
                return DailyDecision('done', status='通灵兽正在巡逻，保留当前任务')
            point = home_entry(screen)
            if point is not None:
                x, y = point
                return DailyDecision.tap('open_pet', point, '进入通灵兽',
                                         guard=(x-25, y-25, x+25, y+25))
            return DailyDecision(status='寻找主页通灵兽入口')
        if screen.page == 'pet-main':
            if _patrol_running(screen):
                status = (f'通灵兽已喂养，开始巡逻 {self.route}' if self.start_sent else
                          '通灵兽正在巡逻，保留当前任务')
                return DailyDecision('done', status=status)
            claim = screen.hits(_CLAIM, _MAIN_ACTION)
            if claim and not self.claim_sent:
                return DailyDecision.tap('claim_patrol', claim[0].point, _CLAIM, guard=_MAIN_ACTION)
            if claim or self.start_sent:
                return DailyDecision(status='等待通灵兽巡逻状态更新')
            if point := patrol_entry(screen):
                return DailyDecision.tap('open_patrol_routes', point, '打开通灵兽巡逻路线',
                                         guard=_MAIN_ACTION)
            return DailyDecision(status='确认通灵兽巡逻状态')
        if screen.page != 'pet-routes':
            return None
        if self.start_sent:
            return DailyDecision(status='已提交喂养，等待巡逻开始')
        if screen.hits('兵粮丸不足', (320, 240, 980, 560)):
            return DailyDecision('unavailable', status='特制兵粮丸不足')

        extra = screen.hits('[额額]外食物', _EXTRA_FOOD)
        route = screen.hits(self.route, _ROUTE_NAMES)
        if not route and self.route_sent and self.route == '雾之湖' and extra:
            # The selected row moves up and its first character can sit under the floating chest.
            y = extra[0].point[1]
            if (screen.hits('2小时', (473, y+40, 575, y+105)) and
                    vision.match(screen.frame, 'pet-exp-reward', _REWARD_PREVIEW, threshold=.96)):
                route = screen.hits('之湖', (40, y+12, 135, y+73))
        if not route:
            if self.scrolls < 2:
                start, end = ((620, 225), (620, 530)) if self.route == '忍村' else ((620, 530), (620, 230))
                return DailyDecision.swipe('scroll_patrol_routes', start, _SCROLL, end, guard=_TITLE)
            return DailyDecision('unavailable', status=f'未找到可选的巡逻路线 {self.route}')
        _, y = route[0].point
        if not extra or not 12 <= y-extra[0].point[1] <= 72:
            if screen.hits('未解锁|解锁后', (0, y-35, 1250, y+75)):
                return DailyDecision('unavailable', status=f'巡逻路线 {self.route} 尚未解锁')
            # The floating chest can intercept taps over the left route name.
            return DailyDecision.tap('select_patrol_route', (620, y+20), _SELECT+self.route,
                                     guard=(0, y-25, 140, y+25))
        if not _empty_food_slot(screen.frame, extra[0]):
            return DailyDecision(status='等待确认额外食物栏为空')
        start = screen.hits('开始巡逻', _START_BUTTON)
        if start:
            return DailyDecision.tap('start_patrol', start[0].point, _START, guard=_REWARD_PREVIEW)
        return DailyDecision(status='等待通灵兽巡逻开始按钮')

    def completed(self, decision, screen):
        if decision.operation == 'claim_patrol':
            self.claim_sent = True
        elif decision.operation == 'select_patrol_route':
            self.route_sent = True
        elif decision.operation == 'start_patrol':
            self.start_sent = True
        elif decision.operation == 'scroll_patrol_routes':
            self.scrolls += 1
