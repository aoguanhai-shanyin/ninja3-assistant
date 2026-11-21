"""Friend rice receipts and a durable cap on gifts sent by the assistant."""
from datetime import date
import json
import re

import cv2
import numpy as np

from ninja3_assistant.activities.daily.flow import DailyDecision
from ninja3_assistant.activities.intelligence.flow import daily_key


LIMIT = 15
FRIEND_HOME = (10,320,94,418)
FRIEND_BACK = (55,46)
INBOX_BACK = (640,700)
INBOX_ENTRY = (1170,20,1268,105)
INBOX_POINT = (1217,49)
INBOX_ACTIONS = {
    'return': ('friend-rice-return',(326,625,531,679),(425,650)),
    'collect': ('friend-rice-collect',(810,625,1012,680),(908,650)),
}


def navigation(frame, vision):
    if vision.match(frame,'friend-guide',(243,300,625,357),threshold=.97):
        return 'friend-guide',(640,331)
    if vision.match(frame,'friend-rice-title',(580,26,740,76)):
        return 'friend-rice-inbox',INBOX_BACK
    if (vision.match(frame,'friend-list',(112,10,240,83)) and
            vision.match(frame,'friend-inbox-entry',INBOX_ENTRY)):
        return 'friend-list',FRIEND_BACK
    return None


def _orange(region):
    red,green,blue = region.astype(np.int16).transpose(2,0,1)
    return ((red > 190) & (green > 65) & (green < red-25) & (blue < 100)).mean()


def _enabled(frame, vision, name, box):
    x1,y1,x2,y2 = box
    if _orange(frame[y1:y2,x1:x2]) < .035:
        return None
    return vision.match(frame,name,box)


def _inbox_rows(frame, vision):
    return tuple((claim,reply) for y in range(82,548,93) if (
        claim := vision.match(frame,'friend-rice-claim',(1039,y,1114,y+89),threshold=.91))
        for reply in (_enabled(frame,vision,'friend-rice-envelope',(922,y,1020,y+89)),))


def _blank_inbox_after(frame, number):
    start = 91 + 93*number
    if start > 560:
        return False
    # The light card background below the last separator excludes another row.
    area = frame[start+5:599,196:1113].astype(np.int16)
    return ((area.min(axis=2) >= 190) & (area.max(axis=2)-area.min(axis=2) < 68)).mean() > .997


def _list_gifts(frame, vision):
    # A row can settle at a nonstandard height after scrolling.
    matches = []
    y = 183
    while y < 630:
        point = _enabled(frame,vision,'friend-row-gift',(1155,y,1253,min(y+150,697)))
        if point:
            matches.append(point)
            y = point[1]+33
        else:
            y += 42
    return matches


def _batch_button(frame, vision, key):
    name,box,point = INBOX_ACTIONS[key]
    return point if _enabled(frame,vision,name,box) else None


def same_gift(frame, vision, decision):
    if decision.operation == 'return_rice_batch':
        rows = _inbox_rows(frame,vision)
        return (len(rows) == decision.rice_gifts and _blank_inbox_after(frame,len(rows)) and
                _batch_button(frame,vision,'return') == decision.point)
    x,y = decision.point
    name = 'friend-rice-envelope' if decision.operation == 'return_rice' else 'friend-row-gift'
    box = (x-47,y-37,x+47,y+37)
    return _enabled(frame,vision,name,box) == decision.point


class FriendRice:
    def __init__(self, budget_path, *, day=daily_key):
        self.path = budget_path
        self.day = day
        self.inbox_complete = False
        self.scroll_from = None

    def remaining(self):
        try:
            saved = json.loads(self.path.read_text(encoding='utf-8'))
            sent = saved['sent_upper_bound']
            saved_day = date.fromisoformat(saved['day'])
            today = self.day()
            if type(sent) is not int or not 0 <= sent <= LIMIT or saved_day > today:
                return 0
            return LIMIT-sent if saved_day == today else LIMIT
        except FileNotFoundError:
            return LIMIT
        except (OSError, ValueError, KeyError, TypeError):
            return 0

    def reserve(self, count):
        today = self.day().isoformat()
        left = self.remaining()
        if not 0 < count <= left or today != self.day().isoformat():
            return False
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'day':today,'sent_upper_bound':LIMIT-left+count}),encoding='utf-8')
        temporary.replace(self.path)
        return True

    def step(self, screen, vision):
        if screen.page == 'home':
            if point := vision.match(screen.frame,'friend-home',FRIEND_HOME):
                return DailyDecision.tap('open_friends',point,'打开好友列表',guard=FRIEND_HOME)
            return DailyDecision(status='等待识别主页好友入口')
        if screen.page == 'friend-list':
            return self._list(screen,vision)
        if screen.page != 'friend-rice-inbox':
            return None
        return self._inbox(screen,vision)

    def _inbox(self, screen, vision):
        frame = screen.frame
        number = re.search(r'(?<!\d)(\d{1,2})[/／]15(?!\d)',screen.text((875,32,1069,66)))
        rows = _inbox_rows(frame,vision)
        empty = '暂无可收取的饭团' in screen.text((473,302,829,389))
        if (number and int(number[1]) == LIMIT) or (not rows and empty):
            self.inbox_complete = True
            return DailyDecision.tap('back',INBOX_BACK,'关闭已检查的好友饭团面板')
        if not rows:
            return DailyDecision(status='等待确认好友饭团收取状态')
        remaining = self.remaining()
        if 0 < len(rows) <= remaining and _blank_inbox_after(frame,len(rows)):
            if point := _batch_button(frame,vision,'return'):
                return DailyDecision.tap('return_rice_batch',point,'批量回赠好友饭团',rice_gifts=len(rows))
        if remaining:
            reply = next((reply for _,reply in rows if reply),None)
            if reply:
                return DailyDecision.tap('return_rice',reply,'回赠一份好友饭团',rice_gifts=1)
        if point := _batch_button(frame,vision,'collect'):
            return DailyDecision.tap('collect_rice_batch',point,'批量收取好友饭团',guard=INBOX_ACTIONS['collect'][1])
        return DailyDecision(status='等待确认好友饭团收取按钮')

    def _list(self, screen, vision):
        if not self.inbox_complete:
            return DailyDecision.tap('open_rice_inbox',INBOX_POINT,'打开好友饭团收取面板',guard=INBOX_ENTRY)
        remaining = self.remaining()
        if not remaining:
            return DailyDecision('done',status='好友饭团已检查，助手今日赠送额度已用完')
        if gifts := _list_gifts(screen.frame,vision):
            return DailyDecision.tap('give_rice',gifts[0],'向好友赠送一份饭团',rice_gifts=1)
        identities = screen.frame[190:687,292:524]
        if self.scroll_from is not None and cv2.absdiff(self.scroll_from,identities).mean() < 2:
            return DailyDecision('done',status='已检查可领取与可赠送的好友饭团')
        return DailyDecision.swipe('scroll_friends',(855,651),'继续检查可赠送饭团的好友',(855,282))

    def completed(self, decision, screen):
        if decision.action == 'swipe' and screen.page == 'friend-list':
            self.scroll_from = screen.frame[190:687,292:524].copy()
