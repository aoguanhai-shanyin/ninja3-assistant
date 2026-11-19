"""Daily bounty menu recognition from fixed SDK frames and local OCR."""
import re

import cv2
import numpy as np

from ninja3_assistant.activities.daily.vision import DailyScreen, DailyVision
from ninja3_assistant.paths import resource_root
from ninja3_assistant.vision.game_update import game_update_banner


REFRESH_BOX = (1075,148,1220,203)
INVITE_SLOT_BOX = (595,300,685,376)
ORDER_COUNTER_BOX = (351,156,378,187)
ORDER_FAILURE_BOX = (420,225,865,295)
ORDER_SUCCESS_BOX = (540,220,740,315)


class DailyBountyScreen(DailyScreen):
    def __init__(self, frame, ocr, page='unknown', **fields):
        super().__init__(frame,ocr,page)
        self.remaining = self.grade = self.join = self.entry = None
        self.ready = self.start = self.create = self.confirm = self.close = None
        self.free_claim = self.free_guard = self.back = self.login = None
        self.free_exhausted = False
        self.minimum_players = False
        self.invite_open = None
        self.invitation = None
        self.recruit_tab = None
        self.recruit_candidates = []
        self.task_states = {}
        self.task_claim = None
        self.join_failed = False
        self.join_pending = False
        self.refresh = None
        self.s_filter = self.ss_filter = '?'
        self.grade_choices = {}
        self.__dict__.update(fields)


class DailyBountyVision:
    def __init__(self, grade='D'):
        self.common = DailyVision()
        self.grade_hint = grade
        self.rank_detector = None
        self.intelligence_tasks = None
        root = resource_root()/'assets'
        self.templates = {p.stem:cv2.imread(str(p),0) for p in (root/'bounty-daily').glob('*.png')}
        started = self.templates.get('order-join-started')
        self.order_started_variants = () if started is None else tuple(
            cv2.resize(started[4:31,4:-4],None,fx=scale,fy=scale)
            for scale in (1.,.992,.986,.98,.974,.968,.85))
        self.order_started_tail = None if started is None else cv2.resize(
            started[4:31,4:-4],None,fx=.365,fy=.365,interpolation=cv2.INTER_AREA)
        joined = self.templates.get('order-join-success')
        self.order_success_early = None if joined is None else cv2.resize(
            joined,None,fx=.84,fy=.84,interpolation=cv2.INTER_AREA)
        for name,path in (('hub-title','menu-title'),('hall-entry','startup-hall-bounty'),
                          ('join','join'),('ready','ready'),('refresh','refresh'),
                          ('join-disbanded','join-disbanded'),('join-closed','join-closed'),
                          ('prepared','prepared'),('result-title','result-title'),
                          ('result-confirm','result-confirm')):
            if name not in self.templates:
                self.templates[name] = cv2.imread(str(root/(path+'.png')),0)

    def match(self, frame, name, box=(0,0,1280,720), threshold=.92):
        # Reuse the daily matcher's error bound and optional local template bank.
        template = self.templates.get(name)
        if template is None:
            return None
        original = self.common.templates.get(name)
        self.common.templates[name] = template
        try:
            return self.common.match(frame,name,box,threshold)
        finally:
            if original is None:
                self.common.templates.pop(name,None)
            else:
                self.common.templates[name] = original

    def navigation(self, frame):
        if self._recruit_tab(frame) is not None:
            return 'bounty_recruit',(750,350)
        if (self.match(frame,'reward-blank-title',(550,120,735,185),.95) and
                self.match(frame,'reward-blank-close',(500,610,790,680),.95)):
            return 'reward',(1150,685)
        if self.tasks_panel(frame):
            return 'bounty_tasks',(1025,155)
        if self.match(frame,'daily-result-title',(600,50,850,130)):
            return 'result',(640,680)
        if point := self.order_result_confirm(frame):
            return 'result',point
        if self.match(frame,'own-empty',(480,290,800,350)):
            return 'bounty_own_empty',(1115,670)
        if self.match(frame,'order-locked',(440,330,875,400)):
            return 'bounty_locked',(65,50)
        if (self.match(frame,'leave-title',(530,340,760,395)) and
                self.match(frame,'leave-room-context',(1030,580,1220,625))):
            return 'bounty_leave',(810,525)
        for name,page,box,back in (
            ('grade-title','bounty_grade',(460,160,820,225),(897,163)),
            ('free-title','bounty_free',(480,165,800,235),(810,170)),
            ('result-title','result',(500,20,920,180),(640,670))):
            if self.match(frame,name,box):
                return page,back
        if self.match(frame,'room-footer',(1030,580,1220,625)):
            return 'bounty_room',(65,50)
        for name,page in (('daily-title','bounty_daily'),('order-title','bounty_order'),
                          ('order-locked-title','bounty_order'),('hub-title','bounty_hub')):
            if self.match(frame,name,(90,0,440,110)):
                return page,(65,50)
        return self.common.navigation(frame)

    def order_result_confirm(self, frame):
        if self.match(frame,'order-result-title',(570,49,825,123),.96):
            return self.match(frame,'result-confirm',(526,630,773,719))
        return None

    def tasks_panel(self, frame):
        return bool(self.match(frame,'tasks-title',(500,155,760,225),.95) and
                    self.match(frame,'tasks-close',(975,115,1075,195),.95))

    @staticmethod
    def _pending_task_label(frame, box):
        # 未完成 is three separate orange glyphs. Claimed marks form one
        # continuous stamp, and claim buttons are wider and taller.
        x1,y1,x2,y2 = box
        hsv = cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv,(4,130,90),(28,255,255))
        x,y,w,h = cv2.boundingRect(mask)
        if not (45 <= x <= 75 and 15 <= y <= 45 and 52 <= w <= 75 and 16 <= h <= 28):
            return False
        painted = (mask != 0).sum(axis=0) >= 3
        edges = np.flatnonzero(np.diff(painted,prepend=False,append=False))
        return len(edges) == 6 and all(12 <= width <= 24 for width in edges[1::2]-edges[::2])

    @staticmethod
    def _recruit_tab(frame):
        # The three tabs have a blue selected background and two dark tabs.
        colors = [frame[76:82,x:x+18,:3].mean(axis=(0,1)) for x in (830,990,1150)]
        blue = [40 < r < 100 and 55 < g < 120 and 90 < b < 155 and b > r+25
                for r,g,b in colors]
        dark = [max(color) < 40 for color in colors]
        for index in range(3):
            if blue[index] and all(dark[other] for other in range(3) if other != index):
                return ('friend','family','chat')[index]
        return None

    @staticmethod
    def _recruit_candidates(screen):
        # A yellow button plus its OCR label and a visible name are required;
        # no fixed row is assumed for the family list.
        hsv = cv2.cvtColor(screen.frame,cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv[90:675,1140:1280],(15,120,150),(40,255,255))
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            x,y = x+1140,y+90
            if not (90 <= w <= 125 and 25 <= h <= 50):
                continue
            if not screen.hits('邀请',(x,y,x+w,y+h)):
                continue
            name = screen.text((910,y-15,1080,y+h+15))
            if name:
                result.append((name,(x+w//2,y+h//2)))
        return sorted(result,key=lambda item:item[1][1])

    @staticmethod
    def _count(text, kind):
        if kind == 'daily':
            hit = re.search(r'今日奖励剩余次数\D{0,3}(\d+)',text)
            return int(hit[1]) if hit and 0 <= int(hit[1]) <= 4 else None
        hit = re.search(r'(?<!S)S(?!S|\+)[^0-9]{0,10}(\d+)',text,re.I)
        return int(hit[1]) if hit and 0 <= int(hit[1]) <= 7 else None

    @staticmethod
    def _filter(frame, x):
        r,g,b = frame[168:175,x-3:x+4,:3].mean(axis=(0,1))
        if 45 <= r <= 135 and g <= 90 and b <= 50 and r >= g+25:
            return 'on'
        if r >= 180 and g >= 140 and b >= 95:
            return 'off'
        return '?'

    @staticmethod
    def _s_creation(screen):
        titles = screen.hits(r'(?<![A-Za-z])S(?![A-Za-z+])(?:级)?(?:召令)?悬赏(?:令)?')
        ss_titles = screen.hits(r'SS\+?(?:级)?(?:召令)?悬赏(?:令)?')
        for button in screen.hits('创建队伍'):
            x,y = button.point
            def same_column(title):
                left,_,width,_ = title.box
                return left-width*.3 <= x <= left+width*1.3
            above = [title for title in titles if title.point[1] < y and same_column(title)]
            if not above:
                continue
            title = max(above,key=lambda item:item.point[1])
            if any(title.point[1] <= ss.point[1] < y and same_column(ss) for ss in ss_titles):
                continue
            return button.point
        return None

    def observe(self, frame, ocr):
        common = self.common.observe(frame,ocr)
        screen = DailyBountyScreen(frame,ocr,common.page)
        text = screen.text()
        if common.page == 'invite':
            screen.point = common.point
            screen.invitation = self.common.invitations.bounty_invitation(
                cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY),ocr)
            return screen
        if (('恭喜获得' in text and '点击空白处关闭' in text) or
                (self.match(frame,'reward-blank-title',(550,120,735,185),.95) and
                 self.match(frame,'reward-blank-close',(500,610,790,680),.95))):
            screen.page,screen.confirm = 'reward',(1150,685)
            return screen
        if common.page == 'reward':
            hits = screen.hits('确定',(200,560,900,715))
            screen.confirm = hits[0].point if hits else self.common.match(frame,'confirm',(200,550,900,715))
            return screen
        tasks_panel = self.tasks_panel(frame)
        if tasks_panel or screen.hits('每日任务',(500,155,750,220)):
            rows = (('friend',(390,235,700,315),(800,235,970,315)),
                    ('family',(390,345,700,425),(800,345,970,425)))
            labels = {name:screen.text(label) for name,label,_ in rows}
            if tasks_panel or '好友' in labels['friend'] or '家族' in labels['family']:
                screen.page = 'bounty_tasks'
                for name,_,status_box in rows:
                    if ('好友' if name == 'friend' else '家族') not in labels[name]:
                        screen.task_states[name] = 'unknown'
                        continue
                    status = screen.text(status_box).replace('颌','领')
                    if '已领取' in status:
                        screen.task_states[name] = 'claimed'
                    elif status in ('领取','领取奖励'):
                        hits = screen.hits(status,status_box)
                        screen.task_states[name] = 'claimable'
                        if hits and screen.task_claim is None:
                            screen.task_claim = hits[0].point
                    elif (status in ('未完成','去完成','前往','去完成任务') or
                          '0/1' in status or '0/1' in labels[name] or
                          (tasks_panel and self._pending_task_label(frame,status_box))):
                        screen.task_states[name] = 'pending'
                    else:
                        screen.task_states[name] = 'unknown'
                return screen
        tab = self._recruit_tab(frame)
        if (tab and screen.hits('好友',(810,20,960,85)) and
                screen.hits('邀请族员',(960,20,1120,85))):
            screen.page,screen.recruit_tab = 'bounty_recruit',tab
            if tab != 'chat':
                screen.recruit_candidates = self._recruit_candidates(screen)
            return screen
        if '通过召令' in text and '等级测试后开启' in text:
            screen.page = 'bounty_locked'
            return screen
        if '当前还未获得' in text and '赏令' in text:
            screen.page = 'bounty_own_empty'
            return screen
        if self.match(frame,'daily-result-title',(600,50,850,130)):
            screen.page,screen.confirm = 'result',(640,680)
            for grade in ('D','C','B','A'):
                if self.match(frame,'result-grade-'+grade,(560,50,850,130),.96):
                    screen.grade = grade
                    break
            if screen.grade is None:
                grade = screen.hits(r'[DCBA]级每日',(560,50,850,130))
                screen.grade = grade[0].text[0] if grade else None
            return screen
        result_grade = screen.hits(r'[DCBA]级每日悬赏',(400,20,1050,145))
        confirm = screen.hits('确定',(400,620,950,715))
        if result_grade and confirm:
            screen.page,screen.grade,screen.confirm = 'result',result_grade[0].text[0],confirm[0].point
            return screen
        if point := self.order_result_confirm(frame):
            screen.page,screen.grade,screen.confirm = 'result','S',point
            return screen
        if (self.match(frame,'leave-title',(530,340,760,395)) and
                self.match(frame,'leave-room-context',(1030,580,1220,625))):
            screen.page,screen.confirm = 'bounty_leave',(810,525)
            return screen
        if self.match(frame,'result-title',(500,20,920,180)):
            screen.page,screen.grade = 'result','S'
            screen.confirm = self.match(frame,'result-confirm',(400,590,950,720))
            return screen
        if '选择每日悬赏难度' in text or self.match(frame,'grade-title',(460,160,820,225)):
            screen.page = 'bounty_grade'
            screen.grade_choices = dict(zip(('D','C','B','A'),((640,246),(640,306),(640,365),(640,423))))
            screen.confirm,screen.close = (640,490),(897,163)
            for grade,(_,y) in screen.grade_choices.items():
                r,g,b = frame[y-5:y+5,460:500,:3].mean(axis=(0,1))
                if r > 225 and g > 185 and b > 75:
                    screen.grade = grade
                    break
            return screen
        purchase_text = screen.text((450,150,825,550))
        if '购买奖励次数' in purchase_text:
            screen.page = 'bounty_free'
            screen.close = (810,170)
            purchases = re.search(r'今日已购买次数\D{0,3}(\d+)/4',purchase_text)
            if purchases:
                screen.free_exhausted = int(purchases[1]) > 0
                if int(purchases[1]) == 0 and self.match(frame,'free-claim',(530,450,746,507),.96):
                    screen.free_claim = (640,480)
                    screen.free_guard = (530,450,746,507)
            return screen
        create = self._s_creation(screen)
        if create is not None:
            screen.page,screen.grade,screen.create = 'bounty_create','S',create
            return screen
        page,back = self.navigation(frame)
        screen.page = page or common.page
        screen.back = back
        if screen.page == 'unknown':
            if self.intelligence_tasks is None:
                from ninja3_assistant.activities.intelligence.tasks import TaskVision
                self.intelligence_tasks = TaskVision()
            if self.intelligence_tasks.detect(frame).kind == 'tasks':
                screen.page = 'intelligence_tasks'
        if screen.page == 'bounty_daily':
            screen.join_failed = '加入队伍失败' in text or '当前暂未找到队伍' in text
            screen.refresh = self.match(frame,'refresh',REFRESH_BOX,.94)
            count = screen.hits(r'\d+',(285,145,323,200))
            screen.remaining = int(count[0].text) if count and 0 <= int(count[0].text) <= 4 else None
            ranks = screen.hits(r'[DCBA]级',(680,625,790,715))
            screen.grade = ranks[0].text[0] if ranks else None
            for grade in ('D','C','B','A'):
                if self.match(frame,'grade-'+grade,(685,625,790,715),.96):
                    screen.grade = grade
                    break
            if screen.remaining:
                screen.join = self._join(frame,screen.grade or self.grade_hint)
        elif screen.page == 'bounty_order':
            screen.remaining = self._count(screen.text((90,140,620,205)),'order')
            if screen.remaining is None:
                screen.remaining = self._count(ocr.get('bounty_order_counter',{}).get('text',''),'order')
            self._order_controls(screen)
        elif screen.page == 'hall':
            screen.entry = self.match(frame,'hall-entry',(0,100,1280,260))
        elif screen.page == 'bounty_room':
            for grade in ('D','C','B','A','S'):
                if self.match(frame,'room-grade-'+grade,(280,0,360,85),.96):
                    screen.grade = grade
                    break
            screen.minimum_players = bool(self.match(frame,'minimum-players',(970,620,1270,710)) or
                                          '至少需要二人' in screen.text((940,620,1280,715)))
            invite = screen.hits('点击邀请好友',(520,300,755,405))
            screen.invite_open = (self.match(frame,'invite-slot',INVITE_SLOT_BOX,.96) or
                                  (invite[0].point if invite else None))
            if not screen.minimum_players:
                prepared = self.match(frame,'prepared',(940,610,1280,720))
                screen.ready = None if prepared else self.match(frame,'ready',(940,610,1280,720))
                starts = screen.hits(r'开始(?:战斗|挑战)?|出发',(940,620,1280,715))
                screen.start = starts[0].point if starts else None
        elif screen.page not in ('home','bounty_hub'):
            if '登录' in text and '点击切换' in text:
                screen.login = (640,515)
        return screen

    @staticmethod
    def order_counter(frame):
        x1,y1,x2,y2 = ORDER_COUNTER_BOX
        return frame[y1:y2,x1:x2]

    def fast_order(self, frame, reference):
        remaining,counter = reference
        # Animated lighting changes the background. The dark outline of
        # the previously OCR-confirmed digit must still occupy the same pixels.
        old_glyph = cv2.cvtColor(counter,cv2.COLOR_RGB2GRAY) < 140
        current_glyph = cv2.cvtColor(self.order_counter(frame),cv2.COLOR_RGB2GRAY) < 140
        shared = np.count_nonzero(old_glyph & current_glyph)
        if (not shared or shared < .90*np.count_nonzero(old_glyph | current_glyph) or
                not (self.match(frame,'order-title',(90,0,440,110)) or
                     self.match(frame,'order-locked-title',(90,0,440,110))) or
                self.common.invitations.detect(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY))):
            return None
        screen = DailyBountyScreen(frame,{},'bounty_order',remaining=remaining)
        self._order_controls(screen)
        return screen if (screen.s_filter,screen.ss_filter) == ('on','off') else None

    def _order_controls(self, screen):
        frame = screen.frame
        screen.s_filter,screen.ss_filter = self._filter(frame,876),self._filter(frame,978)
        screen.join_failed = (self._order_join_failed(frame) or
                              bool(re.search(r'队伍.{0,4}(?:开始战斗|已开战|解散|关闭)',
                                             screen.text(ORDER_FAILURE_BOX))))
        screen.join_pending = self._order_join_pending(frame) or '加入队伍成功' in screen.text(ORDER_SUCCESS_BOX)
        if (screen.remaining and not (screen.join_failed or screen.join_pending) and
                (screen.s_filter,screen.ss_filter) == ('on','off')):
            screen.join = self._join(frame,'S')
        screen.refresh = self.match(frame,'refresh',REFRESH_BOX,.94)

    def _order_join_failed(self, frame):
        x1,y1,x2,y2 = ORDER_FAILURE_BOX
        roi = cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_RGB2GRAY)
        # The toast scales as it appears. Matching only the settled size
        # misses its text while the old card is still visible beneath it.
        if (any(cv2.minMaxLoc(cv2.matchTemplate(roi,variant,cv2.TM_CCOEFF_NORMED))[1] >= .88
                    for variant in self.order_started_variants) or
                self.match(frame,'join-disbanded',ORDER_FAILURE_BOX,.92) or
                self.match(frame,'join-closed',ORDER_FAILURE_BOX,.86)):
            return True
        if self.order_started_tail is None:
            return False
        # The last frame of the animation shrinks toward the center of the board.
        tail = cv2.cvtColor(frame[310:336,558:722],cv2.COLOR_RGB2GRAY)
        return cv2.minMaxLoc(cv2.matchTemplate(tail,self.order_started_tail,cv2.TM_CCOEFF_NORMED))[1] >= .80

    def _order_join_pending(self, frame):
        if self.match(frame,'order-join-success',ORDER_SUCCESS_BOX,.90):
            return True
        if self.order_success_early is None:
            return False
        x1,y1,x2,y2 = ORDER_SUCCESS_BOX
        roi = cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_RGB2GRAY)
        return cv2.minMaxLoc(cv2.matchTemplate(roi,self.order_success_early,cv2.TM_CCOEFF_NORMED))[1] >= .90

    def _join(self, frame, grade):
        gray = None
        if grade == 'S':
            from ninja3_assistant.vision.vision import Vision
            if self.rank_detector is None:
                self.rank_detector = Vision(prewarm=True)
            self.rank_detector.use_prepared_gpu()
            gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        for top in (185,361):
            for left in (95,465,835):
                confirmed = (self.rank_detector.rank_s(gray,left-5,top) if grade == 'S' else
                             self.match(frame,'rank-'+grade,(left-5,top,left+105,top+75),.95))
                if not confirmed:
                    continue
                box = (left,top+13,min(1280,left+360),top+189)
                point = self.match(frame,'join-daily',box,.95) or self.match(frame,'join',box,.95)
                if point:
                    return point
        return None

    def current_action(self, screen, decision, fresh):
        if game_update_banner(fresh):
            return False
        if screen.page == 'result' and screen.grade == 'S' and self.order_result_confirm(screen.frame):
            return self.order_result_confirm(fresh) == decision.point
        if decision.free_kind == 'bounty':
            return bool(decision.guard and self.match(fresh,'free-claim',decision.guard,.96))
        if decision.status == '刷新悬赏队伍列表':
            return (self.navigation(fresh)[0] == screen.page and
                    self.match(fresh,'refresh',REFRESH_BOX,.94) == decision.point)
        if decision.status == '加入S级悬赏队伍':
            return (self.navigation(fresh)[0] == 'bounty_order' and
                    (self._filter(fresh,876),self._filter(fresh,978)) == ('on','off') and
                    not (self._order_join_failed(fresh) or self._order_join_pending(fresh)) and
                    self._join(fresh,'S') == decision.point)
        if screen.page == 'hall':
            if self.navigation(fresh)[0] != 'hall':
                return False
            entry = self.match(fresh,'hall-entry',(0,100,1280,260))
            if decision.action == 'swipe':
                return entry is None
            return entry is not None and max(abs(a-b) for a,b in zip(decision.point,entry)) <= 12
        return self.common.current_action(screen,decision,fresh)
