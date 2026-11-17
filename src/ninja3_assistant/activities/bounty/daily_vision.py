"""Daily bounty menu recognition from fixed SDK frames and local OCR."""
import re

import cv2

from ninja3_assistant.activities.daily.vision import DailyScreen, DailyVision
from ninja3_assistant.paths import resource_root


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
        self.s_filter = self.ss_filter = '?'
        self.grade_choices = {}
        self.__dict__.update(fields)


class DailyBountyVision:
    def __init__(self, grade='D'):
        self.common = DailyVision()
        self.grade_hint = grade
        self.rank_detector = None
        root = resource_root()/'assets'
        self.templates = {p.stem:cv2.imread(str(p),0) for p in (root/'bounty-daily').glob('*.png')}
        for name,path in (('hub-title','menu-title'),('hall-entry','startup-hall-bounty'),
                          ('join','join'),('ready','ready'),
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

    def tasks_panel(self, frame):
        return bool(self.match(frame,'tasks-title',(500,155,760,225),.95) and
                    self.match(frame,'tasks-close',(975,115,1075,195),.95))

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
                    elif status == '领取':
                        hits = screen.hits('领取',status_box)
                        screen.task_states[name] = 'claimable'
                        if hits and screen.task_claim is None:
                            screen.task_claim = hits[0].point
                    elif (status in ('未完成','去完成','前往','去完成任务') or
                          '0/1' in status or '0/1' in labels[name]):
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
        if screen.page == 'bounty_daily':
            screen.join_failed = '加入队伍失败' in text or '当前暂未找到队伍' in text
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
            screen.s_filter,self.ss_filter = self._filter(frame,876),self._filter(frame,978)
            if screen.remaining and (screen.s_filter,screen.ss_filter) == ('on','off'):
                screen.join = self._join(frame,'S')
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
            if invite:
                screen.invite_open = invite[0].point
            if not screen.minimum_players:
                prepared = self.match(frame,'prepared',(940,610,1280,720))
                screen.ready = None if prepared else self.match(frame,'ready',(940,610,1280,720))
                starts = screen.hits(r'开始(?:战斗|挑战)?|出发',(940,620,1280,715))
                screen.start = starts[0].point if starts else None
        elif screen.page not in ('home','bounty_hub'):
            if '登录' in text and '点击切换' in text:
                screen.login = (640,515)
        return screen

    def _join(self, frame, grade):
        gray = None
        if grade == 'S':
            from ninja3_assistant.vision.vision import Vision
            if self.rank_detector is None:
                self.rank_detector = Vision(use_gpu=False)
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
        if decision.free_kind == 'bounty':
            return bool(decision.guard and self.match(fresh,'free-claim',decision.guard,.96))
        return self.common.current_action(screen,decision,fresh)
