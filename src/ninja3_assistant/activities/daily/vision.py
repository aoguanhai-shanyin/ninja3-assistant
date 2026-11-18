"""Daily-menu recognition from SDK frames and local OCR; no device input."""
from dataclasses import dataclass
import re

import cv2
import numpy as np

from ninja3_assistant.paths import resource_root
from ninja3_assistant.vision.invitation import InvitationVision


def compact(text):
    return (re.sub(r'\s+','',text).replace('：',':').replace('夭','天')
            .replace('阝艮','限').replace('剌余','剩余').replace('卜时','小时').replace('下时','小时')
            .replace('铫战','挑战').replace('坎数','次数'))


@dataclass(frozen=True)
class TextHit:
    text: str
    box: tuple

    @property
    def point(self):
        x,y,w,h = self.box
        return round(x+w/2),round(y+h/2)


class DailyScreen:
    def __init__(self, frame, ocr, page='unknown'):
        self.frame,self.ocr,self.page = frame,ocr,page
        self.home_summon = None
        self.home_points = {}

    def hits(self, pattern, box=(0,0,1280,720)):
        x1,y1,x2,y2 = box
        words = [w for w in self.ocr.get('words',()) if
                 x1 <= w['bbox'][0]+w['bbox'][2]/2 < x2 and
                 y1 <= w['bbox'][1]+w['bbox'][3]/2 < y2]
        chars,positions = [],[]
        for word in words:
            text = compact(word['text'])
            chars.extend(text)
            positions.extend([word['bbox']]*len(text))
        value = ''.join(chars)
        result = []
        for match in re.finditer(pattern,value):
            boxes = positions[match.start():match.end()]
            if not boxes:
                continue
            left,top = min(b[0] for b in boxes),min(b[1] for b in boxes)
            right,bottom = max(b[0]+b[2] for b in boxes),max(b[1]+b[3] for b in boxes)
            result.append(TextHit(match[0],(left,top,right-left,bottom-top)))
        return result

    def text(self, box=(0,0,1280,720)):
        x1,y1,x2,y2 = box
        return compact(''.join(w['text'] for w in self.ocr.get('words',()) if
            x1 <= w['bbox'][0]+w['bbox'][2]/2 < x2 and y1 <= w['bbox'][1]+w['bbox'][3]/2 < y2))


class DailyVision:
    def __init__(self):
        self.templates = {p.stem:cv2.imread(str(p),0)
                          for p in (resource_root()/'assets/daily-cleanup').glob('*.png')}
        shop_guide = self.templates.get('shop-guide')
        if shop_guide is not None:
            # The limited-shop welcome variants share their highlighted title.
            self.templates['limited-shop-guide'] = shop_guide[5:38,99:220]
        blessing_free = self.templates.get('blessing-free')
        if blessing_free is not None:
            self.templates['blessing-free-expanded'] = cv2.resize(blessing_free,None,fx=1.12,fy=1.12)
        for name,source in (('home-hall','navigation-home'),('home-summon','navigation-home-summon')):
            path = resource_root()/'assets/intelligence'/(source+'.png')
            self.templates[name] = cv2.imread(str(path),0) if path.is_file() else None
        self.invitations = InvitationVision()

    def match(self, frame, name, box=(0,0,1280,720), threshold=.92):
        template = self.templates.get(name)
        if template is None:
            return None
        x1,y1,x2,y2 = box
        gray = cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_RGB2GRAY)
        if any(a < b for a,b in zip(gray.shape,template.shape)):
            return None
        _,score,_,at = cv2.minMaxLoc(cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED))
        patch = gray[at[1]:at[1]+template.shape[0],at[0]:at[0]+template.shape[1]]
        if score < threshold or np.abs(patch.astype(np.int16)-template).mean() > 28:
            return None
        return (round(x1+at[0]+template.shape[1]/2),round(y1+at[1]+template.shape[0]/2))

    def home(self, frame):
        """Identify fixed home controls while allowing unlocked bottom slots to move."""
        if not self.match(frame,'home-hall',(730,570,1000,705),threshold=.94):
            return None
        box = (180,600,770,720)
        return (self.match(frame,'home-summon',box,threshold=.94) or
                self.match(frame,'home',box,threshold=.90))

    def login(self, frame):
        point = self.match(frame,'login-button',(500,470,800,560),threshold=.95)
        if point and self.match(frame,'login-server',(680,395,840,465),threshold=.95):
            return point
        return None

    def family_welcome(self, frame):
        if not self.match(frame,'family-welcome-members',(1000,0,1140,90)):
            return False
        if (frame[260:480,150:1130].max(axis=2)<25).mean() < .9:
            return False
        for top,bottom in ((210,230),(505,525)):
            band=frame[top:bottom,100:1180].astype(np.int16)
            if ((band[:,:,0]>140)&(band[:,:,1]>90)&(band[:,:,0]>band[:,:,2]+30)).mean() < .25:
                return False
        return True

    def family_entry(self, frame):
        point = self.match(frame,'family',(250,250,650,650))
        if point:
            return point
        template = self.templates.get('family')
        if template is None or template.shape[0] < 36 or template.shape[1] < 46:
            return None
        # The drum's centre keeps its shape when courtyard lighting changes.
        core = template[8:36,18:46]
        gray = cv2.cvtColor(frame[250:650,250:650],cv2.COLOR_RGB2GRAY)
        _,score,_,at = cv2.minMaxLoc(cv2.matchTemplate(gray,core,cv2.TM_CCOEFF_NORMED))
        if score < .96:
            return None
        return (250+at[0]+core.shape[1]//2,250+at[1]+core.shape[0]//2)

    @staticmethod
    def guide_chrome(frame):
        for top,bottom in ((122,128),(590,600)):
            band=frame[top:bottom].astype(np.int16)
            orange=(band[:,:,0]>200)&(band[:,:,1]>110)&(band[:,:,0]>band[:,:,2]+70)
            if orange.mean(axis=1).max() < .7:
                return False
        return True

    def guide_point(self, frame):
        if self.guide_chrome(frame):
            point = self.match(frame,'guide-next',(900,540,1260,645),threshold=.94)
            if point:
                return point
            if self.match(frame,'guide-prev',(600,540,940,645),threshold=.94):
                return (640,100)
        return None

    def shop_guide(self, frame):
        return (self.match(frame,'limited-shop-guide',(330,297,540,365),threshold=.98) or
                self.match(frame,'shop-guide',(250,280,1020,380),threshold=.94) or
                self.match(frame,'shop-stone-guide',(250,280,1020,380),threshold=.94))

    def navigation(self, frame):
        """Fast recognized handoff; never uses a game-wide back key."""
        invite = self.invitations.detect(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY))
        if invite:
            return 'invite',invite
        login = self.login(frame)
        if login:
            return 'login',login
        if self.family_welcome(frame):
            return 'family-welcome',(640,650)
        next_guide = self.guide_point(frame)
        if next_guide:
            return 'paged-guide',next_guide
        if self.match(frame,'summon-guide',(620,220,820,300),threshold=.94):
            return 'summon-guide',(640,100)
        if self.shop_guide(frame):
            return 'menu-guide',(640,100)
        if self.match(frame,'monthly-sidebar',(20,130,250,220)):
            return 'monthly',(1238,48)
        if self.match(frame,'reward-continue',(500,640,800,715)):
            return 'reward_continue',(1130,590)
        if self.match(frame,'reward',(300,0,900,180)):
            return 'reward',self.match(frame,'confirm',(200,560,750,715))
        if self.match(frame,'reward-banner',(500,50,790,175),threshold=.94):
            confirm = self.match(frame,'confirm',(200,560,750,715))
            if confirm:
                return 'reward',confirm
        for page,box,point in (
            ('anniversary',(0,0,650,220),(1195,98)),
            ('ten-signin',(100,0,850,220),(1220,705)),
            ('monthly',(130,20,1100,225),(1238,48)),
            ('rank-preview',(120,0,420,110),(75,50)),
            ('rank',(120,0,320,110),(75,50)),
            ('trial',(0,0,470,120),(58,53)),('prayer',(0,0,470,120),(50,55)),
            ('curse',(0,0,500,120),(78,52)),('hall',(0,0,400,120),(78,53)),
            ('weapon',(800,70,1280,260),(40,48)),('vault',(800,70,1280,260),(40,48)),
            ('family',(250,250,650,650),(90,65)),('house',(950,0,1280,135),(65,50)),
            ('shop',(80,0,400,110),(75,45)),('limited-shop',(80,0,450,110),(75,45)),
            ('daily-shop',(80,0,450,110),(75,45)),('summon',(0,0,220,110),(40,48))):
            if (self.family_entry(frame) if page == 'family' else self.match(frame,page,box)):
                return page,point
        if self.home(frame):
            return 'home',None
        return None,None

    def observe(self, frame, ocr):
        screen = DailyScreen(frame,ocr)
        invite = self.invitations.detect(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY))
        if invite:
            screen.page,screen.point = 'invite',invite
            return screen
        login = self.login(frame)
        if login:
            screen.page,screen.point = 'login',login
            return screen
        text = screen.text()
        next_guide = self.guide_point(frame)
        if not next_guide and self.guide_chrome(frame):
            buttons = screen.hits('下一页|我知道了|知道了|明白了|关闭说明',(850,540,1260,645))
            next_guide = buttons[0].point if buttons else None
        if self.family_welcome(frame):
            screen.page = 'family-welcome'
        elif next_guide:
            screen.page,screen.point = 'paged-guide',next_guide
        elif (('这里是限购商店' in text and '优惠的商品' in text) or
                ('获得的原石可在' in text and '兑换元素石' in text) or
                self.shop_guide(frame)):
            screen.page = 'menu-guide'
        elif (('召唤池存在两种UP模式' in text and '规则界面' in text) or
                self.match(frame,'summon-guide',(620,220,820,300),threshold=.94)):
            screen.page = 'summon-guide'
        elif ('获得奖励' in text and '点击任意位置继续' in text) or self.match(frame,'reward-continue',(500,640,800,715)):
            screen.page = 'reward_continue'
        elif (('恭喜' in text and ('确定' in text or self.match(frame,'confirm',(200,550,750,710)))) or
                ('再抽一次' in text and '确定' in text)):
            screen.page = 'reward'
        elif '本月累计签到' in text:
            screen.page = 'monthly'
        elif '限购数量' in text and '购买' in text:
            screen.page = 'shop_dialog'
        elif re.search(r'获得一个[DC（(]级.*奖册',text,re.I) and '购买' in text:
            screen.page = 'bounty_dialog'
        elif '距离自动刷新' in text and '剩余次数' in text:
            screen.page = 'daily_shop'
        elif '本周勾玉炼化次数' in text:
            screen.page = 'curse'
        elif '今日' in text and '祈' in text and '次数' in text:
            screen.page = 'prayer'
        elif '是否消耗5饭团继续进行修炼' in text:
            screen.page = 'house_continue'
        elif screen.hits('忍阶预览',(120,0,500,110)) or self.match(frame,'rank-preview',(120,0,420,110)):
            screen.page = 'rank-preview'
        elif ('忍阶宝箱' in text and '忍阶预览' in text) or self.match(frame,'rank',(120,0,320,110)):
            screen.page = 'rank'
        elif screen.hits('小屋',(120,0,600,110)) and ('管理' in text or '好友' in text or '修炼' in text):
            screen.page = 'house'
        elif '加入家族' in text and '申请加入' in text:
            screen.page = 'no_family'
        elif self.match(frame,'bounty-tab',(1080,330,1280,420)):
            screen.page = 'bounty_shop'
        else:
            for page,box in (('anniversary',(0,0,650,220)),('ten-signin',(100,0,850,220)),
                ('trial',(0,0,470,120)),('prayer',(0,0,470,120)),('curse',(0,0,500,120)),
                ('hall',(0,0,400,120)),('weapon',(800,70,1280,260)),
                ('vault',(800,70,1280,260)),('family',(250,250,650,650)),
                ('limited-shop',(80,0,450,110)),
                ('house',(950,0,1280,135))):
                if (self.family_entry(frame) if page == 'family' else self.match(frame,page,box)):
                    screen.page = page
                    break
        if screen.page == 'unknown':
            home = self.home(frame)
            if home:
                screen.page,screen.home_summon = 'home',home
                screen.home_points = {'summon':home,'shop':(home[0]-104,home[1])}
                for name,pattern,box in (
                    ('limited',r'答[谢榭]礼',(80,130,330,600)),
                    ('house',r'[小卜]屋',(80,130,330,600)),
                    ('monthly',r'活动',(0,150,100,360)),
                    ('shop',r'商.?店',(180,620,770,715))):
                    hits = screen.hits(pattern,box)
                    if hits:
                        screen.home_points[name] = hits[0].point
                if 'limited' not in screen.home_points:
                    thanks = self.match(frame,'home-thanks',(80,130,330,600),threshold=.94)
                    if thanks:
                        screen.home_points['limited'] = thanks
            elif '悬赏商店' in text and '奖册' in text:
                screen.page = 'bounty_shop'
            elif '超值特惠' in text and ('限购商店' in text or '神秘商店' in text):
                screen.page = 'shop'
            elif '每日签到' in text and '活动' in text:
                screen.page = 'activities'
            elif self.match(frame,'summon',(0,0,220,110)):
                screen.page = 'summon'
        return screen

    def free(self, screen, kind):
        name,box = {'curse':('curse-free',(300,590,565,660)),
                    'weapon':('weapon-free',(620,610,950,695)),
                    'blessing':('blessing-free',(185,620,440,690))}[kind]
        # The cooldown's “下次免费” is outside these exact button-label templates.
        if self.match(screen.frame,name,box,threshold=.95) is not None:
            return True
        # The prayer button expands after its entrance animation.
        return kind == 'blessing' and self.match(screen.frame,'blessing-free-expanded',
                                                (220,643,401,690),threshold=.92) is not None

    def current_action(self, screen, decision, fresh):
        fresh_page,_ = self.navigation(fresh)
        if screen.page == 'login':
            return fresh_page == 'login'
        if fresh_page in ('invite','login','paged-guide','summon-guide','menu-guide','family-welcome','reward','reward_continue') and fresh_page != screen.page:
            return False
        if decision.status.startswith('返回主页，') and self.home(fresh):
            return False
        if decision.status in ('领取忍阶已完成每日任务','领取忍阶每周随机道具'):
            return self.match(fresh,'rank',(120,0,320,110)) is not None
        if decision.free_kind:
            if not self.free(DailyScreen(fresh,screen.ocr,screen.page),decision.free_kind):
                return False
        if decision.guard:
            x1,y1,x2,y2 = decision.guard
            before = cv2.GaussianBlur(screen.frame[y1:y2,x1:x2],(3,3),0)
            after = cv2.GaussianBlur(fresh[y1:y2,x1:x2],(3,3),0)
            if np.abs(before.astype(np.int16)-after).mean() > 12:
                return False
        return True

    def coin(self, screen, box, *, large=False):
        return any(self.match(screen.frame,name,box,threshold=.94) is not None
                   for name in (('coin-buy',) if large else ('coin-slot','coin-bounty','coin-store','coin-spent')))

    def food(self, screen):
        hits = screen.hits(r'\d+',(1160,15,1225,75))
        if hits:
            return int(hits[0].text)
        repeated = screen.ocr.get('food_counter_repeat',{}).get('text','').replace('》',' ')
        count = re.fullmatch(r'\s*(\d+)\s+\1\s+\1\s*',repeated)
        if count:
            return int(count[1])
        for name in self.templates:
            if name.startswith('food-') and self.match(screen.frame,name,(1175,27,1222,67),.98):
                return int(name[5:])
        return None

    def signin(self, screen):
        hits = screen.hits(r'(?:第)?[0-9]+天',(130,270,1250,700))
        sidebar = self.monthly_sidebar(screen)
        first,spacing = (360,200) if sidebar else (228,204)
        if screen.page == 'ten-signin':
            first,spacing = 260,190
        cells = {}
        for hit in hits:
            x,y = hit.point
            number = int(re.search(r'\d+',hit.text)[0])
            column = round((x-first)/spacing)
            if abs(x-(first+column*spacing)) > 25:
                continue
            for target in range(5):
                day = number+target-column
                if 1 <= day <= (10 if screen.page == 'ten-signin' else 25):
                    cells[day] = (first+target*spacing,y)
        highlighted = []
        for number,(x,y) in cells.items():
            y1,y2 = max(270,round(y-30)),min(695,round(y+(140 if screen.page == 'ten-signin' else 175)))
            strips = [screen.frame[y1:y2,max(0,x+d-12):min(1280,x+d+13)] for d in (-88,88)]
            white = max(((s.min(axis=2)>205)&(s.max(axis=2)-s.min(axis=2)<55)).mean(axis=0).max() for s in strips if s.size)
            if white > .35:
                highlighted.append((number,(x,min(y+90,670))))
        return highlighted

    def monthly_sidebar(self, screen):
        return bool(screen.hits('每日签到',(0,100,260,230)) or
                    self.match(screen.frame,'monthly-sidebar',(20,130,250,220)))
