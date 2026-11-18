"""Daily decisions. Free and ninja-coin operations are explicitly separated."""
from dataclasses import dataclass
import re
import time

import numpy as np

from ninja3_assistant.activities.daily.resources import reserve_food, remaining_seconds


@dataclass(frozen=True)
class DailyDecision:
    action: str = 'wait'
    point: tuple | None = None
    status: str = ''
    end: tuple | None = None
    free_kind: str | None = None
    guard: tuple | None = None
    defer_on_timeout: bool = False


class DailyFlow:
    def __init__(self, kind, *, training_finishes_at=None, clock=time.time, task_flow=None):
        self.kind = kind
        self.task_flow = task_flow
        self.free_sent = False
        self.pending_item = None
        self.reward_seen = False
        self.scans = 0
        self.limited_checked = set()
        self.visited_stages = set()
        self.stage = None
        self.bounty_at_start = False
        self.training_finishes_at = training_finishes_at
        self.clock = clock
        self.rank_daily_checked = False
        self.rank_weekly_checked = False

    def tap(self, point, status, **kwargs):
        return DailyDecision('tap',point,status,**kwargs)

    def back(self, point=(65,50)):
        return self.tap(point,'返回主页，继续每日清理')

    def step(self, screen, vision):
        page,text = screen.page,screen.text()
        if page == 'invite':
            return self.tap(screen.point,'拒绝队友或家族邀请',guard=(340,475,940,585))
        if page == 'login':
            return self.tap(screen.point,'登录当前已选账号')
        if page == 'summon-guide':
            return self.tap((640,100),'关闭召唤说明，继续每日清理',guard=(650,230,785,280))
        if page == 'menu-guide':
            return self.tap((640,100),'关闭商店说明，继续每日清理',guard=(280,300,1000,370))
        if page == 'family-welcome':
            return self.tap((640,650),'关闭家族入场展示，继续每日清理',guard=(200,240,1100,490))
        if page == 'paged-guide':
            return self.tap(screen.point,'继续功能说明',guard=(650,225,1220,480))
        if self.task_flow is not None:
            decision = self.task_flow.step(screen,vision)
            if decision is not None:
                return decision
        if page == 'reward_continue':
            return self.tap((1130,590),'关闭已领取奖励，继续每日清理')
        if page == 'reward':
            hits = screen.hits('确定',(200,560,750,715))
            if hits:
                return self.tap(hits[0].point,'确认领取奖励',guard=(250,560,750,715))
            point = vision.match(screen.frame,'confirm',(200,560,750,715))
            return self.tap(point,'确认领取奖励') if point else DailyDecision(status='等待奖励确定按钮')
        if screen.chore_page:
            if screen.point is None:
                return DailyDecision(status='等待当前日常页面的返回入口')
            x,y = screen.point
            return self.tap(screen.point,'返回主页，继续每日清理',
                            guard=(max(0,x-25),max(0,y-25),min(1280,x+25),min(720,y+25)))
        if page == 'home':
            target = ('summon' if self.kind in ('weapon','curse') or
                      (self.kind == 'limited' and 'anniversary' in self.limited_checked) else
                      'shop' if self.kind in ('bounty_shop','daily_shop') else self.kind)
            point = getattr(screen,'home_points',{}).get(target) or {
                'monthly':(50,280),'blessing':(1010,450),'village':(865,600),
                'house':(143,195),'rank':(65,50)}.get(target)
            if point is None:
                return DailyDecision(status='寻找主页每日清理入口：'+target,
                                     defer_on_timeout=target == 'limited')
            return self.tap(point,'进入每日清理项目')
        if page == 'monthly':
            if self.kind != 'monthly':
                sidebar = vision.monthly_sidebar(screen)
                return self.back((1238,48) if sidebar else (1200,710))
            count = re.search(r'本月累计签到(\d+)次',text)
            if count:
                count = int(count[1])
                if count >= 25:
                    return DailyDecision('done',status='本月25天签到已全部领取')
                highlights = vision.signin(screen)
                for day,point in highlights:
                    if day <= count:
                        return DailyDecision('done',status=f'今日已签到，本月累计{count}次')
                    if day == count+1:
                        return self.tap(point,f'领取第{day}天签到奖励')
            if self.scans < 5:
                visible = screen.hits(r'第(\d+)天',(130,270,1250,700))
                if count is not None and visible and min(int(re.search(r'\d+',h.text)[0]) for h in visible) > count:
                    return DailyDecision('swipe',(950,365),'回到月签到当前天数',(950,625))
                return DailyDecision('swipe',(950,625),'滚动检查月签到',(950,365))
            return DailyDecision('unavailable',status='未确认今天签到状态，保留截图')
        if page in ('anniversary','ten-signin'):
            if self.kind != 'limited':
                return self.back((1195,98) if page == 'anniversary' else (1220,705))
            if page == 'anniversary':
                for x1,y1,x2,y2 in ((90,190,275,590),(310,200,495,620),(520,188,760,632)):
                    strips = screen.frame[y1:y2,x1-12:x1+13]
                    white = ((strips.min(axis=2)>205)&(strips.max(axis=2)-strips.min(axis=2)<55)).mean(axis=0).max()
                    if white > .20:
                        return self.tap(((x1+x2)//2,(y1+y2)//2),'领取当前限时签到奖励')
                if not screen.hits(r'第[一二三四五六七八九十0-9]+天') and not re.search(r'已[领倾]取',text):
                    return DailyDecision(status='等待限时签到奖励展示完成')
            else:
                if len(screen.hits(r'(?:第)?\d+天',(130,270,1250,700))) < 3:
                    return DailyDecision(status='等待十日签到卡片展示完成')
                highlights = vision.signin(screen)
                for _,point in highlights:
                    x,y = point
                    if not vision.match(screen.frame,'limited-check',(x+30,max(0,y+5),min(1280,x+115),min(720,y+95)),.9):
                        return self.tap(point,'领取十日签到奖励')
            return DailyDecision('limited_checked',status=page+'当前奖励已检查')
        if self.kind == 'blessing' and page == 'no_family':
            return DailyDecision('unavailable',status='账号未加入家族，暂不能免费祈福')
        if self.kind == 'blessing' and page == 'family':
            point = vision.family_entry(screen.frame)
            return self.tap(point,'进入家族祈福') if point else DailyDecision(status='寻找家族祈福入口')
        expected = {'blessing':'prayer','weapon':'weapon','curse':'curse'}.get(self.kind)
        if page == expected:
            if vision.free(screen,self.kind):
                if self.free_sent:
                    return DailyDecision(status='免费请求已发送，等待结果；不重复抽取')
                point,guard = {'blessing':((310,651),(185,620,440,690)),
                               'weapon':((780,650),(620,610,950,695)),
                               'curse':((430,625),(300,590,565,660))}[self.kind]
                return self.tap(point,'执行明确标为免费的今日次数',free_kind=self.kind,guard=guard)
            if self.kind == 'blessing':
                count = re.search(r'祈福?次数:(\d+)/5',text.replace('已','').replace('己',''))
                done = bool(count and int(count[1]) > 0)
            else:
                cooldown = screen.text((620,570,950,710)) if self.kind == 'weapon' else screen.text((280,575,600,710))
                done = '免费' in cooldown and bool(re.search(r'小时|分钟|\d+分',cooldown))
            if done:
                return DailyDecision('done',status='今日免费次数已领取，付费按钮不操作')
            return DailyDecision(status='等待确认免费或冷却状态')
        if self.kind in ('weapon','curse','limited') and page in ('weapon','vault','summon'):
            if self.kind == 'weapon':
                return self.tap((144,558),'进入百炼武库')
            if self.kind == 'curse':
                return self.tap((1215,365) if page == 'vault' else (140,462),'进入免费咒印炼化')
            if 'anniversary' not in self.limited_checked:
                return self.back((40,48))
            if page != 'summon':
                return self.tap((140,155),'选择当前限时召唤页')
            entries = screen.hits(r'十日签[到角]?',(0,600,580,715))
            point = entries[0].point if entries else vision.match(screen.frame,'ten-signin-entry',(0,600,580,715),threshold=.94)
            return self.tap(point,'打开十日签到') if point else DailyDecision(status='寻找召唤页十日签到入口')
        if self.kind in ('bounty_shop','daily_shop'):
            decision = self.shop(screen,vision)
            if decision:
                return decision
        if self.kind == 'house':
            if page == 'house_continue':
                if '勾玉' not in text and '是否消耗5饭团继续进行修炼' in text:
                    return self.tap((810,530),'消耗5饭团沿用原角色继续修炼',guard=(430,340,920,568))
                return DailyDecision('unavailable',status='续练费用未确认，不操作')
            if page == 'house':
                region = screen.text((270,590,420,715))
                countdown = remaining_seconds(region)
                if countdown is not None and countdown > 0:
                    return DailyDecision('done',status='小屋正在修炼，保留当前角色和倒计时')
                if '修炼完成' in region or vision.match(screen.frame,'house-complete',(260,665,420,715)):
                    return self.tap((320,650),'领取小屋修炼碎片',guard=(270,590,420,715))
                return DailyDecision(status='等待确认小屋修炼状态')
        if self.kind == 'village':
            decision = self.village(screen,vision)
            if decision:
                return decision
        if page == 'rank':
            return self.rank(screen) if self.kind == 'rank' else self.back((75,50))
        if page == 'rank-preview':
            return self.back((75,50))
        if page in ('curse','weapon','vault','prayer','trial','hall','family','shop','limited-shop','daily_shop','bounty_shop','no_family','summon','house'):
            return self.back((90,65) if page == 'family' else (65,50))
        if page == 'activities':
            return self.tap((135,180),'进入每日签到') if self.kind == 'monthly' else self.back((1238,48))
        return DailyDecision(status='等待识别日常页面')

    def rank(self, screen):
        daily = screen.hits('每日任务',(600,210,1240,680))
        main = screen.hits('主线任务',(600,210,1240,680))
        if daily and main and daily[0].point[1] < main[0].point[1]:
            top = daily[0].box[1]+daily[0].box[3]
            bottom = main[0].box[1]
            box = (1060,top,1235,bottom)
            claims = screen.hits('点击[领颌倾]取',box)
            if claims:
                return self.tap(claims[0].point,'领取忍阶已完成每日任务')
            states = screen.hits('已领取|点击前往',box)
            # Claimed rows disappear. Green rows still need a readable claim label.
            buttons = screen.frame[int(top):int(bottom),1060:1235]
            green = ((buttons[:,:,1] > 140)&(buttons[:,:,0] > 80)&
                     (buttons[:,:,2] < 100)).mean() if buttons.size else 0
            if green < .1 and (states or bottom-top < 40 or
                    screen.hits('每日任务(?:已)?全部完成',(600,top,1235,bottom))):
                self.rank_daily_checked = True
                if self.rank_weekly_checked:
                    return DailyDecision('done',status='忍阶每日可领取奖励与每周道具已检查')
                return self.tap((135,685),'检查当前忍阶每周随机道具')
            return DailyDecision(status='等待确认忍阶每日任务领取状态')
        if screen.hits('特殊[忍思]阶',(1080,0,1250,120)):
            weekly = screen.hits('点击[领颌倾]取每周.{0,3}[道遒]具',(600,430,1050,515))
            if weekly:
                return self.tap(weekly[0].point,'领取忍阶每周随机道具')
            if screen.hits(r'每周.{0,2}获得[道遒]具\d+个',(600,350,1100,460)):
                self.rank_weekly_checked = True
            if self.rank_daily_checked and self.rank_weekly_checked:
                return DailyDecision('done',status='忍阶每日可领取奖励与每周道具已检查')
            if not self.rank_weekly_checked:
                return DailyDecision(status='等待确认忍阶每周道具领取状态')
            next_rank = screen.hits('下一[忍思]?阶',(1030,660,1250,720))
            return (self.tap(next_rank[0].point,'从忍阶预览进入下一忍阶任务') if next_rank else
                    DailyDecision(status='等待忍阶预览的任务入口'))
        if self.scans < 2:
            return DailyDecision('swipe',(930,330),'回到忍阶每日任务列表顶部',(930,630))
        return DailyDecision('unavailable',status='未确认完整忍阶每日任务区域，保留截图')

    def shop(self, s, v):
        if s.page == 'limited-shop':
            return self.tap((1190,378) if self.kind == 'bounty_shop' else (1000,690),'进入指定商店分类')
        if s.page == 'shop':
            return self.tap((755,690) if self.kind == 'bounty_shop' else (1000,690),'进入忍币购买页面')
        if s.page == 'bounty_dialog' and self.kind in ('bounty_shop','daily_shop'):
            if self.pending_item is None:
                return self.back((1100,90))
            name = r'[C（(]级.*奖册' if self.pending_item == 'C级' else r'D级.*奖册'
            if self.pending_item and re.search(name,s.text(),re.I) and v.coin(s,(510,640,610,705),large=True):
                return self.tap((640,668),'确认仅使用忍币购买悬赏奖册',guard=(510,632,780,704))
            return DailyDecision('unavailable',status='奖册购买内容或忍币图标未确认，不购买')
        if s.page == 'shop_dialog' and self.kind == 'daily_shop':
            if self.pending_item is None:
                return self.back((1100,90))
            if self.pending_item and v.coin(s,(535,505,635,570),large=True):
                return self.tap((666,537),'确认仅使用忍币购买当前商品',guard=(535,505,800,570))
            return DailyDecision('unavailable',status='购买窗口不是已选忍币商品，不购买')
        if s.page == 'shop_dialog' or s.page == 'bounty_dialog':
            return self.back((1100,90))
        if s.page == 'bounty_shop' and self.kind == 'bounty_shop':
            if not self.bounty_at_start:
                return DailyDecision('swipe',(250,400),'检查悬赏商店最左侧',(950,400))
            found = False
            for name,template in (('D级','bounty-d'),('C级','bounty-c')):
                point = v.match(s.frame,template,(20,250,1080,470))
                if point is None:
                    continue
                found = True
                x = point[0]
                region = (max(0,x-120),130,x+120,635)
                text = s.text(region)
                available = re.search(r'(?:今日|日)剩余(\d+)',text)
                if available and int(available[1]) > 0 and v.coin(s,(x-70,580,x+100,635)):
                    return self.tap((x,360),'选择'+name+'悬赏奖册')
                if available and int(available[1]) > 0:
                    return DailyDecision(status='剩余奖册可买，但忍币图标未确认，不操作')
                return DailyDecision(status='等待确认奖册剩余次数')
            if not found and '神龙之血' in s.text() and '本周' in s.text():
                return DailyDecision('done',status='D、C级忍币悬赏奖册今日均已购买')
            return DailyDecision(status='等待悬赏商店商品展示完成')
        if s.page == 'daily_shop' and self.kind == 'daily_shop':
            confirmed = unknown = 0
            for y in (110,380):
                for x in (20,257,495,731):
                    region = (x,y,x+230,y+250)
                    text = s.text(region)
                    count = re.search(r'剩余次数:(\d+)/',text)
                    if not v.coin(s,(x+35,y+200,x+215,y+250)):
                        continue
                    if count:
                        confirmed += 1
                    else:
                        unknown += 1
                    if count and int(count[1]) > 0:
                        return self.tap((x+115,y+130),'选择每日商店忍币商品')
            if confirmed and not unknown:
                return DailyDecision('done',status='每日商店当前忍币商品均已购买；不付费刷新')
            return DailyDecision(status='等待确认每日商店忍币商品库存')
        if s.page == 'shop' or s.page == 'daily_shop' or s.page == 'bounty_shop':
            return self.tap((1190,378) if self.kind == 'bounty_shop' else (1000,690),'进入指定商店分类')
        return None

    def village(self, s, vision):
        if s.page == 'hall':
            targets = s.hits(r'忍[村材].?[试拭][炼烁]',(200,100,1260,240))
            if targets:
                return self.tap((targets[0].point[0],400),'进入忍村试炼')
            return DailyDecision('swipe',(1120,400),'在集会所向右寻找忍村试炼',(320,400))
        if s.page != 'trial':
            return None
        energy = vision.food(s)
        reserve = reserve_food(self.training_finishes_at,self.clock())
        if energy is None:
            return DailyDecision(status='等待确认饭团数量与小屋预留',defer_on_timeout=True)
        if energy-reserve < 5:
            return DailyDecision('unavailable',status=f'预留{reserve}饭团用于小屋续练后不足5点，停止扫荡；每30分钟恢复1点')
        headers = s.hits(r'第\d+关',(0,190,1280,260))
        for hit in sorted(headers,key=lambda h:h.point[0],reverse=True):
            number = int(re.search(r'\d+',hit.text)[0])
            x = hit.point[0]
            panel = s.text((max(0,x-105),270,min(1280,x+110),570))
            if '首次通关' in panel or number in self.visited_stages:
                continue
            count = re.search(r'挑战次数?\D{0,3}(\d+)',panel)
            if count is None and self.stage == number:
                count = re.search(r'挑战次数?\D{0,3}(\d+)',s.text((1010,570,1270,625)))
            if count and int(count[1]) == 0:
                self.visited_stages.add(number)
                continue
            if count is None:
                if self.stage != number and '关奖励' in panel:
                    return self.tap((x,360),f'选择右侧第{number}关')
                return DailyDecision(status='等待确认右侧关卡剩余次数',defer_on_timeout=True)
            if count:
                if self.stage != number:
                    return self.tap((x,360),f'选择右侧第{number}关')
                return self.tap((895,649),f'扫荡第{number}关（消耗5体力）',guard=(800,620,995,677))
        if self.scans < 7:
            return DailyDecision('swipe',(270,400),'按从右到左的顺序寻找可扫荡关卡',(1080,400))
        return DailyDecision('unavailable',status='未确认其余关卡扫荡状态，保留未完成状态')

    def completed(self, decision, screen):
        completed = getattr(self.task_flow,'completed',None)
        if completed is not None:
            completed(decision,screen)
        if decision.free_kind:
            self.free_sent = True
        if decision.action == 'swipe' and screen.page != 'hall':
            self.scans += 1
        if decision.status == '检查悬赏商店最左侧':
            self.bounty_at_start = True
        if decision.action == 'limited_checked':
            self.limited_checked.add(screen.page)
        if decision.status.startswith('选择右侧第'):
            self.stage = int(re.search(r'第(\d+)关',decision.status)[1])
        if ((decision.status.startswith('选择') and '商' in decision.status) or
                decision.status.startswith('选择D级') or decision.status.startswith('选择C级')):
            self.pending_item = 'D级' if 'D级' in decision.status else 'C级' if 'C级' in decision.status else 'ninbi'
            if decision.status == '选择每日商店忍币商品':
                x,y = decision.point
                name = re.search(r'[DC]级',screen.text((x-115,y-130,x+115,y+120)),re.I)
                if name:
                    self.pending_item = name[0].upper()
