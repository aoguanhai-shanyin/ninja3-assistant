"""Daily bounty menu decisions; combat and touch execution stay in the runner."""
import time

from ninja3_assistant.activities.daily.flow import DailyDecision


class DailyBountyFlow:
    def __init__(self, kind='bounty', grade='D', *, clock=time.monotonic):
        if kind not in ('bounty','bounty_s') or grade not in ('D','C','B','A'):
            raise ValueError('expected a daily bounty kind and grade D/C/B/A')
        self.kind,self.grade = kind,grade
        self.free_checked = False
        self.free_attempted = False
        self.free_baseline = None
        self.confirmed_grade = None
        self.room_trusted = False
        self.battle_seen = False
        self.clock = clock
        self.room_since = None
        self.room_owned = False
        self.try_quick_join = False
        self.quick_join_at = None
        self.no_owned_orders = False
        self.order_quota = None

    @staticmethod
    def tap(point, status, **kwargs):
        return DailyDecision('tap',point,status,**kwargs)

    def step(self, screen, vision):
        page = screen.page
        if page != 'bounty_room':
            self.room_since = None
        if page == 'invite':
            return self.tap(screen.point,'拒绝队友或家族邀请',guard=(340,475,940,585))
        if page == 'battle':
            self.battle_seen = True
            return DailyDecision('battle',status='悬赏战斗：地面滑步，绳索停手，Boss出现后释放忍术')
        if page == 'result':
            expected_grade = 'S' if self.kind == 'bounty_s' else self.grade
            if screen.grade is not None and screen.grade != expected_grade:
                return DailyDecision('unavailable',status='悬赏结算等级与所选等级不符，保留画面')
            if self.room_trusted or self.battle_seen or screen.grade == expected_grade:
                return self.tap(screen.confirm,'确认悬赏结算，返回后复查剩余次数') if screen.confirm else DailyDecision(status='等待悬赏结算按钮')
            return DailyDecision('unavailable',status='未确认当前结算属于本次悬赏，保留画面')
        if page == 'reward':
            return self.tap(screen.confirm,'确认悬赏奖励') if screen.confirm else DailyDecision(status='等待奖励确定按钮')
        if page == 'home':
            return self.tap((865,600),'进入集会所处理每日悬赏')
        if page == 'hall':
            if screen.entry:
                return self.tap(screen.entry,'进入悬赏令')
            return DailyDecision('swipe',(280,400),'寻找集会所的悬赏令入口',(1080,400))
        if page == 'bounty_hub':
            return self.tap((360,400) if self.kind == 'bounty' else (930,400),
                            '进入每日悬赏' if self.kind == 'bounty' else '进入S级召令悬赏')
        if page == 'bounty_locked':
            if self.kind == 'bounty_s':
                return DailyDecision('unavailable',status='召令悬赏尚未通过等级测试，继续其他日常')
            return self.tap((65,50),'返回悬赏分类，继续每日悬赏')
        if page == 'bounty_grade':
            if screen.grade == self.grade and screen.confirm:
                return self.tap(screen.confirm,'确认每日悬赏等级为'+self.grade)
            point = screen.grade_choices.get(self.grade)
            return self.tap(point,'只选择'+self.grade+'级每日悬赏') if point else DailyDecision(status='等待目标等级选项')
        if page == 'bounty_leave':
            return self.tap((810,525),'确认离开等待中的悬赏队伍',guard=(530,340,760,395))
        if page == 'bounty_free':
            if self.kind != 'bounty':
                return self.tap(screen.close,'关闭每日次数窗口') if screen.close else DailyDecision(status='等待次数窗口关闭按钮')
            if screen.free_exhausted:
                self.free_checked = True
                return self.tap(screen.close,'免费次数已使用，不购买付费次数') if screen.close else DailyDecision(status='等待次数窗口关闭按钮')
            if screen.free_claim and not self.free_attempted:
                return self.tap(screen.free_claim,'领取明确免费的每日悬赏次数',free_kind='bounty',guard=screen.free_guard)
            return DailyDecision(status='等待免费次数领取结果，不重复购买')
        if page == 'bounty_daily':
            if self.kind != 'bounty':
                return self.tap((65,50),'返回悬赏分类，继续S级悬赏')
            if screen.join_failed:
                return DailyDecision('retry',status='当前没有'+self.grade+'级队伍，稍后重试')
            if screen.remaining is None:
                return DailyDecision(status='等待确认每日悬赏剩余次数')
            if self.free_attempted and self.free_baseline is not None and screen.remaining > self.free_baseline:
                self.free_checked = True
            if not self.free_checked:
                return self.tap((380,172),'检查每日悬赏免费次数')
            if screen.remaining == 0:
                return DailyDecision('done',status='每日悬赏次数与免费购买次数已清理')
            if self.confirmed_grade != self.grade or (screen.grade is not None and screen.grade != self.grade):
                return self.tap((730,670),'设置每日悬赏等级为'+self.grade)
            if screen.join:
                return self.tap(screen.join,'加入'+self.grade+'级每日悬赏队伍')
            if self.try_quick_join:
                if self.quick_join_at is not None:
                    if self.clock()-self.quick_join_at >= 5:
                        return DailyDecision('retry',status='当前没有'+self.grade+'级队伍，稍后重试')
                    return DailyDecision(status='等待快捷加入目标等级悬赏队伍')
                return self.tap((1120,670),'快捷加入'+self.grade+'级每日悬赏队伍')
            return self.tap((875,670),'创建'+self.grade+'级每日悬赏队伍')
        if page == 'bounty_order':
            if self.kind != 'bounty_s':
                return self.tap((65,50),'返回悬赏分类，继续每日悬赏')
            if screen.remaining is None:
                return DailyDecision(status='等待确认S级悬赏累计次数')
            if screen.remaining == 0:
                return DailyDecision('done',status='S级悬赏累计次数已清理')
            self.order_quota = screen.remaining
            if screen.s_filter == 'off':
                return self.tap((876,171),'开启S级悬赏筛选')
            if screen.ss_filter == 'on':
                return self.tap((978,171),'关闭SS+悬赏筛选')
            if (screen.s_filter,screen.ss_filter) != ('on','off'):
                return DailyDecision(status='等待确认只选择S级悬赏')
            if screen.join:
                return self.tap(screen.join,'加入S级悬赏队伍')
            if self.no_owned_orders:
                return DailyDecision('retry',status='未持有S级召令，暂未找到S级队伍，稍后重试')
            return self.tap((1115,670),'打开我的S级召令悬赏')
        if page == 'bounty_own_empty':
            return self.tap((1115,670),'未持有召令，返回队伍列表寻找其他队伍')
        if page == 'bounty_create':
            if self.kind == 'bounty_s' and not self.order_quota:
                return self.tap((65,50),'先返回召令列表确认S级剩余次数')
            if screen.grade != ('S' if self.kind == 'bounty_s' else self.grade):
                return DailyDecision(status='等待确认创建队伍的悬赏等级')
            return self.tap(screen.create,'确认创建目标等级的悬赏队伍') if screen.create else DailyDecision(status='等待创建队伍按钮')
        if page == 'bounty_room':
            expected_grade = 'S' if self.kind == 'bounty_s' else self.grade
            if screen.grade is not None and screen.grade != expected_grade:
                return DailyDecision('unavailable',status='当前队伍等级与所选悬赏不符')
            if screen.grade == expected_grade:
                self.room_trusted = True
            if not self.room_trusted:
                return DailyDecision('unavailable',status='未确认当前队伍属于所选悬赏等级')
            if self.room_since is None:
                self.room_since = self.clock()
            if screen.minimum_players:
                if self.room_owned and self.clock()-self.room_since >= 20:
                    return self.tap((65,50),'离开等待过久的自建悬赏队伍，尝试加入其他队伍')
                return DailyDecision(status='悬赏队伍至少需要二人，等待其他玩家加入')
            if screen.start:
                return self.tap(screen.start,'开始目标等级悬赏战斗')
            if screen.ready:
                return self.tap(screen.ready,'悬赏队伍准备')
            return DailyDecision(status='悬赏队伍已准备，等待开战')
        if getattr(screen,'back',None):
            return self.tap(screen.back,'返回主页，继续每日悬赏')
        if getattr(screen,'login',None):
            return self.tap(screen.login,'登录当前已选账号')
        return DailyDecision(status='等待识别每日悬赏页面')

    def completed(self, decision, screen):
        if screen.page == 'bounty_grade' and screen.grade == self.grade and decision.status.startswith('确认每日悬赏等级为'):
            self.confirmed_grade = self.grade
        if decision.status == '检查每日悬赏免费次数':
            self.free_baseline = screen.remaining
        if decision.free_kind == 'bounty':
            self.free_attempted = True
        if decision.status.startswith(('加入','创建','确认创建目标')):
            self.room_trusted = True
            self.room_owned = decision.status.startswith(('创建','确认创建目标'))
        if decision.status.startswith('快捷加入'):
            self.room_trusted,self.room_owned = True,False
            self.quick_join_at = self.clock()
        if decision.status == '确认离开等待中的悬赏队伍':
            self.try_quick_join = True
            self.room_trusted,self.room_owned = False,False
        if decision.status == '未持有召令，返回队伍列表寻找其他队伍':
            self.no_owned_orders = True
        if screen.page in ('bounty_daily','bounty_order'):
            self.battle_seen = False
