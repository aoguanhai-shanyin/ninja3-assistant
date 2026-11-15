"""Confirmed navigation and daily-task lifecycle for 情报模拟战."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from activity_policy import Card, choose_purchase

CHINA = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class ActivityScreen:
    kind: str = 'unknown'
    point: tuple | None = None
    daily: tuple = ()  # Same order as task_ids; unknown/incomplete/claimable/claimed.
    claim_points: tuple = ()
    battle: object | None = None
    task: object | None = None
    task_ids: tuple = ('games', 'damage')


@dataclass(frozen=True)
class ActivityDecision:
    action: str = 'wait'
    point: tuple | None = None
    status: str = ''
    end_point: tuple | None = None


def matching_open(now=None):
    now = (now or datetime.now(CHINA)).astimezone(CHINA)
    return 10 <= now.hour < 14 or 18 <= now.hour < 22


def daily_key(now=None):
    now = (now or datetime.now(CHINA)).astimezone(CHINA)
    return (now-timedelta(hours=5)).date()


class ActivityFlow:
    def __init__(self, mode='daily'):
        if mode not in ('daily', 'continuous'):
            raise ValueError('activity mode must be daily or continuous')
        self.mode = mode
        self.checked_day = None
        self.daily_complete = False
        self.last_action = None
        self.last_action_at = float('-inf')
        self.pending = None
        self.purchase = None
        self.refreshed_round = None
        self.round_number = None
        self.await_home = False
        self.completed_matches = 0
        self.claim_target = None
        self.claiming = None
        self.confirmed_claims = set()
        self.claim_day = None
        self.task_states = {}
        self.tasks_scanned = 0
        self.tasks_positioned = False
        self.top_daily_rows = ()
        self.row_based_tasks = False

    def reset_inputs(self):
        self.pending = self.purchase = None

    def step(self, screen, now, wall_time=None):
        day = daily_key(wall_time)
        if self.claim_day != day:
            self.claim_day = day
            self.confirmed_claims.clear()
            self.claiming = None
            self.task_states.clear()
            self.tasks_scanned = 0
            self.tasks_positioned = False
            self.top_daily_rows = ()
            self.row_based_tasks = False
        if self.checked_day != day:
            self.daily_complete = False
        if screen.kind == 'invite':
            return ActivityDecision('invite', screen.point, '拒绝队友邀请')
        if screen.kind == 'battle':
            return self.battle_step(screen.battle, now)
        if now-self.last_action_at < 1:
            return ActivityDecision(status='等待活动页面更新')
        if screen.kind == 'reward' and screen.point:
            return ActivityDecision('confirm_reward', screen.point, '确认活动任务奖励')
        if screen.kind in ('login', 'home', 'hall', 'network_retry'):
            if screen.kind == self.last_action and now-self.last_action_at < 8:
                return ActivityDecision(status='等待游戏页面切换')
            messages = {'login': '登录游戏', 'home': '进入集会所',
                        'hall': '进入情报模拟战', 'network_retry': '重试游戏连接'}
            return ActivityDecision(screen.kind, screen.point, messages[screen.kind])
        if screen.kind == 'tasks':
            task_ids = screen.task_ids
            daily, points = list(screen.daily), screen.claim_points
            rows = getattr(screen.task,'rows',()) if screen.task else ()
            if rows:
                self.row_based_tasks = True
                if not self.tasks_positioned:
                    return ActivityDecision('reset_tasks_scroll',(760,177),'回到日常任务列表顶部',(760,315))
                if not self.tasks_scanned:
                    self.top_daily_rows = tuple(row.signature for row in rows)
                    offset = 0
                elif (len(self.top_daily_rows)>=2 and len(rows)>=2 and
                      self.same_daily_row(self.top_daily_rows[-1],rows[0].signature)):
                    offset = 3-len(rows)
                elif 'incomplete' in self.task_states.values():
                    self.checked_day=day
                    return ActivityDecision('close_tasks',(1120,111),'日常仍未完成，继续对战后复查')
                elif self.tasks_scanned<2:
                    return ActivityDecision('scroll_tasks',(760,315),'继续滚动检查未显示的日常',(760,177))
                else:
                    return ActivityDecision('retry',status='未确认日常滚动结果，交回计划重新检查')
                task_ids=tuple('daily_'+str(offset+i) for i in range(len(rows)))
                daily=[row.state for row in rows]
                points=tuple(row.point for row in rows)
            if not daily or len(daily) != len(task_ids):
                self.daily_complete = False
                return ActivityDecision(status='未确认完整日常任务状态，保留当前画面')
            if screen.task:
                for index, task_id in enumerate(task_ids):
                    if task_id not in screen.task.present and task_id in self.confirmed_claims:
                        daily[index] = 'claimed'
            self.task_states.update((key, state) for key, state in zip(task_ids, daily) if state != 'unknown')
            self.task_states.update((key, 'claimed') for key in self.confirmed_claims)
            if not self.tasks_positioned:
                return ActivityDecision('reset_tasks_scroll', (760, 177), '回到日常任务列表顶部', (760, 315))
            for index, (state, point) in enumerate(zip(daily, points)):
                if state == 'claimable' and point:
                    self.claim_target = task_ids[index]
                    self.daily_complete = False
                    return ActivityDecision('claim', point, '领取情报模拟战日常奖励')
            if 'claimable' in daily:
                self.daily_complete = False
                return ActivityDecision(status='日常可领取，但未确认领取按钮')
            if not self.tasks_scanned or (len(self.task_states) < 3 and self.tasks_scanned < 2):
                return ActivityDecision('scroll_tasks', (760, 315), '滑动检查第三项日常任务', (760, 177))
            required=(('daily_0','daily_1','daily_2') if self.row_based_tasks else
                      ('games','damage','characters'))
            self.daily_complete = all(self.task_states.get(key) == 'claimed' for key in required)
            if not self.daily_complete and 'incomplete' not in self.task_states.values():
                self.daily_complete = False
                if self.mode == 'continuous':
                    self.checked_day = day
                    return ActivityDecision('close_tasks', (1120, 111), '日常状态暂未确认，继续活动对战')
                return ActivityDecision('retry',status='日常进度暂未确认，交回计划重新检查') if rows else ActivityDecision(status='未确认完整日常任务状态，保留当前画面')
            self.checked_day = day
            if self.daily_complete and self.mode == 'daily':
                return ActivityDecision('done', status='情报模拟战日常已全部领取，任务结束')
            return ActivityDecision('close_tasks', (1120, 111), '关闭任务页，准备对战')
        if screen.kind == 'event_home':
            if self.await_home:
                self.await_home = False
                self.completed_matches += 1
                self.checked_day = None
            if self.checked_day != day:
                return ActivityDecision('tasks', (866, 51), '检查情报模拟战每日任务')
            if self.daily_complete and self.mode == 'daily':
                return ActivityDecision('done', status='情报模拟战日常已全部领取，任务结束')
            if not matching_open(wall_time):
                return ActivityDecision(status='等待情报模拟战开放（每日 10–14、18–22 点）')
            return ActivityDecision('match', (640, 563), '开始情报模拟战匹配')
        return ActivityDecision(status='等待识别情报模拟战画面，未知页面不操作')

    @staticmethod
    def same_daily_row(first,second):
        return (bool(first) and len(first)==len(second) and
                sum((a^b).bit_count() for a,b in zip(first,second)) <=
                max(4,sum((a|b).bit_count() for a,b in zip(first,second))*.4))

    def battle_step(self, battle, now):
        if battle is None:
            return ActivityDecision(status='等待确认活动对局')
        if battle.kind == 'result' and battle.result_point:
            self.reset_inputs()
            if now-self.last_action_at < 1:
                return ActivityDecision(status='等待活动结算更新')
            return ActivityDecision('result_confirm', battle.result_point, '确认活动结算并复查日常')
        if battle.kind in ('combat', 'prepared', 'matching'):
            if battle.kind == 'combat':
                self.reset_inputs()
            return ActivityDecision(status={'combat': '活动战斗阶段，等待下一回合或结算',
                'prepared': '活动已准备，等待对方', 'matching': '正在匹配活动对手'}[battle.kind])
        if battle.kind != 'placement':
            return ActivityDecision(status='活动对局暂未确认，停止输入')
        if battle.round_number != self.round_number:
            self.round_number = battle.round_number
            self.refreshed_round = None
            self.reset_inputs()
        if len(battle.board) != 9 or len(battle.shop) != 5:
            return ActivityDecision(status='等待确认活动棋盘和商店布局')
        if battle.coins is None or battle.round_number is None:
            return ActivityDecision(status='等待确认本回合金币和轮次')
        if self.pending:
            shop, cell, card, coins, started = self.pending
            if (battle.selected_shop == shop and
                    ((battle.shop[shop] and battle.shop[shop].id == card.id) or
                     (battle.shop[shop] is None and shop in battle.unknown_shop)) and
                    battle.coins == coins and
                    len(battle.place_points) == 9 and
                    battle.place_points[cell] and battle.board[cell] is None):
                return ActivityDecision('place', battle.place_points[cell], '将已选卡牌放入确认空格')
            elif now-started > 1.5:
                self.reset_inputs()
            else:
                return ActivityDecision(status='等待商店选中标记与空格加号')
        if now-self.last_action_at < .35:
            return ActivityDecision(status='等待活动操作反馈')
        # An unreadable occupied card affects strategy estimates, not the ability
        # to buy into another confirmed empty cell. Keep its slot occupied.
        policy_board = tuple(Card('unrecognized', 0, 0) if index in battle.unknown_board else card
                             for index, card in enumerate(battle.board))
        choice = choose_purchase(policy_board, battle.shop, battle.coins, battle.round_number)
        if choice:
            from activity_battle_vision import SHOP_POINTS
            shop, cell = choice
            if (battle.selected_shop == shop and len(battle.place_points) == 9 and
                    battle.place_points[cell]):
                self.pending = (shop, cell, battle.shop[shop], battle.coins, now)
                return ActivityDecision('place', battle.place_points[cell], '继续放置当前已确认选中的卡牌')
            self.purchase = (shop, cell, battle.shop[shop], battle.coins)
            return ActivityDecision('select_card', SHOP_POINTS[shop], '按当前阵容选择可购买卡牌')
        affordable = any(card and card.cost <= battle.coins for card in battle.shop)
        if (not affordable and battle.coins > 0 and None in policy_board and
                battle.refresh_point and self.refreshed_round != battle.round_number):
            return ActivityDecision('refresh', battle.refresh_point, '使用本回合免费刷新寻找可购买卡牌')
        if battle.ready_point:
            return ActivityDecision('prepare', battle.ready_point, '本回合购牌结束，准备战斗')
        return ActivityDecision(status='等待确认活动准备按钮')

    def completed(self, decision, now):
        self.last_action, self.last_action_at = decision.action, now
        if decision.action == 'select_card' and self.purchase:
            self.pending = (*self.purchase, now)
            self.purchase = None
        elif decision.action == 'place' and self.pending:
            self.pending = None
        elif decision.action == 'refresh':
            self.refreshed_round = self.round_number
        elif decision.action == 'result_confirm':
            self.await_home = True
        elif decision.action == 'claim':
            self.claiming = self.claim_target
        elif decision.action == 'tasks':
            self.task_states.clear()
            self.tasks_scanned = 0
            self.tasks_positioned = False
            self.top_daily_rows = ()
        elif decision.action == 'reset_tasks_scroll':
            self.tasks_positioned = True
        elif decision.action == 'scroll_tasks':
            self.tasks_scanned += 1
        elif decision.action == 'confirm_reward':
            if self.claiming:
                self.confirmed_claims.add(self.claiming)
                self.claiming = None
            self.tasks_scanned = 0
            self.tasks_positioned = False
