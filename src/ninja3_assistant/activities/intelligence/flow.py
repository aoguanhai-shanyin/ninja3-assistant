"""Confirmed navigation and daily-task lifecycle for 情报模拟战."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ninja3_assistant.activities.intelligence.policy import Card, choose_purchase

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
    selection: dict | None = None


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
        self.combo = None
        self.refreshed_round = None
        self.last_refresh_count = None
        self.round_number = None
        self.unknown_shop_since = None
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
        self.combo = None

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
        if screen.kind == 'guide' and screen.point:
            return ActivityDecision('guide', screen.point, '继续情报模拟战玩法说明')
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
            self.round_number = None
            if now-self.last_action_at < 1:
                return ActivityDecision(status='等待活动结算更新')
            return ActivityDecision('result_confirm', battle.result_point, '确认活动结算并复查日常')
        if battle.kind in ('combat', 'prepared', 'matching'):
            if battle.kind == 'combat':
                self.reset_inputs()
            elif battle.kind == 'matching':
                self.round_number = None
                self.reset_inputs()
            return ActivityDecision(status={'combat': '活动战斗阶段，等待下一回合或结算',
                'prepared': '活动已准备，等待对方', 'matching': '正在匹配活动对手'}[battle.kind])
        if battle.kind != 'placement':
            return ActivityDecision(status='活动对局暂未确认，停止输入')
        if battle.round_number != self.round_number:
            self.round_number = battle.round_number
            self.unknown_shop_since = None
            self.refreshed_round = None
            self.last_refresh_count = None
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
        # Continue the original combination only after a new frame matches its predicted state.
        current = (policy_board, tuple(battle.shop), battle.coins)
        if self.combo is not None and current != self.combo[:3]:
            self.combo = None
        if self.combo is not None:
            _, _, _, plan, original = self.combo
            choice = plan[0]['shop'], plan[0]['cell']
            spent = sum(item['cost'] for item in plan)
            selection = {'budget':battle.coins,'spent':spent,
                'remaining_coins':battle.coins-spent,'round':battle.round_number,
                'plan':list(plan),'continued_combo':True,'original_selection':original}
        else:
            selection = {}
            choice = choose_purchase(policy_board, battle.shop, battle.coins, battle.round_number,
                                     explanation=selection)
            if choice and len(selection.get('plan',())) > 1:
                self.combo = (*current,tuple(selection['plan']),dict(selection))
        selection.update(unknown_board=list(battle.unknown_board), unknown_shop=list(battle.unknown_shop))
        resource_reason = ''
        if 'held_coin_value' in selection:
            resource_reason = (f"；保留{selection['remaining_coins']}金币、{selection['empty_cells']}空位，"
                               f"持币机会估值{selection['held_coin_value']:g}（估算）")
        if choice:
            from ninja3_assistant.activities.intelligence.battle_vision import SHOP_POINTS
            shop, cell = choice
            if (battle.selected_shop == shop and len(battle.place_points) == 9 and
                    battle.place_points[cell]):
                self.pending = (shop, cell, battle.shop[shop], battle.coins, now)
                return ActivityDecision('place', battle.place_points[cell], '继续放置当前已确认选中的卡牌',
                                        selection=selection)
            self.purchase = (shop, cell, battle.shop[shop], battle.coins)
            planned = ' → '.join(f"{item['name']}放{item['cell']//3+1}行{item['cell']%3+1}列"
                                  for item in selection['plan'])
            if selection.get('continued_combo'):
                reason = (f"续接已确认组合，预算{battle.coins}，剩余组合花费{selection['spent']}：{planned}"
                          f"（原组合预计增益{selection['original_selection']['estimated_gain']:g}）")
            else:
                reason = (f"预算{battle.coins}，组合花费{selection['spent']}，预计增益{selection['estimated_gain']:g}，"
                          f"组合联动增益{selection['synergy_gain']:g}：{planned}")
                if selection['future_income'] or selection['granted_refreshes']:
                    reason += (f"；每回合收入{selection['income_per_turn']}，后续共{selection['future_income']}金币，"
                               f"回合结束增加{selection['granted_refreshes']}次刷新（未来收益估算）")
                if selection['special_victory']:
                    reason += '；组合可凑齐鬼斩角色和三种不同铠甲，优先特殊胜利'
                reason += resource_reason
            if battle.unknown_board or battle.unknown_shop:
                reason += '；存在未识别卡，仅按已确认信息规划'
            return ActivityDecision('select_card', SHOP_POINTS[shop], reason, selection=selection)
        affordable = any(card and card.cost <= battle.coins for card in battle.shop)
        if (not selection.get('special_victory') and battle.unknown_shop and
                battle.coins > 0 and None in policy_board and not affordable):
            if self.unknown_shop_since is None:
                self.unknown_shop_since = now
            if now-self.unknown_shop_since < 1:
                return ActivityDecision(status='商店卡牌未识别，已留存样本，短暂重读后继续', selection=selection)
        refresh_available = (self.refreshed_round != battle.round_number or
            (battle.refreshes is not None and self.last_refresh_count is not None and
             0 < battle.refreshes < self.last_refresh_count))
        if (not selection.get('special_victory') and battle.coins > 0 and None in policy_board and
                battle.refresh_point and refresh_available):
            selection['refreshes_before'] = battle.refreshes
            return ActivityDecision('refresh', battle.refresh_point,
                                    '使用本回合免费刷新寻找可购买卡牌'+resource_reason, selection=selection)
        if battle.ready_point:
            reason = '鬼斩特殊胜利条件已齐，保留阵容并准备' if selection.get('special_victory') else '本回合购牌结束，准备战斗'
            return ActivityDecision('prepare', battle.ready_point, reason+resource_reason, selection=selection)
        return ActivityDecision(status='等待确认活动准备按钮')

    def completed(self, decision, now):
        self.last_action, self.last_action_at = decision.action, now
        if decision.action == 'select_card' and self.purchase:
            self.pending = (*self.purchase, now)
            self.purchase = None
        elif decision.action == 'place' and self.pending:
            if self.combo is not None:
                board,shop,coins,plan,original = self.combo
                index,cell,card,_,_ = self.pending
                if (plan[0]['shop'],plan[0]['cell']) == (index,cell) and len(plan) > 1:
                    board,shop = list(board),list(shop)
                    board[cell],shop[index] = card,None
                    self.combo = (tuple(board),tuple(shop),coins-card.cost,plan[1:],original)
                else:
                    self.combo = None
            self.pending = None
        elif decision.action == 'refresh':
            self.combo = None
            self.refreshed_round = self.round_number
            self.last_refresh_count = (decision.selection or {}).get('refreshes_before')
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
