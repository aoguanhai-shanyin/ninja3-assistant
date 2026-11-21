"""The reverse-S workflow; it only acts on the current recognized screen."""
from dataclasses import dataclass

from ninja3_assistant.activities.bounty.reverse_menu import FILTER_POINTS, join_control

@dataclass(frozen=True)
class Decision:
    action: str
    point: tuple | None = None
    status: str = ''

class Flow:
    def __init__(self):
        self.phase = 'start'
        self.pending = None
        self.pending_at = 0.0
        self.refreshed_at = 0.0
        self.refresh_was_cooling = False
        self.joined_at = 0.0
        self.ready_at = 0.0
        self.join_point = None
        self.attempted = False
        self.unknown_since = None
        self.reopened = False
        self.trusted_join = False
        self.confirmation_at = None
        self.successful_kills = 0
        self.entering = False
        self.entry_action = None
        self.entry_at = 0.0
        self.entry_attempts = 0
        self.loading_since = None
        self.training_armed = False
        self.training_seen_at = None
        self.training_exit_since = None

    def step(self, screen, now):
        recognized = bool(screen.board or screen.menu or screen.ready or screen.prepared or screen.result or
            screen.battle_loading or screen.battle_active or screen.login or screen.home or screen.hall or
            screen.network_error or screen.invite)
        if screen.training and not recognized:
            self.entering = False
            self.phase, self.pending, self.trusted_join = 'start', None, False
            self.confirmation_at = None
            self.unknown_since = self.loading_since = None
            self.reopened = False
            self.training_armed = True
            self.training_seen_at = now
            self.training_exit_since = None
            return Decision('wait', status='训练所选关中，等待开始训练')
        if recognized:
            self.training_armed = False
            self.training_seen_at = self.training_exit_since = None
        elif self.training_armed:
            if self.training_exit_since is None:
                self.training_exit_since = self.training_seen_at
            if now-self.training_exit_since < 45:
                self.unknown_since = None
                return Decision('wait', status='训练场景过渡中，等待战斗画面')
            self.training_armed = False
            self.training_seen_at = self.training_exit_since = self.loading_since = None
            return self.unknown(now, '训练场景过渡超时', limit=0)
        if screen.game_loading and not recognized:
            if self.loading_since is None:
                self.loading_since = now
            if now-self.loading_since < 30:
                self.unknown_since = None
                return Decision('wait', status='游戏载入中，等待画面恢复')
        elif recognized:
            self.loading_since = None
        if screen.network_error:
            self.entering = True
            if screen.network_retry:
                return self.entry('network_retry', screen.network_retry, now, '游戏网络异常，尝试重新连接')
            return Decision('wait', status='游戏网络异常，等待重试按钮或手动处理')
        if screen.login:
            self.entering = True
            self.phase, self.pending, self.trusted_join = 'start', None, False
            self.confirmation_at = None
            return self.entry('login', screen.login, now, '识别到登录页面，点击登录')
        if screen.home:
            self.entering = True
            self.phase, self.pending, self.trusted_join = 'start', None, False
            return self.entry('home', screen.home, now, '进入集会所')
        if screen.hall:
            self.entering = True
            self.phase, self.pending, self.trusted_join = 'start', None, False
            return self.entry('hall', screen.hall, now, '进入悬赏令')
        if self.entering:
            if screen.board or screen.menu or screen.ready or screen.prepared or screen.result or screen.battle_loading or screen.battle_active:
                self.entering = False
            elif not screen.invite:
                return Decision('wait', status='等待游戏加载；如有账号验证或未知弹窗，请手动处理')
        if self.pending in ('invite', 'menu'):
            still_visible = screen.invite if self.pending == 'invite' else screen.menu
            if still_visible:
                if now - self.pending_at >= 4:
                    return Decision('stop', status='页面未切换，请检查游戏画面')
                return Decision('wait', status='等待页面切换')
            self.pending = None
        if self.confirmation_at is not None:
            if screen.board or screen.menu:
                self.confirmation_at = None
                self.successful_kills += 1
            elif screen.invite:
                return Decision('invite', screen.invite, '拒绝队友邀请')
            elif now - self.confirmation_at >= 8:
                return Decision('stop', status='结算后未能确认返回列表')
            else:
                return Decision('wait', status='等待结算完成并返回列表')
        if screen.invite:
            return Decision('invite', screen.invite, '拒绝队友邀请')
        if screen.menu:
            return Decision('menu', screen.menu, '返回逆袭悬赏列表')
        if screen.result:
            if self.trusted_join or screen.reverse_result:
                return Decision('confirm', screen.result, '战斗结算，点击确定并继续')
            return Decision('stop', status='无法确认当前结算属于逆袭 S')

        if self.phase == 'start' and (screen.battle_loading or screen.battle_active):
            self.phase = 'battle'
            return Decision('wait', status='战斗中，等待结算')

        if self.phase in ('start', 'search'):
            if not screen.board:
                if self.phase == 'start' and (screen.ready or screen.prepared):
                    return Decision('stop', status='无法确认当前房间是逆袭 S，请返回列表')
                return self.unknown(now)
            if self.pending:
                expected = {'a_off': screen.a == 'off', 's_on': screen.s == 'on',
                            'ss_off': screen.ss == 'off'}[self.pending]
                if not expected:
                    if now - self.pending_at >= 4:
                        return Decision('stop', status='等级筛选未确认，请检查画面')
                    return Decision('wait', status='确认等级筛选')
                self.pending = None
                self.phase = 'start'
            if '?' in (screen.a, screen.s, screen.ss):
                return self.unknown(now, '无法确认等级筛选')
            self.unknown_since = None
            self.reopened = False
            if screen.s == 'off': return Decision('s_on', FILTER_POINTS['s_on'], '开启 S 筛选')
            if screen.a == 'on': return Decision('a_off', FILTER_POINTS['a_off'], '关闭 A 筛选')
            if screen.ss == 'on': return Decision('ss_off', FILTER_POINTS['ss_off'], '关闭 SS+ 筛选')
            if self.phase == 'start':
                if screen.join:
                    return Decision('join', screen.join, '发现 S 队伍，立即尝试加入')
                return Decision('refresh', screen.refresh, '刷新逆袭 S 列表') if screen.refresh else Decision('wait', status='等待刷新冷却')
            if screen.refresh is None:
                self.refresh_was_cooling = True
            if not self.attempted and screen.join:
                return Decision('join', screen.join, '发现 S 队伍，尝试加入')
            if screen.refresh and (self.refresh_was_cooling or now - self.refreshed_at >= 1):
                return Decision('refresh', screen.refresh, '继续刷新逆袭 S')
            return Decision('wait', status='等待 S 队伍')

        if self.phase == 'joining':
            if screen.battle_loading or screen.battle_active:
                self.phase = 'battle'
                return Decision('wait', status='战斗中，等待结算')
            if not screen.board and screen.prepared:
                self.phase = 'await_battle'
                return Decision('wait', status='已准备，等待队伍开始战斗')
            if not screen.board and screen.ready:
                return Decision('ready', screen.ready, '进入房间，点击准备')
            if screen.board and screen.failed:
                self.phase = 'search'
                self.trusted_join = False
                if screen.refresh and (self.refresh_was_cooling or now - self.refreshed_at >= 1):
                    return Decision('refresh', screen.refresh, '队伍已开战、解散或关闭，刷新列表')
                return Decision('wait', status='队伍已开战、解散或关闭，等待刷新')
            if screen.board and self.join_point:
                filters = (screen.a, screen.s, screen.ss)
                if '?' in filters:
                    return Decision('wait', status='等待确认 S 筛选')
                if filters != ('off', 'on', 'off'):
                    self.phase = 'start'
                    self.trusted_join = False
                    return Decision('wait', status='筛选发生变化，重新确认')
                if point := join_control(screen, self.join_point):
                    return Decision('join', point, '继续尝试加入 S 队伍')
            if screen.board and now - self.joined_at >= 5:
                self.phase = 'search'
                self.trusted_join = False
                return Decision('wait', status='队伍已不可识别，等待刷新')
            if now - self.joined_at >= 15:
                return self.unknown(now)
            return Decision('wait', status='等待加入房间')

        if screen.battle_loading or screen.battle_active:
            self.phase = 'battle'
            self.unknown_since = None
            self.reopened = False
            return Decision('wait', status='战斗中，等待结算')
        if screen.board:
            self.phase = 'start'
            return Decision('wait', status='已返回列表，准备下一轮')
        if screen.prepared:
            self.phase = 'await_battle'
            self.unknown_since = None
            return Decision('wait', status='已准备，等待队伍开始战斗')
        if self.phase == 'await_battle' and screen.ready:
            return Decision('ready', screen.ready, '重新点击准备')
        if self.phase == 'verify_ready':
            if now - self.ready_at >= 5:
                return self.unknown(now, '等待准备或战斗画面')
            return Decision('wait', status='等待准备状态确认')
        if self.phase == 'battle':
            return self.unknown(now, '等待战斗结束或结算', limit=90)
        return self.unknown(now, '等待队伍开始战斗')

    def entry(self, action, point, now, message):
        if self.entry_action == action:
            if now - self.entry_at < 10:
                return Decision('wait', status='等待游戏页面切换')
            if action != 'network_retry' and self.entry_attempts >= 3:
                return Decision('wait', status='页面未切换，请手动检查；助手会继续等待')
        return Decision(action, point, message)

    def unknown(self, now, message='等待识别逆袭悬赏页面', limit=15):
        if self.unknown_since is None:
            self.unknown_since = now
        if now - self.unknown_since >= limit:
            if self.reopened:
                return Decision('stop', status='无法识别当前画面，请手动返回逆袭悬赏')
            self.reopened = True
            self.unknown_since = now
            if self.phase == 'search':
                self.phase = 'start'
            return Decision('reopen', status='画面中断，尝试返回游戏')
        return Decision('wait', status=message)

    def completed(self, decision, now):
        action = decision.action
        if action in ('login', 'home', 'hall', 'network_retry'):
            self.entry_attempts = self.entry_attempts + 1 if self.entry_action == action else 1
            self.entry_action, self.entry_at = action, now
        elif action in ('invite', 'menu', 'confirm', 'a_off', 's_on', 'ss_off'):
            self.pending, self.pending_at = action, now
            if action in ('menu', 'confirm') or (action == 'invite' and self.phase == 'search'):
                self.phase = 'start'
            if action == 'confirm':
                self.pending = None
                self.confirmation_at = now
                self.trusted_join = False
        elif action == 'refresh':
            self.phase, self.refreshed_at, self.attempted = 'search', now, False
            self.refresh_was_cooling = False
        elif action == 'join':
            self.phase, self.joined_at, self.join_point, self.attempted = 'joining', now, decision.point, True
            self.trusted_join = True
        elif action == 'ready':
            self.phase, self.ready_at = 'verify_ready', now
