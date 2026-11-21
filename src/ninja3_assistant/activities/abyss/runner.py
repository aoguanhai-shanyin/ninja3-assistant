"""Complete the three family preparation dungeons using their weekly rewards."""

from contextlib import closing
import re

from ninja3_assistant.activities.abyss.vision import (
    REWARD_ROWS, REWARDS_CLOSE, TABS, dark_hud, free_relay, nearby_ground_hazard, preparation_page, read_reward,
    ready_ninjutsu, ready_weapons, relay_prompt, result_panel, reward_dungeon, same_region, selected_dungeon,
    visible_ninjutsu,
)
from ninja3_assistant.activities.bounty.combat import BountyCombatVision
from ninja3_assistant.activities.daily.vision import DailyScreen, DailyVision
from ninja3_assistant.activities.intelligence.ocr import WindowsOcr
from ninja3_assistant.device.mumu import MuMu
from ninja3_assistant.device.native_touch import TimedTap, create_native_touch
from ninja3_assistant.game_ui import GameUI, PageAction, PageObservation
from ninja3_assistant.runtime.realtime_battle import RealtimeBattleKernel
from ninja3_assistant.scheduling.task_outcome import TaskOutcome
from ninja3_assistant.vision.game_update import game_update_banner
from ninja3_assistant.vision.promotion_popup import promotion_popup_close


def run_abyss(folder, index, control, emit):
    with closing(MuMu(folder, index)) as device, closing(WindowsOcr(persistent=True)) as ocr:
        native = create_native_touch(folder, index, control.stop, emit)
        if native is None:
            raise RuntimeError('深渊备战需要 NinjaTouch.exe')
        with closing(native):
            control.set_release_input(native.stop_all)
            try:
                return play_abyss(device, native, ocr, control, emit)
            finally:
                control.set_release_input(None)


def play_abyss(device, native, ocr, control, emit):
    daily, battle = DailyVision(), BountyCombatVision()
    kernel = RealtimeBattleKernel(control).attach(native)
    ui = GameUI(control, device.capture, native)
    next_dungeon = zero_rounds = 0
    dungeon = challenge = None
    in_round = engaged = leaving = False
    result_seen = False
    checked_at = cast_at = weapon_at = jumped_at = float('-inf')
    deadline = control.now() + 30

    def press(name, point, *boxes, verify=None):
        def guard(fresh):
            return (fresh is not None and not game_update_banner(fresh) and
                    all(same_region(frame, fresh, box) for box in boxes) and
                    (verify is None or verify(fresh)))

        action = PageAction(name, point, guard=guard)
        observation = PageObservation('abyss', frame, stamp.generation, stamp.captured_at)
        if ui.perform(observation, action) is None:
            return False
        emit('status', '深渊备战 · '+name)
        return True

    while control.wait_active():
        frame = kernel.capture(device)
        stamp = kernel.stamp()
        now = control.now()
        if frame is None:
            if now >= deadline:
                break
            control.stop.wait(.15)
            continue
        if game_update_banner(frame):
            return TaskOutcome.SCENE_CHANGED
        if not in_round and now < deadline and (close := promotion_popup_close(frame)) is not None:
            press('关闭活动弹窗', close, verify=lambda fresh: promotion_popup_close(fresh) == close)
            control.stop.wait(.45)
            continue
        preparation = preparation_page(frame)
        reward_tab = reward_dungeon(frame) if not preparation else None
        relay = relay_prompt(frame)
        settled = result_panel(frame)
        if not leaving and (in_round or (not preparation and reward_tab is None)):
            if not relay and not settled:
                state = battle.detect(frame)
                if state.active and (in_round or dark_hud(frame, (503, 19, 781, 84))):
                    in_round = engaged = True
                    deadline = now+25
                    if now-cast_at >= .32:
                        point = ready_ninjutsu(frame)
                        if point is None and now-cast_at >= .85:
                            point = visible_ninjutsu(frame)
                        if point and kernel.plan(3, (TimedTap(35, point),)) is not None:
                            cast_at = now
                    if now-weapon_at >= .32:
                        actions = tuple(TimedTap(35+i*37, point)
                                        for i, point in enumerate(ready_weapons(frame)))
                        if actions and kernel.plan(4, actions) is not None:
                            weapon_at = now
                    if state.ground and now-jumped_at >= .65 and nearby_ground_hazard(frame):
                        if kernel.plan(5, (TimedTap(30, (1085, 625)),)) is not None:
                            jumped_at = now
                    control.stop.wait(.06)
                    continue
        if engaged:
            native.stop_all()
            engaged = False
        if now >= deadline:
            break
        if relay:
            in_round = True
            for x in (445, 646, 850):
                if free_relay(frame, x):
                    if press('免费接力，继续挑战', (x, 350),
                             verify=lambda fresh: relay_prompt(fresh) and free_relay(fresh, x)):
                        deadline = now+30
                        cast_at = weapon_at = jumped_at = float('-inf')
                        control.stop.wait(.6)
                    break
            control.stop.wait(.10)
            continue
        if now-checked_at < .42:
            control.stop.wait(.10)
            continue
        checked_at = now
        result = {} if preparation else ocr.read(frame, stop=control.stop)
        if result is None and reward_tab is not None:
            result = {}
        if result is None:
            continue
        screen = DailyScreen(frame, result)
        if settled and '任意位置继续' in screen.text((505, 612, 780, 665)):
            score = re.match(r'\d+', screen.text((615, 266, 785, 319)))
            if press('结算并返回副本', (640, 633), (535, 618, 755, 650)):
                if not result_seen and dungeon is not None and score:
                    if dungeon == next_dungeon:
                        zero_rounds = 0 if int(score[0]) > 0 else zero_rounds+1
                    emit('status', f'深渊备战 · 副本 {dungeon+1} 本局贡献 {score[0]}')
                result_seen = True
                in_round, deadline = False, now+30
                battle.reset()
                control.stop.wait(1.5)
            continue
        if in_round:
            continue
        page, point = daily.navigation(frame)
        if page in ('invite', 'reward', 'reward_continue') and point is not None:
            name = '拒绝组队邀请' if page == 'invite' else '确认已领取奖励'
            if press(name, point, verify=lambda fresh: daily.navigation(fresh) == (page, point)):
                deadline = now+15
            continue
        if reward_tab is not None:
            dungeon = reward_tab
            if dungeon != next_dungeon or leaving:
                if press('关闭奖励信息', REWARDS_CLOSE, verify=lambda fresh: reward_dungeon(fresh) == reward_tab):
                    deadline = now+15
            else:
                rewards = [read_reward(screen, y) for y in REWARD_ROWS]
                if 'claim' in rewards:
                    y = REWARD_ROWS[rewards.index('claim')]
                    if press(f'领取副本 {dungeon+1} 奖励', (920, y), (862, y-18, 982, y+20),
                             verify=lambda fresh: reward_dungeon(fresh) == reward_tab):
                        deadline = now+20
                elif all(rewards) and press('关闭奖励信息', REWARDS_CLOSE,
                                             verify=lambda fresh: reward_dungeon(fresh) == reward_tab):
                    if all(reward == 'claimed' for reward in rewards):
                        next_dungeon += 1
                        zero_rounds = 0
                    elif zero_rounds >= 2:
                        leaving = True
                        emit('status', f'深渊备战 · 副本 {dungeon+1} 连续两场贡献为零，保留进度待复查')
                    else:
                        # Wait for the undimmed list before pressing Challenge.
                        challenge = dungeon
                    deadline = now+15
            continue
        if preparation:
            dungeon = selected_dungeon(frame)
            if leaving or next_dungeon == len(TABS):
                if press('退出备战副本列表', (42, 43), (4, 10, 278, 69), (124, 487, 1179, 491)):
                    leaving, deadline = True, now+15
            elif dungeon != next_dungeon:
                if press(f'切换至副本 {next_dungeon+1}', TABS[next_dungeon], (4, 10, 278, 69),
                         (124, 487, 1179, 491)):
                    deadline = now+10
            elif challenge == dungeon:
                if press(f'挑战副本 {dungeon+1}', (1067, 444), (1031, 419, 1168, 466), (124, 487, 1179, 491)):
                    in_round, deadline = True, now+35
                    challenge = None
                    result_seen = False
                    cast_at = weapon_at = jumped_at = float('-inf')
            elif press(f'检查副本 {dungeon+1} 奖励', (494, 441), (447, 412, 581, 470),
                       (124, 487, 1179, 491)):
                deadline = now+10
            continue
        if '深渊冒险' in screen.text((100, 14, 353, 89)):
            label = screen.text((315, 526, 522, 611))
            if leaving:
                if press('返回家族大厅', (52, 56), (16, 20, 343, 94)):
                    return TaskOutcome.DAILY_DONE if next_dungeon == len(TABS) else TaskOutcome.DEFERRED_TODAY
            elif '备战副本' in label and '已开启' in label:
                if press('打开备战副本列表', (415, 565), (355, 539, 477, 595), (100, 20, 344, 84)):
                    deadline = now+15
            elif '备战副本' in label and re.search(r'未开启|已结束|\d+[:点].*开启', label):
                return TaskOutcome.UNAVAILABLE
            continue
        if page == 'home':
            moved = press('进入家族', (1010, 450), (980, 420, 1053, 485))
        elif page == 'family-welcome':
            moved = press('关闭家族入场展示', point, (250, 265, 1030, 430))
        elif page == 'family':
            if '备战副本' in screen.text((1020, 312, 1140, 410)):
                moved = press('前往家族备战入口', (1080, 361), (1035, 325, 1112, 399))
            elif '备战副本' in screen.text((1070, 150, 1230, 230)):
                moved = press('进入深渊冒险', (1141, 435), (1090, 161, 1211, 195))
            else:
                moved = False
        elif '加入家族' in screen.text() and '申请加入' in screen.text():
            return TaskOutcome.UNAVAILABLE
        else:
            moved = False
        if moved:
            deadline = now+15
    if control.stop.is_set():
        return TaskOutcome.STOPPED
    emit('status', '深渊备战 · 未确认下一步，保留未完成状态')
    return TaskOutcome.RECOVER_IN_PLACE if in_round and not preparation_page(frame) else TaskOutcome.ERROR
