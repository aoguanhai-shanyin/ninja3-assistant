"""Run one daily chore with SDK screenshots and the common C++ input process."""
import json
from pathlib import Path
import threading
import time
import sys

import cv2

from ninja3_assistant.paths import project_root
from ninja3_assistant.activities.daily import DEFAULT_PET_FEED_REWARD
from ninja3_assistant.activities.daily.flow import DailyFlow
from ninja3_assistant.game_ui.trial import read_trial_counter
from ninja3_assistant.activities.daily.vision import DailyVision
from ninja3_assistant.activities.daily.resources import resource_path, load_deadline, save_deadline
from ninja3_assistant.activities.intelligence.ocr import WindowsOcr
from ninja3_assistant.device.mumu import MuMu, prepare_game
from ninja3_assistant.device.native_touch import create_native_touch, executable_path
from ninja3_assistant.game_ui import GameUI, NavigationError, PageAction, PageObservation
from ninja3_assistant.game_ui.arena import ArenaRewardPage
from ninja3_assistant.game_ui.house import HousePage, house_view, read_house_clock
from ninja3_assistant.game_ui.pages import MenuVision, ROUTES, page_id
from ninja3_assistant.scheduling.task_outcome import TaskOutcome
from ninja3_assistant.scheduling.invitation_handoff import bounty_handoff, clear_handled_invitation


def run_daily(kind, folder, index, control, emit, *, pet_feed_reward=DEFAULT_PET_FEED_REWARD):
    import subprocess
    root = Path(sys.executable).parent if getattr(sys,'frozen',False) else project_root()/'data/runtime'
    output = root/'daily-cleanup/runs'/(time.strftime('%Y%m%d-%H%M%S')+f'-{time.time_ns()%1000000:06d}-{kind}')
    output.mkdir(parents=True,exist_ok=False)
    device = native = None
    resources = resource_path(folder,index)
    task_flow = None
    if kind == 'pet_feed':
        from ninja3_assistant.activities.daily.pet_feed import PetFeed
        task_flow = PetFeed(pet_feed_reward)
    elif kind == 'arena_reward':
        from ninja3_assistant.activities.daily.arena import ArenaReward
        task_flow = ArenaReward()
    elif kind == 'landlady_quiz':
        from ninja3_assistant.activities.daily.landlady import LandladyQuiz, question_bank_path, recover_question_ocr
        task_flow = LandladyQuiz(question_bank_path(folder,index))
    elif kind == 'friend_rice':
        from ninja3_assistant.activities.daily.friend_rice import FriendRice
        task_flow = FriendRice(resources.with_name(resources.stem+'-friend-rice.json'))
    flow = DailyFlow(kind,training_finishes_at=load_deadline(resources),task_flow=task_flow)
    persistent_ocr = kind in ('blessing','daily_shop','friend_rice','house','village','arena_reward')
    vision,ocr = DailyVision(),WindowsOcr(persistent=persistent_ocr)
    lock = threading.Lock()
    with (output/'events.jsonl').open('w',encoding='utf-8') as log:
        def record(value):
            with lock:
                log.write(json.dumps({'time':time.time(),**value},ensure_ascii=False)+'\n')
                log.flush()
        try:
            if persistent_ocr and not control.stop.is_set():
                ocr.warmup()
            if prepare_game(folder,index,control,lambda text:emit('status',text)) is None:
                return TaskOutcome.STOPPED
            device = MuMu(folder,index)
            native = create_native_touch(folder,index,control.stop,
                lambda event,value:record({'event':event,'value':value}))
            if native is None:
                raise RuntimeError('每日清理需要 NinjaTouch.exe')
            control.set_release_input(native.stop_all)

            def swipe(point, end):
                native.stop_all()
                if control.stop.is_set():
                    return False
                subprocess.run([str(executable_path()),'--folder',str(folder),'--index',str(index),
                    '--swipe',*map(str,(*point,*end))],capture_output=True,check=True,timeout=8,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                return True

            recognize = None
            if kind in ('house','arena_reward'):
                recognize = MenuVision(control.stop)
                recognize.daily = vision
                recognize.ocr = ocr
            ui = GameUI(control,device.capture,native,swipe=swipe,recognize=recognize,routes=ROUTES)
            emit('state','running')
            emit('status','每日清理记录：'+str(output))
            started = control.now()
            sequence = 0
            previous = None
            safe_invite = getattr(control,'bounty_navigation_safe',False)
            waiting_state = waiting_since = waiting_snapshot_at = None
            while control.wait_active() and control.now()-started < 240:
                generation = control.generation()
                captured_at = time.perf_counter()
                frame = device.capture()
                if kind == 'landlady_quiz' and task_flow.pending:
                    # Answer feedback advances automatically while full-frame OCR is still running.
                    until = time.perf_counter()+.55
                    while generation == control.generation() and not control.stop.is_set():
                        if (task_flow.observe_feedback(frame) or time.perf_counter() >= until or
                                control.stop.wait(.12)):
                            break
                        captured_at = time.perf_counter()
                        frame = device.capture()
                # Trial counters are small; enlarging sign-in pages loses whole rows.
                scale = 1.5 if vision.match(frame,'trial',(0,0,470,120)) else 1
                result = ocr.read(cv2.resize(frame,None,fx=scale,fy=scale) if scale != 1 else frame,
                                  stop=control.stop)
                if result is None:
                    continue
                if kind == 'landlady_quiz':
                    result = recover_question_ocr(frame,result,ocr,control.stop)
                if not control.wait_active():
                    return TaskOutcome.STOPPED
                if generation != control.generation():
                    continue
                if scale != 1:
                    for word in result['words']:
                        word['bbox'] = [v/scale for v in word['bbox']]
                    counter = read_trial_counter(frame,ocr,control.stop)
                    if counter is None:
                        continue
                    result['words'] = [w for w in result['words'] if not
                                       (w['bbox'][0] >= 1100 and w['bbox'][1] < 90)]
                    result['food_counter_repeat'] = counter
                if vision.house(frame):
                    result = read_house_clock(frame,result,ocr,control.stop)
                    if result is None:
                        continue
                if not control.wait_active():
                    return TaskOutcome.STOPPED
                if generation != control.generation():
                    continue
                screen = vision.observe(frame,result)
                if screen.page == 'game_update':
                    status = '识别到游戏更新提示，交回调度重启客户端'
                    record({'event':'checkpoint','page':screen.page,'status':status})
                    emit('status',status)
                    return TaskOutcome.SCENE_CHANGED
                if kind == 'daily_shop' and screen.page == 'daily_shop':
                    from ninja3_assistant.activities.daily.shop_stock import recover_coin_stock
                    if not recover_coin_stock(screen,vision,ocr,control.stop):
                        continue
                    if not control.wait_active():
                        return TaskOutcome.STOPPED
                    if generation != control.generation():
                        continue
                if screen.page == 'invite':
                    handoff = bounty_handoff(frame,vision.invitations,ocr,control,
                                             safe=safe_invite,result=result)
                    if handoff:
                        return handoff
                else:
                    safe_invite = screen.page not in ('unknown','trial') and not screen.page.startswith('landlady-')
                    if safe_invite:
                        clear_handled_invitation(control)
                house = None
                if kind == 'house' and screen.page in ('house','house_continue'):
                    house = house_view(vision,screen)
                    if house.training_remaining:
                        observed_at = time.time()
                        flow.training_finishes_at = observed_at+house.training_remaining
                        save_deadline(resources,flow.training_finishes_at,observed_at)
                decision = flow.step(screen,vision,house)
                now = control.now()
                if decision.action == 'wait':
                    current_wait = (screen.page,decision.status)
                    if current_wait != waiting_state:
                        waiting_state,waiting_since,waiting_snapshot_at = current_wait,now,None
                    waiting_timeout = now-waiting_since >= 12
                    waiting_snapshot = (waiting_timeout or waiting_snapshot_at is None or
                                        now-waiting_snapshot_at >= 5)
                else:
                    waiting_state = waiting_since = waiting_snapshot_at = None
                    waiting_timeout = waiting_snapshot = False
                state = (screen.page,decision.action,decision.status)
                if state != previous or decision.action == 'tap' or decision.action == 'swipe' or waiting_snapshot:
                    image = f'{sequence:04d}-{screen.page}-{decision.action}.png'
                    cv2.imwrite(str(output/image),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                    record({'event':'decision','page':screen.page,'action':decision.action,
                            'operation':decision.operation,'point':decision.point,'status':decision.status,'frame':image,
                            'ocr':result,'free_kind':decision.free_kind,'rice_gifts':decision.rice_gifts})
                    sequence += 1
                    emit('status',decision.status)
                    previous = state
                    if decision.action == 'wait':
                        waiting_snapshot_at = now
                if waiting_timeout:
                    if decision.defer_on_timeout:
                        emit('status','日常页面持续无进展，已保留最新画面，今日暂缓并继续其他任务')
                        return TaskOutcome.DEFERRED_TODAY
                    emit('status','日常页面持续无进展，已保留最新画面，稍后重试并继续其他任务')
                    return TaskOutcome.WAITING
                if decision.action == 'done':
                    if kind == 'house' and flow.training_finishes_at is not None:
                        emit('daily_next_check',flow.training_finishes_at)
                    return TaskOutcome.DAILY_DONE
                if decision.action == 'unavailable':
                    return TaskOutcome.UNAVAILABLE
                if decision.action == 'maintenance':
                    return TaskOutcome.MAINTENANCE
                if decision.action == 'limited_checked':
                    flow.completed(decision,screen)
                    if {'anniversary','ten-signin'} <= flow.limited_checked:
                        return TaskOutcome.DAILY_DONE
                    if flow.anniversary_missing and screen.page == 'ten-signin':
                        emit('status','十日签到已检查，主页限时签到入口暂未确认')
                        return TaskOutcome.DEFERRED_TODAY
                    point = (1195,98) if screen.page == 'anniversary' else (1220,705)
                    decision = flow.back(point)
                if decision.operation in ('open_house','open_arena_reward'):
                    page_type = HousePage if decision.operation == 'open_house' else ArenaRewardPage
                    try:
                        page = ui.navigate(page_type)
                    except NavigationError as error:
                        result = error.result
                        record({'event':'navigation','page':result.page,'status':result.status})
                        if result.status == 'stopped':
                            return TaskOutcome.STOPPED
                        if result.page in ('game.invite','game.update','daily.house_continue') or control.paused():
                            continue
                        emit('status','未能确认目标页面，稍后重试')
                        return TaskOutcome.WAITING
                    details = ({'training_remaining':page.training_remaining} if isinstance(page,HousePage)
                               else {'rank':page.rank,'amount':page.amount})
                    record({'event':'navigation','page':page.observation.page,'status':'arrived',**details})
                    flow.completed(decision,screen)
                    continue
                if decision.point is not None and decision.operation:
                    def still_visible(fresh):
                        if not vision.current_action(screen,decision,fresh):
                            record({'event':'input_rejected','reason':'button_changed_or_not_free'})
                            return False
                        return True

                    def reserve_gifts():
                        if decision.rice_gifts and not task_flow.reserve(decision.rice_gifts):
                            record({'event':'input_rejected','reason':'rice_gift_limit'})
                            return False
                        return True

                    identifier = page_id('daily',screen.page)
                    if house is not None and decision.operation in house.controls:
                        observed = PageObservation(identifier,frame,generation,captured_at,house.controls,detail=house)
                        page = HousePage(ui,observed)
                        acted = (page.collect_fragments() if decision.operation == 'claim_training'
                                 else page.continue_training())
                        if not acted:
                            record({'event':'input_rejected','reason':'house_action_not_confirmed'})
                    elif (decision.operation == 'claim_arena_reward' and (reward := ArenaRewardPage.observe(screen))
                          and (action := reward.claim_action(frame))):
                        observed = PageObservation(identifier,frame,generation,captured_at,
                                                   {action.name:action},detail=reward)
                        acted = ArenaRewardPage(ui,observed).claim_reward()
                        if not acted:
                            record({'event':'input_rejected','reason':'arena_reward_not_confirmed'})
                    else:
                        action = PageAction(decision.operation,decision.point,decision.end,
                                            guard=still_visible,hold_seconds=.07,max_age=.25)
                        observed = PageObservation(identifier,frame,generation,captured_at,
                                                   {action.name:action},detail=screen)
                        acted = ui.perform(observed,action.name,before_send=reserve_gifts)
                    if acted:
                        if not decision.end:
                            control.stop.wait(.09)
                        flow.completed(decision,screen)
                        record({'event':'input_sent','operation':decision.operation,'point':decision.point,
                                'free_kind':decision.free_kind,'rice_gifts':decision.rice_gifts})
                control.stop.wait(.17 if screen.page.startswith('landlady-') and decision.action == 'tap' else .65)
            if control.stop.is_set():
                return TaskOutcome.STOPPED
            emit('status','未确认日常完成，已保留画面，继续其他任务')
            return TaskOutcome.UNAVAILABLE
        except Exception as error:
            record({'event':'error','message':str(error)})
            emit('error','每日清理：'+str(error))
            return TaskOutcome.ERROR
        finally:
            control.set_release_input(None)
            if native:
                native.close()
            if device:
                device.close()
            ocr.close()
            emit('finished','')
