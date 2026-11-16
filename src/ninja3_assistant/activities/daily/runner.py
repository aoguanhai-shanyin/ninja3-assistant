"""Run one daily chore with SDK screenshots and the common C++ input process."""
import json
from pathlib import Path
import threading
import time
import sys

import cv2

from ninja3_assistant.paths import project_root
from ninja3_assistant.activities.daily.flow import DailyFlow
from ninja3_assistant.activities.daily.vision import DailyScreen, DailyVision
from ninja3_assistant.activities.daily.resources import resource_path, load_deadline, save_deadline, remaining_seconds
from ninja3_assistant.activities.intelligence.ocr import WindowsOcr
from ninja3_assistant.device.mumu import MuMu, prepare_game
from ninja3_assistant.device.native_touch import create_native_touch, executable_path
from ninja3_assistant.scheduling.task_outcome import TaskOutcome


def run_daily(kind, folder, index, control, emit):
    import subprocess
    root = Path(sys.executable).parent if getattr(sys,'frozen',False) else project_root()/'data/runtime'
    output = root/'daily-cleanup/runs'/(time.strftime('%Y%m%d-%H%M%S')+f'-{time.time_ns()%1000000:06d}-{kind}')
    output.mkdir(parents=True,exist_ok=False)
    device = native = None
    resources = resource_path(folder,index)
    flow,vision,ocr = DailyFlow(kind,training_finishes_at=load_deadline(resources)),DailyVision(),WindowsOcr()
    lock = threading.Lock()
    with (output/'events.jsonl').open('w',encoding='utf-8') as log:
        def record(value):
            with lock:
                log.write(json.dumps({'time':time.time(),**value},ensure_ascii=False)+'\n')
                log.flush()
        try:
            if prepare_game(folder,index,control,lambda text:emit('status',text)) is None:
                return TaskOutcome.STOPPED
            device = MuMu(folder,index)
            native = create_native_touch(folder,index,control.stop,
                lambda event,value:record({'event':event,'value':value}))
            if native is None:
                raise RuntimeError('每日清理需要 NinjaTouch.exe')
            control.set_release_input(native.stop_all)
            emit('state','running')
            emit('status','每日清理记录：'+str(output))
            started = control.now()
            sequence = 0
            previous = None
            while control.wait_active() and control.now()-started < 240:
                generation = control.generation()
                frame = device.capture()
                # Trial counters are small; enlarging sign-in pages loses whole rows.
                scale = 1.5 if vision.match(frame,'trial',(0,0,470,120)) else 1
                result = ocr.read(cv2.resize(frame,None,fx=scale,fy=scale) if scale != 1 else frame,
                                  stop=control.stop)
                if result is None:
                    continue
                if scale != 1:
                    for word in result['words']:
                        word['bbox'] = [v/scale for v in word['bbox']]
                    header = cv2.copyMakeBorder(cv2.resize(frame[:90,1100:],None,fx=3,fy=3),
                                                60,60,100,100,cv2.BORDER_CONSTANT,value=(0,0,0))
                    counter = ocr.read(header,stop=control.stop)
                    if counter is None:
                        continue
                    result['words'] = [w for w in result['words'] if not
                                       (w['bbox'][0] >= 1100 and w['bbox'][1] < 90)]
                    for word in counter['words']:
                        x,y,w,h = word['bbox']
                        result['words'].append({**word,'bbox':[(x-100)/3+1100,(y-60)/3,w/3,h/3]})
                    result['food_counter'] = counter
                    if vision.food(DailyScreen(frame,result)) is None:
                        digits = cv2.resize(frame[38:60,1160:1210],None,fx=4,fy=4)
                        repeated = cv2.copyMakeBorder(cv2.hconcat([digits]*3),40,40,40,40,
                                                     cv2.BORDER_CONSTANT,value=(255,255,255))
                        counter = ocr.read(repeated,stop=control.stop)
                        if counter is None:
                            continue
                        result['food_counter_repeat'] = counter
                if vision.match(frame,'house',(950,0,1280,135)):
                    clock_image = cv2.copyMakeBorder(cv2.resize(frame[675:710,250:410],None,fx=4,fy=4),
                                                     30,30,30,30,cv2.BORDER_CONSTANT,value=(0,0,0))
                    clock_result = ocr.read(clock_image,stop=control.stop)
                    if clock_result is None:
                        continue
                    result['words'] = [w for w in result['words'] if not
                                       (250 <= w['bbox'][0]+w['bbox'][2]/2 < 410 and
                                        675 <= w['bbox'][1]+w['bbox'][3]/2 < 710)]
                    for word in clock_result['words']:
                        x,y,w,h = word['bbox']
                        result['words'].append({**word,'bbox':[(x-30)/4+250,(y-30)/4+675,w/4,h/4]})
                    result['house_clock'] = clock_result
                screen = vision.observe(frame,result)
                if kind == 'house' and screen.page == 'house':
                    countdown = remaining_seconds(screen.text((245,590,460,715)))
                    if countdown is not None and countdown > 0:
                        observed_at = time.time()
                        flow.training_finishes_at = observed_at+countdown
                        save_deadline(resources,flow.training_finishes_at,observed_at)
                decision = flow.step(screen,vision)
                state = (screen.page,decision.action,decision.status)
                if state != previous or decision.action == 'tap' or decision.action == 'swipe':
                    image = f'{sequence:04d}-{screen.page}-{decision.action}.png'
                    cv2.imwrite(str(output/image),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                    record({'event':'decision','page':screen.page,'action':decision.action,
                            'point':decision.point,'status':decision.status,'frame':image,
                            'ocr':result,'free_kind':decision.free_kind})
                    sequence += 1
                    emit('status',decision.status)
                    previous = state
                if decision.action == 'done':
                    if kind == 'house' and flow.training_finishes_at is not None:
                        emit('daily_next_check',flow.training_finishes_at)
                    return TaskOutcome.DAILY_DONE
                if decision.action == 'unavailable':
                    return TaskOutcome.UNAVAILABLE
                if decision.action == 'limited_checked':
                    flow.completed(decision,screen)
                    if {'anniversary','ten-signin'} <= flow.limited_checked:
                        return TaskOutcome.DAILY_DONE
                    point = (1195,98) if screen.page == 'anniversary' else (1220,705)
                    decision = flow.back(point)
                if decision.point is not None:
                    def act():
                        if generation != control.generation():
                            return
                        fresh = device.capture()
                        if not vision.current_action(screen,decision,fresh):
                            record({'event':'input_rejected','reason':'button_changed_or_not_free'})
                            return
                        captured = time.perf_counter()
                        if decision.end:
                            native.stop_all()
                            subprocess.run([str(executable_path()),'--folder',str(folder),'--index',str(index),
                                '--swipe',*map(str,(*decision.point,*decision.end))],capture_output=True,
                                check=True,timeout=8,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                            acted = True
                        else:
                            acted = native.priority_tap(decision.point,captured_at=captured,
                                deadline=captured+.25,wait_seconds=.2,hold_seconds=.07)
                            control.stop.wait(.09)
                        if acted:
                            flow.completed(decision,screen)
                            record({'event':'input_sent','point':decision.point,'free_kind':decision.free_kind})
                    control.run_if_active(act)
                control.stop.wait(.65)
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
            emit('finished','')
