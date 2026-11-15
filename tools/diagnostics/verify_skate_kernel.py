"""Bounded training-opening skate baseline; SDK capture, C++ input only."""
import argparse
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from ninja3_assistant.device.mumu import MuMu
from ninja3_assistant.device.native_touch import create_native_touch,TimedTap
from ninja3_assistant.combat.opening_actions import OpeningActions
from ninja3_assistant.runtime.realtime_battle import RealtimeBattleKernel,FrameDiagnostics
from ninja3_assistant.runtime.runner import RunControl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start',nargs=2,type=int)
    parser.add_argument('--skill',nargs=2,type=int)
    parser.add_argument('--seconds',type=float,default=8)
    parser.add_argument('--walk',action='store_true')
    parser.add_argument('--movement',choices=('skate','slide','jump-two-slide'),default='slide')
    parser.add_argument('--plan-seconds',type=float,default=3.)
    parser.add_argument('--cancel-after',type=float)
    parser.add_argument('--replace-after',type=float)
    parser.add_argument('--replacement-seconds',type=float,default=.6)
    parser.add_argument('--pause-after',action='store_true')
    args = parser.parse_args()
    output = Path('D:/CodexArtifacts/ninja3-realtime-v2')/time.strftime('%Y%m%d-%H%M%S')
    output.mkdir(parents=True)
    control,diagnostics = RunControl(),FrameDiagnostics()
    core = RealtimeBattleKernel(control)
    device = native = None
    timer = None
    started = None
    ended = 'timeout_before_opening'
    cadence,events = [],[]
    ages = []
    next_image = 0.
    cancelled_at = replaced_at = None
    with (output/'events.jsonl').open('w',encoding='utf-8') as log:
        def emit(kind,value):
            events.append((kind,value))
            log.write(json.dumps({'time':time.perf_counter(),'event':kind,'value':value},ensure_ascii=False)+'\n')
            log.flush()
        try:
            opening = OpeningActions()
            device = MuMu('D:/Program Files/Netease/MuMu',1)
            native = create_native_touch('D:/Program Files/Netease/MuMu',1,control.stop,emit,
                                         observation_timeout=.5,lease_ms=1000,movement=args.movement)
            core.attach(native)
            control.set_release_input(native.stop_all)
            timer = threading.Timer(60,control.request_stop)
            timer.start()
            if args.start:
                core.capture(device)
                core.priority_tap(tuple(args.start),captured_at=core.captured_at,
                                  deadline=core.captured_at+.2,hold_seconds=.09,wait_seconds=.1)
            last_skill = float('-inf')
            while control.wait_active():
                before = time.perf_counter()
                frame = core.capture(device)
                if core.recognition_due():
                    if started is None:
                        opening.decide(frame,before)
                        if opening.ground_frames >= 2:
                            started = before
                            if not args.walk:
                                core.schedule_movement(args.plan_seconds)
                            print('training opening confirmed; '+('walking baseline' if args.walk else 'bounded C++ '+args.movement),flush=True)
                    core.recognized(stable=started is not None)
                if started is not None:
                    if args.cancel_after is not None and cancelled_at is None and before-started >= args.cancel_after:
                        core.cancel_movement()
                        cancelled_at=time.perf_counter()
                        emit('trial',{'action':'cancel','at':cancelled_at})
                    if args.replace_after is not None and replaced_at is None and before-started >= args.replace_after:
                        core.schedule_movement(args.replacement_seconds)
                        replaced_at=time.perf_counter()
                        emit('trial',{'action':'replace','at':replaced_at,'seconds':args.replacement_seconds})
                    if args.skill and before-last_skill >= .6:
                        core.plan(4,(TimedTap(25,tuple(args.skill)),TimedTap(100,tuple(args.skill))))
                        last_skill = before
                    if before-started >= args.seconds:
                        ended = 'opening_window_complete'
                        if args.pause_after:
                            core.cancel_movement()
                            core.capture(device)
                            core.priority_tap((1240,25),captured_at=core.captured_at,
                                              deadline=core.captured_at+.2,hold_seconds=.09,wait_seconds=.1)
                            control.stop.wait(.12)
                        break
                if before >= next_image:
                    diagnostics.save(output/f'{before:.6f}.png',frame)
                    next_image = before+(.1 if started is not None else 1.)
                ages.append(core.metrics()['frame_age_ms'])
                cadence.append(time.perf_counter()-before)
                control.stop.wait(max(0,1/60-(time.perf_counter()-before)))
        finally:
            control.request_stop()
            if timer:
                timer.cancel()
            if native:
                native.close()
            if device:
                device.close()
            diagnostics.close()
    cycles = [list(map(int,value.split()[1:])) for kind,value in events if kind == 'native_touch']
    gaps = [(b[1]-a[1])/1000 for a,b in zip(cycles,cycles[1:])]
    summary = {'mode':'walk' if args.walk else args.movement,'ended':ended,'active_seconds':None if started is None else time.perf_counter()-started,
               'plan_seconds':args.plan_seconds,'cancelled_at':cancelled_at,'replaced_at':replaced_at,
               'cycles':len(cycles),'cycle_gap_ms':None if not gaps else
               dict(zip(('p50','p95','max'),map(float,np.percentile(gaps,(50,95,100))))),
               'metrics':core.metrics(),'frames':len(cadence),
               'processing_age_ms':dict(zip(('p50','p95','max'),map(float,np.percentile(ages,(50,95,100))))) if ages else None,
               'controls':[v for k,v in events if k == 'native_control']}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(output,flush=True)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
