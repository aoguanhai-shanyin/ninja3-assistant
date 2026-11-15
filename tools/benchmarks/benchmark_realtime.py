"""Offline recognition/GC and native-mock timeline benchmark; no game input."""
import argparse
from collections import deque
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from ninja3_assistant.device.native_touch import executable_path
from ninja3_assistant.runtime.realtime_battle import RecognitionCadence
from ninja3_assistant.activities.weekend.runner import RaceVision,RaceAvoider


def percentiles(values):
    if not values:
        return None
    return dict(zip(('p50','p95','max'),map(float,np.percentile(values,(50,95,100)))))


def recognition(frame,frames,adaptive):
    vision,avoider,cadence = RaceVision(),RaceAvoider(),RecognitionCadence()
    pauses = deque(maxlen=1024)
    started = 0.
    def callback(phase,info):
        nonlocal started
        if phase == 'start':
            started = time.perf_counter()
        else:
            pauses.append((time.perf_counter()-started)*1000)
    gc.callbacks.append(callback)
    samples = []
    try:
        for i in range(frames):
            at = i/60
            if adaptive and not cadence.due(at,0):
                continue
            before = time.perf_counter()
            gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            vision.detect(frame,gray)
            avoider.decide(frame,at,gray)
            duration = time.perf_counter()-before
            samples.append(duration*1000)
            cadence.completed(at,at+duration,stable=adaptive)
    finally:
        gc.callbacks.remove(callback)
    return {'captures_simulated':frames,'recognitions':len(samples),
            'processing_ms':percentiles(samples),'total_processing_ms':sum(samples),
            'gc_collections_observed':len(pauses),'gc_pause_ms':percentiles(list(pauses)),
            'capture_and_game_latency_measured':False}


def native_timeline():
    process = subprocess.Popen([str(executable_path()),'--mock'],stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
    try:
        if process.stdout.readline().strip() != 'READY':
            raise RuntimeError('native mock failed to start')
        junk = []
        for _ in range(30000):
            cycle = []
            cycle.append(cycle)
            junk.append(cycle)
        del cycle,junk
        captured = time.perf_counter_ns()//1000
        fields = []
        for offset in (40,80,120,160):
            at = captured+offset*1000
            fields.extend((at,380,630,10,at+20000))
        process.stdin.write(f'START 1000\nPLAN 4 1 {captured} {captured+200000} 4 '+
                            ' '.join(map(str,fields))+'\n')
        process.stdin.flush()
        before = time.perf_counter()
        collected = gc.collect()
        pause_ms = (time.perf_counter()-before)*1000
        time.sleep(.24)
        process.stdin.write('STOP\nQUIT\n')
        process.stdin.flush()
        output,_ = process.communicate(timeout=3)
        presses = [line.split() for line in output.splitlines() if line.startswith('PLAN_PRESSED ')]
        lateness = [(int(p[4])-(captured+(40+int(p[3])*40)*1000))/1000 for p in presses]
        return {'backend':'C++ mock, not MuMu','pressed':len(presses),'expected':4,
                'python_forced_gc_ms':pause_ms,'cyclic_objects_collected':collected,
                'native_press_lateness_ms':percentiles(lateness),
                'movement_cycles':sum(line.startswith('CYCLE ') for line in output.splitlines())}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        for stream in (process.stdin,process.stdout):
            if not stream.closed:
                stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames',type=int,default=300)
    parser.add_argument('--image',default='data/weekend-race/live/20261114-170712-recovery-kernel.png')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    frame = cv2.cvtColor(cv2.imread(args.image),cv2.COLOR_BGR2RGB)
    result = {'python':sys.version,'gc_enabled':gc.isenabled(),'gc_threshold':gc.get_threshold(),
              'full':recognition(frame,args.frames,False),
              'adaptive':recognition(frame,args.frames,True),'native_timeline':native_timeline()}
    path = Path('data/experiments/realtime-bench')/time.strftime('%Y-%m-%d')/('benchmark-'+time.strftime('%Y%m%d-%H%M%S')+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(path)
    print(json.dumps(result,ensure_ascii=False))
