"""Bounded native-input acceptance recording for the current race."""
from pathlib import Path
import time
import threading
import cv2
from ninja3_assistant.device.mumu import MuMu
from ninja3_assistant.device.native_touch import create_native_touch
from ninja3_assistant.combat.opening_actions import OpeningActions

root = Path('data/weekend-race/trial')/time.strftime('%Y%m%d-%H%M%S')
root.mkdir(parents=True)
stop = threading.Event()
log = (root/'native.jsonl').open('w', encoding='utf-8')
import json
def emit(kind, value):
    log.write(json.dumps({'event': kind, 'value': value}, ensure_ascii=False)+'\n')
    log.flush()
native = create_native_touch('D:/Program Files/Netease/MuMu', 1, stop, emit)
device = MuMu('D:/Program Files/Netease/MuMu', 1)
opening = OpeningActions()
writer = cv2.VideoWriter(str(root/'race.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280, 720))
start = time.monotonic()
last_card = 0
index = 0
try:
    if native is None:
        raise RuntimeError('C++ touch executable is required')
    while time.monotonic()-start < 90:
        captured = time.perf_counter()
        frame = device.capture()
        opening.decide(frame, time.monotonic())
        if opening.ground_frames >= 2:
            native.note_frame(captured)
            native.renew()
            if time.monotonic()-last_card > .4:
                native.tap(((380, 520, 635)[index % 3], 630), deadline=captured+.2)
                index += 1
                last_card = time.monotonic()
        else:
            native.quiesce()
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if round(time.monotonic()-start) % 5 == 0:
            cv2.imwrite(str(root/'latest.png'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        stop.wait(max(0, .1-(time.perf_counter()-captured)))
finally:
    native.close()
    cv2.imwrite(str(root/'final.png'), cv2.cvtColor(device.capture(), cv2.COLOR_RGB2BGR))
    device.close()
    writer.release()
    log.close()
    print(root)
