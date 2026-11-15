"""Explicit visible-instance sampling for weekend race research."""
import argparse
from pathlib import Path
import threading
import time

import cv2
from ninja3_assistant.device.mumu import MuMu
from ninja3_assistant.device.native_touch import create_native_touch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--tap', nargs=2, type=int)
    parser.add_argument('--wait', type=float, default=1)
    args = parser.parse_args()
    device = MuMu('D:/Program Files/Netease/MuMu', 1)
    try:
        if args.tap:
            native = create_native_touch('D:/Program Files/Netease/MuMu', 1, threading.Event(), lambda *a: None)
            if native is None:
                raise RuntimeError('C++ touch executable required')
            try:
                captured = time.perf_counter()
                native.priority_tap(tuple(args.tap), captured_at=captured, deadline=captured+.5, wait_seconds=.2)
                time.sleep(.1)
            finally:
                native.close()
        time.sleep(args.wait)
        path = Path('data/weekend-race/live')/(time.strftime('%Y%m%d-%H%M%S')+'-'+args.label+'.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(device.capture(), cv2.COLOR_RGB2BGR))
        print(path)
    finally:
        device.close()
