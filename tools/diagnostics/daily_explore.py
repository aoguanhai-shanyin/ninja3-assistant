"""Explicit SDK screenshots and native input for documenting daily chores."""
import argparse
import json
from pathlib import Path
import subprocess
import threading
import time

import cv2

from ninja3_assistant.paths import project_root
from ninja3_assistant.device.mumu import MuMu
from ninja3_assistant.device.native_touch import create_native_touch, executable_path
from ninja3_assistant.activities.intelligence.ocr import WindowsOcr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--folder',required=True)
    parser.add_argument('--index',type=int,required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--tap',nargs=2,type=int)
    actions.add_argument('--swipe',nargs=4,type=int)
    parser.add_argument('--wait',type=float,default=.7)
    parser.add_argument('--ocr',action='store_true')
    args = parser.parse_args()
    output = project_root()/'data/evidence'/time.strftime('%Y-%m-%d')/'daily-exploration'
    output.mkdir(parents=True,exist_ok=True)
    stop = threading.Event()
    if args.swipe:
        subprocess.run([str(executable_path()),'--folder',args.folder,'--index',str(args.index),
                        '--swipe',*map(str,args.swipe)],capture_output=True,check=True,timeout=10,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    elif args.tap:
        touch = create_native_touch(args.folder,args.index,stop,lambda *_:None)
        if touch is None:
            raise RuntimeError('Native input unavailable')
        try:
            captured = time.perf_counter()
            if not touch.priority_tap(tuple(args.tap),captured_at=captured,
                                      deadline=captured+.2,wait_seconds=.2,hold_seconds=.07):
                raise RuntimeError('Tap was not executed')
            stop.wait(.09)
        finally:
            touch.close()
    stop.wait(args.wait)
    device = MuMu(args.folder,args.index)
    try:
        frame = device.capture()
    finally:
        device.close()
    path = output/(time.strftime('%H%M%S')+'-'+args.label+'.png')
    if not cv2.imwrite(str(path),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)):
        raise RuntimeError('Screenshot failed')
    record = {'frame':str(path),'tap':args.tap,'swipe':args.swipe}
    if args.ocr:
        record['ocr'] = WindowsOcr().read(frame)
    path.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({**record,'ocr':record.get('ocr',{}).get('text')},ensure_ascii=False))


if __name__ == '__main__':
    main()
