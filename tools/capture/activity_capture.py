"""Explicit SDK sampling for the intelligence simulation; never controls other instances."""
from ninja3_assistant.paths import project_root
import argparse
import json
from pathlib import Path
import subprocess
import threading
import time

ROOT = project_root()
import cv2
from ninja3_assistant.device.mumu import MuMu, instance_serial
from tools.diagnostics.assistant_control import AssistantController


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default='D:/Program Files/Netease/MuMu')
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--label', required=True)
    parser.add_argument('--tap', type=int, nargs=2)
    parser.add_argument('--swipe', type=int, nargs=4)
    parser.add_argument('--purchase', type=int, nargs=2, metavar=('SHOP', 'CELL'))
    parser.add_argument('--back', action='store_true')
    parser.add_argument('--wait', type=float, default=.7)
    parser.add_argument('--record', type=float, default=0)
    args = parser.parse_args()
    if sum((bool(args.tap), bool(args.swipe), bool(args.purchase), args.back)) > 1:
        parser.error('select only one action')
    output = ROOT/'data/intelligence-simulation/live'
    output.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    path = output/f'{stamp}-{args.label}.png'
    if path.exists():
        raise FileExistsError(path)
    controller = AssistantController(args.folder, args.index)
    controller.hide()
    device = MuMu(args.folder, args.index)
    stop = threading.Event()
    try:
        if args.purchase:
            shop, cell = args.purchase
            if not (0 <= shop < 5 and 0 <= cell < 9):
                raise ValueError('invalid slot')
            source = output/'20261113-181458-match-search-01-end.png'
            selected = output/'20261113-181534-match1-select-purple.png'
            base = cv2.imread(str(source), 0)
            plus = cv2.imread(str(selected), 0)[195:263, 110:172]
            before = device.capture()
            gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
            header = base[152:188, 642:743]
            if cv2.matchTemplate(gray[152:188, 642:743], header, cv2.TM_CCOEFF_NORMED)[0, 0] < .95:
                raise RuntimeError('placement phase not confirmed')
            cx, cy = (140+170*(cell%3), 226+188*(cell//3))
            empty = base[cy-68:cy+68, cx-62:cx+62]
            area = gray[cy-71:cy+71, cx-66:cx+66]
            if cv2.minMaxLoc(cv2.matchTemplate(area, empty, cv2.TM_CCOEFF_NORMED))[1] < .72:
                raise RuntimeError('target empty cell not confirmed')
            device.tap(((731, 923, 1115, 731, 923)[shop], (296, 296, 296, 505, 505)[shop]), stop)
            stop.wait(.15)
            middle = device.capture()
            cv2.imwrite(str(path.with_name(path.stem+'-selected.png')), cv2.cvtColor(middle, cv2.COLOR_RGB2BGR))
            area = cv2.cvtColor(middle, cv2.COLOR_RGB2GRAY)[cy-34:cy+38, cx-34:cx+34]
            if cv2.minMaxLoc(cv2.matchTemplate(area, plus, cv2.TM_CCOEFF_NORMED))[1] < .85:
                raise RuntimeError('selection plus marker not confirmed')
            device.tap((cx, cy), stop)
        elif args.tap:
            device.tap(tuple(args.tap), stop)
        elif args.swipe:
            device.swipe(tuple(args.swipe[:2]), tuple(args.swipe[2:]), stop,
                         steps=20, step_seconds=.025)
        elif args.back:
            info = controller._manager('info', '--vmindex', str(args.index))
            serial = instance_serial(info, args.index)
            adb = Path(args.folder)/'nx_main/adb.exe'
            for command in ([str(adb), 'connect', serial],
                            [str(adb), '-s', serial, 'shell', 'input', 'keyevent', '4']):
                subprocess.run(command, capture_output=True, timeout=8, check=True,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        stop.wait(args.wait)
        frame = device.capture()
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if args.record:
            video = path.with_suffix('.mp4')
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 5, (1280, 720))
            if not writer.isOpened():
                raise RuntimeError('recording could not be opened')
            deadline = time.monotonic()+args.record
            try:
                while time.monotonic() < deadline:
                    started = time.monotonic()
                    writer.write(cv2.cvtColor(device.capture(), cv2.COLOR_RGB2BGR))
                    stop.wait(max(0, .2-(time.monotonic()-started)))
            finally:
                writer.release()
            cv2.imwrite(str(path.with_name(path.stem+'-end.png')),
                        cv2.cvtColor(device.capture(), cv2.COLOR_RGB2BGR))
        with (output/'actions.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps({'time': stamp, 'label': args.label, 'tap': args.tap,
                                 'swipe': args.swipe, 'purchase': args.purchase, 'back': args.back, 'path': str(path)},
                                ensure_ascii=False)+'\n')
        print(path)
    finally:
        device.close()
        controller.hide()


if __name__ == '__main__':
    main()
