"""Run a bounded experimental Boss trial on the copied MuMu instance 0."""
from ninja3_assistant.paths import project_root
import ctypes
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ninja3_assistant.combat.battle import RedBladeTracker
from ninja3_assistant.combat.boss.policy import BossPolicy
from ninja3_assistant.combat.boss.screen import BossScreenDetector
from ninja3_assistant.vision.gpu_vision import GpuAttackSegmenter
from ninja3_assistant.device.mumu import MuMu
from ninja3_assistant.vision.vision import ASSETS


EVIDENCE = (Path(sys.executable).resolve().parent / 'boss-trial-recordings' if getattr(sys, 'frozen', False)
            else project_root().parent / 'android-icon-clicker/build/evidence/battle-inspect')


def challenge_ready(frame):
    if frame.shape != (720, 1280, 3):
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    checks = (
        ('solo-title', (95, 5, 260, 95)),
        ('solo-boss-name', (210, 420, 250, 65)),
        ('solo-start', (1010, 620, 260, 85)),
    )
    for name, (x, y, w, h) in checks:
        template = cv2.imread(str(ASSETS / (name + '.png')), cv2.IMREAD_GRAYSCALE)
        if template is None:
            return False
        _, score, _, point = cv2.minMaxLoc(cv2.matchTemplate(gray[y:y+h, x:x+w], template, cv2.TM_CCOEFF_NORMED))
        if score < .94:
            return False
        patch = gray[y+point[1]:y+point[1]+template.shape[0], x+point[0]:x+point[0]+template.shape[1]]
        if np.abs(patch.astype(np.int16) - template.astype(np.int16)).mean() > 25:
            return False
    return True


def run():
    cv2.setNumThreads(1)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    video = EVIDENCE / ('gpu-boss-trial-' + stamp + '.mp4')
    log = EVIDENCE / ('gpu-boss-trial-' + stamp + '.json')
    segmenter = GpuAttackSegmenter()
    detector = BossScreenDetector()
    tracker = RedBladeTracker(segmenter=segmenter, include_thin=True, include_orbs=True)
    policy = BossPolicy()
    device = MuMu(r'D:\Program Files\Netease\MuMu', 0)
    writer = None
    held = False
    events, frame_times, costs = [], [], []
    counts = {'boss': 0, 'above_located': 0, 'below_located': 0, 'weapon_ready': 0}
    try:
        frame = device.capture()
        if not challenge_ready(frame):
            raise RuntimeError('复制实例不在禅语魔心单人挑战选择页，未启动测试')
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 720))
        if not writer.isOpened():
            raise RuntimeError('无法创建测试录像，未启动挑战')
        if not challenge_ready(device.capture()):
            raise RuntimeError('挑战页面已经变化，未启动测试')
        if device.dll.nemu_input_event_touch_down(device.handle, device.display, 1135, 665):
            raise RuntimeError('无法点击开始挑战')
        held = True
        time.sleep(.07)
        if device.dll.nemu_input_event_touch_up(device.handle, device.display):
            raise RuntimeError('无法松开开始挑战')
        held = False
        start = time.monotonic()
        next_record = start
        release_at = 0.0
        last_boss = None
        count = 0
        print(f'复制实例 0：已开始挑战；GPU：{segmenter.backend}；按 Esc 停止。', flush=True)
        while time.monotonic() - start < 120:
            now = time.monotonic()
            if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
                print('收到 Esc，停止测试。', flush=True)
                break
            if held and now >= release_at:
                result = device.dll.nemu_input_event_touch_up(device.handle, device.display)
                held = False
                if result:
                    raise RuntimeError('无法松开触控')
            before = time.perf_counter()
            frame = device.capture()
            captured = time.monotonic()
            state = detector.detect(frame)
            if state.target_boss:
                counts['boss'] += 1
                if state.player_x is not None and state.side in ('above', 'below'):
                    counts[state.side + '_located'] += 1
                if state.weapon_ready:
                    counts['weapon_ready'] += 1
                last_boss = captured
                tracks = tracker.update(frame, captured)
                action = policy.decide(state, tracks, captured)
                if action and not held:
                    result = device.dll.nemu_input_event_touch_down(device.handle, device.display, *action.point)
                    if result:
                        raise RuntimeError('自动动作触控失败')
                    held = True
                    release_at = time.monotonic() + .07
                    policy.completed(action, captured)
                    threat = next((item for item in tracks if item.id == action.threat_id), None)
                    events.append({'time': captured-start, 'action': action.kind, 'side': state.side,
                                   'player_x': state.player_x, 'rope_y': state.rope_y,
                                   'threat': None if threat is None else {'kind': threat.kind, 'id': threat.id,
                                   'x': int(threat.x), 'y': int(threat.y), 'vx': float(threat.vx)}})
                    print(f'{captured-start:.1f}s {action.kind} side={state.side} player={state.player_x}', flush=True)
            elif last_boss is not None and captured - last_boss > 5:
                print('Boss 画面消失，停止测试。', flush=True)
                break
            if captured >= next_record:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_times.append(captured-start)
                next_record += 1 / 15
                if captured - next_record > 1 / 15:
                    next_record = captured + 1 / 15
            costs.append((time.perf_counter() - before) * 1000)
            count += 1
        print(f'识别帧数：{count}；动作数：{len(events)}；录像：{video}', flush=True)
    finally:
        if held:
            device.dll.nemu_input_event_touch_up(device.handle, device.display)
        if writer is not None:
            writer.release()
        device.close()
        summary = {'gpu': segmenter.backend, 'events': events, 'frame_times': frame_times, 'counts': counts,
                   'loop_ms_p50': float(np.percentile(costs, 50)) if costs else None,
                   'loop_ms_p95': float(np.percentile(costs, 95)) if costs else None}
        log.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'测试记录：{log}', flush=True)


def self_test(path):
    try:
        segmenter = GpuAttackSegmenter()
        strong, weak = segmenter.masks(np.zeros((300, 590, 3), dtype=np.uint8))
        if strong.shape != (300, 590) or weak.shape != (300, 590) or strong.any() or weak.any():
            raise RuntimeError('GPU 攻击分割自检失败')
        BossScreenDetector()
        RedBladeTracker(segmenter=segmenter, include_thin=True, include_orbs=True)
        report = {'ok': True, 'gpu': segmenter.backend, 'providers': segmenter.session.get_providers()}
    except Exception as error:
        report = {'ok': False, 'error': str(error)}
    Path(path).write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')


def wait_for_console():
    if sys.stdin.isatty():
        try:
            input('按回车关闭窗口。')
        except EOFError:
            pass


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--self-test':
        self_test(sys.argv[2])
    else:
        try:
            run()
        except Exception as error:
            print(f'测试已停止：{error}', flush=True)
        finally:
            wait_for_console()
