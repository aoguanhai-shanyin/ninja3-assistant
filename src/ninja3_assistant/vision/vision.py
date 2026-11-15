"""Screen recognition for the 1280x720 reverse-bounty flow."""
from ninja3_assistant.paths import resource_root
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from ninja3_assistant.device.dxgi_adapters import adapters
from ninja3_assistant.vision.invitation import InvitationVision

ROOT = resource_root()
ASSETS = ROOT / 'assets'

@dataclass(frozen=True)
class Screen:
    login: tuple | None = None
    home: tuple | None = None
    hall: tuple | None = None
    network_error: bool = False
    network_retry: tuple | None = None
    board: bool = False
    a: str = '?'
    s: str = '?'
    ss: str = '?'
    refresh: tuple | None = None
    join: tuple | None = None
    ready: tuple | None = None
    prepared: bool = False
    invite: tuple | None = None
    menu: tuple | None = None
    failed: bool = False
    battle_loading: bool = False
    battle_active: bool = False
    game_loading: bool = False
    training: bool = False
    result: tuple | None = None
    reverse_result: bool = False
    card_cells: tuple = ()

class Vision:
    def __init__(self, use_gpu=True):
        self.invitations = InvitationVision()
        names = ['board-title', 'refresh', 'join', 'join-bottom', 'ready', 'prepared', 'invite-label',
                 'invite-collapse', 'invite-collapse-alt', 'menu-title', 'menu-reverse',
                 'join-disbanded', 'battle-word', 'battle-loading-percent', 'battle-slide', 'battle-down', 'battle-jump',
                 'result-title', 'result-confirm', 'result-boss', 'join-closed', 'card-header',
                 'startup-login', 'startup-server', 'startup-home', 'startup-hall-title', 'startup-hall-bounty',
                 'startup-network', 'startup-retry', 'training-title', 'training-parameters', 'training-loading']
        self.images = {}
        for name in names:
            image = cv2.imread(str(ASSETS / (name + '.png')), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError('找不到识别图片：' + name)
            self.images[name] = image
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        providers = ['CPUExecutionProvider']
        self.backend = 'CPU'
        if use_gpu and 'DmlExecutionProvider' in ort.get_available_providers():
            available = [a for a in adapters() if a[2] != 0x1414]
            if available:
                index, name, _, _ = max(available, key=lambda item: item[3])
                providers.insert(0, ('DmlExecutionProvider', {'device_id': index}))
                self.backend = 'GPU: ' + name
        self.session = ort.InferenceSession(str(ASSETS / 'rank-s.onnx'), sess_options=options, providers=providers)
        if use_gpu and self.backend.startswith('GPU') and self.session.get_providers()[0] != 'DmlExecutionProvider':
            raise RuntimeError('GPU 推理初始化失败')

    def match(self, gray, name, x, y, w, h, threshold=.92, max_error=None):
        template = self.images[name]
        area = gray[y:y+h, x:x+w]
        if area.shape[0] < template.shape[0] or area.shape[1] < template.shape[1]:
            return None
        _, score, _, at = cv2.minMaxLoc(cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED))
        if score < threshold:
            return None
        if max_error is not None:
            patch = area[at[1]:at[1]+template.shape[0], at[0]:at[0]+template.shape[1]]
            if np.abs(patch.astype(np.int16) - template.astype(np.int16)).mean() > max_error:
                return None
        return (x + at[0] + template.shape[1] // 2, y + at[1] + template.shape[0] // 2)

    def rank_s(self, gray, left, top):
        region = np.ascontiguousarray(gray[top:top+75, left:left+110], dtype=np.float32)[None, None] / 255.0
        if region.shape != (1, 1, 75, 110):
            return False
        # The model computes the minimum mean squared error over the card's badge area.
        error = float(self.session.run(None, {'image': region})[0]) / (74 * 45)
        return error < .012

    @staticmethod
    def filter(frame, x):
        r, g, b = frame[168:175, x-3:x+4, :3].mean(axis=(0, 1))
        if 45 <= r <= 135 and g <= 90 and b <= 50 and r >= g + 25:
            return 'on'
        if r >= 180 and g >= 140 and b >= 95:
            return 'off'
        return '?'

    def detect(self, frame):
        if frame.shape != (720, 1280, 3):
            return Screen()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        invite = self.invitations.detect(gray)
        if invite:
            return Screen(invite=invite)
        if self.match(gray, 'menu-title', 80, 0, 250, 120, max_error=30):
            menu = self.match(gray, 'menu-reverse', 900, 410, 320, 130)
            if menu:
                return Screen(menu=menu)
        if self.match(gray, 'result-title', 540, 40, 280, 95):
            confirm = self.match(gray, 'result-confirm', 500, 620, 300, 100)
            if confirm:
                reverse = bool(self.match(gray, 'result-boss', 460, 20, 140, 110))
                return Screen(result=confirm, reverse_result=reverse)
        if not self.match(gray, 'board-title', 80, 0, 340, 120, threshold=.95, max_error=30):
            if ((cv2.mean(gray)[0] if gray.dtype == np.uint8 and gray.ndim == 2 else gray.mean()) < 5 and
                    self.match(gray, 'training-loading', 1100, 640, 180, 80, threshold=.9, max_error=20)):
                return Screen(game_loading=True)
            if (self.match(gray, 'training-title', 80, 0, 240, 110, threshold=.95, max_error=25) and
                    self.match(gray, 'training-parameters', 1130, 0, 150, 125, threshold=.95, max_error=25)):
                return Screen(training=True)
            if self.match(gray, 'startup-network', 480, 240, 320, 100, threshold=.92, max_error=25):
                retry = self.match(gray, 'startup-retry', 500, 490, 280, 100, threshold=.92, max_error=25)
                retry = retry or self.match(gray, 'result-confirm', 500, 480, 280, 110, threshold=.92, max_error=25)
                return Screen(network_error=True, network_retry=retry)
            login = self.match(gray, 'startup-login', 500, 470, 300, 90, threshold=.93, max_error=25)
            if login and self.match(gray, 'startup-server', 620, 390, 230, 80, threshold=.93, max_error=25):
                return Screen(login=login)
            home = self.match(gray, 'startup-home', 700, 540, 300, 150, threshold=.92, max_error=25)
            if home:
                return Screen(home=home)
            if self.match(gray, 'startup-hall-title', 80, 0, 300, 110, threshold=.93, max_error=30):
                hall = self.match(gray, 'startup-hall-bounty', 910, 100, 330, 120, threshold=.92, max_error=30)
                if hall:
                    return Screen(hall=hall)
            if gray[:50, :100].mean() < 15 and gray[:50, 1180:].mean() < 15:
                percentages = sum(bool(self.match(gray, 'battle-loading-percent', 278 + i*210, 525, 40, 45, threshold=.72))
                                  for i in range(5))
                if percentages >= 3:
                    return Screen(battle_loading=True)
            if ((self.match(gray, 'battle-slide', 110, 570, 170, 130, threshold=.85)
                 or self.match(gray, 'battle-down', 110, 570, 170, 130, threshold=.9))
                    and self.match(gray, 'battle-jump', 1010, 550, 180, 150, threshold=.85)):
                return Screen(battle_active=True)
            prepared = bool(self.match(gray, 'prepared', 940, 610, 330, 106))
            ready = None if prepared else self.match(gray, 'ready', 940, 610, 330, 106)
            return Screen(ready=ready, prepared=prepared)
        failed = bool(self.match(gray, 'join-disbanded', 420, 220, 440, 75, threshold=.92)
                      or self.match(gray, 'join-closed', 420, 220, 440, 75, threshold=.86)
                      or self.match(gray, 'battle-word', 680, 220, 180, 75, threshold=.85))
        refresh = self.match(gray, 'refresh', 1040, 120, 210, 100)
        join = None
        card_cells = []
        if not failed:
            for top in (185, 361):
                for left in (95, 465, 835):
                    if self.match(gray, 'card-header', left+55, top+13, 140, 42, threshold=.90):
                        card_cells.append((left, top+13))
                    if self.rank_s(gray, left-5, top):
                        found = self.match(gray, 'join', left, top+13, 360, 176)
                        if not found:
                            bottom = self.match(gray, 'join-bottom', left, top+13, 360, 176, threshold=.95)
                            if bottom:
                                found = (bottom[0], bottom[1] - 16)
                        if found and not join:
                            join = found
        return Screen(board=True, a=self.filter(frame, 766), s=self.filter(frame, 876),
                      ss=self.filter(frame, 978), refresh=refresh, join=join, failed=failed,
                      card_cells=tuple(card_cells))
