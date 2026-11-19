"""Experimental Hayabusa perception; RGB input, calibrated at 1280 x 720.

Optional YOLO ONNX models detect attacks and the visible player body.
Absent models retain the earlier color rules and player templates.
Teaching captions and chapter labels are never recognition inputs.
"""
from dataclasses import dataclass, replace
from math import hypot

import cv2
import numpy as np

from ninja3_assistant.combat.battle import BladeTrack
from ninja3_assistant.vision.vision import ASSETS


@dataclass(frozen=True)
class HayabusaScreen:
    active: bool = False
    rope_y: float | None = None
    player_x: float | None = None
    player_y: float | None = None
    side: str = 'unknown'
    airborne: bool = False
    weapon_ready: bool = False
    weapon_point: tuple | None = None
    result: str | None = None
    board_hint: str = 'unknown'
    phase_hint: str | None = None
    board_id: str | None = None
    board_name: str = ''
    board_confidence: float = 0.
    board_margin: float = 0.
    board_progress: float = 0.
    board_evidence: str = ''
    board_locked: bool = False
    board_onset: bool = False
    board_ended: bool = False
    board_epoch: int = 0
    phase_source: str = 'unknown'
    left_control: str = 'unknown'
    right_control: str = 'unknown'
    grounded: bool | None = None
    player_head_x: float | None = None
    player_head_y: float | None = None
    player_box: tuple | None = None
    player_source: str = 'unknown'


def color_masks(field):
    r, g, b = (field[:, :, channel].astype(np.float32) for channel in range(3))
    return {
        'blue': ((b >= 225) & (g >= 170) & (b-r >= 30) & (g-r >= 20)).astype(np.uint8)*255,
        'fire': ((r >= 205) & (g >= 120) & (r-b >= 40)).astype(np.uint8)*255,
        'blackfire': ((r >= 70) & (b >= 110) & (b-g >= 55) & (r-g >= 35)).astype(np.uint8)*255,
    }


class HayabusaSegmenter:
    def __init__(self, use_gpu=True):
        self.session = None
        self.backend = 'CPU color rules'
        if not use_gpu:
            return
        import onnxruntime as ort
        from ninja3_assistant.vision.winml import model_session
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        self.session, self.backend = model_session(ASSETS/'hayabusa-colors.onnx', require_gpu=True, options=options)

    def masks(self, field):
        if self.session is None:
            return color_masks(field)
        tensor = np.ascontiguousarray(field.transpose(2, 0, 1)[None], dtype=np.float32)
        values = self.session.run(None, {'image': tensor})
        return {name: (value[0, 0]*255).astype(np.uint8)
                for name, value in zip(('blue', 'fire', 'blackfire'), values)}


class HayabusaVision:
    def __init__(self, segmenter=None, use_gpu=True, recognize_boards=True, detector=None, player_detector=None):
        use_player_model = detector is None and segmenter is None
        if detector is None and segmenter is None and (ASSETS/'hayabusa-detector.onnx').is_file():
            from ninja3_assistant.combat.hayabusa.detector import HayabusaDetector
            detector = HayabusaDetector(ASSETS/'hayabusa-detector.onnx', use_gpu=use_gpu)
        self.detector = detector if detector is not False else None
        self.segmenter = segmenter
        if self.detector is None and self.segmenter is None:
            self.segmenter = HayabusaSegmenter(use_gpu)
        self.backend = (self.detector.backend if self.detector is not None
                        else getattr(self.segmenter, 'backend', 'injected segmenter'))
        if player_detector is None and use_player_model and (ASSETS/'hayabusa-player.onnx').is_file():
            from ninja3_assistant.combat.hayabusa.player_detector import HayabusaPlayerDetector
            player_detector = HayabusaPlayerDetector(ASSETS/'hayabusa-player.onnx', use_gpu=use_gpu)
        self.player_locator = None
        if player_detector is not None and player_detector is not False:
            from ninja3_assistant.combat.hayabusa.player_detector import PlayerBodyLocator
            self.player_locator = PlayerBodyLocator(player_detector)
            self.backend += ' + ' + player_detector.backend
        self.templates = {}
        names = ('name', 'portrait', 'shadow-name', 'shadow-portrait', 'down', 'victory',
                 'result-name', 'shadow-result-name', 'defeat', 'right-jump-live', 'right-up', 'right-jump')
        if self.player_locator is None:
            names += ('player-head-live', 'player-head-live-2', 'player-head-live-3',
                      'player-head-air-tuck', 'player-head-air', 'player-head-below',
                      'player-head-run', 'player-head-below-2', 'player-head-apex',
                      'face-1', 'face-2', 'face-3', 'face-4', 'face-5')
        for name in names:
            image = cv2.imread(str(ASSETS/f'hayabusa-{name}.png'), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f'缺少隼白识别模板：{name}')
            self.templates[name] = image
        self.live_heads = [] if self.player_locator is not None else [(self.templates[name], offset, scales) for name, offset, scales in (
            ('player-head-live', (-25, 25), (.9, 1, 1.1)),
            ('player-head-live-2', (-25, 25), (.9, 1, 1.1)),
            ('player-head-live-3', (-25, 25), (.9, 1, 1.1)),
            ('player-head-air-tuck', (-25, -5), (1,)),
            ('player-head-air', (-15, 30), (1,)),
            ('player-head-below', (-22, -36), (1,)),
            ('player-head-run', (-25, 25), (1,)),
            ('player-head-below-2', (-22, -36), (1,)),
            ('player-head-apex', (-20, 32), (1,)))]
        self.weapon = cv2.imread(str(ASSETS/'weapon-ready.png'), cv2.IMREAD_GRAYSCALE)
        self.slide = cv2.imread(str(ASSETS/'battle-slide.png'), cv2.IMREAD_GRAYSCALE)
        self.recognizer = None
        if recognize_boards:
            from ninja3_assistant.combat.hayabusa.recognition import HayabusaBoardRecognizer
            self.recognizer = HayabusaBoardRecognizer()
        self.rotated_faces = [] if self.player_locator is not None else [cv2.warpAffine(self.templates[f'face-{i}'],
            cv2.getRotationMatrix2D((15, 15), angle, 1), (30, 30))
            for i in range(1, 6) for angle in range(30, 360, 30)]
        self.reset()

    def warmup(self):
        """Run the selected models at their live input sizes without tracking a scene."""
        if self.detector is None and self.segmenter is not None:
            self.segmenter.masks(np.zeros((250, 640, 3), dtype=np.uint8))
        if self.detector is not None or self.player_locator is not None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            if self.detector is not None:
                self.detector.predict(frame)
            if self.player_locator is not None:
                self.player_locator.detector.predict(frame)

    def reset(self):
        self.tracks = {}
        self.next_id = 1
        self.last_time = None
        self.last_rope = None
        self.last_player_state = None
        self._reset_player()
        if self.recognizer is not None:
            self.recognizer.reset()

    def _reset_player(self):
        self.live_player_track = None
        self.live_player_seen = False
        self.current_player_head = None
        self.current_player_box = None
        self.current_player_source = 'unknown'
        if self.player_locator is not None:
            self.player_locator.reset()

    def _recognized(self, frame, screen, tracks, now):
        if self.recognizer is None:
            return screen, tracks
        last_seen = self.recognizer.last_time
        if (screen.result or frame.shape != (720, 1280, 3) or
                (last_seen is not None and not 0 <= now-last_seen <= .35)):
            self.recognizer.reset()
        if not screen.active:
            return screen, tracks
        match = self.recognizer.update(frame, screen, tracks, now)
        return replace(screen, board_id=match.board_id, board_name=getattr(match, 'name', ''),
            phase_hint=match.phase, board_confidence=match.confidence,
            board_margin=match.margin, board_progress=match.progress,
            board_evidence=match.evidence, phase_source=getattr(match, 'phase_source', 'board_match'),
            board_locked=getattr(match, 'locked', False), board_onset=getattr(match, 'onset', False),
            board_ended=getattr(match, 'ended', False), board_epoch=getattr(match, 'epoch', 0)), tracks

    @staticmethod
    def _match(gray, template, bounds, scales=(1.0,)):
        x, y, width, height = bounds
        area = gray[y:y+height, x:x+width]
        best = (0.0, None, None)
        for scale in scales:
            target = cv2.resize(template, None, fx=scale, fy=scale)
            h, w = target.shape
            if area.shape[0] < h or area.shape[1] < w:
                continue
            _, score, _, at = cv2.minMaxLoc(cv2.matchTemplate(area, target, cv2.TM_CCOEFF_NORMED))
            error = float(np.abs(area[at[1]:at[1]+h, at[0]:at[0]+w].astype(np.float32)-target).mean())
            if score > best[0]:
                best = (score, (x+at[0]+w/2, y+at[1]+h/2), error)
        return best

    @classmethod
    def _has_hud(cls, gray, templates):
        for name_key, portrait_key, score, error in (('name', 'portrait', (.83, .8), 32),
                                                    ('shadow-name', 'shadow-portrait', (.9, .9), 25)):
            name = cls._match(gray, templates[name_key], (280, 8, 360, 90), (.8, 1, 1.2))
            if name[0] < score[0] or name[2] >= error:
                continue
            portrait = cls._match(gray, templates[portrait_key], (270, 4, 250, 110), (.8, 1, 1.2))
            if portrait[0] >= score[1] and portrait[2] < error:
                return True
        return False

    @classmethod
    def _result(cls, gray, templates):
        victory = cls._match(gray, templates['victory'], (350, 220, 650, 360), (.85, 1, 1.15))
        if victory[0] >= .88:
            result_name = cls._match(gray, templates['result-name'], (350, 0, 700, 260), (.85, 1, 1.15))
            if result_name[0] >= .85:
                return 'victory'
        defeat = cls._match(gray, templates['defeat'], (350, 10, 600, 200), (.9, 1, 1.1))
        if defeat[0] >= .9 and defeat[2] < 25:
            shadow = cls._match(gray, templates['shadow-result-name'], (240, 240, 420, 160), (.9, 1, 1.1))
            if shadow[0] >= .9 and shadow[2] < 25:
                return 'defeat'
        return None

    def detect(self, frame, now):
        if self.last_time is not None and not 0 < now-self.last_time <= .5:
            self.reset()
        self.last_time = now
        if frame.shape != (720, 1280, 3):
            self.tracks.clear()
            self._reset_player()
            return self._recognized(frame, HayabusaScreen(), [], now)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        hud = self._has_hud(gray, self.templates)
        if not hud:
            self.tracks.clear()
            self._reset_player()
            result = self._result(gray, self.templates)
            return self._recognized(frame, HayabusaScreen(result=result), [], now)
        down = self._match(gray, self.templates['down'], (80, 550, 230, 165), (.8, 1, 1.2))
        slide = self._match(gray, self.slide, (80, 550, 230, 165), (.65, .8, 1, 1.2))
        left = down if down[0] >= slide[0] else slide
        left_control = 'down' if down[0] >= slide[0] else 'slide'
        if not (left[0] >= .82 and left[2] < 35):
            self.tracks.clear()
            self._reset_player()
            result = self._result(gray, self.templates)
            return self._recognized(frame, HayabusaScreen(result=result), [], now)
        dark = (gray[190:555, 130:1150] < 32).mean(axis=1)
        smooth = np.convolve(dark, np.ones(5)/5, mode='same')
        contrast = smooth-(np.roll(smooth, 10)+np.roll(smooth, -10))/2
        index = int(np.argmax(contrast[12:-12]))+12
        if smooth[index] < .6 or contrast[index] < .38:
            if self.last_rope is None or now-self.last_rope[1] > .35:
                self.tracks.clear()
                self._reset_player()
                return self._recognized(frame, HayabusaScreen(), [], now)
            rope = self.last_rope[0]
        else:
            rope = float(index+190)
            self.last_rope = (rope, now)
        point = self._locate_player(frame, gray, rope)
        head = self.current_player_head
        side = ('above' if point[1] < rope else 'below') if point else 'unknown'
        up = self._match(gray, self.templates['right-up'], (1010, 550, 180, 165), (.8, 1, 1.2))
        jump = self._match(gray, self.templates['right-jump'], (1010, 550, 180, 165), (.8, 1, 1.2))
        live_jump = self._match(gray, self.templates['right-jump-live'], (1010, 550, 180, 165), (.9, 1, 1.1))
        if live_jump[0] > jump[0]:
            jump = live_jump
        right = up if up[0] >= jump[0] else jump
        right_control = ('up' if up[0] >= jump[0] else 'jump') if right[0] >= .84 and right[2] < 35 else 'unknown'
        airborne = bool(point and side == 'above' and (right_control == 'up' or rope-point[1] > 85))
        if point:
            height = rope-point[1]
            if self.last_player_state is not None:
                previous_height, previous_time = self.last_player_state
                dt = now-previous_time
                if (side == 'above' and 0 < dt <= .2 and abs(height-previous_height) >= 5
                        and abs((height-previous_height)/dt) > 120):
                    airborne = True
            self.last_player_state = (height, now)
        grounded = ((not airborne and 10 <= rope-point[1] <= 55)
                    if point and side == 'above' and right_control != 'unknown' else None)
        weapon = self._match(gray, self.weapon, (280, 550, 220, 165)) if self.weapon is not None else (0, None, None)
        ready = weapon[0] >= .94 and weapon[2] < 25
        tracks = self._detect_tracks(frame, rope, point, now)
        kinds = {track.kind for track in tracks}
        hint = 'compound' if len(kinds) > 1 else next(iter(kinds), 'unknown')
        screen = HayabusaScreen(True, rope, point[0] if point else None,
            point[1] if point else None, side, airborne,
            ready, (int(weapon[1][0]), int(weapon[1][1])) if ready else None, board_hint=hint,
            left_control=left_control, right_control=right_control, grounded=grounded,
            player_head_x=float(head[0]) if head is not None else None,
            player_head_y=float(head[1]) if head is not None else None,
            player_box=self.current_player_box, player_source=self.current_player_source)
        return self._recognized(frame, screen, tracks, now)

    def _locate_player(self, frame, gray, rope):
        self.current_player_head = None
        self.current_player_box = None
        self.current_player_source = 'unknown'
        if self.player_locator is not None:
            box = self.player_locator.locate(frame, self.last_time)
            self.current_player_box = box
            if box is not None:
                # This is the center of visible body extent, not a measured damage hitbox.
                self.current_player_source = 'body-detector'
                return (box[0]+box[2])/2, (box[1]+box[3])/2
            return None
        recent = self.live_player_track
        if recent is not None and 0 < self.last_time-recent[3] <= .5 and self.last_time-recent[4] <= 1:
            left, right = max(140, int(recent[2][0])-140), min(1140, int(recent[2][0])+140)
            reach = min(180, 1200*(self.last_time-recent[3])+12)+20
            top, bottom = max(70, int(recent[2][1]-reach)), min(535, int(recent[2][1]+reach))
            bounds = (left, top, right-left, bottom-top)
            local_scales = (.9, 1, 1.1)
        else:
            bounds = (140, 70, 1000, 465)
            local_scales = None
        live = max(((self._match(gray, template, bounds, local_scales or scales), offset)
                    for template, offset, scales in self.live_heads),
                   key=lambda value: value[0][0])
        match, offset = live
        if match[0] >= .83 and match[2] < 25:
            self._seed_live_player(gray, match[1], offset)
            return match[1][0]+offset[0], match[1][1]+offset[1]
        followed = self._follow_live_player(gray)
        if followed is not None:
            return followed
        # A confirmed live head cannot become an old masked face during one HUD segment.
        # Keep loss of its visual evidence unknown; the legacy search is for initial acquisition.
        if self.live_player_seen:
            return None
        top, bottom = max(70, int(rope-390)), min(535, int(rope+170))
        area = frame[top:bottom, 140:1140]
        minimum, maximum = area.min(axis=2), area.max(axis=2)
        white = ((minimum > 175) & (maximum.astype(np.int16)-minimum < 65)).astype(np.uint8)
        _, _, stats, centers = cv2.connectedComponentsWithStats(white)
        best = (0.0, None, None)
        candidates = []
        for (_, _, w, h, count), (cx, cy) in zip(stats[1:], centers[1:]):
            if not (45 <= count <= 320 and 7 <= w <= 30 and 7 <= h <= 30):
                continue
            left, upper = max(0, int(cx)+140-23), max(0, int(cy)+top-23)
            candidates.append((left, upper))
            for i in range(1, 6):
                found = self._match(gray, self.templates[f'face-{i}'], (left, upper, 46, 46))
                if found[0] > best[0] and found[2] < 35:
                    best = found
        if best[0] < .83:
            for left, upper in candidates:
                local = frame[max(0, upper-20):upper+66, max(0, left-20):left+66].astype(np.int16)
                purple = (local[:, :, 0]-local[:, :, 1] > 35) & (local[:, :, 2]-local[:, :, 1] > 55)
                if np.count_nonzero(purple) < 15:
                    continue
                for template in self.rotated_faces:
                    found = self._match(gray, template, (left, upper, 46, 46))
                    if found[0] > best[0] and found[2] < 35:
                        best = found
        if best[0] >= .83:
            self.live_player_track = None
            self.current_player_head = best[1]
            self.current_player_source = 'legacy-mask'
            return best[1]
        return None

    def _seed_live_player(self, gray, head, offset):
        self.live_player_seen = True
        self.current_player_head = head
        self.current_player_source = 'head-template'
        x, y = map(round, head)
        left, top = x-12, y-12
        points = cv2.goodFeaturesToTrack(gray[top:y+13, left:x+13], maxCorners=12,
                                        qualityLevel=.03, minDistance=3, blockSize=3)
        self.live_player_track = None
        if points is not None and len(points) >= 4:
            points += np.array([left, top], dtype=np.float32)
            self.live_player_track = (gray.copy(), points, np.array(head), self.last_time, self.last_time, offset)

    def _follow_live_player(self, gray):
        if self.live_player_track is None:
            return None
        previous, points, head, seen, verified, offset = self.live_player_track
        self.live_player_track = None
        if not 0 < self.last_time-seen <= .5 or self.last_time-verified > 1:
            return None
        options = dict(winSize=(15, 15), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .02))
        forward, status, error = cv2.calcOpticalFlowPyrLK(previous, gray, points, None, **options)
        if forward is None or np.count_nonzero(status) < 4:
            return None
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, forward, None, **options)
        if backward is None:
            return None
        valid = ((status[:, 0] == 1) & (back_status[:, 0] == 1) & (error[:, 0] < 20) &
                 (np.linalg.norm(backward[:, 0]-points[:, 0], axis=1) < 1.5))
        if np.count_nonzero(valid) < 4:
            return None
        motions = forward[valid, 0]-points[valid, 0]
        motion = np.median(motions, axis=0)
        head = head+motion
        if (np.linalg.norm(motion) > min(180, 1200*(self.last_time-seen)+12) or
                np.median(np.linalg.norm(motions-motion, axis=1)) > 2.5 or
                not (152 <= head[0] <= 1127 and 85 <= head[1] <= 522)):
            return None
        self.live_player_track = (gray.copy(), forward[valid], head, self.last_time, verified, offset)
        self.current_player_head = head
        self.current_player_source = 'head-flow'
        return float(head[0]+offset[0]), float(head[1]+offset[1])

    def _detect_tracks(self, frame, rope, player, now):
        if self.detector is not None:
            detections = []
            for item in self.detector.predict(frame):
                x1, y1, x2, y2 = item.xyxy
                x, y = (x1+x2)/2, (y1+y2)/2
                if (x < 130 and y < 360) or y > min(575, rope+150):
                    continue
                detections.append((x, y, max(1, round(x2-x1)), max(1, round(y2-y1)), item.kind))
            return self._track(detections, now)
        field = cv2.resize(frame[90:590], (640, 250))
        field_gray = cv2.cvtColor(field, cv2.COLOR_RGB2GRAY)
        masks = self.segmenter.masks(field)
        bright = cv2.inRange(field, (220, 210, 155), (255, 255, 255))
        # The orange perimeter alone splits a white-cored flame into fragments.
        # Join only nearby bright cores, rather than treating yellow text as fire.
        near_fire = cv2.dilate(masks['fire'], np.ones((7, 7), np.uint8))
        masks['fire'] = cv2.bitwise_or(masks['fire'], cv2.bitwise_and(bright, near_fire))
        detections = []
        for name, mask in masks.items():
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            for x, y, w, h, area in stats[1:]:
                cx, cy, width, height = 2*x+w, 90+2*y+h, 2*w, 2*h
                if area < 12 or width < 8 or height < 6:
                    continue
                if player and hypot(cx-player[0], cy-player[1]) < 45:
                    continue
                if cx < 130 and cy < 360:
                    continue
                if name == 'blue':
                    if width >= 450 and height < 160:
                        kind = 'flash'
                    elif width > 165 or height > 190 or area > 6500:
                        continue  # Huge foreground effects are not live projectile hitboxes.
                    elif max(width, height) >= 42 and (height/width > 1.25 or width/height > 2):
                        kind = 'slash'
                    elif 9 <= width <= 90 and 8 <= height <= 70:
                        kind = 'shuriken'
                    else:
                        continue
                    if kind != 'flash' and np.mean(field_gray[y:y+h, x:x+w] < 70) < .08:
                        continue  # Lit rectangular window panes have no dark projectile body/arc gap.
                elif name == 'fire':
                    if not (38 <= width <= 190 and 8 <= height <= 110 and width >= 1.05*height):
                        continue
                    core = bright[y:y+h, x:x+w]
                    if np.count_nonzero(core) < max(8, area*.08):
                        continue
                    if cv2.distanceTransform(core, cv2.DIST_L2, 3).max() < 2:
                        continue  # Letter strokes and decorative sparks lack a thick flame core.
                    kind = 'fire'
                else:
                    if not (40 <= width <= 160 and 12 <= height <= 95 and width >= 1.25*height
                            and area >= 120 and area/(w*h) >= .2):
                        continue
                    kind = 'blackfire'
                if cy > min(575, rope+150):
                    continue
                detections.append((float(cx), float(cy), int(width), int(height), kind))
        return self._track(detections, now)

    def _track(self, detections, now):
        self.tracks = {key: value for key, value in self.tracks.items() if 0 <= now-value.seen <= .18}
        pairs = []
        for old in self.tracks.values():
            dt = now-old.seen
            for i, (x, y, _, _, kind) in enumerate(detections):
                if kind == old.kind:
                    distance = hypot(x-old.x-old.vx*dt, y-old.y-old.vy*dt)
                    if distance <= 45+1000*dt:
                        pairs.append((distance, old.id, i))
        used_tracks, used_detections = set(), set()
        for _, ident, i in sorted(pairs):
            if ident in used_tracks or i in used_detections:
                continue
            old = self.tracks[ident]
            x, y, w, h, kind = detections[i]
            dt = now-old.seen
            vx = .6*((x-old.x)/dt)+.4*old.vx if dt > 0 else 0
            vy = .6*((y-old.y)/dt)+.4*old.vy if dt > 0 else 0
            self.tracks[ident] = BladeTrack(ident, x, y, w, h, vx, vy, now, old.hits+1, kind)
            used_tracks.add(ident)
            used_detections.add(i)
        for i, (x, y, w, h, kind) in enumerate(detections):
            if i not in used_detections:
                ident = self.next_id
                self.next_id += 1
                self.tracks[ident] = BladeTrack(ident, x, y, w, h, 0, 0, now, 1, kind)
                used_tracks.add(ident)
        return [self.tracks[key] for key in sorted(used_tracks)]
