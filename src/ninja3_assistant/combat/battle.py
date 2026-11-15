"""Experimental tracking of red attacks observed in the boss recording."""
from dataclasses import dataclass
from math import hypot

import cv2

from ninja3_assistant.vision.vision import ASSETS


@dataclass(frozen=True)
class BladeTrack:
    id: int
    x: float
    y: float
    width: int
    height: int
    vx: float
    vy: float
    seen: float
    hits: int
    kind: str = 'spin'


class RedBladeTracker:
    """Detect one observed attack type; output must not be used as a dodge command."""

    def __init__(self, segmenter=None, include_thin=False, include_orbs=False):
        self.tracks = {}
        self.next_id = 1
        self.segmenter = segmenter
        self.include_thin = include_thin
        self.include_orbs = include_orbs
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.thin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.orb_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.orb_template = cv2.imread(str(ASSETS / 'boss-orb.png'), cv2.IMREAD_GRAYSCALE) if include_orbs else None
        if include_orbs and self.orb_template is None:
            raise RuntimeError('缺少 Boss 符文弹幕模板')

    def update(self, frame, now):
        # Restrict the prototype to the visible battlefield and process it at half size.
        field = cv2.resize(frame[110:710, 100:1280], (590, 300))
        if self.segmenter:
            mask, weak = self.segmenter.masks(field)
        else:
            hsv = cv2.cvtColor(field, cv2.COLOR_RGB2HSV)
            mask = cv2.bitwise_or(cv2.inRange(hsv, (0, 150, 130), (10, 255, 255)),
                                  cv2.inRange(hsv, (172, 150, 130), (179, 255, 255)))
            weak = cv2.bitwise_or(cv2.inRange(hsv, (0, 60, 150), (10, 255, 255)),
                                  cv2.inRange(hsv, (172, 60, 150), (179, 255, 255)))
        raw_mask = mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        detections = [(100 + 2*x + w, 110 + 2*y + h, 2*w, 2*h, 'spin')
                      for x, y, w, h, area in stats[1:]
                      if area >= 2000 and w >= 40 and h >= 40]
        if self.include_thin:
            weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, self.thin_kernel)
            _, _, stats, _ = cv2.connectedComponentsWithStats(weak)
            for x, y, w, h, area in stats[1:]:
                if area < 45 or not (5 <= w <= 45 and h >= 25 and h >= 1.2*w):
                    continue
                cx, cy = 100 + 2*x + w, 110 + 2*y + h
                if any(abs(cx-sx) <= sw/2 and abs(cy-sy) <= sh/2 for sx, sy, sw, sh, _ in detections):
                    continue
                detections.append((cx, cy, 2*w, 2*h, 'thin'))
        if self.include_orbs:
            orb_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, self.orb_kernel)
            _, _, stats, _ = cv2.connectedComponentsWithStats(orb_mask)
            gray = cv2.cvtColor(field, cv2.COLOR_RGB2GRAY)
            for x, y, w, h, area in stats[1:]:
                if not (15 <= area <= 500 and 5 <= w <= 35 and 5 <= h <= 35 and .45 <= w/h <= 2.2):
                    continue
                cx, cy = x + w//2, y + h//2
                left, top = max(0, cx-12), max(0, cy-12)
                region = gray[top:min(300, cy+12), left:min(590, cx+12)]
                if region.shape[0] < 16 or region.shape[1] < 16:
                    continue
                _, score, _, pos = cv2.minMaxLoc(cv2.matchTemplate(region, self.orb_template, cv2.TM_CCOEFF_NORMED))
                if score < .5:
                    continue
                px, py = 100 + 2*(left+pos[0]+8), 110 + 2*(top+pos[1]+8)
                if any(abs(px-sx) < 20 and abs(py-sy) < 20 for sx, sy, _, _, kind in detections if kind == 'orb'):
                    continue
                if any(abs(px-sx) <= sw/2 and abs(py-sy) <= sh/2 for sx, sy, sw, sh, kind in detections if kind == 'spin'):
                    continue
                detections.append((px, py, 32, 32, 'orb'))

        self.tracks = {key: track for key, track in self.tracks.items() if 0 <= now - track.seen <= .4}
        pairs = []
        for track in self.tracks.values():
            elapsed = now - track.seen
            for index, (x, y, _, _, kind) in enumerate(detections):
                if track.kind != kind:
                    continue
                distance = hypot(x - (track.x + track.vx * elapsed), y - (track.y + track.vy * elapsed))
                if distance <= 85 + 1200 * elapsed:
                    pairs.append((distance, track.id, index))
        matched_tracks, matched_detections = set(), set()
        for _, track_id, index in sorted(pairs):
            if track_id in matched_tracks or index in matched_detections:
                continue
            old = self.tracks[track_id]
            x, y, width, height, kind = detections[index]
            elapsed = now - old.seen
            vx = (old.vx + (x - old.x) / elapsed) / 2 if elapsed else old.vx
            vy = (old.vy + (y - old.y) / elapsed) / 2 if elapsed else old.vy
            self.tracks[track_id] = BladeTrack(track_id, x, y, width, height, vx, vy, now, old.hits + 1, kind)
            matched_tracks.add(track_id)
            matched_detections.add(index)
        for index, (x, y, width, height, kind) in enumerate(detections):
            if index not in matched_detections:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = BladeTrack(track_id, x, y, width, height, 0, 0, now, 1, kind)
                matched_tracks.add(track_id)
        return [self.tracks[track_id] for track_id in sorted(matched_tracks)]
