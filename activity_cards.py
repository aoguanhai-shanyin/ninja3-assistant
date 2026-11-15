"""Offline card-face recognition for intelligence simulation.

The caller supplies a complete, already located card rectangle in RGB order.
This module does not locate slots, read live attack values, or control a device.
Its thresholds have only been checked on resized/compressed catalog sources;
the placement screen's crop alignment and rendering still need live evidence.
"""
from functools import lru_cache
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from activity_policy import Card


ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
CATALOG_PATH = ROOT / 'assets/intelligence-cards.json'
FACE_SIZE = (150, 181)
# At catalog scale, attack/cost badges finish above this crop. The effect
# lettering starts below it. Names, attack values and prices are never scored.
ARTWORK = (8, 49, 149, 108)
MIN_SCORE = .93
MIN_MARGIN = .08


def load_catalog(path=CATALOG_PATH):
    """Load base values and visually transcribed effects as immutable Cards."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return tuple(Card(**{key: entry[key] for key in
                         ('id', 'attack', 'cost', 'element', 'profession', 'effect')})
                 for entry in data['cards'])


def _feature(rgb):
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or
            rgb.ndim != 3 or rgb.shape[2] != 3 or
            rgb.shape[0] < 58 or rgb.shape[1] < 48):
        return None
    ratio = rgb.shape[1] / rgb.shape[0]
    if abs(ratio / (FACE_SIZE[0] / FACE_SIZE[1]) - 1) > .06:
        return None
    face = cv2.resize(rgb, FACE_SIZE, interpolation=cv2.INTER_AREA)
    x1, y1, x2, y2 = ARTWORK
    art = cv2.resize(face[y1:y2, x1:x2], (72, 30), interpolation=cv2.INTER_AREA)
    values = art.astype(np.float32).ravel()
    values -= values.mean()
    norm = np.linalg.norm(values)
    if norm < 8 * np.sqrt(values.size):
        return None
    return values / norm


class CardRecognizer:
    def __init__(self, path=CATALOG_PATH):
        path = Path(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        self.cards = load_catalog(path)
        features = []
        for entry in data['cards']:
            # Catalog paths are relative to the project, not the working dir.
            template = path.parent.parent / entry['template']
            bgr = cv2.imread(str(template))
            if bgr is None:
                raise ValueError(f'Cannot read intelligence card template: {template}')
            feature = _feature(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if feature is None:
                raise ValueError(f'Invalid intelligence card artwork: {template}')
            features.append(feature)
        self.features = np.stack(features)

    def recognize_card(self, rgb_crop):
        """Return a catalog Card only when both score and separation pass."""
        feature = _feature(rgb_crop)
        if feature is None:
            return None
        scores = self.features @ feature
        order = np.argsort(scores)
        best = int(order[-1])
        if scores[best] < MIN_SCORE or scores[best] - scores[order[-2]] < MIN_MARGIN:
            return None
        return self.cards[best]


@lru_cache(maxsize=1)
def _recognizer():
    return CardRecognizer()


def recognize_card(rgb_crop) -> Card | None:
    return _recognizer().recognize_card(rgb_crop)
