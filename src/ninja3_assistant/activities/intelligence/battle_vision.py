"""Live-sampled 1280x720 intelligence battle layout; no device access."""
from ninja3_assistant.paths import resource_root
from dataclasses import dataclass
import json
import re

import cv2
import numpy as np

from ninja3_assistant.activities.intelligence.policy import Card

ROOT = resource_root()
ASSETS = ROOT/'assets/intelligence-battle'
BOARD_CENTERS = tuple((x, y) for y in (232, 419, 606) for x in (140, 310, 480))
BOARD_BOXES = tuple((x-73, y-94, x+72, y+78) for x, y in BOARD_CENTERS)
SHOP_BOXES = ((645, 196, 812, 395), (836, 196, 1003, 395), (1027, 196, 1194, 395),
              (645, 403, 812, 602), (836, 403, 1003, 602))
SHOP_POINTS = tuple(((x1+x2)//2, (y1+y2)//2) for x1, y1, x2, y2 in SHOP_BOXES)


@dataclass(frozen=True)
class BattleObservation:
    kind: str = 'unknown'
    board: tuple = ()
    shop: tuple = ()
    unknown_board: tuple = ()
    unknown_shop: tuple = ()
    coins: int | None = None
    round_number: int | None = None
    refreshes: int | None = None
    selected_shop: int | None = None
    place_points: tuple = ()
    ready_point: tuple | None = None
    refresh_point: tuple | None = None
    result_point: tuple | None = None
    outcome: str | None = None
    score: int | None = None
    score_gain: int | None = None


class BattleVision:
    def __init__(self):
        self.templates = {p.stem: cv2.imread(str(p), 0) for p in ASSETS.glob('*.png')}
        data = json.loads((ROOT/'assets/intelligence-cards.json').read_text(encoding='utf-8'))
        self.cards = tuple(Card(**{key: entry[key] for key in Card.__dataclass_fields__})
                           for entry in data['cards'])
        self.art = []
        self.names = []
        for entry in data['cards']:
            image = cv2.cvtColor(cv2.imread(str(ROOT/entry['template'])), cv2.COLOR_BGR2RGB)
            # Lock/withdraw icons in live cards obscure the catalog's right edge.
            self.art.append(image[40:105, 10:120])
            self.names.append(image[154:175, 30:135])

    def _match(self, gray, name, box, threshold=.94, error=25):
        if name not in self.templates:
            return False
        x1, y1, x2, y2 = box
        area, template = gray[y1:y2, x1:x2], self.templates[name]
        if any(a < b for a, b in zip(area.shape, template.shape)):
            return False
        _, score, _, at = cv2.minMaxLoc(cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED))
        if score < threshold and (name.startswith('empty-board-') or name == 'empty-shop'):
            # White sparkles animate over the dark empty-card pattern. Preserve
            # that pattern while limiting the sparkle's influence on matching.
            cap = 100 if name == 'empty-shop' else 70
            _, score, _, at = cv2.minMaxLoc(cv2.matchTemplate(
                np.minimum(area, cap), np.minimum(template, cap), cv2.TM_CCOEFF_NORMED))
            threshold = max(threshold, .95 if name == 'empty-shop' else .94)
        patch = area[at[1]:at[1]+template.shape[0], at[0]:at[0]+template.shape[1]]
        return score >= threshold and np.abs(patch.astype(np.int16)-template).mean() <= error

    def _card(self, frame, box, selected=False):
        x1, y1, x2, y2 = box
        face = cv2.resize(frame[y1:y2, x1:x2], (150, 181), interpolation=cv2.INTER_AREA)
        search = face[30:115, 0:130]
        scores = np.array([cv2.minMaxLoc(cv2.matchTemplate(search, art, cv2.TM_CCOEFF_NORMED))[1]
                           for art in self.art])
        order = np.argsort(scores)
        best, runner_up = order[-1], order[-2]
        # The confirmed green selection tint slightly changes the portrait's
        # color. Keep the candidate separation gate and only relax this state.
        if scores[best] < (.87 if selected else .88) or scores[best]-scores[runner_up] < .15:
            # On-board damage text covers the lower portrait. Require an
            # independent upper portrait and name match for the same candidate.
            upper = np.array([cv2.minMaxLoc(cv2.matchTemplate(search, art[:30], cv2.TM_CCOEFF_NORMED))[1]
                              for art in self.art])
            ranked = np.argsort(upper)
            name = cv2.minMaxLoc(cv2.matchTemplate(face[144:181, 20:145], self.names[best],
                                                   cv2.TM_CCOEFF_NORMED))[1]
            selected_name = selected and scores[best] >= .85 and scores[best]-scores[runner_up] >= .15 and name >= .9
            if not selected_name and (ranked[-1] != best or upper[best] < .89 or
                    upper[best]-upper[ranked[-2]] < .25 or name < .86):
                return None
        return self.cards[best]

    def _round(self, gray):
        values = []
        for number in range(1, 6):
            name = f'round-{number}'
            if name not in self.templates:
                continue
            area, template = gray[15:69, 581:615], self.templates[name]
            score = cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED)[0, 0]
            error = np.abs(area.astype(np.int16)-template).mean()
            values.append((float(score), float(error), number))
        values.sort(reverse=True)
        if not values or values[0][0] < .92 or values[0][1] > 22:
            return None
        if len(values) > 1 and values[0][0]-values[1][0] < .05:
            return None
        return values[0][2]

    @staticmethod
    def numbers_image(frame):
        """One small OCR pass for coins/refresh count; decorative rounds use templates."""
        image = np.full((230, 540, 3), 255, np.uint8)
        coins = cv2.resize(frame[153:191, 1095:1220], (375, 114), interpolation=cv2.INTER_CUBIC)
        refresh = cv2.resize(frame[562:591, 1038:1206], (504, 87), interpolation=cv2.INTER_CUBIC)
        image[:114, :375] = coins
        image[135:222, :504] = refresh
        return image

    @staticmethod
    def parse_numbers(ocr):
        if ocr is None:
            return None, None
        texts = [''.join(word['text'] for word in ocr.get('words', [])
                         if top <= word['bbox'][1] < bottom).replace(' ', '')
                 for top, bottom in ((0, 120), (130, 230))]
        coin = re.search(r'金币[:：]?(\d{1,3})', texts[0])
        refresh = re.search(r'刷新次数[:：]?(\d{1,2})', texts[1])
        return (int(coin[1]) if coin else None, int(refresh[1]) if refresh else None)

    @staticmethod
    def result_numbers_image(frame):
        return cv2.resize(frame[105:202, 520:790], (810, 291), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def parse_result_numbers(ocr):
        text = ''.join(word['text'] for word in (ocr or {}).get('words', [])).replace(' ', '')
        score = re.search(r'当前积分[:：]?(\d+)', text)
        gain = re.search(r'本局[+＋]?(\d+)', text)
        return (int(score[1]) if score else None, int(gain[1]) if gain else None)

    def detect(self, frame, ocr=None):
        if not isinstance(frame, np.ndarray) or frame.shape != (720, 1280, 3):
            return BattleObservation()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if (self._match(gray, 'result-title', (480, 104, 760, 170)) and
                self._match(gray, 'result-confirm', (588, 512, 694, 570))):
            score, gain = self.parse_result_numbers(ocr)
            return BattleObservation('result', result_point=(640, 541), score=score, score_gain=gain)
        if self._match(gray, 'matching', (595, 535, 690, 592)):
            return BattleObservation('matching')
        round_number = self._round(gray)
        if not self._match(gray, 'round-suffix', (610, 10, 760, 75)):
            return BattleObservation()
        if self._match(gray, 'combat-corner', (664, 104, 718, 164)):
            return BattleObservation('combat', round_number=round_number)
        if not self._match(gray, 'shop-title', (637, 148, 795, 195)):
            return BattleObservation()
        board, unknown_board, place_points = [], [], []
        for index, ((x, y), box) in enumerate(zip(BOARD_CENTERS, BOARD_BOXES)):
            region = (x-55, y-55, x+55, y+55)
            empty = any(self._match(gray, f'empty-board-{number}', region, .88, 32)
                        for number in range(9))
            selected = any(self._match(gray, f'selected-board-{number}', region, .88, 32)
                           for number in range(9))
            if empty or selected:
                board.append(None)
                plus = self._match(gray, 'plus', (x-23, y-24, x+24, y+25), .92, 30)
                place_points.append((x, y) if selected and plus else None)
            else:
                card = self._card(frame, box)
                board.append(card)
                place_points.append(None)
                if card is None:
                    unknown_board.append(index)
        shop, unknown_shop, green = [], [], []
        for index, box in enumerate(SHOP_BOXES):
            x1, y1, x2, y2 = box
            strip = frame[y1+8:y2, x2-6:x2+3].astype(np.int16)
            green_score = np.mean((strip[:, :, 1] > 150) &
                                  (strip[:, :, 1] > strip[:, :, 0]+35) &
                                  (strip[:, :, 1] > strip[:, :, 2]+15))
            green.append(green_score)
            empty = self._match(gray, 'empty-shop', (x1+25, y1+40, x2-20, y2-40), .92, 27)
            if empty:
                shop.append(None)
            else:
                card = self._card(frame, box, selected=green_score > .18 and any(place_points))
                shop.append(card)
                if card is None:
                    unknown_shop.append(index)
        selected = [index for index, score in enumerate(green) if score > .18]
        selected_shop = selected[0] if len(selected) == 1 and any(place_points) else None
        ready = self._match(gray, 'ready', (1048, 627, 1197, 682))
        coins, refreshes = self.parse_numbers(ocr)
        if refreshes is None and self._match(gray, 'refresh-one', (1168, 560, 1196, 591), .96, 15):
            refreshes = 1
        return BattleObservation('placement' if ready else 'prepared', tuple(board), tuple(shop),
            tuple(unknown_board), tuple(unknown_shop), coins, round_number, refreshes,
            selected_shop, tuple(place_points), (1118, 654) if ready else None,
            (1115, 535) if refreshes is not None and refreshes > 0 and ready else None)
