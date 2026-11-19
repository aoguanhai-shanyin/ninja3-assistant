"""Live-sampled 1280x720 intelligence battle layout; no device access."""
from ninja3_assistant.paths import resource_root
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import json
import logging
import re

import cv2
import numpy as np

from ninja3_assistant.activities.intelligence.policy import Card
from ninja3_assistant.activities.intelligence.portrait_batch import portrait_scores, warm_portrait_session

ROOT = resource_root()
ASSETS = ROOT/'assets/intelligence-battle'
BOARD_CENTERS = tuple((x, y) for y in (232, 419, 606) for x in (140, 310, 480))
BOARD_BOXES = tuple((x-73, y-94, x+72, y+78) for x, y in BOARD_CENTERS)
SHOP_BOXES = ((645, 196, 812, 395), (836, 196, 1003, 395), (1027, 196, 1194, 395),
              (645, 403, 812, 602), (836, 403, 1003, 602))
SHOP_POINTS = tuple(((x1+x2)//2, (y1+y2)//2) for x1, y1, x2, y2 in SHOP_BOXES)
_LOG = logging.getLogger(__name__)


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
    card_candidates: dict = field(default_factory=dict)


class BattleVision:
    def __init__(self, *, use_winml=False):
        self._use_winml = use_winml
        self._portrait_ready = None
        self._portrait_session = None
        self.portrait_warmup_ms = None
        self.templates = {p.stem: cv2.imread(str(p), 0) for p in ASSETS.glob('*.png')}
        data = json.loads((ROOT/'assets/intelligence-cards.json').read_text(encoding='utf-8'))
        self.cards = tuple(Card(**{key: entry[key] for key in Card.__dataclass_fields__})
                           for entry in data['cards'])
        self.art = []
        self.names = []
        self.patches = []
        for entry in data['cards']:
            image = cv2.cvtColor(cv2.imread(str(ROOT/entry['template'])), cv2.COLOR_BGR2RGB)
            # Lock/withdraw icons in live cards obscure the catalog's right edge.
            self.art.append(image[40:105, 10:120])
            self.names.append(image[154:175, 30:135])
            gray = cv2.cvtColor(self.art[-1],cv2.COLOR_RGB2GRAY)
            self.patches.append(tuple((gray[a:b,c:d],c,a) for a,b,c,d in
                ((0,20,0,30),(0,20,80,110),(45,65,0,30),(45,65,80,110))))
        self.empty_patterns = {area:tuple((t,cv2.Canny(t,15,50)) for name,t in self.templates.items()
            if name.startswith('empty-'+area) and not name.endswith('-variant')) for area in ('board','shop')}
        self.empty_variants = {area:tuple((t,self._shape_edges(t)) for name,t in self.templates.items()
            if name.startswith('empty-'+area) and name.endswith('-variant')) for area in ('board','shop')}

    @property
    def backend(self):
        return 'GPU: Windows ML / TensorRT RTX' if self._portrait_session is not None else 'CPU: OpenCV'

    def warmup(self):
        if not self._use_winml or self._portrait_session is not None:
            return
        try:
            if self._portrait_ready is None:
                self._portrait_ready = warm_portrait_session(len(self.art), np.stack(self.art).tobytes())
            if self._portrait_ready.done():
                self._portrait_session, self.portrait_warmup_ms = self._portrait_ready.result()
                self._use_winml = self._portrait_session is not None
        except Exception as error:
            _LOG.warning('Windows ML portrait warmup unavailable: %s', error)
            self._use_winml = False

    @staticmethod
    def _shape_edges(gray):
        # Dark card backs otherwise lose their shuriken outline at fixed Canny thresholds.
        contrast = cv2.normalize(cv2.GaussianBlur(gray,(3,3),0),None,0,255,cv2.NORM_MINMAX)
        return cv2.Canny(contrast,25,75)

    def _empty_pattern(self, gray, area, box):
        """Positive card-back pattern, robust to glow; not 'unknown means empty'."""
        x1,y1,x2,y2 = box
        crop = gray[y1:y2,x1:x2]
        edges = cv2.Canny(crop,15,50)
        for template,pattern in self.empty_patterns[area]:
            if any(a < b for a,b in zip(crop.shape,template.shape)):
                continue
            score = cv2.minMaxLoc(cv2.matchTemplate(crop,template,cv2.TM_CCOEFF_NORMED))[1]
            if score >= .6 and cv2.minMaxLoc(cv2.matchTemplate(edges,pattern,cv2.TM_CCOEFF_NORMED))[1] >= .65:
                return True
        for template,pattern in self.empty_variants[area]:
            if any(a < b for a,b in zip(crop.shape,template.shape)):
                continue
            _,score,_,at = cv2.minMaxLoc(cv2.matchTemplate(crop,template,cv2.TM_CCOEFF_NORMED))
            if score < .8:
                continue
            patch = crop[at[1]:at[1]+template.shape[0],at[0]:at[0]+template.shape[1]]
            outline = self._shape_edges(patch)
            # Both directions must cover the same aligned outline, with one-pixel tolerance.
            expanded = cv2.dilate(pattern,np.ones((3,3),np.uint8))
            reverse = cv2.dilate(outline,np.ones((3,3),np.uint8))
            forward_score = np.count_nonzero(outline & expanded)/max(1,np.count_nonzero(outline))
            reverse_score = np.count_nonzero(pattern & reverse)/max(1,np.count_nonzero(pattern))
            if min(forward_score,reverse_score) >= .7:
                return True
        return False

    def _named_card(self, face, scores, candidates):
        """Use a distinctive name and independently aligned portrait patches."""
        names = np.array([cv2.minMaxLoc(cv2.matchTemplate(face[138:181,:],t,
                            cv2.TM_CCOEFF_NORMED))[1] for t in self.names])
        order = np.argsort(names)
        best = int(order[-1])
        margin = names[best]-names[order[-2]]
        named = names[best] >= .75 and margin >= (.06 if names[best] >= .84 else .12)
        ranked = np.argsort(scores)
        # Golden selection effects can tint the name. In that case also require
        # a distinctive whole portrait and three agreeing portrait patches.
        glowing = (names[best] >= .65 and margin >= .1 and ranked[-1] == best and
                   scores[best] >= .8 and scores[best]-scores[ranked[-2]] >= .2)
        if not named and not glowing:
            return None
        search = cv2.cvtColor(face[25:125,:135],cv2.COLOR_RGB2GRAY)
        matches = []
        for template,x,y in self.patches[best]:
            _,score,_,at = cv2.minMaxLoc(cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED))
            matches.append((score,at[0]-x,at[1]-y))
        def aligned(count, threshold):
            return any(a[0] >= threshold and
                       sum(score >= threshold and abs(x-a[1]) <= 4 and abs(y-a[2]) <= 4
                           for score,x,y in matches) >= count for a in matches)
        agreed = (named and aligned(2,.8)) or (glowing and aligned(3,.83))
        if candidates is not None:
            candidates[:] = ([{'card':self.cards[best].id,'score':round(float(scores[best]),4),
                'name_score':round(float(names[best]),4),
                'patch_scores':[round(float(m[0]),4) for m in matches]}] +
                [c for c in candidates if c['card'] != self.cards[best].id])[:3]
        return self.cards[best] if agreed else None

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

    @staticmethod
    def _card_face(frame, box):
        x1, y1, x2, y2 = box
        return cv2.resize(frame[y1:y2, x1:x2], (150, 181), interpolation=cv2.INTER_AREA)

    def _card(self, frame, box, selected=False, candidates=None):
        return self._classify_card(self._card_face(frame, box), selected, candidates)

    def _classify_card(self, face, selected, candidates, scores=None, upper=None):
        search = face[30:115, 0:130]
        if scores is not None:
            runner_up, best = np.sort(scores)[-2:]
            upper_runner_up, upper_best = np.sort(upper)[-2:]
            # Preserve OpenCV decisions near a confidence threshold or rank tie.
            tolerance = .001
            near_gate = (any(abs(best-cutoff) < tolerance for cutoff in (.8, .85, .87, .88)) or
                         any(abs(best-runner_up-cutoff) < tolerance for cutoff in (0, .15, .2)) or
                         abs(upper_best-.89) < tolerance or
                         any(abs(upper_best-upper_runner_up-cutoff) < tolerance for cutoff in (0, .25)))
            if near_gate:
                scores = upper = None
        if scores is None:
            scores = np.array([cv2.minMaxLoc(cv2.matchTemplate(search, art, cv2.TM_CCOEFF_NORMED))[1]
                               for art in self.art])
        order = np.argsort(scores)
        if candidates is not None:
            candidates.extend({'card':self.cards[int(i)].id,'score':round(float(scores[i]),4)}
                              for i in order[-3:][::-1])
        best, runner_up = order[-1], order[-2]
        # The confirmed green selection tint slightly changes the portrait's
        # color. Keep the candidate separation gate and only relax this state.
        if scores[best] < (.87 if selected else .88) or scores[best]-scores[runner_up] < .15:
            # On-board damage text covers the lower portrait. Require an
            # independent upper portrait and name match for the same candidate.
            if upper is None:
                upper = np.array([cv2.minMaxLoc(cv2.matchTemplate(search, art[:30], cv2.TM_CCOEFF_NORMED))[1]
                                  for art in self.art])
            ranked = np.argsort(upper)
            name = cv2.minMaxLoc(cv2.matchTemplate(face[144:181, 20:145], self.names[best],
                                                   cv2.TM_CCOEFF_NORMED))[1]
            selected_name = selected and scores[best] >= .85 and scores[best]-scores[runner_up] >= .15 and name >= .9
            if not selected_name and (ranked[-1] != best or upper[best] < .89 or
                    upper[best]-upper[ranked[-2]] < .25 or name < .86):
                return self._named_card(face,scores,candidates)
        return self.cards[best]

    def _batch_portraits(self, frame, jobs):
        if not self._use_winml or not jobs:
            return None
        self.warmup()
        if self._portrait_session is None:
            return None
        try:
            faces = [self._card_face(frame, box) for _, _, box, _ in jobs]
            scores, upper = portrait_scores(self._portrait_session, faces)
            if not np.isfinite(scores).all() or not np.isfinite(upper).all():
                raise RuntimeError('nonfinite portrait scores')
            return faces, scores, upper
        except Exception as error:
            _LOG.warning('Windows ML portrait matching failed: %s', error)
            self._portrait_session = None
            self._use_winml = False
            return None

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

    @classmethod
    def with_numbers(cls, observation, ocr):
        """Apply OCR from the same frame without repeating visual recognition."""
        if observation.kind == 'result':
            score, gain = cls.parse_result_numbers(ocr)
            return replace(observation, score=score, score_gain=gain)
        coins, refreshes = cls.parse_numbers(ocr)
        if refreshes is None:
            refreshes = observation.refreshes
        return replace(observation, coins=coins, refreshes=refreshes,
                       refresh_point=(1115, 535) if refreshes else None)

    def detect(self, frame, ocr=None):
        if not isinstance(frame, np.ndarray) or frame.shape != (720, 1280, 3):
            return BattleObservation()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if (self._match(gray, 'result-title', (480, 104, 760, 170)) and
                self._match(gray, 'result-confirm', (588, 512, 694, 570))):
            result = BattleObservation('result', result_point=(640, 541))
            return self.with_numbers(result, ocr) if ocr is not None else result
        if self._match(gray, 'matching', (595, 535, 690, 592)):
            return BattleObservation('matching')
        round_number = self._round(gray)
        if not self._match(gray, 'round-suffix', (610, 10, 760, 75)):
            return BattleObservation()
        if self._match(gray, 'combat-corner', (664, 104, 718, 164)):
            return BattleObservation('combat', round_number=round_number)
        if not self._match(gray, 'shop-title', (637, 148, 795, 195)):
            return BattleObservation()
        if not self._match(gray, 'ready', (1048, 627, 1197, 682)):
            return BattleObservation('prepared', round_number=round_number)
        board, unknown_board, place_points = [], [], []
        card_jobs = []
        for index, ((x, y), box) in enumerate(zip(BOARD_CENTERS, BOARD_BOXES)):
            region = (x-55, y-55, x+55, y+55)
            empty = any(self._match(gray, f'empty-board-{number}', region, .88, 32)
                        for number in range(9))
            if not empty:
                empty = self._empty_pattern(gray,'board',region)
            selected = any(self._match(gray, f'selected-board-{number}', region, .88, 32)
                           for number in range(9))
            board.append(None)
            if empty or selected:
                plus = any(self._match(gray,name,(x-23,y-24,x+24,y+25),.92,30)
                           for name in ('plus','plus-variant'))
                place_points.append((x, y) if plus else None)
            else:
                place_points.append(None)
                card_jobs.append(('board', index, box, False))
        shop, unknown_shop, green = [], [], []
        has_visible_plus = any(place_points)
        for index, box in enumerate(SHOP_BOXES):
            x1, y1, x2, y2 = box
            strip = frame[y1+8:y2, x2-6:x2+3].astype(np.int16)
            green_score = np.mean((strip[:, :, 1] > 150) &
                                  (strip[:, :, 1] > strip[:, :, 0]+35) &
                                  (strip[:, :, 1] > strip[:, :, 2]+15))
            green.append(green_score)
            empty = self._match(gray, 'empty-shop', (x1+25, y1+40, x2-20, y2-40), .92, 27)
            if not empty:
                empty = self._empty_pattern(gray,'shop',(x1+25,y1+40,x2-20,y2-40))
            shop.append(None)
            if not empty:
                card_jobs.append(('shop', index, box, green_score > .18 and has_visible_plus))

        batch = self._batch_portraits(frame, card_jobs)

        def read_card(index):
            _, _, box, selected = card_jobs[index]
            candidates = []
            if batch is None:
                card = self._card(frame, box, selected=selected, candidates=candidates)
            else:
                faces, scores, upper = batch
                card = self._classify_card(faces[index], selected, candidates, scores[index], upper[index])
            return card, candidates

        indices = range(len(card_jobs))
        if len(card_jobs) > 1:
            # Match independent slots from this frame in parallel, then preserve slot order.
            with ThreadPoolExecutor(max_workers=6) as pool:
                card_results = list(pool.map(read_card, indices))
        else:
            card_results = list(map(read_card, indices))
        card_candidates = {}
        for (area, index, _, _), (card, candidates) in zip(card_jobs, card_results):
            slots, unknown = (board, unknown_board) if area == 'board' else (shop, unknown_shop)
            slots[index] = card
            if card is None:
                unknown.append(index)
                card_candidates[f'{area}:{index}'] = candidates
        selected = [index for index, score in enumerate(green) if score > .18]
        selected_shop = selected[0] if len(selected) == 1 and has_visible_plus else None
        refreshes = 1 if self._match(gray, 'refresh-one', (1168, 560, 1196, 591), .96, 15) else None
        placement = BattleObservation('placement', tuple(board), tuple(shop),
            tuple(unknown_board), tuple(unknown_shop), None, round_number, refreshes,
            selected_shop, tuple(place_points), (1118, 654), (1115, 535) if refreshes else None,
            card_candidates=card_candidates)
        return self.with_numbers(placement, ocr) if ocr is not None else placement
