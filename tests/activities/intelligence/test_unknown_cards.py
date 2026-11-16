"""Replay of manually reviewed private crops, not unseen-scene accuracy."""
import json
import unittest

import cv2
import numpy as np

from ninja3_assistant.paths import project_root
from ninja3_assistant.activities.intelligence.battle_vision import BattleVision


DATA = project_root()/'data/evidence/2026-11-15/intelligence-economy'


class UnknownCardReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = BattleVision()
        cls.cases = json.loads((DATA/'reviewed-cards.json').read_text(encoding='utf-8'))

    def test_reviewed_cards_recover_without_assigning_identities_to_empty_backs(self):
        self.assertEqual(len(self.cases),48)
        recovered = empty = 0
        for case in self.cases:
            with self.subTest(crop=case['crop']):
                frame = cv2.cvtColor(cv2.imread(str(DATA/case['crop'])),cv2.COLOR_BGR2RGB)
                h,w = frame.shape[:2]
                gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                region = (18,39,128,149) if case['area'] == 'board' else (25,40,w-20,h-40)
                is_empty = self.vision._empty_pattern(gray,case['area'],region)
                card = None if is_empty else self.vision._card(frame,(0,0,w,h))
                self.assertEqual(card.id if card else None,case['expected'])
                if case['expected']:
                    self.assertFalse(is_empty)
                    recovered += 1
                else:
                    empty += is_empty
        self.assertEqual(recovered,31)
        self.assertEqual(empty,16)  # One heavily animated back remains unknown.

    def test_real_cards_blank_and_noise_cannot_match_the_positive_empty_pattern(self):
        random = np.random.default_rng(482)
        raw = json.loads((project_root()/'assets/intelligence-cards.json').read_text(encoding='utf-8'))
        images = [cv2.imread(str(project_root()/c['template'])) for c in raw['cards']]
        images += [np.zeros((181,150,3),np.uint8),np.full((181,150,3),255,np.uint8),
                   random.integers(0,256,(181,150,3),dtype=np.uint8)]
        for image in images:
            for area,size,region in (('shop',(167,199),(25,40,147,159)),
                                     ('board',(145,172),(18,39,128,149))):
                gray = cv2.cvtColor(cv2.resize(image,size),cv2.COLOR_BGR2GRAY)
                self.assertFalse(self.vision._empty_pattern(gray,area,region))

    def test_name_only_or_portrait_only_is_not_enough_for_the_fallback(self):
        # The mask portrait is hidden by combat text; it needs the new fallback.
        frame = cv2.cvtColor(cv2.imread(str(DATA/self.cases[28]['crop'])),cv2.COLOR_BGR2RGB)
        face = cv2.resize(frame,(150,181),interpolation=cv2.INTER_AREA)
        scores = np.zeros(len(self.vision.cards))
        self.assertEqual(self.vision._named_card(face,scores,None).id,'c037')
        no_name = face.copy(); no_name[138:] = 0
        self.assertIsNone(self.vision._named_card(no_name,scores,None))
        no_portrait = face.copy(); no_portrait[25:125] = 0
        self.assertIsNone(self.vision._named_card(no_portrait,scores,None))
        different_name = face.copy()
        different_name[144:181] = 0
        different_name[154:175,30:135] = self.vision.names[40]
        self.assertIsNone(self.vision._named_card(different_name,scores,None))
