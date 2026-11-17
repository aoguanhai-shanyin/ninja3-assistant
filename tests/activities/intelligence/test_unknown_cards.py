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

    def test_recorded_occluded_and_glowing_board_cards(self):
        root = project_root()/'data/runtime/intelligence-simulation/runs'
        cases = (
            ('20261117-113902-154100','0000-round4-board5.png','c005'),
            ('20261117-113902-154100','0001-round4-board7.png','c053'),
            ('20261117-115557-940100','0000-round2-board3.png','c014'),
            ('20261117-133710-399700','0000-round1-board0.png','c021'),
        )
        for run,name,expected in cases:
            with self.subTest(run=run,frame=name):
                image = cv2.imread(str(root/run/'unknown-cards'/name))
                frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                card = self.vision._card(frame,(0,0,frame.shape[1],frame.shape[0]))
                self.assertIsNotNone(card)
                self.assertEqual(card.id,expected)

    def test_foreign_name_and_portrait_do_not_form_a_match(self):
        # Exercise unrelated catalog cards, including the weaker-name glow path.
        raw = json.loads((project_root()/'assets/intelligence-cards.json').read_text(encoding='utf-8'))
        cards = [raw['cards'][i] for i in (0,4,13,20,30,39,45,52)]
        for portrait in cards:
            image = cv2.cvtColor(cv2.imread(str(project_root()/portrait['template'])),cv2.COLOR_BGR2RGB)
            for other in cards:
                if other['id'] == portrait['id']:
                    continue
                with self.subTest(portrait=portrait['id'],name=other['id']):
                    name = cv2.cvtColor(cv2.imread(str(project_root()/other['template'])),cv2.COLOR_BGR2RGB)
                    mixed = image.copy()
                    mixed[138:181] = name[138:181]
                    scores = np.zeros(len(self.vision.cards))
                    i = next(i for i,card in enumerate(self.vision.cards) if card.id == other['id'])
                    scores[i] = .85
                    self.assertIsNone(self.vision._named_card(mixed,scores,None))

    def test_current_back_variants_recover_reviewed_empty_slots_without_hiding_cards(self):
        cases = json.loads((project_root()/'data/evidence/2026-11-16/intelligence-recognition/'
                            'reviewed-empty-variants.json').read_text(encoding='utf-8'))
        recovered = 0
        for case in cases:
            with self.subTest(source=case['source']):
                image = cv2.imread(str(project_root()/case['source']))
                h,w = image.shape[:2]
                gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                region = (18,39,128,149) if case['area'] == 'board' else (25,40,w-20,h-40)
                empty = self.vision._empty_pattern(gray,case['area'],region)
                if empty:
                    self.assertTrue(case['expected_empty'])
                    recovered += 1
                elif case['expected_empty']:
                    frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    self.assertIsNone(self.vision._card(frame,(0,0,w,h)))
        self.assertEqual(len(cases),150)
        self.assertEqual(sum(case['expected_empty'] for case in cases),65)
        self.assertEqual(recovered,58)

    def test_partial_card_back_cannot_match_the_complete_positive_outline(self):
        source = project_root()/'data/runtime/intelligence-simulation/runs/20261116-182901-170000/'
        gray = cv2.cvtColor(cv2.imread(str(source/'unknown-cards/0006-round2-board3.png')),cv2.COLOR_BGR2GRAY)
        self.assertTrue(self.vision._empty_pattern(gray,'board',(18,39,128,149)))
        gray[:,75:] = 0
        self.assertFalse(self.vision._empty_pattern(gray,'board',(18,39,128,149)))
