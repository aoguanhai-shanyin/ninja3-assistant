"""Same-source catalog replay, not unseen live-screen generalization."""
import json
import unittest

import cv2
import numpy as np

from activity_cards import CATALOG_PATH, CardRecognizer, load_catalog, recognize_card
from activity_policy import Card


class ActivityCardsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = CardRecognizer()
        cls.entries = json.loads(CATALOG_PATH.read_text(encoding='utf-8'))['cards']
        cls.images = [cv2.cvtColor(cv2.imread(str(CATALOG_PATH.parent.parent / c['template'])),
                                  cv2.COLOR_BGR2RGB) for c in cls.entries]

    def test_catalog_base_values_and_card_count_match_reviewed_library(self):
        cards = load_catalog()
        self.assertEqual(len(cards), 53)
        self.assertEqual(sum(entry['copies'] for entry in self.entries), 81)
        self.assertTrue(all(isinstance(card, Card) for card in cards))
        self.assertEqual(len({card.id for card in cards}), 53)
        self.assertEqual(cards[0], Card('c001', 1, 1, 'fire', 'character',
                                       '忍村5虎。上场2虎后自身攻击+2'))

    def test_all_53_source_cards_and_uniform_resize_replay(self):
        for entry, image in zip(self.entries, self.images):
            for scale in (.5, .75, 1, 1.4):
                with self.subTest(card=entry['id'], scale=scale):
                    resized = cv2.resize(image, None, fx=scale, fy=scale,
                                         interpolation=cv2.INTER_LINEAR)
                    self.assertEqual(self.recognizer.recognize_card(resized).id, entry['id'])

    def test_all_53_cards_with_light_jpeg_replay(self):
        for entry, image in zip(self.entries, self.images):
            with self.subTest(card=entry['id']):
                smaller = cv2.resize(image, None, fx=.75, fy=.75)
                ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(smaller, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, 82])
                self.assertTrue(ok)
                decoded = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                self.assertEqual(recognize_card(decoded).id, entry['id'])

    def test_dynamic_badges_and_effect_text_are_excluded(self):
        for entry, image in zip(self.entries, self.images):
            with self.subTest(card=entry['id']):
                changed = image.copy()
                changed[:49] = 0
                changed[108:] = 255
                self.assertEqual(self.recognizer.recognize_card(changed).id, entry['id'])

    def test_black_uniform_noise_and_invalid_rectangles_are_rejected(self):
        random = np.random.default_rng(73)
        for image in (None, np.zeros((181, 150, 3), np.uint8),
                      np.full((181, 150, 3), 120, np.uint8),
                      random.integers(0, 256, (181, 150, 3), dtype=np.uint8),
                      np.zeros((20, 15, 3), np.uint8), self.images[0][:, :75],
                      self.images[0][:, :, 0], self.images[0].astype(float)):
            self.assertIsNone(self.recognizer.recognize_card(image))

    def test_equal_blends_and_split_artwork_are_rejected(self):
        # Check every distinct pair; a nearest catalog face is not enough to act.
        for first in range(len(self.images)):
            for second in range(first):
                with self.subTest(first=first, second=second):
                    blend = cv2.addWeighted(self.images[first], .5, self.images[second], .5, 0)
                    self.assertIsNone(self.recognizer.recognize_card(blend))
                    split = self.images[first].copy()
                    split[:, 75:] = self.images[second][:, 75:]
                    self.assertIsNone(self.recognizer.recognize_card(split))


if __name__ == '__main__':
    unittest.main()
