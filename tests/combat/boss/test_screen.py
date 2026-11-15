from ninja3_assistant.paths import project_root
import unittest

import cv2

from ninja3_assistant.combat.boss.screen import BossScreenDetector


EVIDENCE = project_root().parent / 'android-icon-clicker/build/evidence/battle-inspect'


@unittest.skipUnless((EVIDENCE / 'boss-fight.mp4').is_file(), 'local boss recording unavailable')
class BossScreenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = BossScreenDetector()

    def frame(self, filename, index):
        capture = cv2.VideoCapture(str(EVIDENCE / filename))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
        finally:
            capture.release()
        self.assertTrue(ok)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def test_target_boss_on_top_of_rope(self):
        result = self.detector.detect(self.frame('boss-fight.mp4', 12 * 30))
        self.assertTrue(result.target_boss)
        self.assertEqual(result.side, 'above')
        self.assertAlmostEqual(result.rope_y, 483, delta=4)
        self.assertAlmostEqual(result.player_x, 311, delta=35)
        self.assertFalse(result.weapon_ready)

    def test_weapon_ready_after_cooldown(self):
        result = self.detector.detect(self.frame('boss-fight.mp4', 34 * 30))
        self.assertTrue(result.target_boss)
        self.assertTrue(result.weapon_ready)

    def test_under_rope_on_another_boss(self):
        if not (EVIDENCE / 'ninjutsu-full.mp4').is_file():
            self.skipTest('local full-energy recording unavailable')
        result = self.detector.detect(self.frame('ninjutsu-full.mp4', 5 * 30))
        self.assertFalse(result.target_boss)
        self.assertEqual(result.side, 'below')
        self.assertAlmostEqual(result.rope_y, 426, delta=4)

    def test_pause_and_countdown_never_trigger_skills(self):
        if not (EVIDENCE / 'ninjutsu-full.mp4').is_file():
            self.skipTest('local full-energy recording unavailable')
        result = self.detector.detect(self.frame('ninjutsu-full.mp4', 2 * 30))
        self.assertFalse(result.weapon_ready)
        self.assertEqual(result.side, 'unknown')

    def test_player_remains_located_after_live_rope_flip(self):
        if not (EVIDENCE / 'below-31.png').is_file():
            self.skipTest('local live rope-flip screenshot unavailable')
        image = cv2.imread(str(EVIDENCE / 'below-31.png'))
        result = self.detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.assertTrue(result.target_boss)
        self.assertEqual(result.side, 'below')
        self.assertAlmostEqual(result.player_x, 315, delta=30)

    def test_weapon_can_be_ready_after_live_rope_flip(self):
        if not (EVIDENCE / 'below-55.png').is_file():
            self.skipTest('local live weapon screenshot unavailable')
        image = cv2.imread(str(EVIDENCE / 'below-55.png'))
        result = self.detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.assertTrue(result.target_boss)
        self.assertEqual(result.side, 'below')
        self.assertIsNotNone(result.player_x)
        self.assertTrue(result.weapon_ready)


if __name__ == '__main__':
    unittest.main()
