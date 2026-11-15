import unittest
from pathlib import Path

import cv2

from battle import RedBladeTracker


VIDEO = Path(__file__).resolve().parents[2] / 'android-icon-clicker/build/evidence/battle-inspect/boss-fight.mp4'


@unittest.skipUnless(VIDEO.is_file(), 'local boss recording unavailable')
class RedBladeTrackerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = cv2.VideoCapture(str(VIDEO))
        if not cls.capture.isOpened():
            raise RuntimeError('could not open boss recording')

    @classmethod
    def tearDownClass(cls):
        cls.capture.release()

    def frame(self, index):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        self.assertTrue(ok, f'could not decode frame {index}')
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def test_finds_two_red_blades_without_detecting_background(self):
        tracker = RedBladeTracker()
        self.assertEqual(tracker.update(self.frame(0), 0), [])
        self.assertEqual(tracker.update(self.frame(270), 9), [])
        tracks = tracker.update(self.frame(360), 12)
        self.assertTrue(any(110 <= item.x <= 250 for item in tracks))
        self.assertTrue(any(380 <= item.x <= 570 for item in tracks))

    def test_tracks_an_actual_blade_moving_towards_the_player(self):
        tracker = RedBladeTracker()
        observed = []
        for index in range(330, 355, 3):
            tracks = tracker.update(self.frame(index), index / 30)
            observed.append(tracks)
        initial = min(observed[0], key=lambda item: abs(item.x - 1063))
        same_track = [item for item in observed[-1] if item.id == initial.id]
        self.assertEqual(len(same_track), 1)
        self.assertLess(same_track[0].x, initial.x - 300)
        self.assertLess(same_track[0].vx, -200)


if __name__ == '__main__':
    unittest.main()
