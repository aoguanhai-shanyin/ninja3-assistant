from ninja3_assistant.paths import project_root
import bisect
import json
import unittest

import cv2
import numpy as np
import onnxruntime as ort

from ninja3_assistant.combat.battle import RedBladeTracker
from ninja3_assistant.vision.gpu_vision import GpuAttackSegmenter


VIDEO = project_root().parent / 'android-icon-clicker/build/evidence/battle-inspect/boss-fight.mp4'


@unittest.skipUnless(VIDEO.is_file(), 'local boss recording unavailable')
@unittest.skipUnless('DmlExecutionProvider' in ort.get_available_providers(), 'DirectML unavailable')
class GpuBattleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = cv2.VideoCapture(str(VIDEO))
        cls.segmenter = GpuAttackSegmenter()

    @classmethod
    def tearDownClass(cls):
        cls.capture.release()

    def frame(self, index):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        self.assertTrue(ok)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def test_gpu_masks_match_recorded_attack_colors(self):
        self.assertEqual(self.segmenter.session.get_providers()[0], 'DmlExecutionProvider')
        frame = self.frame(360)
        field = cv2.resize(frame[110:710, 100:1280], (590, 300))
        strong, weak = self.segmenter.masks(field)
        hsv = cv2.cvtColor(field, cv2.COLOR_RGB2HSV)
        for actual, saturation, value in ((strong, 150, 130), (weak, 60, 150)):
            expected = cv2.bitwise_or(cv2.inRange(hsv, (0, saturation, value), (10, 255, 255)),
                                     cv2.inRange(hsv, (172, saturation, value), (179, 255, 255))) > 0
            actual = actual > 0
            iou = np.logical_and(expected, actual).sum() / np.logical_or(expected, actual).sum()
            self.assertGreater(iou, .94)

    def test_finds_thin_and_spinning_blades(self):
        tracker = RedBladeTracker(segmenter=self.segmenter, include_thin=True)
        thin = tracker.update(self.frame(270), 9)
        self.assertTrue(any(item.kind == 'thin' and 590 <= item.x <= 660 for item in thin))
        self.assertTrue(any(item.kind == 'thin' and 1130 <= item.x <= 1190 for item in thin))
        spin = tracker.update(self.frame(360), 12)
        self.assertTrue(any(item.kind == 'spin' and 110 <= item.x <= 250 for item in spin))
        self.assertTrue(any(item.kind == 'spin' and 380 <= item.x <= 570 for item in spin))

    def test_tracks_gpu_blade_through_recorded_frames(self):
        tracker = RedBladeTracker(segmenter=self.segmenter, include_thin=True)
        observations = []
        for index in range(330, 355, 3):
            observations.append(tracker.update(self.frame(index), index / 30))
        initial = min((item for item in observations[0] if item.kind == 'spin'),
                      key=lambda item: abs(item.x - 1063))
        same = [item for item in observations[-1] if item.id == initial.id]
        self.assertEqual(len(same), 1)
        self.assertLess(same[0].x, initial.x - 300)
        self.assertLess(same[0].vx, -200)

    def test_detects_red_orbs_in_live_boss_phase(self):
        image = VIDEO.parent / 'orb-83.png'
        if not image.is_file():
            self.skipTest('local orb screenshot unavailable')
        frame = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        tracker = RedBladeTracker(segmenter=self.segmenter, include_thin=True, include_orbs=True)
        tracks = tracker.update(frame, 1)
        self.assertTrue(any(item.kind == 'orb' and abs(item.x - 700) < 20 and abs(item.y - 330) < 20
                            for item in tracks))
        self.assertTrue(any(item.kind == 'orb' and abs(item.x - 700) < 20 and abs(item.y - 414) < 20
                            for item in tracks))

    def test_orb_recognizer_does_not_classify_background_as_orbs(self):
        tracker = RedBladeTracker(segmenter=self.segmenter, include_thin=True, include_orbs=True)
        self.assertFalse(any(item.kind == 'orb' for item in tracker.update(self.frame(270), 9)))

    def test_tracks_orb_barrage_from_live_recording(self):
        base = VIDEO.parent / 'gpu-boss-trial-20261109-151501'
        if not base.with_suffix('.mp4').is_file() or not base.with_suffix('.json').is_file():
            self.skipTest('local orb barrage recording unavailable')
        metadata = json.loads(base.with_suffix('.json').read_text(encoding='utf-8'))
        times = metadata['frame_times']
        start, end = bisect.bisect_left(times, 80), bisect.bisect_left(times, 82)
        capture = cv2.VideoCapture(str(base.with_suffix('.mp4')))
        tracker = RedBladeTracker(segmenter=self.segmenter, include_thin=True, include_orbs=True)
        trajectories = {}
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            for index in range(start, end):
                ok, image = capture.read()
                self.assertTrue(ok)
                tracks = tracker.update(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), times[index])
                for item in tracks:
                    if item.kind == 'orb':
                        trajectories.setdefault(item.id, []).append(item)
        finally:
            capture.release()
        self.assertTrue(any(len(items) >= 4 and items[-1].x < items[0].x - 150 and items[-1].vx < -300
                            for items in trajectories.values()))


if __name__ == '__main__':
    unittest.main()
