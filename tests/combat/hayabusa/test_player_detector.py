from ninja3_assistant.paths import project_root
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ninja3_assistant.combat.hayabusa.player_detector import HayabusaPlayerDetector, PlayerBodyLocator, PlayerDetection
from ninja3_assistant.combat.hayabusa.vision import HayabusaVision


class Session:
    def __init__(self, output):
        self.output = output

    def get_inputs(self):
        return [SimpleNamespace(name='images', shape=[1, 3, 640, 640], type='tensor(float)')]

    def run(self, _, feeds):
        self.tensor = feeds['images']
        return [self.output]


class SequenceDetector:
    backend = 'test player'

    def __init__(self, frames):
        self.frames = iter(frames)
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        return [PlayerDetection(tuple(box), .9) for box in next(self.frames)]


class HayabusaPlayerDetectorTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.model = Path(self.folder.name)/'player.onnx'
        self.spec = {'classes': ['player'], 'crop': [0, 70, 1280, 555],
                     'imgsz': 640, 'conf': .25, 'iou': .5}
        self.write_spec()

    def write_spec(self):
        self.model.with_suffix('.json').write_text(json.dumps(self.spec), encoding='utf-8')

    def detector(self, boxes):
        output = np.zeros((1, 5, len(boxes)), np.float32)
        # Native coordinates map through the fixed crop's ratio=.5 and top=199.
        for index, (x1, y1, x2, y2, score) in enumerate(boxes):
            output[0, :, index] = ((x1+x2)/4, (y1+y2-140)/4+199,
                                   (x2-x1)/2, (y2-y1)/2, score)
        return HayabusaPlayerDetector(self.model, session=Session(output))

    def test_player_crop_rgb_letterbox_and_native_box_roundtrip(self):
        detector = self.detector([(340, 110, 440, 220, .9)])
        frame = np.full((720, 1280, 3), (255, 128, 0), np.uint8)
        frame[:70] = (0, 0, 255)
        boxes = detector.predict(frame)
        np.testing.assert_allclose(boxes[0].xyxy, (340, 110, 440, 220))
        tensor = detector.session.tensor
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_allclose(tensor[0, :, 199, 0], (1, 128/255, 0))
        np.testing.assert_allclose(tensor[0, :, 198, 0], (114/255,)*3)

    def test_nms_removes_duplicate_body_but_keeps_separate_conflicting_candidate(self):
        detector = self.detector([(340, 110, 440, 220, .9), (342, 112, 442, 222, .8),
                                  (1000, 300, 1100, 410, .7), (700, 300, 750, 350, .1),
                                  (700, 300, 750, 350, float('nan'))])
        boxes = detector.predict(np.zeros((720, 1280, 3), np.uint8))
        self.assertEqual(len(boxes), 2)
        self.assertAlmostEqual(boxes[0].confidence, .9)
        self.assertAlmostEqual(boxes[1].confidence, .7)

    def test_invalid_metadata_and_outputs_fail_without_silent_template_fallback(self):
        for key, value in (('classes', ['boss']), ('crop', [0, 90, 1280, 500]),
                           ('imgsz', 320), ('conf', 0), ('iou', 1)):
            with self.subTest(key=key):
                original = self.spec[key]
                self.spec[key] = value
                self.write_spec()
                with self.assertRaises(ValueError):
                    HayabusaPlayerDetector(self.model, session=Session(None))
                self.spec[key] = original
        self.write_spec()
        detector = self.detector([])
        detector.session.output = np.zeros((1, 9, 8400), np.float32)
        with self.assertRaisesRegex(ValueError, r'\[1, 5, N\]'):
            detector.predict(np.zeros((720, 1280, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, '1280'):
            detector.predict(np.zeros((360, 640, 3), np.uint8))

    def test_present_model_with_bad_hash_is_rejected_before_runtime_load(self):
        self.model.write_bytes(b'broken model')
        self.spec['onnx_sha256'] = 'incorrect'
        self.write_spec()
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            HayabusaPlayerDetector(self.model, use_gpu=False)

    def test_session_factory_preserves_gpu_requirement_and_explicit_cpu(self):
        self.model.write_bytes(b'test player')
        self.spec['onnx_sha256'] = hashlib.sha256(b'test player').hexdigest()
        self.write_spec()
        for use_gpu, backend in ((True, 'Windows ML'), (False, 'CPU')):
            with self.subTest(use_gpu=use_gpu), patch('ninja3_assistant.vision.winml.model_session',
                    return_value=(Session(None), backend)) as create:
                detector = HayabusaPlayerDetector(self.model, use_gpu=use_gpu)
                self.assertTrue(detector.backend.endswith(backend))
                self.assertEqual(create.call_args.args, (self.model,))
                self.assertEqual(create.call_args.kwargs['use_gpu'], use_gpu)
                self.assertEqual(create.call_args.kwargs['require_gpu'], use_gpu)
                self.assertEqual(create.call_args.kwargs['options'].intra_op_num_threads, 1)


class PlayerBodyLocatorTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((720, 1280, 3), np.uint8)
        self.body = (340, 350, 430, 440)
        self.boss = (1020, 330, 1120, 440)

    def test_cold_multiple_candidates_and_missed_frame_are_unknown(self):
        detector = SequenceDetector([[self.body, self.boss], [self.body], [], [self.body]])
        locator = PlayerBodyLocator(detector)
        self.assertIsNone(locator.locate(self.frame, 1.))
        self.assertEqual(locator.locate(self.frame, 1.05), self.body)
        self.assertIsNone(locator.locate(self.frame, 1.10))
        self.assertEqual(locator.locate(self.frame, 1.15), self.body)
        self.assertEqual(detector.calls, 4)

    def test_continuity_selects_player_and_rejects_sudden_boss_handoff(self):
        airborne = (342, 230, 432, 330)
        detector = SequenceDetector([[self.body], [airborne, self.boss], [self.boss]])
        locator = PlayerBodyLocator(detector)
        self.assertEqual(locator.locate(self.frame, 1.), self.body)
        self.assertEqual(locator.locate(self.frame, 1.1), airborne)
        self.assertIsNone(locator.locate(self.frame, 1.15))

    def test_no_unbounded_identity_or_partial_offscreen_location(self):
        cropped = (340, 70, 430, 130)
        detector = SequenceDetector([[self.body], [self.boss, self.body], [cropped]])
        locator = PlayerBodyLocator(detector)
        self.assertEqual(locator.locate(self.frame, 1.), self.body)
        self.assertIsNone(locator.locate(self.frame, 2.))
        self.assertIsNone(locator.locate(self.frame, 2.1))

    def test_model_path_does_not_use_head_templates_to_fill_missing_detection(self):
        detector = SequenceDetector([[(340, 350, 430, 440)], []])
        vision = HayabusaVision(use_gpu=False, recognize_boards=False, detector=False,
                                player_detector=detector)
        image = cv2.imread(str(project_root()/'tests/fixtures/hayabusa/live-shadow-battle-072.jpg'))
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        first, _ = vision.detect(frame, 1.)
        self.assertTrue(first.active)
        self.assertEqual(first.player_source, 'body-detector')
        self.assertEqual(first.player_box, (340, 350, 430, 440))
        self.assertEqual((first.player_x, first.player_y), (385, 395))
        self.assertIsNone(first.player_head_x)
        missing, _ = vision.detect(frame, 1.05)
        self.assertTrue(missing.active)
        self.assertIsNone(missing.player_x)
        self.assertEqual(missing.player_source, 'unknown')

    def test_invalid_hud_clears_body_identity_before_next_acquisition(self):
        detector = SequenceDetector([[self.body], [self.boss, self.body]])
        vision = HayabusaVision(use_gpu=False, recognize_boards=False, detector=False,
                                player_detector=detector)
        image = cv2.imread(str(project_root()/'tests/fixtures/hayabusa/live-shadow-battle-072.jpg'))
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.assertIsNotNone(vision.detect(frame, 1.)[0].player_x)
        paused, _ = vision.detect((frame*.3).astype(np.uint8), 1.05)
        self.assertFalse(paused.active)
        self.assertIsNone(vision.player_locator.previous)
        self.assertIsNone(vision.detect(frame, 1.1)[0].player_x)

    def test_body_model_does_not_require_old_character_template_assets(self):
        read = cv2.imread

        def read_ui_only(path, *args):
            if 'player-head' in path or 'hayabusa-face-' in path:
                raise AssertionError('body model must not depend on old character appearance')
            return read(path, *args)

        with patch('ninja3_assistant.combat.hayabusa.vision.cv2.imread', side_effect=read_ui_only):
            vision = HayabusaVision(use_gpu=False, recognize_boards=False, detector=False,
                                    player_detector=SequenceDetector([]))
        self.assertEqual(vision.live_heads, [])
        self.assertEqual(vision.rotated_faces, [])


if __name__ == '__main__':
    unittest.main()
