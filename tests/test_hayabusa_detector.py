import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hayabusa_detector import ATTACK_CLASSES, AttackDetection, HayabusaDetector
from hayabusa_vision import HayabusaVision


class Session:
    def __init__(self, output):
        self.output = output
        self.tensor = None

    def get_inputs(self):
        return [SimpleNamespace(name='images', shape=[1, 3, 640, 640], type='tensor(float)')]

    def run(self, outputs, feeds):
        self.tensor = feeds['images']
        return [self.output]


class HayabusaDetectorTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = Path(self.directory.name)/'detector.onnx'
        self.metadata = self.model.with_suffix('.json')
        self.spec = {'classes': list(ATTACK_CLASSES), 'crop': [0, 90, 1280, 500],
                     'imgsz': 640, 'conf': .25, 'iou': .45}
        self.metadata.write_text(json.dumps(self.spec), encoding='utf-8')

    def detector(self, rows):
        # Input rows are native 1280x720 xyxy boxes with kind and score.
        output = np.zeros((1, 9, len(rows)), np.float32)
        for index, (x1, y1, x2, y2, kind, score) in enumerate(rows):
            output[0, :4, index] = ((x1+x2)/4, (y1+y2-180)/4+217,
                                    (x2-x1)/2, (y2-y1)/2)
            output[0, 4+ATTACK_CLASSES.index(kind), index] = score
        return HayabusaDetector(self.model, session=Session(output))

    def test_rgb_crop_letterbox_and_coordinate_roundtrip(self):
        detector = self.detector([(400, 190, 520, 230, 'fire', .9)])
        frame = np.full((720, 1280, 3), (255, 128, 0), np.uint8)
        frame[:90] = (0, 0, 255)
        frame[500:] = (0, 255, 0)
        detections = detector.predict(frame)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].kind, 'fire')
        np.testing.assert_allclose(detections[0].xyxy, (400, 190, 520, 230))
        tensor = detector.session.tensor
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(tensor[0, :, 217, 0], (1, 128/255, 0))
        np.testing.assert_allclose(tensor[0, :, 216, 0], (114/255,)*3)
        np.testing.assert_allclose(tensor[0, :, 422, 0], (114/255,)*3)

    def test_nms_is_per_class_and_keeps_only_confident_finite_boxes(self):
        detector = self.detector([(400, 190, 520, 230, 'fire', .9),
                                  (402, 192, 522, 232, 'fire', .8),
                                  (402, 192, 522, 232, 'blackfire', .7),
                                  (700, 300, 720, 330, 'shuriken', .1),
                                  (800, 300, 820, 330, 'slash', float('nan'))])
        detections = detector.predict(np.zeros((720, 1280, 3), np.uint8))
        self.assertEqual([item.kind for item in detections], ['fire', 'blackfire'])
        self.assertAlmostEqual(detections[0].confidence, .9)

    def test_padding_and_inverted_boxes_do_not_become_attacks(self):
        detector = self.detector([(300, -50, 350, -20, 'fire', .9),
                                  (300, 520, 350, 560, 'blackfire', .9),
                                  (500, 250, 450, 300, 'slash', .9)])
        self.assertEqual(detector.predict(np.zeros((720, 1280, 3), np.uint8)), [])

    def test_non_yolo11_output_and_wrong_frame_fail_explicitly(self):
        detector = self.detector([])
        detector.session.output = np.zeros((1, 10, 8), np.float32)
        with self.assertRaisesRegex(ValueError, 'nms=False'):
            detector.predict(np.zeros((720, 1280, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, '1280'):
            detector.predict(np.zeros((360, 640, 3), np.uint8))

    def test_incompatible_metadata_and_input_contract_are_rejected(self):
        for key, value in (('classes', list(reversed(ATTACK_CLASSES))),
                           ('crop', [0, 90, 1280, 510]), ('imgsz', 320), ('conf', 0), ('iou', 1)):
            with self.subTest(key=key):
                self.metadata.write_text(json.dumps({**self.spec, key: value}), encoding='utf-8')
                with self.assertRaises(ValueError):
                    HayabusaDetector(self.model, session=Session(None))
        self.metadata.write_text(json.dumps(self.spec), encoding='utf-8')
        session = Session(None)
        session.get_inputs = lambda: [SimpleNamespace(name='images', shape=[1, 3, 640, 640], type='tensor(float16)')]
        with self.assertRaisesRegex(ValueError, 'float32'):
            HayabusaDetector(self.model, session=session)

    def test_inference_failure_is_not_silently_replaced_with_rules(self):
        detector = self.detector([])
        with patch.object(detector.session, 'run', side_effect=RuntimeError('inference failed')):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                detector.predict(np.zeros((720, 1280, 3), np.uint8))

    def test_changed_model_bytes_cannot_use_another_models_metadata(self):
        self.model.write_bytes(b'changed model')
        self.metadata.write_text(json.dumps({**self.spec, 'onnx_sha256': '0'*64}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            HayabusaDetector(self.model, use_gpu=False)


class HayabusaDetectorIntegrationTest(unittest.TestCase):
    def test_model_boxes_reuse_tracking_velocity_without_dropping_near_player_attacks(self):
        detector = SimpleNamespace(backend='test learned ONNX')
        vision = HayabusaVision(detector=detector, recognize_boards=False)
        self.assertIsNone(vision.segmenter)
        self.assertEqual(vision.backend, 'test learned ONNX')
        frame = np.zeros((720, 1280, 3), np.uint8)
        for index in range(4):
            x = 650-index*12
            detector.predict = lambda frame, x=x: [
                AttackDetection((x-30, 370, x+30, 390), 'fire', .9),
                AttackDetection((280, 355, 320, 395), 'slash', .9)]
            tracks = vision._detect_tracks(frame, 428, (300, 380), 1+index*.04)
            self.assertEqual(len(tracks), 2)
            fire = next(item for item in tracks if item.kind == 'fire')
            self.assertEqual(fire.id, 1)
        self.assertEqual(fire.hits, 4)
        self.assertLess(fire.vx, -200)

    def test_missing_model_uses_rules_but_present_broken_model_stops_initialization(self):
        with patch.object(Path, 'is_file', return_value=False):
            vision = HayabusaVision(use_gpu=False, recognize_boards=False)
        self.assertIsNone(vision.detector)
        self.assertEqual(vision.backend, 'CPU color rules')
        with patch.object(Path, 'is_file', return_value=True), \
                patch('hayabusa_detector.HayabusaDetector', side_effect=RuntimeError('broken model')):
            with self.assertRaisesRegex(RuntimeError, 'broken model'):
                HayabusaVision(use_gpu=False, recognize_boards=False)


if __name__ == '__main__':
    unittest.main()
