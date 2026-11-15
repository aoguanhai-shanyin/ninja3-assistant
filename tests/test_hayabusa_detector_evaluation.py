import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools.evaluate_hayabusa_detector import CLASSES, clip_predictions, evaluate, load_reviewed, predict, validate_boxes


def box(kind='fire', coordinates=(10, 20, 100, 80), confidence=1.):
    return {'class': kind, 'xyxy': list(coordinates), 'confidence': confidence}


class DetectorEvaluationTest(unittest.TestCase):
    def test_model_roundoff_is_clipped_without_relaxing_reviewed_truth_bounds(self):
        predictions = clip_predictions([box(coordinates=(-.00001, 20, 1280.0001, 410.00001)),
                                        box(coordinates=(1280.00001, 20, 1281, 30))])
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]['xyxy'], [0., 20, 1280, 410])
        validate_boxes(predictions)
        with self.assertRaisesRegex(ValueError, 'outside'):
            validate_boxes([box(coordinates=(-.00001, 20, 1280.0001, 410.00001))])

    def test_duplicate_predictions_do_not_match_one_truth_twice(self):
        frames = [{'file': 'a.jpg', 'boxes': [box()]}]
        metrics = evaluate(frames, {'a.jpg': [box(confidence=.9), box(confidence=.8)]})
        self.assertEqual(metrics['total']['tp'], 1)
        self.assertEqual(metrics['total']['fp'], 1)
        self.assertEqual(metrics['total']['fn'], 0)
        self.assertEqual(metrics['total']['precision'], .5)
        self.assertEqual(metrics['total']['map50'], 1.)

    def test_negative_frame_predictions_and_wrong_classes_are_false_positives(self):
        frames = [{'file': 'a.jpg', 'boxes': [box()]}, {'file': 'negative.jpg', 'boxes': []}]
        metrics = evaluate(frames, {'a.jpg': [box('slash')], 'negative.jpg': [box()]})
        self.assertEqual(metrics['total']['tp'], 0)
        self.assertEqual(metrics['total']['fp'], 2)
        self.assertEqual(metrics['total']['fn'], 1)
        self.assertEqual(metrics['total']['map50'], 0.)

    def test_low_confidence_detections_affect_ap_but_not_threshold_counts(self):
        frames = [{'file': 'a.jpg', 'boxes': [box()]}]
        metrics = evaluate(frames, {'a.jpg': [box(confidence=.1)]})
        self.assertEqual(metrics['total']['tp'], 0)
        self.assertEqual(metrics['total']['fn'], 1)
        self.assertEqual(metrics['total']['map50'], 1.)
        self.assertIsNone(metrics['per_class']['slash']['ap50'])

    def test_iou_thresholds_distinguish_imprecise_boxes(self):
        frames = [{'file': 'a.jpg', 'boxes': [box(coordinates=(0, 0, 100, 100))]}]
        metrics = evaluate(frames, {'a.jpg': [box(coordinates=(0, 0, 80, 100))]})
        self.assertEqual(metrics['total']['map50'], 1.)
        self.assertLess(metrics['total']['map50_95'], 1.)

    def test_unreviewed_manifest_is_rejected_even_with_nonempty_boxes(self):
        payload = {'classes': list(CLASSES), 'crop_xyxy': [0, 90, 1280, 500],
                   'frames': [{'file': 'a.jpg', 'split': 'test', 'reviewed': False, 'boxes': [box()]}]}
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder)/'manifest.json'
            manifest.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'unreviewed'):
                load_reviewed(manifest, 'test')
            payload['frames'][0]['reviewed'] = True
            manifest.write_text(json.dumps(payload), encoding='utf-8')
            self.assertEqual(len(load_reviewed(manifest, 'test')[1]), 1)

    def test_missing_predictions_count_as_misses_and_foreign_frames_are_rejected(self):
        frames = [{'file': 'a.jpg', 'boxes': [box()]}]
        self.assertEqual(evaluate(frames, {})['total']['fn'], 1)
        with self.assertRaisesRegex(ValueError, 'outside'):
            evaluate(frames, {'foreign.jpg': [box()]})

    def test_onnx_evaluation_uses_runtime_crop_coordinates_and_ap_confidence(self):
        with tempfile.TemporaryDirectory() as folder, patch('hayabusa_detector.HayabusaDetector') as detector:
            root = Path(folder)
            frame = np.zeros((410, 1280, 3), np.uint8)
            frame[:, :, 0] = 200
            cv2.imwrite(str(root/'frame.png'), frame)
            detector.return_value.predict.return_value = [SimpleNamespace(
                kind='fire', xyxy=(10., 100., 60., 150.), confidence=.8)]
            predictions, timings = predict(root/'best.onnx', root, [{'file': 'frame.png'}], 'cpu')
            self.assertEqual(predictions['frame.png'][0]['xyxy'], [10., 10., 60., 60.])
            self.assertEqual(detector.return_value.confidence, .001)
            self.assertEqual(detector.return_value.iou, .7)
            detector.assert_called_once_with(root/'best.onnx', use_gpu=False)
            passed = detector.return_value.predict.call_args.args[0]
            self.assertEqual(passed.shape, (720, 1280, 3))
            self.assertEqual(tuple(passed[100, 10]), (0, 0, 200))
            self.assertEqual(len(timings), 1)


if __name__ == '__main__':
    unittest.main()
