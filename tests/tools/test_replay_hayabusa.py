import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from tools.replay.replay_hayabusa import analyze_video, readable_video


def screen():
    return SimpleNamespace(active=True, rope_y=400, player_x=300, player_y=350,
                           side='above', airborne=False, board_hint='shuriken', result=None)


class ReplayHayabusaTest(unittest.TestCase):
    def test_cache_copy_cancellation_removes_partial_file_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'cache.m4s'
            source.write_bytes(b'000000000\x00\x00\x00\x24ftyp'+b'data'*(1024*1024))
            before = hashlib.sha256(source.read_bytes()).digest()
            copies, checks = [],[0]
            create = tempfile.NamedTemporaryFile
            def temporary(**kwargs):
                result = create(**kwargs)
                copies.append(Path(result.name))
                return result
            def cancelled():
                checks[0] += 1
                return checks[0] >= 3
            with patch('tools.replay.replay_hayabusa.tempfile.NamedTemporaryFile',side_effect=temporary), \
                    self.assertRaises(InterruptedError):
                with readable_video(source,cancelled=cancelled):
                    self.fail('cancelled video must not be yielded')
            self.assertEqual(len(copies),1)
            self.assertFalse(copies[0].exists())
            self.assertEqual(hashlib.sha256(source.read_bytes()).digest(),before)

    def test_cache_header_copy_is_removed_even_when_analysis_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'cache.m4s'
            source.write_bytes(b'000000000\x00\x00\x00\x24ftyp' + b'data'*20)
            before = hashlib.sha256(source.read_bytes()).digest()
            with self.assertRaisesRegex(RuntimeError, 'analysis failure'):
                with readable_video(source) as normalized:
                    copy = normalized
                    self.assertNotEqual(copy, source)
                    self.assertEqual(copy.read_bytes(), source.read_bytes()[9:])
                    raise RuntimeError('analysis failure')
            self.assertFalse(copy.exists())
            self.assertEqual(hashlib.sha256(source.read_bytes()).digest(), before)

    def test_real_video_exports_timestamped_hypothetical_actions_without_device(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder)/'source.mp4', Path(folder)/'replay'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 36))
            self.assertTrue(writer.isOpened())
            for value in range(6):
                writer.write(np.full((36, 64, 3), value*30, np.uint8))
            writer.release()
            before = source.read_bytes()
            vision = Mock(backend='test CPU')
            vision.detect.return_value = (screen(), [])
            action = SimpleNamespace(kind='jump', point=(1090, 630), threat_id=4)
            policy = Mock(last_reason='test crossing')
            policy.decide.return_value = action
            with patch('ninja3_assistant.combat.hayabusa.vision.HayabusaVision', return_value=vision) as factory, \
                    patch('ninja3_assistant.combat.hayabusa.policy.HayabusaPolicy', return_value=policy):
                report = analyze_video(source, output, start=.1, end=.4, fps=10, cpu=True)
            factory.assert_called_once_with(use_gpu=False)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['executed_actions'], 0)
            self.assertEqual(len(report['frames']), 4)
            self.assertAlmostEqual(report['frames'][0]['video_seconds'], .1, places=3)
            self.assertAlmostEqual(report['frames'][-1]['video_seconds'], .4, places=3)
            self.assertEqual(report['summary']['suggested_actions_by_kind'], {'jump': 4})
            self.assertEqual(policy.completed.call_count, 4)
            self.assertEqual(vision.detect.call_args.args[0].shape, (720, 1280, 3))
            self.assertEqual(json.loads((output/'report.json').read_text())['backend'], 'test CPU')
            self.assertGreater((output/'annotated.mp4').stat().st_size, 100)
            self.assertEqual(source.read_bytes(), before)

    def test_seek_returning_last_cached_frame_is_rejected_before_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder)/'partial.mp4', Path(folder)/'replay'
            source.write_bytes(b'test file')
            capture = Mock()
            capture.isOpened.return_value = True
            capture.set.return_value = True
            capture.grab.return_value = True
            capture.get.side_effect = lambda prop: {cv2.CAP_PROP_FPS: 60,
                cv2.CAP_PROP_FRAME_COUNT: 39600, cv2.CAP_PROP_POS_MSEC: 303367}[prop]
            vision, policy = Mock(backend='test CPU'), Mock()
            with patch('ninja3_assistant.combat.hayabusa.vision.HayabusaVision', return_value=vision), \
                    patch('ninja3_assistant.combat.hayabusa.policy.HayabusaPolicy', return_value=policy), \
                    patch('tools.replay.replay_hayabusa.cv2.VideoCapture', return_value=capture), \
                    self.assertRaisesRegex(RuntimeError, 'Seek timestamp mismatch'):
                analyze_video(source, output, start=500, end=510, cpu=True)
            vision.detect.assert_not_called()
            policy.decide.assert_not_called()
            capture.release.assert_called()
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report['status'], 'error')
            self.assertEqual(report['summary']['analyzed_frames'], 0)

    def test_timestamp_gap_resets_both_temporal_modules(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder)/'gapped.mp4', Path(folder)/'replay'
            source.write_bytes(b'test file')
            timestamps, current = iter((0, .1, .8)), [None]

            def grab():
                try:
                    current[0] = next(timestamps)
                    return True
                except StopIteration:
                    return False

            capture = Mock()
            capture.isOpened.return_value = capture.set.return_value = True
            capture.grab.side_effect = grab
            capture.retrieve.return_value = (True, np.zeros((36, 64, 3), np.uint8))
            capture.get.side_effect = lambda prop: {cv2.CAP_PROP_FPS: 10,
                cv2.CAP_PROP_FRAME_COUNT: 9, cv2.CAP_PROP_POS_MSEC: (current[0] or 0)*1000}[prop]
            vision, policy = Mock(backend='test CPU'), Mock(last_reason='tracking')
            vision.detect.return_value = (screen(), [])
            policy.decide.return_value = None
            with patch('ninja3_assistant.combat.hayabusa.vision.HayabusaVision', return_value=vision), \
                    patch('ninja3_assistant.combat.hayabusa.policy.HayabusaPolicy', return_value=policy), \
                    patch('tools.replay.replay_hayabusa.cv2.VideoCapture', return_value=capture):
                report = analyze_video(source, output, end=.8, fps=10, cpu=True)
            vision.reset.assert_called_once()
            policy.reset.assert_called_once()
            self.assertEqual(report['timeline_resets'], [{'from_seconds': .1, 'to_seconds': .8}])
            self.assertEqual([frame['video_seconds'] for frame in report['frames']], [0, .1, .8])


if __name__ == '__main__':
    unittest.main()
