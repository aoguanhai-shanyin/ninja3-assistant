from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from ninja3_assistant.activities.intelligence.ocr import WindowsOcr, WindowsOcrError, _OcrSession, compact_text, region_words


def stream_command(code):
    return [sys.executable, '-u', '-c', code]


ECHO_IMAGE = stream_command(
    'import json, pathlib, sys\n'
    'for request in sys.stdin:\n'
    '    image = json.loads(request)["path"]\n'
    '    assert pathlib.Path(image).is_file()\n'
    '    print(json.dumps({"result": {"text": image, "words": []}}))\n'
)
BLOCK_READ = stream_command('import sys, time; sys.stdin.readline(); time.sleep(30)')


class ActivityOcrTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((20, 30, 3), (255, 128, 0), np.uint8)
        self.result = {'text': '已 领 取 0 / 2', 'words': [
            {'text': '已', 'bbox': [10, 5, 4, 6], 'line': 0},
            {'text': '领', 'bbox': [14, 5, 4, 6], 'line': 0},
            {'text': '取', 'bbox': [18, 5, 4, 6], 'line': 0},
            {'text': '0 / 2', 'bbox': [40, 5, 15, 6], 'line': 0}]}

    def completed(self, **changes):
        return SimpleNamespace(returncode=0, stdout=json.dumps(self.result, ensure_ascii=False), stderr='', **changes)

    def test_reads_rgb_with_hidden_process_and_removes_temporary_image(self):
        paths = []
        def run(command, **kwargs):
            image = Path(command[command.index('-ImagePath')+1])
            paths.append(image)
            np.testing.assert_array_equal(cv2.imread(str(image))[0, 0], (0, 128, 255))
            self.assertTrue(command[command.index('-File')+1].endswith('assets\\windows_ocr.ps1'))
            self.assertEqual(kwargs['encoding'], 'utf-8')
            self.assertEqual(kwargs['creationflags'], getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.assertIs(kwargs['stdin'], subprocess.DEVNULL)
            return self.completed()
        with patch('ninja3_assistant.activities.intelligence.ocr.subprocess.run', side_effect=run):
            self.assertEqual(WindowsOcr().read(self.frame), self.result)
        self.assertFalse(paths[0].exists())

    def test_cancelled_before_read_never_starts_a_process(self):
        stop = threading.Event()
        stop.set()
        with patch('ninja3_assistant.activities.intelligence.ocr.subprocess.run') as run, patch('ninja3_assistant.activities.intelligence.ocr.cv2.imwrite') as write:
            self.assertIsNone(WindowsOcr().read(self.frame, stop=stop))
            run.assert_not_called()
            write.assert_not_called()

    def test_cancelled_during_read_discards_result(self):
        stop = threading.Event()
        def run(*args, **kwargs):
            stop.set()
            return self.completed()
        with patch('ninja3_assistant.activities.intelligence.ocr.subprocess.run', side_effect=run):
            self.assertIsNone(WindowsOcr().read(self.frame, stop=stop))

    def test_timeout_is_clear_and_does_not_expose_child_output(self):
        with patch('ninja3_assistant.activities.intelligence.ocr.subprocess.run', side_effect=subprocess.TimeoutExpired('secret-command', 1, output='secret-output')):
            with self.assertRaisesRegex(WindowsOcrError, '超时') as error:
                WindowsOcr(timeout=1).read(self.frame)
        self.assertNotIn('secret', str(error.exception))

    def test_missing_language_has_readable_error_without_raw_stderr(self):
        failed = SimpleNamespace(returncode=1, stdout='', stderr="OCR language 'zh-Hans-CN' is unavailable. secret")
        with patch('ninja3_assistant.activities.intelligence.ocr.subprocess.run', return_value=failed):
            with self.assertRaisesRegex(WindowsOcrError, '缺少简体中文OCR语言') as error:
                WindowsOcr().read(self.frame)
        self.assertNotIn('secret', str(error.exception))

    def test_missing_powershell_has_readable_error(self):
        with patch('ninja3_assistant.activities.intelligence.ocr.subprocess.run', side_effect=FileNotFoundError('secret')):
            with self.assertRaisesRegex(WindowsOcrError, '无法启动系统Windows PowerShell'):
                WindowsOcr().read(self.frame)

    def test_rejects_unreadable_json_and_invalid_word_geometry(self):
        for payload in ('not json', '{}', '{"text":"x","words":[{"text":"x","bbox":[0,0,-1,4]}]}'):
            with self.subTest(payload=payload), patch('ninja3_assistant.activities.intelligence.ocr.subprocess.run', return_value=SimpleNamespace(
                    returncode=0, stdout=payload, stderr='')):
                with self.assertRaisesRegex(WindowsOcrError, '无法读取'):
                    WindowsOcr().read(self.frame)

    def test_rejects_wrong_image_contract_and_failed_save(self):
        for frame in (self.frame[:, :, 0], self.frame.astype(float), None):
            with self.subTest(frame_type=type(frame)), self.assertRaisesRegex(ValueError, 'RGB uint8'):
                WindowsOcr().read(frame)
        with patch('ninja3_assistant.activities.intelligence.ocr.cv2.imwrite', return_value=False), patch('ninja3_assistant.activities.intelligence.ocr.subprocess.run') as run:
            with self.assertRaisesRegex(WindowsOcrError, '临时图片'):
                WindowsOcr().read(self.frame)
            run.assert_not_called()

    def test_persistent_session_reuses_the_process_for_new_temporary_images(self):
        ocr = WindowsOcr(persistent=True)
        with patch.object(ocr, '_command', return_value=ECHO_IMAGE) as command:
            try:
                first = ocr.read(self.frame)
                session = ocr._session
                second = ocr.read(self.frame)
                command.assert_called_once_with('-Stream')
            finally:
                ocr.close()
        self.assertNotEqual(first['text'], second['text'])
        self.assertFalse(Path(first['text']).exists())
        self.assertFalse(Path(second['text']).exists())
        self.assertIsNotNone(session.process.returncode)
        self.assertFalse(session.thread.is_alive())

    def test_persistent_read_timeout_recovers_with_a_fresh_process(self):
        ocr = WindowsOcr(timeout=0.2, persistent=True)
        with patch.object(ocr, '_command', side_effect=[BLOCK_READ, ECHO_IMAGE]) as command:
            try:
                with self.assertRaisesRegex(WindowsOcrError, '超时'):
                    ocr.read(self.frame)
                self.assertIsNone(ocr._session)
                ocr.timeout = 3
                self.assertTrue(ocr.read(self.frame)['text'].endswith('frame.png'))
                self.assertEqual(command.call_count, 2)
            finally:
                ocr.close()

    def test_persistent_stop_and_close_clear_a_pending_read(self):
        read = _OcrSession.read
        for cancel_with_stop in (True, False):
            with self.subTest(cancel_with_stop=cancel_with_stop):
                ocr = WindowsOcr(persistent=True)
                stop, started = threading.Event(), threading.Event()

                def notify_read(session, *args):
                    started.set()
                    return read(session, *args)

                with patch.object(ocr, '_command', return_value=BLOCK_READ), \
                        patch.object(_OcrSession, 'read', notify_read), ThreadPoolExecutor(max_workers=1) as pool:
                    try:
                        pending = pool.submit(ocr.read, self.frame, stop)
                        self.assertTrue(started.wait(3))
                        session = ocr._session
                        if cancel_with_stop:
                            stop.set()
                        else:
                            ocr.close()
                        self.assertIsNone(pending.result(timeout=3))
                    finally:
                        ocr.close()
                self.assertIsNone(ocr._session)
                self.assertIsNotNone(session.process.returncode)
                self.assertFalse(session.thread.is_alive())

    def test_persistent_recognizes_structured_language_error_from_exiting_process(self):
        ocr = WindowsOcr(persistent=True)
        command = stream_command('import sys; sys.stdin.readline(); print(\'{"error":"language_unavailable"}\')')
        with patch.object(ocr, '_command', return_value=command):
            try:
                with self.assertRaisesRegex(WindowsOcrError, '缺少简体中文OCR语言'):
                    ocr.read(self.frame)
                self.assertIsNone(ocr._session)
            finally:
                ocr.close()

    def test_region_selects_by_center_and_compacts_split_chinese_characters(self):
        words = region_words(self.result, 10, 5, 12, 6)
        self.assertEqual(compact_text(words), '已领取')
        self.assertEqual(compact_text(' 0 /\t2\n'), '0/2')
        self.assertEqual(region_words(self.result, 0, 0, 1, 1), [])


if __name__ == '__main__':
    unittest.main()
