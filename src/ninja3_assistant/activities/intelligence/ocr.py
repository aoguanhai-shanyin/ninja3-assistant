"""Local Windows OCR for activity screenshots; never connects to an emulator."""
from ninja3_assistant.paths import resource_root
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import tempfile
import threading
import time

import cv2
import numpy as np


class WindowsOcrError(RuntimeError):
    pass


_START_ERROR = '无法启动系统Windows PowerShell 5.1进行OCR识别'
_LANGUAGE_ERROR = '系统缺少简体中文OCR语言（zh-Hans-CN），无法识别活动任务'
_EXECUTION_ERROR = 'Windows OCR执行失败，请检查系统OCR组件和简体中文识别语言'
_RESULT_ERROR = 'Windows OCR返回了无法读取的识别结果'


class _OcrSession:
    def __init__(self, command):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace',
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.requests = Queue()
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._serve, name='windows-ocr', daemon=True)
        self.thread.start()

    def _serve(self):
        try:
            with self.process.stdin as stdin, self.process.stdout as stdout:
                while (request := self.requests.get()) is not None:
                    image, response = request
                    try:
                        stdin.write(json.dumps({'path': str(image)})+'\n')
                        stdin.flush()
                        line = stdout.readline()
                    except OSError:
                        line = ''
                    response.put(line)
                    if not line:
                        return
        except OSError:
            pass  # The child may exit while its input pipe is being closed.

    def read(self, image, stop, deadline):
        response = Queue()
        self.requests.put((image, response))
        while not self.closed.is_set() and not (stop is not None and stop.is_set()):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            try:
                line = response.get(timeout=min(0.05, remaining))
            except Empty:
                continue
            if self.closed.is_set() or (stop is not None and stop.is_set()):
                return None
            return line
        return None

    def close(self):
        self.closed.set()
        self.requests.put(None)
        try:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            pass  # Preserve the read error if process cleanup also fails.
        self.thread.join(timeout=1)


class WindowsOcr:
    def __init__(self, timeout=8, *, persistent=False):
        self.timeout = timeout
        root = resource_root()
        self.script = root/'assets/windows_ocr.ps1'
        self.powershell = Path(os.environ.get('SystemRoot', r'C:\Windows'))/ \
            'System32/WindowsPowerShell/v1.0/powershell.exe'
        self._persistent = persistent
        self._session = None
        self._read_lock = threading.Lock()
        self._session_lock = threading.Lock()

    def _command(self, *arguments):
        return [str(self.powershell), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                '-File', str(self.script), *arguments]

    def close(self):
        """End the current OCR session and cancel its pending read, if any."""
        self._close_session()

    def _close_session(self, session=None):
        with self._session_lock:
            if self._session is not None and (session is None or session is self._session):
                self._session.close()
                self._session = None

    def _get_session(self):
        with self._session_lock:
            if self._session is None:
                try:
                    self._session = _OcrSession(self._command('-Stream'))
                except OSError:
                    raise WindowsOcrError(_START_ERROR) from None
            return self._session

    def warmup(self):
        if self._persistent and self.script.is_file():
            try:
                self._get_session()
            except WindowsOcrError:
                pass  # Let the first read retry and report any startup error.

    def _read_stream(self, image, stop):
        with self._read_lock:
            if stop is not None and stop.is_set():
                return None
            deadline = time.monotonic()+self.timeout
            session = self._get_session()
            try:
                line = session.read(image, stop, deadline)
                if line is None:
                    self._close_session(session)
                    return None
                if not line:
                    raise WindowsOcrError(_EXECUTION_ERROR)
                payload = _load_result(line)
                if 'error' in payload:
                    raise WindowsOcrError(_LANGUAGE_ERROR if payload['error'] == 'language_unavailable'
                                          else _EXECUTION_ERROR)
                return _validate_result(payload.get('result'))
            except TimeoutError:
                self._close_session(session)
                if stop is not None and stop.is_set():
                    return None
                raise WindowsOcrError(f'Windows OCR识别超时（{self.timeout}秒）') from None
            except WindowsOcrError:
                self._close_session(session)
                raise

    def _read_once(self, image, stop):
        command = self._command('-ImagePath', str(image))
        try:
            process = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                     text=True, encoding='utf-8', errors='replace', timeout=self.timeout,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired:
            if stop is not None and stop.is_set():
                return None
            raise WindowsOcrError(f'Windows OCR识别超时（{self.timeout}秒）') from None
        except OSError:
            raise WindowsOcrError(_START_ERROR) from None
        if stop is not None and stop.is_set():
            return None
        if process.returncode:
            if 'OCR language' in process.stderr and 'unavailable' in process.stderr:
                raise WindowsOcrError(_LANGUAGE_ERROR)
            raise WindowsOcrError(_EXECUTION_ERROR)
        return _validate_result(_load_result(process.stdout))

    def read(self, frame, stop=None):
        """Read an RGB uint8 frame, or return None when cancellation is set."""
        if stop is not None and stop.is_set():
            return None
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Windows OCR需要RGB uint8图像')
        if not self.script.is_file():
            raise WindowsOcrError('找不到Windows OCR脚本 assets/windows_ocr.ps1')
        with tempfile.TemporaryDirectory(prefix='ninja3-ocr-') as directory:
            image = Path(directory)/'frame.png'
            if not cv2.imwrite(str(image), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                raise WindowsOcrError('无法保存Windows OCR临时图片')
            if stop is not None and stop.is_set():
                return None
            return self._read_stream(image, stop) if self._persistent else self._read_once(image, stop)


def _load_result(line):
    try:
        result = json.loads(line.lstrip('\ufeff'))
        if isinstance(result, dict):
            return result
    except ValueError:
        pass
    raise WindowsOcrError(_RESULT_ERROR)


def _validate_result(result):
    try:
        if not isinstance(result, dict) or not isinstance(result['text'], str) or not isinstance(result['words'], list):
            raise ValueError
        for word in result['words']:
            box = word['bbox']
            if (not isinstance(word['text'], str) or len(box) != 4 or
                    not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box) or
                    box[2] < 0 or box[3] < 0):
                raise ValueError
    except (ValueError, KeyError, TypeError):
        raise WindowsOcrError(_RESULT_ERROR) from None
    return result


def region_words(result, x, y, width, height):
    """Select words whose centers lie inside a source-image rectangle, in OCR order."""
    return [word for word in result.get('words', [])
            if x <= word['bbox'][0]+word['bbox'][2]/2 < x+width and
            y <= word['bbox'][1]+word['bbox'][3]/2 < y+height]


def compact_text(value):
    """Remove whitespace from OCR text or a sequence of OCR words."""
    text = value if isinstance(value, str) else ''.join(word['text'] for word in value)
    return ''.join(text.split())
