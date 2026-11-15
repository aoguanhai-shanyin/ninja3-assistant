"""Local Windows OCR for activity screenshots; never connects to an emulator."""
from ninja3_assistant.paths import resource_root
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np


class WindowsOcrError(RuntimeError):
    pass


class WindowsOcr:
    def __init__(self, timeout=8):
        self.timeout = timeout
        root = resource_root()
        self.script = root/'assets/windows_ocr.ps1'
        self.powershell = Path(os.environ.get('SystemRoot', r'C:\Windows'))/ \
            'System32/WindowsPowerShell/v1.0/powershell.exe'

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
            command = [str(self.powershell), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                       '-File', str(self.script), '-ImagePath', str(image)]
            try:
                process = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                         text=True, encoding='utf-8', errors='replace', timeout=self.timeout,
                                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            except subprocess.TimeoutExpired:
                if stop is not None and stop.is_set():
                    return None
                raise WindowsOcrError(f'Windows OCR识别超时（{self.timeout}秒）') from None
            except OSError:
                raise WindowsOcrError('无法启动系统Windows PowerShell 5.1进行OCR识别') from None
            if stop is not None and stop.is_set():
                return None
            if process.returncode:
                if 'OCR language' in process.stderr and 'unavailable' in process.stderr:
                    raise WindowsOcrError('系统缺少简体中文OCR语言（zh-Hans-CN），无法识别活动任务')
                raise WindowsOcrError('Windows OCR执行失败，请检查系统OCR组件和简体中文识别语言')
            try:
                result = json.loads(process.stdout.lstrip('\ufeff'))
                if not isinstance(result, dict) or not isinstance(result['text'], str) or not isinstance(result['words'], list):
                    raise ValueError
                for word in result['words']:
                    box = word['bbox']
                    if (not isinstance(word['text'], str) or len(box) != 4 or
                            not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box) or
                            box[2] < 0 or box[3] < 0):
                        raise ValueError
            except (ValueError, KeyError, TypeError):
                raise WindowsOcrError('Windows OCR返回了无法读取的识别结果') from None
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
