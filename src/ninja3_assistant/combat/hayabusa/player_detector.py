"""Independent ONNX detection of the visible player body, without pose templates."""
from dataclasses import dataclass
import hashlib
import json
from math import hypot
from pathlib import Path

import cv2
import numpy as np


PLAYER_CROP = (0, 70, 1280, 555)


@dataclass(frozen=True)
class PlayerDetection:
    xyxy: tuple
    confidence: float


class HayabusaPlayerDetector:
    def __init__(self, model, *, metadata=None, use_gpu=True, session=None):
        model = Path(model)
        metadata = Path(metadata) if metadata is not None else model.with_suffix('.json')
        spec = json.loads(metadata.read_text(encoding='utf-8'))
        if spec['classes'] != ['player'] or tuple(spec['crop']) != PLAYER_CROP or spec['imgsz'] != 640:
            raise ValueError('玩家模型的类别、裁剪区域或输入尺寸与运行程序不匹配')
        self.confidence, self.iou = float(spec['conf']), float(spec['iou'])
        if not 0 < self.confidence < 1 or not 0 < self.iou < 1:
            raise ValueError('玩家模型的置信度或重叠阈值无效')
        self.backend = 'player ONNX / injected session'
        if session is None:
            if hashlib.sha256(model.read_bytes()).hexdigest() != spec['onnx_sha256']:
                raise ValueError('玩家模型与描述文件的 SHA256 不匹配')
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.intra_op_num_threads = 1
            providers = ['CPUExecutionProvider']
            self.backend = 'player ONNX / CPU'
            if use_gpu:
                from ninja3_assistant.device.dxgi_adapters import adapters
                available = [item for item in adapters() if item[2] != 0x1414]
                if 'DmlExecutionProvider' not in ort.get_available_providers() or not available:
                    raise RuntimeError('玩家模型需要 DirectML GPU；离线回放可指定 CPU')
                index, name, _, _ = max(available, key=lambda item: item[3])
                providers.insert(0, ('DmlExecutionProvider', {'device_id': index}))
                self.backend = 'player ONNX / ' + name
            session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
            if use_gpu and session.get_providers()[0] != 'DmlExecutionProvider':
                raise RuntimeError('玩家模型的 DirectML 初始化失败')
        self.session = session
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, 640, 640] or inputs[0].type != 'tensor(float)':
            raise ValueError('玩家模型必须使用固定 640×640 RGB float32 输入')
        self.input_name = inputs[0].name

    @staticmethod
    def preprocess(frame):
        if frame.shape != (720, 1280, 3) or frame.dtype != np.uint8:
            raise ValueError('玩家模型需要 1280×720 RGB uint8 画面')
        x1, y1, x2, y2 = PLAYER_CROP
        crop = frame[y1:y2, x1:x2]
        ratio = min(640/crop.shape[1], 640/crop.shape[0])
        width, height = round(crop.shape[1]*ratio), round(crop.shape[0]*ratio)
        left, top = round((640-width)/2-.1), round((640-height)/2-.1)
        image = np.full((640, 640, 3), 114, np.uint8)
        image[top:top+height, left:left+width] = cv2.resize(crop, (width, height))
        return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)/255., (ratio, left, top)

    def predict(self, frame):
        tensor, (ratio, left, top) = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: tensor})
        if len(outputs) != 1 or np.asarray(outputs[0]).ndim != 3 or np.asarray(outputs[0]).shape[:2] != (1, 5):
            raise ValueError('玩家模型输出应为 [1, 5, N]，请使用单类 YOLO11 nms=False 导出')
        rows = np.asarray(outputs[0])[0].T
        rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] >= self.confidence) & (rows[:, 4] <= 1)]
        boxes = np.column_stack((rows[:, 0]-rows[:, 2]/2, rows[:, 1]-rows[:, 3]/2,
                                 rows[:, 0]+rows[:, 2]/2, rows[:, 1]+rows[:, 3]/2))
        boxes[:, (0, 2)] = np.clip((boxes[:, (0, 2)]-left)/ratio, 0, 1280)
        boxes[:, (1, 3)] = np.clip((boxes[:, (1, 3)]-top)/ratio+70, 70, 555)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores = boxes[valid], rows[valid, 4]
        areas = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
        order, kept = np.argsort(-scores, kind='stable'), []
        while len(order):
            current, rest = order[0], order[1:]
            kept.append(current)
            intersection = np.maximum(0, np.minimum(boxes[current, 2:], boxes[rest, 2:])-
                                     np.maximum(boxes[current, :2], boxes[rest, :2])).prod(axis=1)
            order = rest[intersection/(areas[current]+areas[rest]-intersection) <= self.iou]
        return [PlayerDetection(tuple(float(v) for v in boxes[i]), float(scores[i])) for i in kept]


class PlayerBodyLocator:
    """Associate current visual detections; missing or ambiguous evidence stays unknown."""
    def __init__(self, detector):
        self.detector = detector
        self.reset()

    def reset(self):
        self.previous = None

    def locate(self, frame, now):
        detections = self.detector.predict(frame)
        boxes = [item.xyxy for item in detections if
                 0 < item.xyxy[0] < item.xyxy[2] < 1280 and
                 70 < item.xyxy[1] < item.xyxy[3] < 555 and
                 item.xyxy[2]-item.xyxy[0] >= 12 and item.xyxy[3]-item.xyxy[1] >= 12]
        if self.previous is not None:
            previous, seen = self.previous
            dt = now-seen
            if 0 < dt <= .35:
                x, y = (previous[0]+previous[2])/2, (previous[1]+previous[3])/2
                reach = min(240, 1600*dt+32)
                boxes = [box for box in boxes if hypot((box[0]+box[2])/2-x, (box[1]+box[3])/2-y) <= reach]
            else:
                self.previous = None
        if len(boxes) != 1:
            return None
        self.previous = (boxes[0], now)
        return boxes[0]
