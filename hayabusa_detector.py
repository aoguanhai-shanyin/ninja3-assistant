"""YOLO11 ONNX attack detection for the annotated Hayabusa battlefield crop."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ATTACK_CLASSES = ('fire', 'blackfire', 'shuriken', 'slash', 'flash')
BATTLE_CROP = (0, 90, 1280, 500)


@dataclass(frozen=True)
class AttackDetection:
    xyxy: tuple
    kind: str
    confidence: float


class HayabusaDetector:
    def __init__(self, model, *, metadata=None, use_gpu=True, session=None):
        model = Path(model)
        metadata = Path(metadata) if metadata is not None else model.with_suffix('.json')
        spec = json.loads(metadata.read_text(encoding='utf-8'))
        if tuple(spec['classes']) != ATTACK_CLASSES or tuple(spec['crop']) != BATTLE_CROP or spec['imgsz'] != 640:
            raise ValueError('隼白检测模型的类别、裁剪区域或输入尺寸与运行程序不匹配')
        self.confidence, self.iou = float(spec['conf']), float(spec['iou'])
        if not 0 < self.confidence < 1 or not 0 < self.iou < 1:
            raise ValueError('隼白检测模型的置信度或重叠阈值无效')
        self.backend = 'YOLO11n ONNX / injected session'
        if session is None:
            if hashlib.sha256(model.read_bytes()).hexdigest() != spec['onnx_sha256']:
                raise ValueError('隼白检测模型与描述文件的 SHA256 不匹配')
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.intra_op_num_threads = 1
            providers = ['CPUExecutionProvider']
            self.backend = 'YOLO11n ONNX / CPU'
            if use_gpu:
                from dxgi_adapters import adapters
                available = [item for item in adapters() if item[2] != 0x1414]
                if 'DmlExecutionProvider' not in ort.get_available_providers() or not available:
                    raise RuntimeError('隼白检测模型需要 DirectML GPU；离线回放可指定 CPU')
                index, name, _, _ = max(available, key=lambda item: item[3])
                providers.insert(0, ('DmlExecutionProvider', {'device_id': index}))
                self.backend = 'YOLO11n ONNX / ' + name
            session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
            if use_gpu and session.get_providers()[0] != 'DmlExecutionProvider':
                raise RuntimeError('隼白检测模型的 DirectML 初始化失败')
        self.session = session
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, 640, 640] or inputs[0].type != 'tensor(float)':
            raise ValueError('隼白检测模型必须使用固定 640×640 RGB float32 输入')
        self.input_name = inputs[0].name

    @staticmethod
    def preprocess(frame):
        if frame.shape != (720, 1280, 3) or frame.dtype != np.uint8:
            raise ValueError('隼白检测模型需要 1280×720 RGB uint8 画面')
        x1, y1, x2, y2 = BATTLE_CROP
        crop = frame[y1:y2, x1:x2]
        ratio = min(640/crop.shape[1], 640/crop.shape[0])
        width, height = round(crop.shape[1]*ratio), round(crop.shape[0]*ratio)
        left, top = round((640-width)/2-.1), round((640-height)/2-.1)
        image = np.full((640, 640, 3), 114, dtype=np.uint8)
        image[top:top+height, left:left+width] = cv2.resize(crop, (width, height))
        tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)/255.
        return tensor, (ratio, left, top)

    def predict(self, frame):
        tensor, geometry = self.preprocess(frame)
        output = self.session.run(None, {self.input_name: tensor})
        if len(output) != 1:
            raise ValueError('隼白检测模型应导出为单个未内置 NMS 的输出')
        return self.decode(output[0], geometry)

    def decode(self, output, geometry):
        output = np.asarray(output)
        if output.ndim != 3 or output.shape[:2] != (1, 4+len(ATTACK_CLASSES)):
            raise ValueError('隼白检测模型的输出应为 [1, 9, N]，请使用 YOLO11 nms=False 导出')
        rows = output[0].T
        classes = rows[:, 4:].argmax(axis=1)
        scores = rows[np.arange(len(rows)), classes+4]
        valid = np.isfinite(rows).all(axis=1) & (scores >= self.confidence) & (scores <= 1)
        rows, classes, scores = rows[valid], classes[valid], scores[valid]
        if not len(rows):
            return []
        ratio, left, top = geometry
        boxes = np.column_stack((rows[:, 0]-rows[:, 2]/2, rows[:, 1]-rows[:, 3]/2,
                                 rows[:, 0]+rows[:, 2]/2, rows[:, 1]+rows[:, 3]/2))
        boxes[:, (0, 2)] = np.clip((boxes[:, (0, 2)]-left)/ratio, 0, 1280)
        boxes[:, (1, 3)] = np.clip((boxes[:, (1, 3)]-top)/ratio+90, 90, 500)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, classes, scores = boxes[valid], classes[valid], scores[valid]
        areas = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
        kept = []
        for kind in np.unique(classes):
            order = np.flatnonzero(classes == kind)
            order = order[np.argsort(-scores[order], kind='stable')]
            while len(order):
                current = order[0]
                kept.append(current)
                rest = order[1:]
                intersection = np.maximum(0, np.minimum(boxes[current, 2:], boxes[rest, 2:])-
                                             np.maximum(boxes[current, :2], boxes[rest, :2])).prod(axis=1)
                overlap = intersection/(areas[current]+areas[rest]-intersection)
                order = rest[overlap <= self.iou]
        return [AttackDetection(tuple(float(v) for v in boxes[i]), ATTACK_CLASSES[classes[i]], float(scores[i]))
                for i in sorted(kept, key=lambda i: -scores[i])[:300]]
