"""GPU color segmentation for the currently observed red Boss attacks."""
import numpy as np
import onnxruntime as ort

from dxgi_adapters import adapters
from vision import ASSETS


class GpuAttackSegmenter:
    def __init__(self):
        if 'DmlExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('当前环境没有 DirectML GPU 推理支持')
        available = [adapter for adapter in adapters() if adapter[2] != 0x1414]
        if not available:
            raise RuntimeError('找不到可用的 GPU')
        index, self.backend, _, _ = max(available, key=lambda adapter: adapter[3])
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(ASSETS / 'boss-red-segmentation.onnx'),
                sess_options=options, providers=[('DmlExecutionProvider', {'device_id': index}), 'CPUExecutionProvider'])
        if self.session.get_providers()[0] != 'DmlExecutionProvider':
            raise RuntimeError('GPU 推理初始化失败')

    def masks(self, field):
        tensor = np.ascontiguousarray(field.transpose(2, 0, 1)[None], dtype=np.float32)
        strong, weak = self.session.run(None, {'image': tensor})
        return (np.asarray(strong[0, 0] * 255, dtype=np.uint8),
                np.asarray(weak[0, 0] * 255, dtype=np.uint8))
