"""GPU color segmentation for the currently observed red Boss attacks."""
import numpy as np
import onnxruntime as ort

from ninja3_assistant.vision.vision import ASSETS
from ninja3_assistant.vision.winml import model_session


class GpuAttackSegmenter:
    def __init__(self):
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        self.session, self.backend = model_session(ASSETS / 'boss-red-segmentation.onnx',
                                                   require_gpu=True,options=options)

    def masks(self, field):
        tensor = np.ascontiguousarray(field.transpose(2, 0, 1)[None], dtype=np.float32)
        strong, weak = self.session.run(None, {'image': tensor})
        return (np.asarray(strong[0, 0] * 255, dtype=np.uint8),
                np.asarray(weak[0, 0] * 255, dtype=np.uint8))
