"""Batched normalized portrait matching on the worker's WinML device."""
from concurrent.futures import Future
from functools import lru_cache
import logging
from threading import Thread
from time import perf_counter

import numpy as np
import onnxruntime as ort

from ninja3_assistant.paths import resource_root
from ninja3_assistant.vision.winml import model_session


_LOG = logging.getLogger(__name__)
_MODEL = resource_root()/'assets'/'intelligence-portrait-batch.onnx'


def portrait_session(card_count, pixels):
    if not _MODEL.is_file():
        return None
    try:
        art = np.frombuffer(pixels, dtype=np.uint8).reshape(card_count, 65, 110, 3)
        art = art.transpose(0, 3, 1, 2).astype(np.float64)
        filters = []
        for height in (65, 30):
            crop = art[:, :, :height, :]
            centered = crop - crop.mean(axis=(2, 3), keepdims=True)
            norm = np.sqrt((centered * centered).sum(axis=(1, 2, 3), keepdims=True))
            if not norm.all():
                return None
            values = np.ascontiguousarray(centered / norm, dtype=np.float32)
            filters.append(ort.OrtValue.ortvalue_from_numpy(values))
        options = ort.SessionOptions()
        options.add_external_initializers(['filters', 'upper_filters'], filters)
        session, _ = model_session(_MODEL.read_bytes(), require_gpu=True, options=options)
        return session
    except Exception as error:
        _LOG.warning('Windows ML portrait model unavailable: %s', error)
        return None


@lru_cache(maxsize=1)
def warm_portrait_session(card_count, pixels):
    """Prepare one reusable session and its first inference off the frame loop."""
    ready = Future()

    def warmup():
        start = perf_counter()
        session = portrait_session(card_count, pixels)
        if session is not None:
            try:
                portrait_scores(session, ())
            except Exception as error:
                _LOG.warning('Windows ML portrait warmup failed: %s', error)
                session = None
        ready.set_result((session, (perf_counter()-start)*1000))

    Thread(target=warmup, name='portrait-warmup', daemon=True).start()
    return ready


def portrait_scores(session, faces):
    pixels = np.zeros((14, 3, 85, 130), dtype=np.float32)
    for index, face in enumerate(faces):
        pixels[index] = face[30:115, :130].transpose(2, 0, 1)
    scores, upper = session.run(None, {'images': pixels})
    return scores[:len(faces)], upper[:len(faces)]
