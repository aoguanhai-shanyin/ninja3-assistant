"""Project-wide WinML template matching, shared by every activity.

One ONNX graph scores every registered template against its own native-size
screen region in a single inference. Template pixels are external initializers
supplied when the session is created, so the checked-in graph contains no
game-derived art.

Why one model, not one per activity
-----------------------------------
Weekend rotations, activity pages and combat HUDs all need the same primitive:
"is this artwork present in this region, and where". Keeping one catalog and
one graph means a new obstacle or a rotated layout is a template entry, never a
new model file or a version switch.

Accuracy contract
-----------------
Each branch reproduces ``cv2.matchTemplate(..., TM_CCOEFF_NORMED)`` over its
region, so thresholds already verified in live runs carry over unchanged. The
branch subtracts the region mean before convolving: summing ``I^2 - (sum I)^2/n``
directly cancels catastrophically in float32 on near-flat windows and inflates
float noise into a false 1.0. Windows too flat to normalize are scored 0, which
is what OpenCV does.
"""
from dataclasses import dataclass
from functools import lru_cache
import logging

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from ninja3_assistant.paths import resource_root
from ninja3_assistant.vision.winml import model_session

_LOG = logging.getLogger(__name__)
MODEL_NAME = 'template-match.onnx'
_GPU_LOCK_FAILED = object()


@dataclass(frozen=True, slots=True)
class Template:
    """One catalog entry.

    ``box`` is the search region in the 1280x720 frame. ``scale`` resizes the
    template (not the region) to cover size drift between near and far
    instances. ``region_scale`` downscales both region and template for speed;
    reported positions are mapped back to frame pixels.
    """
    name: str
    asset: str
    box: tuple[int, int, int, int]
    threshold: float
    scale: float = 1.0
    region_scale: float = 1.0
    locate: bool = False

    @lru_cache(maxsize=None)
    def geometry(self):
        """Return (region_h, region_w, template_h, template_w) in graph pixels.

        Cached: peak mapping asks for it once per template per frame, and
        re-reading the template file at 60 Hz is pure overhead.
        """
        _, _, w, h = self.box
        image = cv2.imread(str(resource_root() / 'assets' / self.asset), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError('缺少模板匹配素材：' + self.asset)
        th, tw = image.shape
        factor = self.region_scale
        return (max(1, round(h * factor)), max(1, round(w * factor)),
                max(1, round(th * factor * self.scale)),
                max(1, round(tw * factor * self.scale)))


def _page(name, asset, box, threshold, scale=1.0):
    return Template(name, asset, box, threshold, scale=scale)


def _sprite(name, asset, box, threshold, scale=1.0, region_scale=.5):
    return Template(name, asset, box, threshold, scale=scale,
                    region_scale=region_scale, locate=True)


CATALOG: tuple[Template, ...] = (
    # --- weekend pages: every observed rotation, no version switch ----------
    _page('entry', 'weekend-race/entry.png', (900, 110, 380, 120), .93),
    _page('entry-current', 'weekend-race/entry-current.png', (900, 100, 380, 130), .78),
    _page('intro-go', 'weekend-race/intro-go.png', (900, 570, 330, 140), .78),
    _page('start-current', 'weekend-race/start-current.png', (820, 610, 340, 110), .70),
    _page('tasks-current', 'weekend-race/tasks-current.png', (500, 70, 300, 100), .78),
    _page('claim-all-current', 'weekend-race/claim-all-current.png', (1000, 640, 200, 80), .78),
    _page('race-marker-current', 'weekend-race/race-marker-current.png', (850, 0, 160, 90), .78),
    # The distance board is the rotation-independent race confirmation: it
    # scores 0.98..1.00 in every race frame and 0.00..0.52 on every menu page.
    _page('race-distance', 'weekend-race/race-distance.png', (1130, 20, 120, 45), .93),
    # When a smoke card or the standings list covers the distance board, the
    # top-centre effect marker still identifies the earlier rotation's race:
    # 0.93..1.00 on every race frame, 0.00..0.18 on every page.
    _page('race-effect', 'weekend-race/race-effect.png', (875, 0, 65, 65), .90),
    _page('summary-return', 'weekend-race/summary-return-current.png', (580, 620, 140, 90), .90),
    # The spectator replay has its own trace page. Its back button sits at
    # (540,570) instead of the post-match page's (640,667), so it needs its own
    # template: 1.00 here, 0.00 on the post-match trace and <=0.06 elsewhere.
    # Without it the post-match template fires at 1.00 on this page and the
    # runner taps empty space.
    _page('spectate-back', 'weekend-race/spectate-back.png', (450, 545, 180, 60), .90),
    # Settlement closes itself and parks on a spectator ranking view with no
    # start button. Its 进入观战 control scores 1.00 here and at most 0.56 on
    # every other page, so it identifies that view without a version switch.
    _page('ranking-spectate', 'weekend-race/ranking-spectate.png',
          (810, 610, 280, 78), .85),
    _page('finish', 'weekend-race/finish.png', (500, 370, 300, 80), .93),
    _page('result', 'weekend-race/result.png', (1030, 645, 230, 65), .93),
    _page('standings', 'weekend-race/standings.png', (1190, 145, 90, 65), .93),
    _page('reward', 'weekend-race/reward.png', (510, 70, 260, 90), .93),
    _page('ready', 'weekend-race/ready.png', (510, 540, 270, 80), .93),
    _page('home', 'weekend-race/home.png', (90, 5, 340, 90), .93),
    _page('hall', 'weekend-race/hall.png', (100, 0, 260, 100), .93),
    _page('match', 'weekend-race/match.png', (1030, 655, 230, 55), .77),
    _page('tasks', 'weekend-race/tasks.png', (280, 40, 210, 60), .93),
    # Weapon readiness is recognised by the same model; the runner releases it.
    _page('weapon-ready', 'weapon-ready.png', (310, 560, 160, 150), .93),
    # --- claim grids: locate every row so one model counts them -------------
    _sprite('claim', 'weekend-race/claim.png', (900, 110, 200, 520), .78, region_scale=.5),
    _sprite('claimed', 'weekend-race/claimed.png', (900, 110, 200, 520), .93, region_scale=.5),
    _sprite('claimed-current', 'weekend-race/claimed-current.png', (940, 120, 240, 540),
            .78, region_scale=.5),
    # --- obstacles: old sprite set plus the 2026-11-21 needle tracks --------
    # One scale per obstacle: self-match is 0.97..1.0 at scale 1.0, and the
    # extra 0.9/1.1 branches only added inference cost without new detections.
    _sprite('spike', 'weekend-obstacles/spike.png', (300, 180, 800, 400), .82),
    _sprite('spike-needle', 'weekend-obstacles/spike-needle.png', (300, 180, 800, 400), .82),
    _sprite('gate', 'weekend-obstacles/gate.png', (300, 180, 800, 400), .82),
    _sprite('plant', 'weekend-obstacles/plant.png', (300, 180, 800, 400), .82),
    _sprite('bomb', 'weekend-obstacles/bomb.png', (300, 180, 800, 400), .82),
)

# Templates that report a full score map (locate) instead of one scalar.
SPRITES = tuple(i for i, t in enumerate(CATALOG) if t.locate)
PAGES = tuple(i for i, t in enumerate(CATALOG) if not t.locate)


def _region_size(box, region_scale):
    _, _, w, h = box
    return max(1, round(h * region_scale)), max(1, round(w * region_scale))


def _roi_layout():
    """Map each template to a shared ROI input, so identical regions are
    cropped, centred and squared once instead of once per template."""
    order, mapping = [], {}
    for index, template in enumerate(CATALOG):
        key = (tuple(template.box), template.region_scale)
        if key not in mapping:
            mapping[key] = len(order)
            order.append(key)
        template_index = index
        mapping.setdefault('template_roi', {})[template_index] = mapping[key]
    return order, mapping['template_roi']


def build_model():
    """Build the whole project template-matching graph (no game pixels inside).

    Templates sharing a search region share one ROI input and one
    centre/square pass; each template then needs only one correlation Conv and
    two AveragePools for the sliding-window mean and variance.
    """
    node = helper.make_node
    f32 = TensorProto.FLOAT
    nodes, weights, inputs = [], [], []
    order, template_roi = _roi_layout()
    for group, (box, region_scale) in enumerate(order):
        rh, rw = _region_size(box, region_scale)
        prefix = f'r{group}_'
        inputs.append(helper.make_tensor_value_info(prefix + 'roi', f32, [1, 1, rh, rw]))
        nodes.extend([
            node('ReduceMean', [prefix + 'roi'], [prefix + 'mean'], axes=[2, 3], keepdims=1),
            node('Sub', [prefix + 'roi', prefix + 'mean'], [prefix + 'centered']),
            node('Mul', [prefix + 'centered', prefix + 'centered'], [prefix + 'sq']),
        ])

    page_best, sprite_maps = [], []
    for index, template in enumerate(CATALOG):
        source = f'r{template_roi[index]}_'
        prefix = f't{index}_'
        rh, rw, th, tw = template.geometry()
        if th > rh or tw > rw:
            raise RuntimeError(f'模板大于搜索区域：{template.name} ({tw}x{th} in {rw}x{rh})')
        weights.extend([
            _external(prefix + 'kernel', (1, 1, th, tw)),
            _external(prefix + 'energy', (1, 1, 1, 1)),
            helper.make_tensor(prefix + 'count', f32, [], [float(th * tw)]),
            helper.make_tensor(prefix + 'flat', f32, [], [float(th * tw) * 1e-5]),
        ])
        nodes.extend([
            node('Conv', [source + 'centered', prefix + 'kernel'], [prefix + 'num'],
                 kernel_shape=[th, tw]),
            # Sliding-window mean and mean-of-squares; var = n*(E[x^2]-E[x]^2).
            node('AveragePool', [source + 'centered'], [prefix + 'win_mean'],
                 kernel_shape=[th, tw], strides=[1, 1]),
            node('AveragePool', [source + 'sq'], [prefix + 'win_sq'],
                 kernel_shape=[th, tw], strides=[1, 1]),
            node('Mul', [prefix + 'win_mean', prefix + 'win_mean'], [prefix + 'win_mean_sq']),
            node('Sub', [prefix + 'win_sq', prefix + 'win_mean_sq'], [prefix + 'var_mean']),
            node('Mul', [prefix + 'var_mean', prefix + 'count'], [prefix + 'var']),
            node('Max', [prefix + 'var', 'variance_floor'], [prefix + 'var_safe']),
            node('Mul', [prefix + 'var_safe', prefix + 'energy'], [prefix + 'den2']),
            node('Sqrt', [prefix + 'den2'], [prefix + 'den']),
            node('Div', [prefix + 'num', prefix + 'den'], [prefix + 'corr']),
            node('Clip', [prefix + 'corr', 'minus_one', 'plus_one'], [prefix + 'bounded']),
            # Flat windows cannot be normalized reliably: score them 0 like OpenCV
            # instead of amplifying float32 cancellation noise to a false 1.0.
            node('Greater', [prefix + 'var', prefix + 'flat'], [prefix + 'informative']),
            node('Where', [prefix + 'informative', prefix + 'bounded', 'zero'],
                 [prefix + 'valid']),
        ])
        if template.locate:
            sprite_maps.append(helper.make_tensor_value_info(prefix + 'valid', f32, None))
        else:
            nodes.append(node('ReduceMax', [prefix + 'valid'], [prefix + 'best'],
                              axes=[2, 3], keepdims=0))
            page_best.append(prefix + 'best')

    nodes.append(node('Concat', page_best, ['scores'], axis=1))
    weights.extend([
        helper.make_tensor('variance_floor', f32, [], [1e-6]),
        helper.make_tensor('minus_one', f32, [], [-1.0]),
        helper.make_tensor('plus_one', f32, [], [1.0]),
        helper.make_tensor('zero', f32, [], [0.0]),
    ])
    outputs = [helper.make_tensor_value_info('scores', f32, [1, len(page_best)]), *sprite_maps]
    graph = helper.make_graph(nodes, 'Project template matching', inputs, outputs, weights)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)


def _external(name, shape):
    tensor = TensorProto(name=name, dims=list(shape), data_type=TensorProto.FLOAT,
                         data_location=TensorProto.EXTERNAL)
    tensor.external_data.add(key='location', value=f'{name}.bin')
    return tensor


def _weights():
    """Template pixels as external initializers, one pair per catalog entry."""
    names, values = [], []
    root = resource_root() / 'assets'
    for index, template in enumerate(CATALOG):
        image = cv2.imread(str(root / template.asset), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError('缺少模板匹配素材：' + template.asset)
        factor = template.region_scale * template.scale
        if abs(factor - 1.0) > 1e-6:
            image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        _, _, th, tw = template.geometry()
        if image.shape != (th, tw):
            image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
        pixels = image.astype(np.float32) / 255.0
        centered = pixels - pixels.mean()
        energy = float((centered * centered).sum())
        if energy <= 0:
            raise RuntimeError('模板没有有效对比度：' + template.name)
        names += [f't{index}_kernel', f't{index}_energy']
        values += [np.ascontiguousarray(centered[None, None]),
                   np.array([[[[energy]]]], np.float32)]
    return names, values


def _model_path():
    return resource_root() / 'assets' / MODEL_NAME


@lru_cache(maxsize=2)
def matcher_session(use_gpu=True):
    """Create and warm one reusable session for the whole process."""
    path = _model_path()
    source = path.read_bytes() if path.is_file() else build_model().SerializeToString()
    names, values = _weights()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_external_initializers(
        names, [ort.OrtValue.ortvalue_from_numpy(value) for value in values])
    if use_gpu:
        try:
            session, backend = model_session(source, use_gpu=True, require_gpu=True,
                                             options=options)
        except Exception as error:
            _LOG.warning('模板匹配 GPU 会话不可用，回退 CPU 识别：%s', error)
            options = ort.SessionOptions()
            options.add_external_initializers(
                names, [ort.OrtValue.ortvalue_from_numpy(value) for value in values])
            session = ort.InferenceSession(source, sess_options=options,
                                           providers=['CPUExecutionProvider'])
            backend = 'CPU'
    else:
        session = ort.InferenceSession(source, sess_options=options,
                                       providers=['CPUExecutionProvider'])
        backend = 'CPU'
    feeds = {f'r{group}_roi': np.zeros((1, 1, *size), np.float32)
             for group, size in enumerate(_region_size(box, scale)
                                          for box, scale in _roi_layout()[0])}
    session.run(None, feeds)
    return session, backend


def _feeds(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    feeds = {}
    for group, (box, region_scale) in enumerate(_roi_layout()[0]):
        x, y, w, h = box
        rh, rw = _region_size(box, region_scale)
        patch = gray[y:y + h, x:x + w]
        if patch.shape[:2] != (rh, rw):
            patch = cv2.resize(patch, (rw, rh), interpolation=cv2.INTER_AREA)
        feeds[f'r{group}_roi'] = np.ascontiguousarray(patch[None, None])
    return feeds


# Output layout: 'scores' holds page templates in catalog order, then one map
# per locate template. Precompute both index mappings once.
_PAGE_POSITION = {index: position for position, index in enumerate(PAGES)}
_SPRITE_POSITION = {index: position for position, index in enumerate(SPRITES)}


class FrameScores:
    """Scores for one frame, computed lazily and once."""

    __slots__ = ('_pages', '_maps')

    def __init__(self, pages, maps):
        self._pages = pages
        self._maps = maps

    def score(self, name):
        scores = [self._pages[_PAGE_POSITION[i]] for i, t in enumerate(CATALOG)
                  if t.name == name and not t.locate]
        if scores:
            return max(scores)
        peaks = self.peaks(name)
        return max((item[0] for item in peaks), default=0.0)

    def matched(self, name):
        for i, template in enumerate(CATALOG):
            if (template.name == name and not template.locate and
                    self._pages[_PAGE_POSITION[i]] >= template.threshold):
                return True
        return False

    def peaks(self, name):
        """Return [(score, x, y, w, h)] for a locate template, best first."""
        found = []
        for i, template in enumerate(CATALOG):
            if template.name != name or not template.locate:
                continue
            found.extend(_peaks(self._maps[_SPRITE_POSITION[i]], template))
        return found

    def count(self, name, threshold=None):
        return len(self.peaks(name))

    def obstacles(self, ground):
        """Merge sprite peaks that stand on the road into obstacle boxes."""
        merged = []
        for template in CATALOG:
            if not template.locate or template.name in ('claim', 'claimed', 'claimed-current'):
                continue
            for score, x, y, w, h in self.peaks(template.name):
                floor = y + h
                if template.name == 'bomb' or ground is None or abs(floor - ground) < 130:
                    if not any(item['kind'] == template.name and abs(item['box'][0] - x) < 45
                               and abs(item['box'][1] - y) < 45 for item in merged):
                        merged.append({'kind': template.name, 'box': (x, y, w, h),
                                       'score': score})
        return tuple(sorted(merged, key=lambda item: item['box'][0]))


def _peaks(score_map, template):
    """Non-maximum suppression over a score map, mapped back to frame pixels."""
    map2d = np.asarray(score_map)[0, 0]
    factor = template.region_scale
    _, _, th, tw = template.geometry()
    box_w = max(1, round(tw / factor))
    box_h = max(1, round(th / factor))
    found = []
    scores = map2d.copy()
    x0, y0, _, _ = template.box
    for _ in range(6):
        _, best, _, (cx, cy) = cv2.minMaxLoc(scores)
        if best < template.threshold:
            break
        found.append((float(best), int(round(x0 + cx / factor)), int(round(y0 + cy / factor)),
                      box_w, box_h))
        scores[max(0, cy - 4):cy + 5, max(0, cx - 4):cx + 5] = -1
    return found


def scores_for(frame, session=None):
    """Run one inference and return a :class:`FrameScores`."""
    if session is None:
        session, _ = matcher_session()
    outputs = session.run(None, _feeds(frame))
    pages = outputs[0][0]
    maps = {position: outputs[position + 1] for position in range(len(SPRITES))}
    return FrameScores(pages, maps)


def matched(frame, name, session=None):
    return scores_for(frame, session).matched(name)
