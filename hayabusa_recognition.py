"""Formation/sequence reference matching for the observed Hayabusa boards.

Runtime features contain verified attack geometry only. They contain no video
clock, chapter text, HUD, player skin, background texture or learned OCR cues.
Confidence is formation similarity, not a measured probability of correctness.
"""
from collections import deque
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from vision import ASSETS


KINDS = ('fire', 'blackfire', 'shuriken', 'slash', 'flash')
GRID = (8, 12)
CHANNEL_SIZE = GRID[0] * GRID[1] + 4
FEATURE_SIZE = len(KINDS) * CHANNEL_SIZE
FEATURE_VERSION = 'attack-formation-subsequence-dtw-v3'


@dataclass(frozen=True)
class BoardMatch:
    board_id: str | None = None
    phase: str | None = None
    confidence: float = 0.
    margin: float = 0.
    progress: float = 0.
    evidence: str = 'no distinct attack formation'
    phase_source: str = 'unknown'
    name: str = ''
    locked: bool = False
    onset: bool = False
    ended: bool = False
    epoch: int = 0


def formation_features(screen, tracks):
    """Rope-relative attack positions and sizes, independent of teaching text."""
    if screen.rope_y is None:
        return np.zeros(FEATURE_SIZE, np.float32)
    output = np.zeros((len(KINDS), CHANNEL_SIZE), np.float32)
    gy, gx = np.mgrid[:GRID[0], :GRID[1]]
    for channel, kind in enumerate(KINDS):
        items = [item for item in tracks if item.kind == kind and
                 item.width > 0 and item.height > 0 and
                 all(np.isfinite(value) for value in (item.x, item.y, item.width, item.height))]
        if not items:
            continue
        grid = np.zeros(GRID, np.float32)
        for item in items:
            x = item.x / 1280 * GRID[1] - .5
            y = (item.y - screen.rope_y + 320) / 520 * GRID[0] - .5
            sx = max(.55, min(3., item.width / 1280 * GRID[1] / 2))
            sy = max(.5, min(2., item.height / 520 * GRID[0] / 2))
            amplitude = min(2., max(.5, (item.width * item.height) ** .5 / 40))
            grid += np.exp(-.5 * (((gx-x)/sx)**2 + ((gy-y)/sy)**2)) * amplitude
        output[channel, :GRID[0]*GRID[1]] = np.minimum(grid, 3).ravel()
        output[channel, -4:] = (min(len(items), 16)/8,
            min(np.mean([item.width for item in items]), 400)/200,
            min(np.mean([item.height for item in items]), 250)/125,
            np.clip(np.mean([item.y-screen.rope_y for item in items])/320, -1, 1))
    vector = output.ravel()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-6 else vector


def mirror_features(features):
    values = np.array(features, dtype=np.float32, copy=True).reshape(-1, len(KINDS), CHANNEL_SIZE)
    grids = values[:, :, :GRID[0]*GRID[1]].reshape(-1, len(KINDS), *GRID)
    grids[:] = grids[:, :, :, ::-1].copy()
    return values.reshape(np.shape(features))


def normalize(features):
    values = np.asarray(features, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-6)


def sequence_costs(references, tokens, sequences):
    """Short subsequence DTW, with independent mirrored/normal paths.

    Stay/advance-one/advance-two permits different observation/playback speeds.
    Paths cannot jump to another board, clip, or a different wave each frame.
    """
    scores = references @ tokens.T
    local = 1-np.clip(scores,-1,1)
    blank_refs = np.linalg.norm(references,axis=1)<.1
    for index,token in enumerate(tokens):
        if np.linalg.norm(token)<.1:
            local[blank_refs,index]=0
    cost=local[:,0]
    same_one=sequences[1:]==sequences[:-1]
    same_two=sequences[2:]==sequences[:-2]
    for index in range(1,len(tokens)):
        one=np.full(len(cost),np.inf,np.float32)
        two=one.copy()
        one[1:]=np.where(same_one,cost[:-1],np.inf)
        two[2:]=np.where(same_two,cost[:-2]+.015,np.inf)
        cost=local[:,index]+np.minimum(cost+.008,np.minimum(one,two))
    return cost/len(tokens)


class HayabusaBoardRecognizer:
    def __init__(self, asset_dir=None, *, min_confidence=None, min_margin=None):
        folder = Path(asset_dir) if asset_dir is not None else ASSETS
        metadata_file = folder/'hayabusa-boards.json'
        arrays_file = folder/'hayabusa-boards.npz'
        if not metadata_file.is_file() or not arrays_file.is_file():
            raise RuntimeError('缺少隼白板子参考库，请先运行 tools/build_hayabusa_boards.py')
        self.metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
        if self.metadata.get('feature_version') != FEATURE_VERSION:
            raise RuntimeError('隼白板子参考库版本与识别器不一致')
        with np.load(arrays_file, allow_pickle=False) as bank:
            self.features = normalize(bank['features'])
            self.labels = bank['labels'].astype(np.int32)
            self.progress = bank['progress'].astype(np.float32)
            self.sequences = bank['sequences'].astype(np.int32)
        count = len(self.features)
        if (not count or self.features.shape != (count, FEATURE_SIZE) or
                len(self.labels) != count or len(self.progress) != count or
                len(self.sequences) != count or not np.isfinite(self.features).all()):
            raise RuntimeError('隼白板子参考库内容无效')
        self.boards = self.metadata['boards']
        if self.labels.min() < 0 or self.labels.max() >= len(self.boards):
            raise RuntimeError('隼白板子参考库标签越界')
        self.min_confidence = (self.metadata.get('min_confidence', .78) if min_confidence is None
                               else min_confidence)
        self.min_margin = self.metadata.get('min_margin', .055) if min_margin is None else min_margin
        self.groups = [np.flatnonzero(self.labels == index) for index in range(len(self.boards))]
        self.board_families=[]
        for board,group in zip(self.boards,self.groups):
            if board.get('families'):
                self.board_families.append(frozenset(board['families']))
            else:
                energy=np.abs(self.features[group]).reshape(-1,len(KINDS),CHANNEL_SIZE).sum(axis=(0,2))
                self.board_families.append(frozenset(kind for kind,value in zip(KINDS,energy)
                                                     if value>max(energy.max()*.25,1e-6)))
        self.reset()

    def reset(self):
        self.epoch=getattr(self,'epoch',0)+1
        self.history = deque()
        self.last_time = None
        self.candidate = None
        self.candidate_since = None
        self.candidate_hits = 0
        self.candidate_at = None
        self.locked = None
        self.locked_confidence = 0.
        self.locked_margin = 0.
        self.locked_progress = 0.
        self.last_supported = None
        self.quiet_since = None
        self.last_match = BoardMatch()
        self.last_candidate = None

    def update(self, frame, screen, tracks, now):
        if not screen.active or screen.result:
            self.reset()
            return BoardMatch(evidence='inactive, paused, result or unknown screen')
        return self._update_features(formation_features(screen, tracks), now)

    def _update_features(self, feature, now):
        if self.last_time is not None and not 0 < now-self.last_time <= .5:
            self.reset()
        self.last_time = now
        while self.history and now-self.history[0][0] > 2.:
            self.history.popleft()
        if not np.isfinite(feature).all() or np.linalg.norm(feature) < .1:
            self.history.append((now, np.zeros(FEATURE_SIZE, np.float32)))
            self.quiet_since = now if self.quiet_since is None else self.quiet_since
            if self.locked is not None and now-self.quiet_since<.55:
                self.last_match=self._locked_match('confirmed board retained through a short visible attack gap')
                return self.last_match
            ended=self.locked is not None
            self.locked = None
            self.candidate = None
            self.candidate_hits = 0
            self.last_match=BoardMatch(evidence='empty attack geometry; visible board end or unknown formation',
                                      ended=ended,epoch=self.epoch)
            return self.last_match
        self.quiet_since=None
        if not self.history or now-self.history[-1][0] >= .09:
            self.history.append((now, feature))
        else:
            return replace(self.last_match,onset=False,ended=False)
        tokens=np.stack([entry[1] for entry in list(self.history)[-8:]])
        scores=1-np.minimum(sequence_costs(self.features,tokens,self.sequences),
                            sequence_costs(self.features,mirror_features(tokens),self.sequences))
        best_rows = [int(group[np.argmax(scores[group])]) if len(group) else -1 for group in self.groups]
        class_scores = np.array([scores[row] if row >= 0 else -1 for row in best_rows])
        order = np.argsort(class_scores)
        label = int(order[-1])
        confidence = float(class_scores[label])
        second = float(class_scores[order[-2]]) if len(order) > 1 else 0.
        margin = confidence-second
        self.last_candidate={'board_id':self.boards[label]['id'],'confidence':confidence,'margin':margin}
        if self.locked is not None:
            own_score=class_scores[self.locked]
            if own_score>=.65:
                self.last_supported=now
            elif now-self.last_supported>.8:
                self.locked=None
        if confidence < self.min_confidence or margin < self.min_margin:
            if self.locked is not None:
                if self.candidate!=self.locked:
                    self.candidate=None
                    self.candidate_hits=0
                self.last_match=self._locked_match('confirmed board continuation; current formation alone is ambiguous')
            else:
                self.last_match=BoardMatch(confidence=confidence, margin=margin,
                                          evidence='formation ambiguous or outside reference coverage')
            return self.last_match
        switching=self.locked is not None and label!=self.locked
        overlapping=bool(switching and self.board_families[label]&self.board_families[self.locked])
        if switching and (confidence < (.90 if overlapping else .86) or
                          margin < (.15 if overlapping else .10) or
                          confidence-float(own_score) < (.15 if overlapping else .12)):
            self.candidate=None
            self.candidate_hits=0
            self.last_match=self._locked_match('confirmed board retained; competing board evidence is too weak')
            return self.last_match
        if self.candidate != label or self.candidate_at is None or now-self.candidate_at>.65:
            self.candidate, self.candidate_since, self.candidate_hits = label, now, 1
        else:
            self.candidate_hits += 1
        self.candidate_at=now
        required_hits=(5 if overlapping else 4) if switching else 3
        required_span=(.4 if overlapping else .25) if switching else .12
        if self.candidate_hits < required_hits or now-self.candidate_since < required_span:
            self.last_match=(self._locked_match('confirmed board retained while a different candidate is checked')
                if self.locked is not None else BoardMatch(confidence=confidence, margin=margin,
                evidence='candidate formation awaiting temporal confirmation'))
            return self.last_match
        onset=self.locked!=label
        if onset:
            self.epoch+=1
        self.locked=label
        self.locked_confidence,self.locked_margin=confidence,margin
        self.locked_progress=float(self.progress[best_rows[label]])
        self.last_supported=now
        self.last_match=self._locked_match('ordered attack-sequence match; phase follows confirmed board',onset=onset)
        return self.last_match

    def _locked_match(self,evidence,onset=False):
        board = self.boards[self.locked]
        return BoardMatch(board_id=board['id'],phase=board.get('phase'),name=board['name'],
            confidence=self.locked_confidence,margin=self.locked_margin,progress=self.locked_progress,
            phase_source=board.get('phase_evidence', 'board_reference_inference'),
            evidence=evidence,locked=True,onset=onset,epoch=self.epoch)
