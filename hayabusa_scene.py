"""Automatic Hayabusa scene; shared by the assistant and manual diagnostics."""
from collections import Counter, deque
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from hayabusa_policy import HayabusaPolicy
from hayabusa_fixed_opening import FixedOpening
from hayabusa_windup import FireSetupCue
from hayabusa_vision import HayabusaScreen, HayabusaVision
from vision import ASSETS


RECORDINGS = ((Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False)
               else Path(__file__).resolve().parent / 'data') / 'hayabusa-trials')


@dataclass(frozen=True)
class SceneDecision:
    claimed: bool = True
    active: bool = False
    result: str | None = None
    action: object = None
    status: str = ''
    backend: str | None = None
    state: HayabusaScreen | None = None
    stalled: bool = False


class HayabusaScene:
    def __init__(self, *, clock=None, debug=False, record_root=None, policy=None):
        cv2.setNumThreads(1)
        self.clock = clock
        self.debug = debug
        self.record_root = Path(record_root) if record_root is not None else RECORDINGS
        self.vision = None
        self.policy = policy
        self.owned = False
        self.log = self.stream = self.pending_record = None
        self.frames = self.clicks = self.suggestions = self.confirmed = 0
        self.kinds, self.boards, self.phases = Counter(), Counter(), Counter()
        self.costs = deque(maxlen=10000)
        self.timing_costs = {name: deque(maxlen=10000) for name in ('capture', 'vision', 'policy', 'input')}
        self.last_pixels = None
        self.changed_at = None
        self.gates = {}
        pair = os.environ.get('NINJA_FIRE_PAIR') == '1'
        self.fire_probe = (FixedOpening(max_start_age=.8, ack_timeout=.65, fire_probe_delay=.55,
                                        fire_air_jump_delay=.30 if pair else None,
                                        fire_air_jump_deadline=.70 if pair else None)
                           if os.environ.get('NINJA_FIRE_PROBE') == '1' else None)
        self.fire_cue = FireSetupCue() if self.fire_probe is not None else None
        self.shuri_probe = os.environ.get('NINJA_SHURI_PROBE') == '1'
        self.probe_only = os.environ.get('NINJA_PROBE_ONLY') == '1'
        self.fire_probe_attempted = False
        self.fire_probe_cue = None
        for name in ('name', 'portrait', 'shadow-name', 'shadow-portrait', 'victory',
                     'result-name', 'shadow-result-name', 'defeat'):
            template = cv2.imread(str(ASSETS/f'hayabusa-{name}.png'), cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise RuntimeError('缺少隼白场景识别模板：'+name)
            self.gates[name] = template
        if debug:
            self._initialize()

    @property
    def backend(self):
        return self.vision.backend if self.vision is not None else None

    def _initialize(self):
        if self.vision is None:
            self.vision = HayabusaVision()
            if self.policy is None:
                self.policy = HayabusaPolicy()

    def _eligible(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if HayabusaVision._has_hud(gray, self.gates):
            return True
        return HayabusaVision._result(gray, self.gates) is not None

    def _start_log(self, now):
        if self.stream is not None:
            return
        self.record_root.mkdir(parents=True, exist_ok=True)
        self.log = self.record_root/('hayabusa-'+time.strftime('%Y%m%d-%H%M%S')+'-'+str(time.time_ns()%1000000)+'.jsonl')
        self.stream = self.log.open('w', encoding='utf-8')
        self.started = now
        self.stream.write(json.dumps({'event':'start','debug':self.debug,'backend':self.backend,
            'scope':'Scene-driven board recognition and guarded actions; live clear unverified.'},ensure_ascii=False)+'\n')

    def _flush(self):
        if self.stream is not None and self.pending_record is not None:
            self.stream.write(json.dumps(self.pending_record,ensure_ascii=False)+'\n')
            self.pending_record = None
            self.stream.flush()

    def step(self, frame, now):
        if not isinstance(frame, np.ndarray) or frame.shape != (720, 1280, 3):
            return None
        if not self.owned and not self.debug:
            if not self._eligible(frame):
                return None
            self.owned = True
        self._initialize()
        self._flush()
        before = time.perf_counter()
        state, tracks = self.vision.detect(frame, now)
        vision_ms = (time.perf_counter()-before)*1000
        if state.active or state.result:
            self.owned = True
        if not self.owned and not self.debug:
            return None
        self._start_log(now)
        self.frames += 1
        pixels = frame[100:550:8,120:1160:8]
        if self.last_pixels is None or not np.array_equal(pixels,self.last_pixels):
            self.changed_at = now
        self.last_pixels = pixels.copy()
        stalled = now-self.changed_at >= .25
        self.confirmed = self.confirmed+1 if state.active and not stalled else 0
        policy_started = time.perf_counter()
        deciding = self.clock() if self.clock is not None else now+(time.perf_counter()-before)
        action = None
        action_source = None
        if self.fire_probe is not None:
            if state.active and not state.result and not stalled and self.confirmed >= 3:
                if not self.fire_probe_attempted:
                    if (self.shuri_probe and state.board_id in FixedOpening.CLASSES and
                            state.board_locked and state.board_confidence >= .8 and
                            state.board_margin >= .1 and state.board_progress <= .3):
                        self.fire_probe_cue = {'family':'shuriken', 'response_class':state.board_id,
                            'observed_at':now, 'experimental':True,
                            'scope':'late formation match; subtype not independently verified'}
                        self.fire_probe_attempted = True
                        if self.fire_probe.start((state.board_epoch, now), state.board_id, now, deciding):
                            self.policy.suspend(deciding)
                    else:
                        cue = self.fire_cue.update(frame, state, now)
                        if cue is not None:
                            self.fire_probe_cue = asdict(cue)
                            self.fire_probe_attempted = True
                            if self.fire_probe.start(cue.first_observed_at, 'fire_probe', cue.observed_at, deciding):
                                self.policy.suspend(deciding)
                fixed_active = self.fire_probe.state in ('ready', 'proposed', 'waiting_ack',
                    'waiting_air_jump', 'proposed_air_jump')
                action = self.fire_probe.decide(state, deciding, now)
                if action is not None:
                    action_source = 'fixed'
                if not fixed_active and not self.probe_only:
                    action = self.policy.decide(state, tracks, deciding)
                    if action is not None:
                        action_source = 'generic'
            elif not state.active or state.result or stalled:
                self.fire_probe.abort('battle not active or frame stalled')
                self.fire_cue.reset()
        elif self.confirmed >= 3:
            action = self.policy.decide(state,tracks,deciding)
            if action is not None:
                action_source = 'generic'
        if state.result:
            self.policy.reset(preserve_weapon=True,preserve_jumps=True)
        elif not state.active or stalled:
            self.policy.suspend(deciding)
        policy_ms = (time.perf_counter()-policy_started)*1000
        elapsed = (time.perf_counter()-before)*1000
        self.costs.append(elapsed)
        self.timing_costs['vision'].append(vision_ms)
        self.timing_costs['policy'].append(policy_ms)
        self.kinds.update(item.kind for item in tracks)
        if state.board_id:
            self.boards.update((state.board_id,))
        if state.phase_hint:
            self.phases.update((state.phase_hint,))
        recipe = getattr(self.policy,'recipe',None)
        self.pending_record = {'time':round(now-self.started,4),'capture_time':round(now,4),'screen':asdict(state),
            'tracks':[asdict(item) for item in tracks],
            'suggested_action':asdict(action) if action else None,'action_source':action_source,
            'sent':False,'sent_time':None,
            'analysis_ms':elapsed,'stalled':stalled,'touch_error':None,
            'reason':self.fire_probe.reason if action_source == 'fixed' else self.policy.last_reason,
            'timing_ms':{'capture':None,'vision':vision_ms,'policy':policy_ms,'input':None},
            'recipe':None if recipe is None else {'id':recipe.recipe_id,'step':recipe.stage,
                'wave':recipe.wave,'mode_active':recipe.mode_active,'mode_state':recipe.mode_state,
                'reason':recipe.last_reason}}
        if self.fire_probe is not None:
            self.pending_record['fire_probe'] = {'cue':self.fire_probe_cue,
                'state':self.fire_probe.state, 'reason':self.fire_probe.reason,
                'suggested_button':None if action is None else asdict(action),
                'on_pressed_at':None,
                'parameters':{'delay_s':self.fire_probe.fire_probe_delay,'max_start_age_s':.8,'ack_timeout_s':.65,
                    'air_jump_delay_s':self.fire_probe.fire_air_jump_delay,
                    'air_jump_deadline_s':self.fire_probe.fire_air_jump_deadline,
                              'scope':'experimental values; not calibrated'}}
        if state.result:
            message = '隼白战斗已结束，等待选择下一场景'
        elif not state.active:
            message = '等待隼白场景恢复或切换到已识别界面'
        elif stalled:
            message = '隼白画面停滞，等待恢复'
        elif state.player_x is None or state.side == 'unknown':
            message = '隼白场景已启用，等待重新定位角色'
        else:
            label = ((state.phase_hint or '阶段待判别')+' '+state.board_name) if state.board_id else '板子待判别，实时轨迹躲避'
            if recipe is not None and recipe.mode_active:
                label += ('；中途进入，当前波次避让' if recipe.late_entry else f'；流程第 {recipe.stage+1} 步')
            message = '隼白自动战斗：'+label
        if elapsed > 200:
            if self.fire_probe is not None and action is not None and action_source == 'fixed':
                self.fire_probe.rejected()
                self.pending_record['fire_probe']['state'] = self.fire_probe.state
                self.pending_record['fire_probe']['reason'] = 'rejected: analysis exceeded 200ms'
            action = None
            message = '隼白当前帧处理过慢，跳过动作'
        return SceneDecision(True,state.active,state.result,action,message,self.backend,state,stalled)

    def completed(self, action, now):
        self._accept(action,now,True)

    def observe(self, action, now):
        self._accept(action,now,False)

    def _accept(self, action, now, sent):
        if self.pending_record is None:
            return
        if self.fire_probe is not None and self.pending_record.get('action_source') == 'fixed':
            if sent:
                accepted = self.fire_probe.completed(action, now)
                self.pending_record['fire_probe']['on_pressed_at'] = now if accepted else None
                self.pending_record['fire_probe']['state'] = self.fire_probe.state
                self.pending_record['fire_probe']['reason'] = self.fire_probe.reason
            else:
                self.fire_probe.rejected()
                self.pending_record['fire_probe']['state'] = self.fire_probe.state
                self.pending_record['fire_probe']['reason'] = 'observe only; no press'
        self.policy.completed(action,now)
        self.suggestions += 1
        self.clicks += int(sent)
        self.pending_record['sent'] = sent
        self.pending_record['sent_time'] = round(now-self.started,4) if sent else None
        self.pending_record['simulated'] = not sent

    def failed_touch(self, error):
        if (self.fire_probe is not None and self.pending_record is not None and
                self.pending_record.get('action_source') == 'fixed' and
                self.fire_probe.state in ('proposed', 'proposed_air_jump')):
            self.fire_probe.rejected()
        if self.pending_record is not None:
            self.pending_record['touch_error'] = str(error)

    def frame_age(self, seconds):
        if self.pending_record is not None:
            self.pending_record['frame_age_ms'] = seconds*1000
            if (seconds > .2 and self.fire_probe is not None and
                    self.pending_record.get('action_source') == 'fixed' and
                    self.fire_probe.state in ('proposed', 'proposed_air_jump')):
                self.fire_probe.rejected()
                self.pending_record['fire_probe']['state'] = self.fire_probe.state
                self.pending_record['fire_probe']['reason'] = 'rejected: frame age exceeded 200ms'

    def capture_duration(self, seconds):
        self._duration('capture', seconds)

    def input_duration(self, seconds):
        self._duration('input', seconds)

    def _duration(self, name, seconds):
        if self.pending_record is not None:
            self.pending_record['timing_ms'][name] = seconds*1000
            self.timing_costs[name].append(seconds*1000)

    def reset(self, *, new_battle=False):
        if self.fire_probe is not None:
            self.fire_probe.abort('scene reset')
            self.fire_cue.reset()
            if new_battle:
                self.fire_probe_attempted = False
                self.fire_probe_cue = None
        if self.vision is not None:
            self.vision.reset()
            if new_battle:
                self.policy = HayabusaPolicy()
            else:
                self.policy.reset(preserve_weapon=True,preserve_jumps=True)
        self.confirmed = 0
        self.last_pixels = None

    def release(self):
        self.close()
        self.owned = False
        self.reset(new_battle=True)
        self.frames = self.clicks = self.suggestions = 0
        self.kinds.clear()
        self.boards.clear()
        self.phases.clear()
        self.costs.clear()
        for values in self.timing_costs.values():
            values.clear()

    def close(self):
        if self.stream is None:
            return
        self._flush()
        self.stream.write(json.dumps({'event':'summary','frames':self.frames,'clicks':self.clicks,
            'suggestions':self.suggestions,'track_observations':dict(self.kinds),
            'board_observations':dict(self.boards),'phase_observations':dict(self.phases),
            'analysis_ms_p50':float(np.percentile(self.costs,50)) if self.costs else None,
            'analysis_ms_p95':float(np.percentile(self.costs,95)) if self.costs else None,
            'timing_ms':{name:{'count':len(values),'p50':float(np.percentile(values,50)),
                'p95':float(np.percentile(values,95))} if values else {'count':0,'p50':None,'p95':None}
                for name,values in self.timing_costs.items()},
            'timing_scope':'input is caller-side dispatch duration: Python SDK includes hold/release; '
                           'C++ ends at press acknowledgement or cancellation completion; '
                           'neither measures visible game response'},ensure_ascii=False)+'\n')
        self.stream.close()
        self.stream = None
