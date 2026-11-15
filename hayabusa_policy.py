"""Experimental, closed-loop dodging for Hayabusa; movement needs live calibration."""
import math
from types import SimpleNamespace

from boss_policy import BossAction
from hayabusa_board_policy import BoardDodgePolicy


class HayabusaPolicy:
    # Screen-space estimates, not timings copied from the slowed teaching video.
    REACTION = .06
    FLIP_TIME = .18
    GRAVITY = 1400.
    JUMP_SPEED = 650.
    HORIZON = .6
    MAX_JUMPS = 4
    ATTACKS = frozenset(('fire', 'blackfire', 'shuriken', 'slash', 'flash'))

    def __init__(self):
        self.recipe = BoardDodgePolicy()
        self.reset()

    def reset(self, preserve_weapon=False, preserve_jumps=False):
        weapon_used = getattr(self, 'weapon_used', False) if preserve_weapon else False
        jumps = getattr(self, 'jumps', 0) if preserve_jumps else 0
        last_jump = getattr(self, 'last_jump', float('-inf')) if preserve_jumps else float('-inf')
        self.previous = None
        self.velocity = (0., 0.)
        self.last_action = float('-inf')
        self.last_jump = last_jump
        self.jumps = jumps
        self.grounded_since = None
        self.pending_flip = None
        self.weapon_used = weapon_used
        self.weapon_missing = None
        self.last_reason = 'waiting for confirmed battle and player'
        self.recipe.reset()

    def suspend(self, now):
        """Discard motion estimates while a short unconfirmed frame is observed."""
        self.previous = None
        self.velocity = (0., 0.)
        self.pending_flip = None
        self.grounded_since = None
        self.weapon_missing = None
        self.recipe.suspend(now)
        self.last_reason = 'battle, rope or player unconfirmed'

    def decide(self, screen, tracks, now):
        if (not screen.active or screen.result or screen.rope_y is None or
                screen.side not in ('above', 'below') or
                screen.player_x is None or screen.player_y is None):
            self.suspend(now)
            if screen.result:
                self.recipe.reset()
            return None
        if self.previous:
            dt = now - self.previous[2]
            if dt <= 0:
                self.reset()
            elif dt > .35:
                self.velocity = (0., 0.)
                self.pending_flip = None
                self.grounded_since = None
            else:
                self.velocity = (max(-250., min(250., (screen.player_x - self.previous[0]) / dt)),
                                 max(-1000., min(1000., (screen.player_y - self.previous[1]) / dt)))
        self.previous = (screen.player_x, screen.player_y, now)
        grounded = getattr(screen, 'grounded', not screen.airborne) is True
        if grounded:
            if self.grounded_since is None:
                self.grounded_since = now
        else:
            self.grounded_since = None
        if screen.side == 'below' or (self.grounded_since is not None and
                now-self.grounded_since >= .06 and now-self.last_jump > .22):
            self.jumps = 0
        if screen.weapon_ready:
            self.weapon_missing = None
        elif self.weapon_missing is None:
            self.weapon_missing = now
        elif now - self.weapon_missing >= 1:
            self.weapon_used = False
        match = SimpleNamespace(board_id=getattr(screen, 'board_id', None),
            phase=getattr(screen, 'phase_hint', None), confidence=getattr(screen, 'board_confidence', 0),
            margin=getattr(screen, 'board_margin', 0), progress=getattr(screen, 'board_progress', 0),
            evidence=getattr(screen, 'board_evidence', ''), locked=getattr(screen, 'board_locked', False),
            onset=getattr(screen, 'board_onset', False), ended=getattr(screen, 'board_ended', False),
            epoch=getattr(screen, 'board_epoch', 0))
        proposal = self.recipe.decide(screen, tracks, match, now, self.jumps)
        if proposal and not self._valid_movement(screen, proposal.kind):
            proposal = None
        if self.pending_flip:
            if screen.side == self.pending_flip:
                self.pending_flip = None
            elif now - self.last_action < .35:
                self.last_reason = 'waiting for rope-side confirmation'
                return None
            else:
                self.pending_flip = None
        if now - self.last_action < .16:
            self.last_reason = 'movement cooldown'
            return None

        threats = [track for track in tracks if self._usable(track, now)]
        baseline, threat_id, exposure = self._clearance(screen, threats, now, 'wait')
        holding_mode = (proposal is None and getattr(self.recipe, 'mode_active', False))
        if getattr(self.recipe, 'hold_for_approach', False) or holding_mode:
            immediate, _, _ = self._clearance(screen, threats, now, 'wait', horizon=.24)
            if immediate >= 1.15:
                self.last_reason = self.recipe.last_reason
                return None
        if proposal:
            clearance, _, proposed_exposure = self._clearance(screen, threats, now, proposal.kind)
            if baseline >= 1.2 and clearance >= 1.2 and proposed_exposure <= .02:
                self.last_reason = self.recipe.last_reason
                return proposal
        if baseline < 1.2:
            kinds = ['up', 'roll'] if screen.side == 'below' else ['drop' if screen.airborne else 'down']
            if screen.side == 'above' and self.jumps < self.MAX_JUMPS:
                kinds.append('jump')
            kinds = [kind for kind in kinds if self._valid_movement(screen, kind)]
            if not kinds:
                self.last_reason = 'movement or landing state unconfirmed'
                return None
            assessments = [(kind, self._clearance(screen, threats, now, kind)) for kind in kinds]
            candidates = [(-item[2] + .05*item[0] - (.04 if kind == 'jump' else .02), kind)
                          for kind, item in assessments]
            score, kind = candidates[0] if assessments[0][1][0] >= 1.3 else max(candidates)
            if proposal:
                clearance, _, candidate_exposure = self._clearance(screen, threats, now, proposal.kind)
                recipe_score = -candidate_exposure+.05*clearance-(.04 if proposal.kind == 'jump' else .02)
                if recipe_score >= score-.04 and recipe_score > -exposure+.05*baseline+.02:
                    self.last_reason = self.recipe.last_reason
                    return proposal
            if score > -exposure + .05*baseline + .02:
                self.last_reason = f'predicted crossing; {kind} improves clearance'
                point = (1090, 630) if kind in ('jump', 'up') else (190, 630)
                return BossAction(kind, point, threat_id)
            self.last_reason = 'no clearly safer movement predicted'
            return None
        if (screen.weapon_ready and screen.weapon_point is not None and not self.weapon_used
                and baseline >= 1.5 and now - self.last_action >= .35 and
                not self.recipe.pending and not (match.board_id or '').endswith('_transition')):
            x, y = screen.weapon_point
            if 0 <= x < 1280 and 0 <= y < 720:
                self.last_reason = 'weapon ready; no nearby predicted crossing'
                return BossAction('weapon', (int(x), int(y)))
        self.last_reason = 'tracking; no imminent crossing'
        return None

    def _valid_movement(self, screen, kind):
        if kind == 'jump':
            return screen.side == 'above' and self.jumps < self.MAX_JUMPS
        if kind == 'up':
            return screen.side == 'below'
        if kind == 'roll':
            return screen.side == 'below'
        if kind == 'drop':
            return screen.side == 'above' and screen.airborne
        grounded = getattr(screen, 'grounded', not screen.airborne) is True
        return kind == 'down' and screen.side == 'above' and not screen.airborne and grounded

    @classmethod
    def _usable(cls, track, now):
        values = (track.x, track.y, track.width, track.height, track.vx, track.vy, track.seen)
        return (track.kind in cls.ATTACKS and track.hits >= 2 and
                0 <= now - track.seen <= .18 and all(math.isfinite(x) for x in values) and
                track.width > 0 and track.height > 0 and
                abs(track.vx) <= 6000 and abs(track.vy) <= 6000)

    def _position(self, screen, kind, t):
        px = screen.player_x + self.velocity[0] * min(t, .15)
        if screen.side == 'above' and screen.airborne:
            base_y = min(screen.rope_y - 32,
                         screen.player_y + self.velocity[1] * t + .5 * self.GRAVITY * t * t)
        else:
            base_y = screen.player_y
        if kind == 'wait' or t <= self.REACTION:
            return px, base_y
        elapsed = t - self.REACTION
        if kind == 'jump':
            py = screen.player_y - self.JUMP_SPEED * elapsed + .5 * self.GRAVITY * elapsed * elapsed
            return px, max(90., min(screen.rope_y - 32, py))
        if kind == 'drop':
            return px, min(screen.rope_y-32, screen.player_y+1200*elapsed)
        if kind == 'roll':
            fraction = max(0., 1.-abs(elapsed-.12)/.18)
            return px, screen.player_y+35*fraction
        target_y = screen.rope_y + (34 if kind == 'down' else -34)
        fraction = min(1., elapsed / self.FLIP_TIME)
        return px, screen.player_y + (target_y - screen.player_y) * fraction

    def _clearance(self, screen, tracks, now, kind, horizon=None):
        nearest, ident = 3., None
        exposure = [0.] * 18
        horizon = self.HORIZON if horizon is None else horizon
        for track in tracks:
            # Foreground/background shuriken must already have been filtered by vision.
            rx, ry = track.width / 2 + 18, track.height / 2 + 24
            for step in range(3, 21):
                t = self.REACTION+(horizon-self.REACTION)*(step-2)/18
                px, py = self._position(screen, kind, t)
                age = now - track.seen + t
                tx, ty = track.x + track.vx * age, track.y + track.vy * age
                distance = max(abs(tx - px) / rx, abs(ty - py) / ry)
                exposure[step-3] = max(exposure[step-3], max(0., 1.-distance))
                if distance < nearest:
                    nearest, ident = distance, track.id
        return nearest, ident, sum(exposure) / len(exposure)

    def completed(self, action, now):
        self.recipe.completed(action, now)
        self.last_action = now
        if action.kind == 'jump':
            self.jumps = min(self.MAX_JUMPS, self.jumps+1)
            self.last_jump = now
        elif action.kind in ('up', 'down'):
            self.pending_flip = 'above' if action.kind == 'up' else 'below'
        elif action.kind == 'weapon':
            self.weapon_used = True
