"""Visual-event recipes for the boards shown in the Hayabusa teaching video.

Recipes only propose inputs. The caller must compare their future collision risk,
enforce the character's jump budget, and call completed only for an accepted input.
No recipe schedules inputs from elapsed teaching-video time.
"""
from dataclasses import dataclass
import math

from ninja3_assistant.combat.boss.policy import BossAction


@dataclass(frozen=True)
class BoardRecipe:
    board_id: str
    phase: str
    mode: str
    actions: tuple[str, ...]
    kinds: tuple[str, ...]
    evidence: str
    repeat: bool = True
    calibration_needed: bool = True


RECIPES = {item.board_id: item for item in (
    BoardRecipe('p1_fire_1', 'P1', 'late_triple', ('jump', 'jump', 'jump'), ('fire',),
                '00:30: third group, after the first two flames descend, three jumps'),
    BoardRecipe('p1_fire_2', 'P1', 'gap', (), ('fire',),
                '00:38: visible demonstration; exact input sequence unconfirmed'),
    BoardRecipe('p1_fire_3', 'P1', 'cycle', ('land', 'jump', 'land'), ('fire',),
                '00:56: descend through gap, jump through gap, descend again'),
    BoardRecipe('p1_fire_4', 'P1', 'first_gap_then_flip', ('below', 'above'), ('fire',),
                '01:10: first group uses gap, second group alternates rope sides'),
    BoardRecipe('p1_sword_1', 'P1', 'gap_jump', (), ('slash',),
                '01:24: jump during descent; exact meaning of double-jump cue needs live calibration'),
    BoardRecipe('p1_sword_2', 'P1', 'tracking', ('jump', 'below', 'above'), ('slash',),
                '01:36: tracking sword, move the character to change its target'),
    BoardRecipe('p1_sword_3', 'P1', 'cycle', ('jump', 'below', 'above'), ('slash',),
                '01:48: last three groups use high jumps and downward flips'),
    BoardRecipe('p1_shuriken_1', 'P1', 'gap', (), ('shuriken',),
                '02:10: fixed routes; select the observed safe gap'),
    BoardRecipe('p1_shuriken_2', 'P1', 'cycle', ('below', 'above'), ('shuriken',),
                '02:08: begin below the rope and pass through gaps'),
    BoardRecipe('p1_shuriken_3', 'P1', 'cycle', ('jump', 'below', 'roll', 'above'), ('shuriken',),
                '02:16: small jump and roll, repeat'),
    BoardRecipe('p1_shuriken_4', 'P1', 'cycle', ('below', 'roll', 'above', 'land'), ('shuriken',),
                '02:44: start with downward flip, descend through first and second gaps, repeat'),
    BoardRecipe('p1_shuriken_34_variant', 'P1', 'gap', (), ('shuriken',),
                '03:10: similar third/fourth variants must not share a blind input replay'),
    BoardRecipe('p1_flash', 'P1', 'tracking', ('below', 'above'), ('flash',),
                '03:24: a small position change avoids the flash'),
    BoardRecipe('p1_flash_shuriken', 'P1', 'cycle', ('below', 'above', 'jump'), ('flash', 'shuriken'),
                '03:28: first group switches sides, second group flips up then small jump'),
    BoardRecipe('p2_transition', 'P2', 'middle_swords', ('below', 'above', 'below', 'above'), ('slash',),
                '03:38: two downward flips, watch the middle two sword waves', repeat=False),
    BoardRecipe('p2_blackfire_1', 'P2', 'cycle', ('below', 'above'), ('blackfire',),
                '03:50: two groups and one group alternate'),
    BoardRecipe('p2_blackfire_2', 'P2', 'gap', (), ('blackfire',),
                '04:00: mirrored attack presentation; exact full input sequence unconfirmed'),
    BoardRecipe('p2_blackfire_tracking_short', 'P2', 'tracking', ('jump', 'below', 'above'), ('blackfire',),
                '04:20: jump to bait, flip down when black fire is about to reach the character'),
    BoardRecipe('p2_blackfire_tracking_long', 'P2', 'tracking_gap', (), ('blackfire', 'slash'),
                '04:30: jump through gaps; author explicitly reports an element of chance'),
    BoardRecipe('p2_tracking_shuriken', 'P2', 'cycle', ('below', 'above', 'jump'), ('shuriken',),
                '04:40: demonstrated after a hit; not evidence of a no-hit solution'),
    BoardRecipe('p2_illusion_shuriken_1', 'P2', 'depth_gap', (), ('shuriken',),
                '04:55: only the middle visual depth layer is damaging according to author'),
    BoardRecipe('p2_illusion_shuriken_2', 'P2', 'depth_gap', (), ('shuriken',),
                '05:05: vertical three-shot pattern; observe the middle-size layer'),
    BoardRecipe('p2_fork_sword', 'P2', 'cycle', ('below', 'above'), ('slash',),
                '05:25: alternate rope sides to avoid the three-pronged swords'),
    BoardRecipe('p2_snake_sword', 'P2', 'gap_jump', (), ('slash',),
                '05:40: second group jumps through the gap'),
    BoardRecipe('p2_tracking_sword', 'P2', 'tracking_gap', (), ('slash',),
                '05:55: one jump, fall into gap; repeated slow-motion demonstration'),
    BoardRecipe('p3_transition', 'P3', 'tracking', ('jump', 'below', 'roll', 'above'), ('flash',),
                '06:20: flash tracks the character; repeatedly change position'),
    BoardRecipe('p3_ground_slam_1', 'P3', 'cycle', ('jump', 'land', 'below', 'above'), ('slash', 'flash'),
                '06:27 onward: ground slam demonstration; collision timing needs live calibration'),
    BoardRecipe('p3_ground_slam_2', 'P3', 'cycle', ('jump', 'land', 'below', 'above'), ('slash', 'flash'),
                '06:45: author says the dodge is the same as ground-slam one'),
    BoardRecipe('p3_orbit_shuriken', 'P3', 'orbit', ('below', 'above'), ('shuriken', 'flash'),
                '07:10: first two groups share a dodge, final group needs one down flip'),
    BoardRecipe('p3_fire_sword_1', 'P3', 'tracking_gap', (), ('blackfire', 'slash'),
                '07:25: compound black-fire and sword demonstration'),
    BoardRecipe('p3_fire_sword_2', 'P3', 'tracking_gap', (), ('blackfire',),
                '07:40: author follows black fire and ignores flash in this particular board'),
    BoardRecipe('p3_fire_sword_3', 'P3', 'gap', (), ('blackfire', 'slash'),
                '08:00: wide gaps; compare the current actual threat geometry'),
    BoardRecipe('p4_transition', 'P4', 'two_jumps', ('jump', 'land', 'jump', 'land'), ('slash', 'flash'),
                '08:40: two separate one-jumps through gaps', repeat=False),
    BoardRecipe('p4_tracking_compound', 'P4', 'tracking', ('jump', 'below', 'roll', 'above'), ('blackfire', 'slash', 'flash'),
                '09:20: jump to bait, roll below rope to bait, repeat, then jump/down through final swords'),
    BoardRecipe('p4_clone_flash', 'P4', 'cycle', ('below', 'above'), ('flash',),
                '10:05: fixed clone flash, not tracking'),
    BoardRecipe('p4_clone_fire', 'P4', 'tracking', ('jump', 'below', 'roll', 'above'), ('blackfire',),
                '10:20: large jumps bait fire, rise quickly again, avoid prolonged hovering'),
)}


class BoardDodgePolicy:
    MAX_JUMPS = 4
    MIN_CONFIDENCE = .72
    MIN_MARGIN = .04
    # A conservative alignment estimate from reference formation progress, not
    # an elapsed-video/input timer. Late confirmation cannot restart step zero.
    LATE_ENTRY_PROGRESS = .45
    ROUND_LATE_PROGRESS = .65

    def __init__(self, allow_air_drop=True):
        self.allow_air_drop = allow_air_drop
        self.reset()

    def reset(self):
        blocked_epoch = getattr(self, 'blocked_epoch', None)
        blocked_board = getattr(self, 'blocked_board', None)
        if getattr(self, 'mode_active', False) and getattr(self, '_locked_protocol', False):
            blocked_epoch, blocked_board = self.mode_epoch, self.recipe_id
        # Losing execution feedback cannot make the same recognizer epoch a new
        # attack. Preserve its retirement marker through further resets/occlusion.
        self.blocked_epoch, self.blocked_board = blocked_epoch, blocked_board
        self.recipe_id = None
        self.stage = 0
        self.wave = 0
        self.pending = None
        self.hold_for_approach = False
        self.mode_active = False
        self.mode_state = 'idle'
        self.mode_epoch = None
        self.late_entry = False
        self.done = False
        self.last_reason = 'waiting for confident board recognition'
        self._candidate = None
        self._stable = 0
        self._last_time = None
        self._last_action = float('-inf')
        self._last_proposal = None
        self._wave_ids = set()
        self._cycle_ids = set()
        self._wait_new_cycle = False
        self._descended_ids = set()
        self._locked_protocol = False
        self._last_locked_at = None
        self._last_player_at = None
        self._occluded_since = None
        self._onset_ready = False
        self._chain_ready = False
        self._late_round_ids = set()
        self._late_quiet_since = None
        self._late_round_clear = False
        self._late_max_progress = 0.

    def suspend(self, now):
        """Wait briefly for a fresh player observation; never confirm movement here."""
        self.hold_for_approach = False
        self._last_proposal = None
        if not self.mode_active:
            self.reset()
            return
        if self._occluded_since is None:
            self._occluded_since = self._last_player_at if self._last_player_at is not None else now
        if now-self._occluded_since > .35 or now < self._occluded_since:
            self.reset()
            self.mode_state = 'resync_required'
            self.last_reason = 'player occlusion exceeded the short wait; resynchronize from current screen'
            return
        self._last_time = now
        self.mode_state = 'late_entry_occluded' if self.late_entry else 'occluded'
        self.last_reason = 'attack mode retained briefly; player hidden, no input or movement confirmation'

    def decide(self, screen, tracks, match, now, jumps=0):
        self.hold_for_approach = False
        if self._last_time is not None and not 0 < now-self._last_time <= .5:
            self.reset()
        self._last_time = now
        if not screen.active or screen.result:
            self.reset()
            return None
        protocol = hasattr(match, 'locked')
        epoch = getattr(match, 'epoch', None)
        if getattr(match, 'ended', False):
            self.reset()
            self.mode_state = 'ended'
            self.last_reason = 'visible attack mode ended'
            return None
        if self.mode_active and self._locked_protocol:
            if not protocol or epoch != self.mode_epoch:
                self.reset()
                self._last_time = now
            elif getattr(match, 'locked', False):
                self._last_locked_at = now
            elif self._last_locked_at is None or now-self._last_locked_at > .35:
                self.reset()
                self.mode_state = 'resync_required'
                self.last_reason = 'attack lock unsupported beyond the short ambiguity window'
                return None
        if (screen.rope_y is None or screen.player_x is None or screen.player_y is None or
                screen.side not in ('above', 'below')):
            self.suspend(now)
            return None
        if self._occluded_since is not None and now-self._occluded_since > .35:
            self.reset()
            self._last_time = now
        self._occluded_since = None
        self._last_player_at = now
        board = getattr(match, 'board_id', None)
        recipe = RECIPES.get(board)
        phase = getattr(match, 'phase', None)
        if self.mode_active and self._locked_protocol:
            # The recognizer owns the attack identity for this epoch. A single
            # ambiguous formation must not restart a previously confirmed flow.
            recipe = RECIPES[self.recipe_id]
        else:
            if (recipe is None or getattr(match, 'confidence', 0) < self.MIN_CONFIDENCE or
                    getattr(match, 'margin', 0) < self.MIN_MARGIN or
                    (phase is not None and phase != recipe.phase) or
                    (protocol and not getattr(match, 'locked', False))):
                self.reset()
                self.last_reason = 'unknown or ambiguous board; reactive safety policy remains in charge'
                return None
            if protocol:
                if board != self.recipe_id or epoch != self.mode_epoch:
                    self.reset()
                    self._last_time = now
                    self._last_player_at = now
                self._candidate = self.recipe_id = board
                self._locked_protocol = True
                self.mode_epoch = epoch
                if self.blocked_epoch is not None and epoch != self.blocked_epoch:
                    self.blocked_epoch = self.blocked_board = None
                self._last_locked_at = now
                self._onset_ready = self._onset_ready or bool(getattr(match, 'onset', False))
            else:
                if board != self._candidate:
                    self.reset()
                    self._candidate, self._stable, self._last_time = board, 1, now
                    self._last_player_at = now
                    self.mode_state = 'candidate'
                    return None
                self._stable += 1
                if self._stable < 2:
                    return None
                self.recipe_id = board
        if self.pending:
            kind, old_x, old_y, issued, advance, was_airborne = self.pending
            confirmed = ((kind == 'up' and screen.side == 'above') or
                         (kind == 'down' and screen.side == 'below') or
                         (kind == 'jump' and screen.side == 'above' and screen.player_y < old_y-6) or
                         (kind == 'drop' and was_airborne and now-issued >= .02 and self._grounded(screen)) or
                         (kind == 'roll' and screen.side == 'below' and
                          max(abs(screen.player_x-old_x), abs(screen.player_y-old_y)) >= 6))
            if confirmed:
                self.pending = None
                self._chain_ready = True
                if advance:
                    self._advance(recipe)
            elif now-issued > .65:
                self.reset()
                self.mode_state = 'resync_required'
                self.last_reason = 'input movement unconfirmed; discard recipe and resynchronize'
                return None
            else:
                self.mode_state = 'late_entry_waiting_movement' if self.late_entry else 'waiting_movement'
                self.last_reason = 'waiting for observed movement confirmation'
                return None
        if self.done:
            self.mode_state = 'sequence_complete'
            return None
        if now-self._last_action < .16:
            return None
        candidates = self._incoming(screen, tracks, recipe, now)
        near_ids = {track.id for _, track in candidates}
        if near_ids and (not self._wave_ids or near_ids.isdisjoint(self._wave_ids)):
            self.wave += 1
            self._wave_ids = near_ids
        elif near_ids:
            self._wave_ids.update(near_ids)
        elif not any(track.id in self._wave_ids and 0 <= now-track.seen <= .18 for track in tracks):
            self._wave_ids.clear()
        for track in tracks:
            if (track.kind == 'fire' and track.hits >= 2 and 0 <= now-track.seen <= .18
                    and track.vy > 25 and track.y > screen.rope_y+10):
                self._descended_ids.add(track.id)
        if not self.mode_active:
            if not candidates and not self._onset_ready:
                self.mode_state = 'waiting_start'
                self.last_reason = 'board label alone cannot start inputs; wait for visible attack onset'
                return None
            self.mode_active = True
            self._chain_ready = True
            self._onset_ready = False
            progress = getattr(match, 'progress', 0.)
            self.late_entry = ((self.blocked_epoch is not None and self.mode_epoch == self.blocked_epoch) or
                               not math.isfinite(progress) or progress > self.LATE_ENTRY_PROGRESS)
            if self.late_entry:
                self._late_round_ids = {track.id for track in tracks if track.kind in recipe.kinds}
        self.mode_state = 'active'
        if self.late_entry:
            self._resync_late_round(tracks, candidates, match, recipe, now)
            if self.late_entry:
                blocked = self.blocked_epoch is not None and self.mode_epoch == self.blocked_epoch
                state = 'late_entry_blocked' if blocked else 'late_entry'
                self.mode_state = state
                self.last_reason = (('execution feedback lost in this epoch' if blocked else 'late visual confirmation') +
                                    '; use current gaps, not the beginning of the fixed flow')
                if not candidates:
                    return None
                action = self._gap(screen, candidates, now, jumps, prefer_jump=False)
                self.mode_state = state + '_response' if action is not None else state
                self.last_reason = (('same epoch cannot restart fixed flow' if blocked else 'late visual confirmation') +
                                    '; current-side/current-gap response only')
                return action
        if self._wait_new_cycle:
            if not near_ids or not near_ids.isdisjoint(self._cycle_ids):
                self.mode_state = 'waiting_attack'
                self.last_reason = 'cycle complete; waiting for a new visible attack group'
                return None
            self._wait_new_cycle = False
            self._cycle_ids.clear()
            self._chain_ready = True
        threat = candidates[0][1] if candidates else None
        eta = candidates[0][0] if candidates else float('inf')
        if recipe.mode == 'late_triple' and not (self.wave >= 3 and len(self._descended_ids) >= 2):
            return self._gap(screen, candidates, now, jumps, prefer_jump=False) if candidates else None
        if recipe.mode == 'first_gap_then_flip' and self.wave <= 1:
            return self._gap(screen, candidates, now, jumps, prefer_jump=False) if candidates else None
        if recipe.mode in ('gap', 'gap_jump', 'depth_gap', 'tracking_gap'):
            if not candidates:
                self.mode_state = 'waiting_attack'
                self.last_reason = 'attack mode retained; adaptive gap step needs current threat geometry'
                return None
            return self._gap(screen, candidates, now, jumps, prefer_jump=recipe.mode in ('gap_jump', 'tracking_gap'))
        if not candidates and not self._chain_ready:
            self.mode_state = 'waiting_attack'
            return None
        if recipe.mode == 'orbit' and self.wave >= 3:
            intent = 'below'
        else:
            intent = recipe.actions[self.stage]
        if recipe.mode == 'tracking' and intent in ('below', 'roll') and eta > .34:
            self.hold_for_approach = True
            self.mode_state = 'holding_approach'
            self.last_reason = 'baiting tracking attack; wait until it approaches before changing lane'
            return None
        self._cycle_ids.update(near_ids)
        if recipe.mode == 'two_jumps' and intent == 'jump' and screen.side == 'above' and not self._grounded(screen):
            return self._intent('land', screen, threat, now, jumps, advance=False)
        return self._intent(intent, screen, threat, now, jumps)

    def _resync_late_round(self, tracks, candidates, match, recipe, now):
        progress = getattr(match, 'progress', 1.)
        trusted = (getattr(match, 'board_id', None) == recipe.board_id and
                   (not hasattr(match, 'locked') or match.locked) and math.isfinite(progress))
        if trusted:
            self._late_max_progress = max(self._late_max_progress, progress)
        visible = {track.id for track in tracks if track.kind in recipe.kinds and
                   track.hits >= 2 and 0 <= now-track.seen <= .18}
        if not visible:
            if self._late_quiet_since is None:
                self._late_quiet_since = now
            if now-self._late_quiet_since >= .12:
                self._late_round_clear = True
            return
        new_round = (self._late_round_clear and candidates and
                     not visible.intersection(self._late_round_ids) and
                     trusted and self._late_max_progress >= self.ROUND_LATE_PROGRESS and 0 <= progress <= .25)
        self._late_quiet_since = None
        self._late_round_clear = False
        self._late_round_ids.update(visible)
        if new_round:
            self.late_entry = False
            if self.blocked_epoch == self.mode_epoch:
                self.blocked_epoch = self.blocked_board = None
            self.stage = 0
            self.wave = 1
            self._wave_ids = {track.id for _, track in candidates}
            self._cycle_ids.clear()
            self._descended_ids.clear()
            self._wait_new_cycle = False
            self._chain_ready = True
            self.mode_state = 'active'
            self.last_reason = 'visible clear and new early formation; synchronize to a new attack round'

    @staticmethod
    def _grounded(screen):
        if hasattr(screen, 'grounded'):
            return screen.side == 'above' and screen.grounded is True
        return (screen.side == 'above' and not screen.airborne and
                0 <= screen.rope_y-screen.player_y <= 50)

    @staticmethod
    def _incoming(screen, tracks, recipe, now):
        found = []
        for track in tracks:
            if (track.kind not in recipe.kinds or track.hits < 2 or
                    not 0 <= now-track.seen <= .18 or
                    track.width <= 0 or track.height <= 0 or
                    not all(math.isfinite(v) for v in (track.x, track.y, track.vx, track.vy,
                                                      track.width, track.height)) or
                    abs(track.vx) > 6000 or abs(track.vy) > 6000):
                continue
            if recipe.mode == 'depth_gap' and not (18 <= track.width <= 110 and 8 <= track.height <= 75):
                continue
            age = now-track.seen
            dx = track.x+track.vx*age-screen.player_x
            dy = track.y+track.vy*age-screen.player_y
            rx, ry = max(12., track.width/2+22), max(16., track.height/2+28)
            speed = (track.vx/rx)**2+(track.vy/ry)**2
            eta = max(0., min(.9, -(dx*track.vx/rx**2+dy*track.vy/ry**2)/speed)) if speed else 0.
            distance = max(abs(dx+track.vx*eta)/rx, abs(dy+track.vy*eta)/ry)
            if distance <= 2 and eta <= .8:
                found.append((eta, track))
        if recipe.mode == 'middle_swords':
            found = sorted(found, key=lambda item: abs(item[1].y-screen.rope_y))[:2]
        return sorted(found, key=lambda item: item[0])

    def _gap(self, screen, candidates, now, jumps, prefer_jump):
        threat = candidates[0][1]
        if screen.side == 'below':
            intent = 'above'
        elif prefer_jump or threat.y >= screen.rope_y-45:
            intent = 'jump' if jumps < self.MAX_JUMPS else 'land'
        else:
            intent = 'below'
        return self._intent(intent, screen, threat, now, jumps, advance=False)

    def _intent(self, intent, screen, threat, now, jumps, advance=True):
        if intent == 'above':
            if screen.side == 'above':
                if advance:
                    self._advance(RECIPES[self.recipe_id])
                return None
            kind = 'up'
        elif intent == 'below':
            if screen.side == 'below':
                if advance:
                    self._advance(RECIPES[self.recipe_id])
                return None
            if hasattr(screen, 'grounded') and screen.grounded is None and not screen.airborne:
                self.last_reason = 'wait for ground/air control confirmation before down input'
                return None
            kind = 'drop' if screen.airborne or not self._grounded(screen) else 'down'
            advance = advance and kind == 'down'
        elif intent == 'jump':
            if screen.side == 'below':
                kind, advance = 'up', False
            elif jumps >= self.MAX_JUMPS:
                self.last_reason = 'jump budget exhausted; wait for confirmed landing'
                return None
            else:
                kind = 'jump'
        elif intent == 'land':
            if self._grounded(screen):
                if advance:
                    self._advance(RECIPES[self.recipe_id])
                return None
            if hasattr(screen, 'grounded') and screen.grounded is None and not screen.airborne:
                self.last_reason = 'landing/control state unconfirmed; do not advance recipe'
                return None
            kind, advance = ('up', False) if screen.side == 'below' else ('drop', advance)
        elif intent == 'roll':
            if screen.side == 'above':
                kind = 'drop' if screen.airborne or not self._grounded(screen) else 'down'
                advance = False
            else:
                kind = 'roll'
        else:
            return None
        if kind == 'drop' and not self.allow_air_drop:
            self.last_reason = 'wait for natural descent; air drop disabled'
            return None
        point = (1090, 630) if kind in ('jump', 'up') else (190, 630)
        action = BossAction(kind, point, threat.id if threat is not None else None)
        self._last_proposal = (action, now, screen.player_x, screen.player_y, advance, screen.airborne)
        self.mode_state = 'proposing_step'
        self.last_reason = f'{self.recipe_id}: locked attack flow, intent={intent}, step={self.stage}'
        return action

    def _advance(self, recipe):
        self.stage += 1
        if self.stage >= len(recipe.actions):
            if recipe.repeat:
                self.stage = 0
                self._wait_new_cycle = True
                self._chain_ready = False
            else:
                self.done = True

    def completed(self, action, now):
        if self._last_proposal is None:
            return
        proposed, at, x, y, advance, was_airborne = self._last_proposal
        if action != proposed or not 0 <= now-at <= .5:
            return
        self._last_action = now
        self.pending = (action.kind, x, y, now, advance, was_airborne)
        self.mode_state = 'late_entry_waiting_movement' if self.late_entry else 'waiting_movement'
        self._last_proposal = None
