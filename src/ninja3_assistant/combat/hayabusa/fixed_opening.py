"""Single confirmed opening action; event subtype and timing need calibration."""
from ninja3_assistant.combat.boss.policy import BossAction


class FixedOpening:
    """Consume a unique classified onset once; this does not execute a full board."""

    CLASSES = frozenset(('p1_shuriken_2', 'p1_shuriken_4'))

    def __init__(self, *, max_start_age, ack_timeout, fire_probe_delay=None, jump_ack_pixels=15,
                 fire_air_jump_delay=None, fire_air_jump_deadline=None):
        if max_start_age <= 0 or ack_timeout <= 0:
            raise ValueError('positive calibrated limits required')
        self.max_start_age = max_start_age
        self.ack_timeout = ack_timeout
        if fire_probe_delay is not None and not 0 <= fire_probe_delay <= max_start_age:
            raise ValueError('probe delay must fit within the onset window')
        if jump_ack_pixels <= 0:
            raise ValueError('positive jump acknowledgement threshold required')
        self.fire_probe_delay = fire_probe_delay
        self.jump_ack_pixels = jump_ack_pixels
        if (fire_air_jump_delay is None) != (fire_air_jump_deadline is None):
            raise ValueError('air jump requires both delay and deadline')
        if fire_air_jump_delay is not None:
            if fire_probe_delay is None or not 0 < fire_air_jump_delay < fire_air_jump_deadline:
                raise ValueError('air jump requires an enabled fire probe and bounded timing window')
        self.fire_air_jump_delay = fire_air_jump_delay
        self.fire_air_jump_deadline = fire_air_jump_deadline
        self.seen = set()
        self.abort('idle')

    def abort(self, reason='interrupted'):
        self.event_id = None
        self.observed_at = None
        self.pending = None
        self.sent_at = None
        self.response_class = None
        self.initial_y = None
        self.first_sent_at = None
        self.state = 'idle'
        self.reason = reason

    def start(self, event_id, response_class, observed_at, now):
        allowed = response_class in self.CLASSES or (response_class == 'fire_probe' and self.fire_probe_delay is not None)
        if event_id is None or event_id in self.seen or not allowed:
            return False
        if self.state in ('ready', 'proposed', 'waiting_ack', 'waiting_air_jump', 'proposed_air_jump'):
            return False
        self.seen.add(event_id)
        self.abort('event received')
        if not 0 <= now-observed_at <= self.max_start_age:
            self.reason = 'late onset'
            return False
        self.event_id, self.observed_at = event_id, observed_at
        self.response_class = response_class
        self.state = 'ready'
        return True

    def decide(self, screen, now, captured_at):
        if self.state in ('idle', 'complete', 'aborted'):
            return None
        if not screen.active or screen.result:
            self.abort('battle not active')
            return None
        if not 0 <= now-captured_at <= .2:
            self.abort('stale frame')
            return None
        if self.state == 'waiting_ack':
            player_y = getattr(screen, 'player_y', None)
            acknowledged = (player_y is not None and self.initial_y is not None and
                            player_y <= self.initial_y-self.jump_ack_pixels) if self.response_class == 'fire_probe' else screen.right_control == 'up'
            if captured_at > self.sent_at and acknowledged:
                if self.response_class == 'fire_probe' and self.fire_air_jump_delay is not None:
                    self.state = 'waiting_air_jump'
                    self.reason = 'first jump ascent confirmed; waiting for experimental air jump'
                else:
                    self.state = 'complete'
                    self.reason = 'jump ascent confirmed' if self.response_class == 'fire_probe' else 'below control confirmed'
            elif now-self.sent_at > self.ack_timeout:
                self.abort('missing control confirmation')
            return None
        if self.state in ('waiting_air_jump', 'proposed_air_jump'):
            elapsed = now-self.first_sent_at
            if elapsed > self.fire_air_jump_deadline:
                self.abort('air jump window expired')
                return None
            if getattr(screen, 'grounded', None) is True:
                self.abort('landed before air jump')
                return None
            if self.state == 'proposed_air_jump':
                return None
            if elapsed < self.fire_air_jump_delay:
                self.reason = 'waiting for experimental air jump delay'
                return None
            if (screen.right_control != 'jump' or getattr(screen, 'grounded', None) is not False or
                    getattr(screen, 'airborne', None) is not True):
                self.reason = 'airborne jump state unconfirmed'
                return None
            self.pending = BossAction('jump', (1090, 630))
            self.state = 'proposed_air_jump'
            self.reason = 'experimental air jump proposed'
            return self.pending
        if now-self.observed_at > self.max_start_age:
            self.abort('late onset')
            return None
        if self.state == 'proposed':
            return None
        if self.response_class == 'fire_probe':
            if now-self.observed_at < self.fire_probe_delay:
                self.reason = 'waiting for calibrated probe delay'
                return None
            if screen.right_control != 'jump' or getattr(screen, 'grounded', None) is not True:
                self.reason = 'probe requires supported jump mode'
                return None
            self.initial_y = getattr(screen, 'player_y', None)
            if self.initial_y is None:
                self.reason = 'probe requires visible initial body for measurement'
                return None
            self.pending = BossAction('jump', (1090, 630))
            self.state = 'proposed'
            self.reason = 'experimental fire-family first jump'
            return self.pending
        if screen.right_control == 'up':
            self.state = 'complete'
            self.reason = 'already below by control mode'
            return None
        if screen.right_control != 'jump' or screen.left_control != 'down':
            self.reason = 'control mode unknown'
            return None
        if getattr(screen, 'grounded', None) is not True:
            self.reason = 'supported start unconfirmed'
            return None
        self.pending = BossAction('down', (190, 630))
        self.state = 'proposed'
        self.reason = 'confirmed opening down'
        return self.pending

    def completed(self, action, now):
        if self.state not in ('proposed', 'proposed_air_jump') or action is not self.pending:
            return False
        if self.state == 'proposed_air_jump':
            self.sent_at = now
            self.pending = None
            self.state = 'complete'
            self.reason = 'air jump pressed; movement effect unverified'
            return True
        self.sent_at = now
        self.first_sent_at = now
        self.pending = None
        self.state = 'waiting_ack'
        return True

    def rejected(self):
        """A proposed touch was not sent; do not retry the same onset."""
        self.abort('proposed touch not sent')
