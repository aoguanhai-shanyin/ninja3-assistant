"""Conservative experimental actions for the observed rope Boss."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BossAction:
    kind: str
    point: tuple
    threat_id: int | None = None


class BossPolicy:
    def __init__(self):
        self.last_action = float('-inf')
        self.handled = {}
        self.weapon_used = False
        self.weapon_missing = None
        self.last_player = None

    def decide(self, screen, tracks, now):
        if not screen.target_boss:
            self.last_player = None
            return None
        if screen.rope_y is None or screen.side == 'unknown':
            return None
        if screen.player_x is not None:
            self.last_player = (screen.player_x, now)
        elif self.last_player and 0 <= now - self.last_player[1] <= .25:
            screen = replace(screen, player_x=self.last_player[0])
        else:
            return None
        self._rearm(screen, now)
        if now - self.last_action < .35:
            return None
        nearest = None
        for track in tracks:
            if now < self.handled.get(track.id, 0):
                continue
            eta = self._threat_eta(track, screen, now, screen.side)
            if eta is not None and (nearest is None or eta < nearest[0]):
                nearest = eta, track
        if nearest:
            _, track = nearest
            if track.kind == 'thin' and track.y >= screen.rope_y - 80:
                if screen.side == 'above':
                    return BossAction('jump', (1090, 630), track.id)
            destination = 'above' if screen.side == 'below' else 'below'
            if any(other.id != track.id and self._threat_eta(other, screen, now, destination) is not None
                   for other in tracks):
                return None
            return BossAction('up' if screen.side == 'below' else 'down',
                              (1090, 630) if screen.side == 'below' else (190, 630), track.id)
        if screen.weapon_ready and not self.weapon_used:
            return BossAction('weapon', (380, 630))
        return None

    @staticmethod
    def _threat_eta(track, screen, now, side):
        min_hits = 2 if track.kind == 'orb' else 4
        if track.hits < min_hits or not 0 <= now - track.seen <= .15:
            return None
        distance = track.x - screen.player_x
        if abs(track.vx) < 120 or distance * track.vx >= 0:
            return None
        gap = max(0, abs(distance) - track.width / 2 - 30)
        eta = gap / abs(track.vx)
        if eta > (1.1 if track.kind == 'orb' else .8):
            return None
        predicted_y = track.y
        if track.kind == 'orb':
            predicted_y += max(-150, min(150, track.vy * eta))
        if side == 'above' and predicted_y > screen.rope_y + 20:
            return None
        if side == 'below' and predicted_y < screen.rope_y - 20:
            return None
        top, bottom = predicted_y - track.height / 2, predicted_y + track.height / 2
        lane_top, lane_bottom = ((screen.rope_y - 140, screen.rope_y - 5) if side == 'above'
                                 else (screen.rope_y + 5, screen.rope_y + 140))
        return eta if bottom >= lane_top and top <= lane_bottom else None

    def completed(self, action, now):
        self.last_action = now
        if action.threat_id is not None:
            self.handled[action.threat_id] = now + .9
        elif action.kind == 'weapon':
            self.weapon_used = True

    def _rearm(self, screen, now):
        if screen.weapon_ready:
            self.weapon_missing = None
        elif self.weapon_missing is None:
            self.weapon_missing = now
        elif now - self.weapon_missing >= 1:
            self.weapon_used = False
