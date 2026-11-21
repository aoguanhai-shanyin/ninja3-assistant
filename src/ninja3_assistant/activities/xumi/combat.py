"""Xumi combat controls and defensive actions on the shared battle kernel."""

import math
from dataclasses import replace

import numpy as np

from ninja3_assistant.activities.bounty.combat import BountyCombat, BountyCombatVision
from ninja3_assistant.activities.xumi.crow import CrowOpening
from ninja3_assistant.activities.xumi.ghost import GhostAvoidance
from ninja3_assistant.device.native_touch import TimedTap


NINJUTSU_POINT = (55, 60)
_ANGLES = np.linspace(0, 2 * np.pi, 48, endpoint=False)
_RADII = np.arange(34, 55, 2)[:, None]
_RING_Y = np.rint(NINJUTSU_POINT[1] + _RADII * np.sin(_ANGLES)).astype(np.intp)
_RING_X = np.rint(NINJUTSU_POINT[0] + _RADII * np.cos(_ANGLES)).astype(np.intp)
_LEFT_ARC = slice(12, 37)
_UPPER_LEFT = slice(24, 37)
_LOWER_LEFT = slice(12, 24)
_FORMATION_POINT = (635, 618)
_FORMATION_RADII = np.arange(42, 51, 2)[:, None]
_FORMATION_Y = np.rint(_FORMATION_POINT[1]+_FORMATION_RADII*np.sin(_ANGLES)).astype(np.intp)
_FORMATION_X = np.rint(_FORMATION_POINT[0]+_FORMATION_RADII*np.cos(_ANGLES)).astype(np.intp)


class XumiCombatVision(BountyCombatVision):
    def detect(self, frame, *, battle_allowed=True):
        state = super().detect(frame,battle_allowed=battle_allowed)
        if state.boss_active:
            ring = frame[_FORMATION_Y,_FORMATION_X].min(axis=2).max(axis=0)
            if (ring >= 205).mean() >= .85:
                # This slot's complete white rim appears when its countdown disappears.
                state = replace(state,weapon_points=state.weapon_points+(_FORMATION_POINT,))
        return state

    def ninjutsu_point(self, frame, gray):
        if self._red_ninjutsu_point(frame):
            return NINJUTSU_POINT

        # Fangcun and Daqian change the glyph and charge color. The control
        # keeps a dark inner rim and a brighter band around that rim.
        ring = gray[_RING_Y, _RING_X].astype(np.int16)
        inner_rim = ring[2]  # 38 pixels from the center.
        band = np.median(ring[4:7], axis=0)  # 42–46 pixels from the center.
        edges = np.abs(np.diff(ring, axis=0)).max(axis=0) >= 55
        contrast = band - inner_rim >= 20
        if (inner_rim < 45).mean() >= .52 and gray[39:85, 28:84].std() >= 35:
            if contrast.mean() >= .58 and edges.mean() >= .72:
                return NINJUTSU_POINT
            if contrast[_LEFT_ARC].mean() >= .8 and edges[_LEFT_ARC].mean() >= .8:
                # The black-wing version has a dark right half. Confirm its
                # gold upper and pink lower arcs instead of relaxing the ring.
                red, green, blue = np.median(frame[_RING_Y[4:7], _RING_X[4:7]], axis=0).T
                gold = (red >= 170) & (green >= 100) & (blue < 100) & (red - green >= 20) & (green - blue >= 30)
                pink = (red >= 175) & (red - green >= 35) & (red - blue >= 35) & (green >= 65) & (blue >= 45)
                if gold[_UPPER_LEFT].mean() >= .8 and pink[_LOWER_LEFT].mean() >= .8:
                    return NINJUTSU_POINT

        # The large middle-left icon in the saved casts is an animation.
        return None


class XumiCombat(BountyCombat):
    def __init__(self, kernel, vision):
        super().__init__(kernel)
        self.crow = CrowOpening(enable_feather=True)
        self.ghost = GhostAvoidance(vision.down, vision.ground_controls[1])
        self.board_until = None

    def cancel_board(self):
        if self.board_until is not None:
            self.kernel.cancel_plan(5)
            self.board_until = None

    def stop(self):
        self.cancel_board()
        super().stop()

    def reset_boss(self):
        self.stop()
        self.crow.reset()
        self.ghost.reset()

    def update(self, frame, state, now, boss_elapsed=None):
        if self.board_until is not None:
            if not (state.on_rope and state.boss_active and self.kernel.fresh()):
                self.cancel_board()
            elif self.kernel.clock() < self.board_until:
                return '八咫乌 · 开场飞羽短翻', None
            else:
                self.cancel_board()
        if state.on_rope and state.boss_active and self.kernel.fresh():
            crow = self.crow.propose(frame, boss_active=True, boss_elapsed=boss_elapsed)
            captured_at = self.kernel.captured_at
            if crow is not None and crow.kind == 'up':
                self._stop_movement()
                self._stop_skills()
                first = math.ceil((self.kernel.clock()-captured_at)*1000)+8
                taps = (TimedTap(first,crow.point,25,40), TimedTap(first+crow.return_delay_ms,(190,630),25,40))
                if self.kernel.plan(5,taps) is not None:
                    last = taps[-1]
                    self.board_until = captured_at+(last.offset_ms+last.slack_ms+last.hold_ms)/1000
                    self.crow.accepted(crow)
                    board = 'crow-feathers-3' if crow.feather_wave == 3 else 'crow-feathers'
                    return '八咫乌 · 开场飞羽短翻', (board,crow.point,taps)
                return super().update(state,now), None
            point = crow.point if crow is not None else self.ghost.update(frame,captured_at)
            if point is not None:
                self._stop_movement()
                receipt = self.kernel.priority_tap(point, captured_at=captured_at,
                                                   deadline=captured_at+.15)
                if receipt is not None:
                    if crow is not None:
                        self.crow.accepted(crow)
                        message = '八咫乌 · 下翻躲开场飞羽' if crow.kind == 'down' else '八咫乌 · 起跳躲冲刺'
                        return message, ('crow-'+crow.kind,point,())
                    self.ghost.accepted()
                    direction = 'down' if point[0] < 640 else 'up'
                    message = '鬼斩狂 · 下翻躲近身刀球' if direction == 'down' else '鬼斩狂 · 上翻躲近身刀球'
                    return message, ('ghost-'+direction,point,())
        return super().update(state, now), None
