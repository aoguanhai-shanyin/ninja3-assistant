"""Replay the opening under both trial modes and both rope positions."""

import unittest

import cv2
import numpy as np

from ninja3_assistant.activities.xumi.crow import CrowAction, CrowOpening, _crosses_lane, _feather_contours
from ninja3_assistant.activities.xumi.crow import _feather_tracks, _rope_y
from ninja3_assistant.paths import project_root


RUNS = project_root() / 'data/runtime/xumi/runs'
DAQIAN = RUNS / '20261119-135458-511800'
FANGCUN = RUNS / '20261119-172722-780500'
BELOW = RUNS / '20261119-173128-841800'
OPENING_DODGE = RUNS / '20261119-175126-264800'
THIRD_FAN = project_root() / 'data/evidence/2026-11-19/xumi-152/crow-third-fan'


def frame(run, number):
    path = run / f'battle-{number:04d}.png'
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def third_fan_frame(number):
    path = THIRD_FAN / f'{number:05d}-find_rope-below.png'
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def with_feathers(background, *positions):
    image = background.copy()
    outline = np.array(((0, 20), (18, 9), (58, 0), (40, 13), (15, 22)))
    for x, y in positions:
        cv2.fillConvexPoly(image, (outline + (x, y)).astype(np.int32), (230, 50, 30))
    return image


@unittest.skipUnless(all((run / name).is_file() for run, name in (
    (DAQIAN, 'battle-0012.png'), (FANGCUN, 'battle-0009.png'), (BELOW, 'battle-0006.png'),
)), 'local Xumi opening replays unavailable')
class CrowOpeningTest(unittest.TestCase):
    def test_proposes_down_before_health_bar_fills_and_latches_on_acceptance(self):
        for run, number in ((DAQIAN, 12), (FANGCUN, 9)):
            with self.subTest(run=run.name):
                policy = CrowOpening()
                opening = frame(run, number)
                self.assertIsNone(policy.propose(opening, boss_active=False))
                action = policy.propose(opening, boss_active=True)
                self.assertEqual(action, CrowAction('down', (190, 630)))
                self.assertEqual(policy.propose(opening, boss_active=True), action)
                policy.accepted(action)
                self.assertIsNone(policy.propose(frame(run, number + 1), boss_active=True))
                policy.reset()
                self.assertEqual(policy.propose(opening, boss_active=True), action)

    def test_already_below_does_not_flip_on_subsequent_above_frames(self):
        policy = CrowOpening()
        self.assertIsNone(policy.propose(frame(BELOW, 6), boss_active=True))
        self.assertIsNone(policy.propose(frame(FANGCUN, 9), boss_active=True))

    def test_other_boss_and_pre_boss_rope_do_not_propose_down(self):
        policy = CrowOpening()
        for run, number in ((DAQIAN, 64), (BELOW, 75), (FANGCUN, 8)):
            with self.subTest(run=run.name, number=number):
                self.assertIsNone(policy.propose(frame(run, number), boss_active=True))

    def test_overlay_obscuring_right_control_blocks_the_action(self):
        policy = CrowOpening()
        opening = frame(FANGCUN, 9)
        opening[575:685, 1035:1145] = 0
        self.assertIsNone(policy.propose(opening, boss_active=True))

    def test_charge_jump_only_when_above_and_the_two_summoned_crows_are_visible(self):
        policy = CrowOpening()
        charge = frame(FANGCUN, 50)
        self.assertIsNone(policy.propose(frame(FANGCUN, 48), boss_active=True, boss_elapsed=9.84))
        self.assertIsNone(policy.propose(frame(FANGCUN, 51), boss_active=True, boss_elapsed=10.63))
        self.assertIsNone(policy.propose(charge, boss_active=True, boss_elapsed=11.2))
        action = policy.propose(charge, boss_active=True, boss_elapsed=10.37)
        self.assertEqual(action, CrowAction('jump', (1090, 630)))
        policy.accepted(action)
        self.assertIsNone(policy.propose(charge, boss_active=True, boss_elapsed=10.40))

    def test_charge_with_below_rope_control_handles_window_without_a_flip(self):
        policy = CrowOpening()
        self.assertIsNone(policy.propose(frame(BELOW, 45), boss_active=True, boss_elapsed=9.77))
        self.assertFalse(policy.charge_handled)
        self.assertIsNone(policy.propose(frame(BELOW, 46), boss_active=True, boss_elapsed=10.0))
        self.assertTrue(policy.charge_handled)

    def test_opening_flip_expires_before_the_feathers(self):
        policy = CrowOpening()
        self.assertIsNone(policy.propose(frame(DAQIAN, 14), boss_active=True, boss_elapsed=3.5))

    @unittest.skipUnless((OPENING_DODGE / 'battle-0031.png').is_file(), 'local first-feather replay unavailable')
    def test_live_feather_crossing_also_has_an_unsafe_upper_lane(self):
        past, current = frame(OPENING_DODGE, 29), frame(OPENING_DODGE, 30)
        tracks = _feather_tracks(_feather_contours(past, _rope_y(past)),
                                 _feather_contours(current, _rope_y(current)), .256)
        low = [track for track in tracks if .28 <= track[5] / track[4] <= .49]
        self.assertTrue(any(_crosses_lane(track, (43, 89), (.08, .25)) for track in low))
        self.assertTrue(any(_crosses_lane(track, (-57, -12), (.14, .28)) for track in tracks))

        policy = CrowOpening(enable_feather=True)
        self.assertIsNone(policy.propose(past, boss_active=True, boss_elapsed=4.175))
        self.assertIsNone(policy.propose(current, boss_active=True, boss_elapsed=4.431))
        self.assertIsNone(policy.propose(frame(OPENING_DODGE, 31), boss_active=True, boss_elapsed=4.682))
        self.assertEqual(policy.handled_feather_waves, {1})

    def test_dodge_requires_flight_and_proposes_up_only_once(self):
        background = frame(BELOW, 19)
        previous = with_feathers(background, (550, 347))
        current = with_feathers(background, (404, 402))
        policy = CrowOpening(enable_feather=True)
        self.assertIsNone(policy.propose(previous, boss_active=True, boss_elapsed=4.))
        action = policy.propose(current, boss_active=True, boss_elapsed=4.16)
        self.assertEqual(action, CrowAction('up', (1090, 630), feather_wave=1))
        policy.accepted(action)
        self.assertIsNone(policy.propose(current, boss_active=True, boss_elapsed=4.18))

        disabled = CrowOpening()
        self.assertIsNone(disabled.propose(previous, boss_active=True, boss_elapsed=4.))
        self.assertIsNone(disabled.propose(current, boss_active=True, boss_elapsed=4.16))

    @unittest.skipUnless((THIRD_FAN / '01987-find_rope-below.png').is_file(), 'local training replay unavailable')
    def test_third_fan_waits_for_top_gap_and_remains_independent_of_first(self):
        for first_accepted in (False, True):
            with self.subTest(first_accepted=first_accepted):
                policy = CrowOpening(enable_feather=True)
                if first_accepted:
                    policy.accepted(CrowAction('up', (1090, 630), feather_wave=1))
                # Same-boss training at 18:05:35. The upper feather has not
                # passed in 01983, then clears the landing in 01987.
                for number, elapsed in ((1979, 6.300), (1983, 6.378)):
                    self.assertIsNone(policy.propose(third_fan_frame(number), boss_active=True, boss_elapsed=elapsed))
                action = policy.propose(third_fan_frame(1987), boss_active=True, boss_elapsed=6.457)
                self.assertEqual(action, CrowAction('up', (1090, 630), feather_wave=3))
                policy.accepted(action)
                self.assertIsNone(policy.propose(third_fan_frame(1991), boss_active=True, boss_elapsed=6.535))
                self.assertIn(3, policy.handled_feather_waves)

    def test_third_fan_delays_return_until_a_trailing_feather_clears(self):
        background = frame(BELOW, 19)
        clear = CrowOpening(enable_feather=True)
        clear.propose(with_feathers(background, (452, 383)), boss_active=True, boss_elapsed=6.30)
        self.assertEqual(clear.propose(with_feathers(background, (380, 414)), boss_active=True, boss_elapsed=6.38),
                         CrowAction('up', (1090, 630), feather_wave=3))

        previous = with_feathers(background, (452, 383), (862, 246))
        current = with_feathers(background, (380, 414), (790, 277))
        policy = CrowOpening(enable_feather=True)
        self.assertIsNone(policy.propose(previous, boss_active=True, boss_elapsed=6.30))
        action = policy.propose(current, boss_active=True, boss_elapsed=6.38)
        self.assertEqual(action, CrowAction('up',(1090,630),feather_wave=3,return_delay_ms=480))

    def test_unassigned_wave_cannot_reuse_third_fan_observation(self):
        background = frame(BELOW, 19)
        previous = with_feathers(background, (550, 347))
        current = with_feathers(background, (404, 402))
        policy = CrowOpening(enable_feather=True)
        self.assertIsNone(policy.propose(previous, boss_active=True, boss_elapsed=5.2))
        self.assertIsNone(policy.propose(current, boss_active=True, boss_elapsed=5.36))
        self.assertIsNone(policy.propose(previous, boss_active=True, boss_elapsed=6.70))
        self.assertIsNone(policy.propose(current, boss_active=True, boss_elapsed=6.86))
