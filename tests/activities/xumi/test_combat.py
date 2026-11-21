"""Offline Xumi combat replays from MuMu instance 152."""

import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ninja3_assistant.activities.bounty.combat import BountyBattleState, BountyCombatVision
from ninja3_assistant.activities.xumi.combat import NINJUTSU_POINT, XumiCombat, XumiCombatVision
from ninja3_assistant.activities.xumi.crow import CrowAction
from ninja3_assistant.paths import project_root


EVIDENCE = project_root() / 'data/evidence/2026-11-19/xumi-152'
RUN = project_root() / 'data/runtime/xumi/runs/20261119-135458-511800'
FORMATION_RUN = project_root() / 'data/runtime/xumi/runs/20261119-175126-264800'
THIRD = project_root() / 'data/runtime/xumi/runs/20261119-172722-780500'


def rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


class XumiBoardPlanTest(unittest.TestCase):
    def test_feather_roll_submits_both_taps_and_cancels_when_rope_is_lost(self):
        kernel = Mock(captured_at=100.)
        kernel.clock.return_value = 100.02
        kernel.fresh.return_value = True
        kernel.plan.return_value = 1
        vision = Mock(down=np.zeros((32,32),np.uint8), ground_controls=(None,None))
        with patch('ninja3_assistant.activities.xumi.combat.CrowOpening'):
            combat = XumiCombat(kernel,vision)
        combat.crow.propose.return_value = CrowAction('up',(1090,630),return_delay_ms=480)
        state = BountyBattleState(active=True,on_rope=True,boss_active=True)
        _, dodge = combat.update(None,state,100.02,3.)
        finger,taps = kernel.plan.call_args.args
        self.assertEqual(finger,5)
        self.assertEqual([tap.point for tap in taps],[(1090,630),(190,630)])
        self.assertEqual(taps[1].offset_ms-taps[0].offset_ms,480)
        self.assertEqual(dodge[0],'crow-feathers')
        # The final press could have used all its slack and still be held.
        clock = 100.+(taps[-1].offset_ms+taps[-1].slack_ms+1)/1000
        kernel.clock.return_value = clock
        combat.update(None,state,clock,3.4)
        kernel.plan.assert_called_once()
        kernel.cancel_plan.assert_not_called()
        combat.update(None,BountyBattleState(),clock+.001)
        kernel.cancel_plan.assert_called_once_with(5)


@unittest.skipUnless((RUN / 'battle-0044.png').is_file(), 'local Xumi battle replay unavailable')
class XumiCombatVisionTest(unittest.TestCase):
    @unittest.skipUnless((FORMATION_RUN / 'battle-0076.png').is_file(), 'local Xumi skill replay unavailable')
    def test_adds_formation_skill_only_when_ready_rim_is_visible(self):
        vision = XumiCombatVision()
        self.assertNotIn((635,618),vision.detect(rgb(FORMATION_RUN/'battle-0073.png')).weapon_points)
        self.assertIn((635,618),vision.detect(rgb(FORMATION_RUN/'battle-0076.png')).weapon_points)

    def test_fangcun_both_character_controls_resolve_on_active_boss_frames(self):
        for filename in ('battle-0044.png', 'battle-0064.png'):
            with self.subTest(frame=filename):
                frame = rgb(RUN / filename)
                generic = BountyCombatVision().detect(frame)
                self.assertTrue(generic.boss_active)
                self.assertIsNone(generic.ninjutsu_point)

                xumi = XumiCombatVision().detect(frame)
                self.assertTrue(xumi.boss_active)
                self.assertEqual(xumi.ninjutsu_point, NINJUTSU_POINT)

    def test_daqian_keeps_top_control_for_both_character_variants(self):
        for filename in ('daqian-blue-ninjutsu-boss1.png', 'daqian-red-ninjutsu-boss2.png'):
            with self.subTest(frame=filename):
                state = XumiCombatVision().detect(rgb(EVIDENCE / filename))
                self.assertTrue(state.boss_active)
                self.assertEqual(state.ninjutsu_point, NINJUTSU_POINT)

    def test_relay_settlement_and_lower_cast_icon_do_not_resolve_top_control(self):
        vision = XumiCombatVision()
        for filename in ('0018-unknown-wait.png', '0030-result_done-settle.png', 'battle-0074.png'):
            with self.subTest(frame=filename):
                frame = rgb(RUN / filename)
                self.assertIsNone(vision.ninjutsu_point(frame, cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)))

    def test_an_empty_ring_without_a_visible_glyph_is_rejected(self):
        vision = XumiCombatVision()
        gray = np.zeros((720, 1280), dtype=np.uint8)
        cv2.circle(gray, NINJUTSU_POINT, 44, 240, thickness=6)
        self.assertIsNone(vision.ninjutsu_point(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), gray))

    @unittest.skipUnless((THIRD / 'battle-0009.png').is_file(), 'local third-difficulty Xumi replay unavailable')
    def test_fangcun_gold_pink_arc_resolves_before_and_during_boss_one(self):
        vision = XumiCombatVision()
        intro = rgb(THIRD / 'battle-0009.png')
        self.assertEqual(vision.ninjutsu_point(intro, cv2.cvtColor(intro, cv2.COLOR_RGB2GRAY)), NINJUTSU_POINT)
        for number in (10, 13, 19, 34, 45, 60):
            with self.subTest(frame=number):
                state = vision.detect(rgb(THIRD / f'battle-{number:04d}.png'))
                self.assertTrue(state.boss_active)
                self.assertEqual(state.ninjutsu_point, NINJUTSU_POINT)

        relay = rgb(THIRD / '0070-boss-1-relay.png')
        self.assertIsNone(vision.ninjutsu_point(relay, cv2.cvtColor(relay, cv2.COLOR_RGB2GRAY)))

    @unittest.skipUnless((THIRD / 'battle-0100.png').is_file(), 'local second-boss replay unavailable')
    def test_fangcun_orange_control_keeps_current_position_in_boss_two(self):
        vision = XumiCombatVision()
        for number in (77, 80, 100):
            with self.subTest(frame=number):
                state = vision.detect(rgb(THIRD / f'battle-{number:04d}.png'))
                self.assertTrue(state.boss_active)
                self.assertEqual(state.ninjutsu_point, NINJUTSU_POINT)
