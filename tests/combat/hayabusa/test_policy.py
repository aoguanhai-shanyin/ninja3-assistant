from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ninja3_assistant.combat.battle import BladeTrack
from ninja3_assistant.combat.boss.policy import BossAction
from ninja3_assistant.combat.hayabusa.policy import HayabusaPolicy


def screen(**changes):
    values = dict(active=True, rope_y=430., player_x=300., player_y=394., side='above',
                  airborne=False, weapon_ready=False, weapon_point=None, result=None)
    values.update(changes)
    return SimpleNamespace(**values)


def attack(**changes):
    values = dict(id=1, x=650., y=385., width=70, height=65, vx=-800., vy=0.,
                  seen=1., hits=4, kind='slash')
    values.update(changes)
    return BladeTrack(**values)


class HayabusaPolicyTest(unittest.TestCase):
    def test_upper_attack_prompts_down_flip(self):
        action = HayabusaPolicy().decide(screen(), [attack()], 1.)
        self.assertEqual(action.kind, 'down')
        self.assertEqual(action.point, (190, 630))

    def test_lower_attack_prompts_up_flip(self):
        action = HayabusaPolicy().decide(screen(side='below', player_y=466), [attack(y=470)], 1.)
        self.assertEqual(action.kind, 'up')

    def test_wide_flash_does_not_freeze_the_planner_on_initial_overlap(self):
        action = HayabusaPolicy().decide(screen(),
            [attack(x=640, y=394, width=1200, height=20, vx=0, kind='flash')], 1.)
        self.assertIsNotNone(action)
        self.assertIn(action.kind, ('jump', 'down'))

    def test_attack_from_left_is_predicted(self):
        action = HayabusaPolicy().decide(screen(player_x=950), [attack(x=600, vx=800)], 1.)
        self.assertEqual(action.kind, 'down')

    def test_jump_when_both_rope_lanes_are_threatened(self):
        action = HayabusaPolicy().decide(screen(), [attack(y=394), attack(id=2, y=475)], 1.)
        self.assertEqual(action.kind, 'jump')

    def test_vertical_attack_requires_velocity_in_both_axes(self):
        policy = HayabusaPolicy()
        falling = attack(x=300, y=160, vx=0, vy=600)
        stationary = attack(x=300, y=160, vx=0, vy=0)
        self.assertLess(policy._clearance(screen(), [falling], 1., 'wait')[0], 1)
        self.assertGreater(policy._clearance(screen(), [stationary], 1., 'wait')[0], 2)

    def test_moving_away_and_distant_overhead_attacks_do_not_cause_jump(self):
        policy = HayabusaPolicy()
        self.assertIsNone(policy.decide(screen(), [attack(vx=800)], 1.))
        self.assertIsNone(policy.decide(screen(), [attack(y=100)], 1.01))

    def test_unconfirmed_or_stale_tracks_are_not_dodged(self):
        for track in (attack(hits=1), attack(seen=.7), attack(seen=2),
                      attack(kind='decoration'), attack(vx=float('nan'))):
            with self.subTest(track=track):
                self.assertIsNone(HayabusaPolicy().decide(screen(), [track], 1.))

    def test_menus_results_and_missing_player_never_trigger_weapon(self):
        for changes in ({'active': False}, {'player_x': None}, {'player_y': None},
                        {'rope_y': None}, {'side': 'unknown'}, {'result': 'success'}):
            self.assertIsNone(HayabusaPolicy().decide(
                screen(weapon_ready=True, weapon_point=(380, 630), **changes), [attack()], 1.))

    def test_dodge_precedes_weapon_and_unsafe_weapon_is_not_released(self):
        ready = screen(weapon_ready=True, weapon_point=(380, 630))
        self.assertNotEqual(HayabusaPolicy().decide(ready, [attack()], 1.).kind, 'weapon')
        self.assertEqual(HayabusaPolicy().decide(ready, [], 1.).kind, 'weapon')

    def test_weapon_rearms_only_after_readiness_disappears(self):
        policy = HayabusaPolicy()
        ready = screen(weapon_ready=True, weapon_point=(380, 630))
        policy.completed(policy.decide(ready, [], 1.), 1.)
        self.assertIsNone(policy.decide(ready, [], 2.))
        policy.decide(screen(), [], 2.1)
        policy.decide(screen(), [], 3.2)
        self.assertEqual(policy.decide(ready, [], 3.3).kind, 'weapon')

    def test_pending_flip_waits_for_confirmation_or_timeout(self):
        policy = HayabusaPolicy()
        first = policy.decide(screen(), [attack()], 1.)
        policy.completed(first, 1.)
        self.assertIsNone(policy.decide(screen(), [attack(seen=1.2)], 1.2))
        self.assertIn('confirmation', policy.last_reason)
        policy.decide(screen(side='below', player_y=464), [], 1.25)
        self.assertIsNone(policy.pending_flip)

    def test_jump_count_is_bounded_until_landing(self):
        policy = HayabusaPolicy()
        for now in (1., 1.17, 1.34, 1.51):
            policy.completed(BossAction('jump', (1090, 630)), now)
        self.assertEqual(policy.jumps, 4)
        airborne = screen(airborne=True, player_y=300)
        policy.decide(airborne, [], 1.6)
        self.assertEqual(policy.jumps, 4)
        policy.decide(screen(), [], 1.8)
        policy.decide(screen(), [], 1.9)
        self.assertEqual(policy.jumps, 0)

    def test_no_fifth_jump_when_ground_not_confirmed(self):
        policy = HayabusaPolicy()
        policy.jumps = 4
        policy.last_jump = .9
        action = policy.decide(screen(airborne=True, player_y=394),
                               [attack(y=394), attack(id=2, y=475)], 1.)
        self.assertTrue(action is None or action.kind != 'jump')

    def test_reset_discards_motion_and_pending_actions(self):
        policy = HayabusaPolicy()
        policy.completed(BossAction('down', (190, 630)), 1.)
        policy.reset()
        self.assertIsNone(policy.pending_flip)
        self.assertEqual(policy.jumps, 0)
        self.assertEqual(policy.decide(screen(), [attack()], 1.).kind, 'down')

    def test_transient_reset_keeps_weapon_deduplication(self):
        policy = HayabusaPolicy()
        ready = screen(weapon_ready=True, weapon_point=(380, 630))
        policy.completed(policy.decide(ready, [], 1), 1)
        policy.reset(preserve_weapon=True)
        self.assertIsNone(policy.decide(ready, [], 2))
        policy.decide(screen(), [], 2.1)
        policy.decide(screen(), [], 3.2)
        self.assertEqual(policy.decide(ready, [], 3.3).kind, 'weapon')

    def test_fourth_air_jump_is_available_but_fifth_is_not(self):
        policy = HayabusaPolicy()
        policy.jumps = 3
        air = screen(airborne=True, player_y=320, grounded=False)
        self.assertTrue(policy._valid_movement(air, 'jump'))
        policy.completed(BossAction('jump', (1090, 630)), 1)
        self.assertFalse(policy._valid_movement(air, 'jump'))

    def test_unknown_frame_and_pause_reset_preserve_air_jump_budget(self):
        policy = HayabusaPolicy()
        policy.jumps = 4
        policy.last_jump = 1
        policy.reset(preserve_weapon=True, preserve_jumps=True)
        self.assertEqual(policy.jumps, 4)
        policy.decide(screen(player_x=None), [], 2)
        policy.decide(screen(grounded=None, airborne=False), [], 2.1)
        policy.decide(screen(grounded=None, airborne=False), [], 2.2)
        self.assertEqual(policy.jumps, 4)
        policy.decide(screen(grounded=True), [], 2.3)
        self.assertEqual(policy.jumps, 4)
        policy.decide(screen(grounded=True), [], 2.4)
        self.assertEqual(policy.jumps, 0)

    def test_air_down_is_drop_and_rope_down_is_a_flip(self):
        policy = HayabusaPolicy()
        air = screen(airborne=True, player_y=230, grounded=False)
        self.assertTrue(policy._valid_movement(air, 'drop'))
        self.assertFalse(policy._valid_movement(air, 'down'))
        self.assertTrue(policy._valid_movement(screen(), 'down'))
        self.assertFalse(policy._valid_movement(screen(), 'drop'))

    def test_missing_player_cancels_prior_landing_confirmation(self):
        policy = HayabusaPolicy()
        policy.decide(screen(grounded=True), [], 1)
        policy.jumps, policy.last_jump = 4, .9
        policy.decide(screen(player_x=None, grounded=None), [], 1.05)
        policy.decide(screen(grounded=True), [], 1.4)
        self.assertEqual(policy.jumps, 4)
        policy.decide(screen(grounded=True), [], 1.5)
        self.assertEqual(policy.jumps, 0)

    def test_unknown_grounding_never_turns_air_down_into_a_rope_flip(self):
        policy = HayabusaPolicy()
        uncertain = screen(grounded=None)
        self.assertFalse(policy._valid_movement(uncertain, 'down'))
        policy.jumps = 4
        self.assertIsNone(policy.decide(uncertain, [attack()], 1))

    def test_recipe_movement_is_subject_to_the_same_collision_check(self):
        policy = HayabusaPolicy()
        unsafe = BossAction('down', (190, 630), 1)
        with patch.object(policy.recipe, 'decide', return_value=unsafe):
            action = policy.decide(screen(), [attack(y=394), attack(id=2, y=475)], 1)
        self.assertEqual(action.kind, 'jump')

    def test_tracking_bait_wait_is_respected_unless_danger_is_immediate(self):
        policy = HayabusaPolicy()
        def hold(*args):
            policy.recipe.hold_for_approach = True
            return None
        with patch.object(policy.recipe, 'decide', side_effect=hold):
            self.assertIsNone(policy.decide(screen(), [attack(x=700)], 1))
            imminent = policy.decide(screen(), [attack(x=425, seen=1.04)], 1.04)
            self.assertIsNotNone(imminent)

    def test_weapon_waits_during_confirmed_transition(self):
        state = screen(weapon_ready=True, weapon_point=(380, 630), board_id='p4_transition',
                       phase_hint='P4', board_confidence=.95, board_margin=.2)
        self.assertIsNone(HayabusaPolicy().decide(state, [], 1))

    def test_short_suspend_retains_fixed_step_and_actual_input_budget(self):
        policy = HayabusaPolicy()
        known = screen(grounded=True, board_id='p4_transition', phase_hint='P4',
                       board_confidence=.95, board_margin=.2, board_progress=.2,
                       board_locked=True, board_onset=True, board_epoch=7)
        action = policy.decide(known, [], 1.)
        self.assertEqual(action.kind, 'jump')
        policy.completed(action, 1.)
        pending = policy.recipe.pending
        policy.weapon_used = True
        policy.velocity = (100., -250.)
        policy.pending_flip = 'below'
        policy.grounded_since = .9
        policy.weapon_missing = .8
        self.assertIsNone(policy.suspend(1.1))
        self.assertTrue(policy.recipe.mode_active)
        self.assertEqual(policy.recipe.mode_epoch, 7)
        self.assertEqual(policy.recipe.pending, pending)
        self.assertEqual(policy.recipe.stage, 0)
        self.assertEqual(policy.jumps, 1)
        self.assertTrue(policy.weapon_used)
        self.assertIsNone(policy.previous)
        self.assertEqual(policy.velocity, (0., 0.))
        self.assertIsNone(policy.pending_flip)
        self.assertIsNone(policy.grounded_since)
        self.assertIsNone(policy.weapon_missing)

    def test_unconfirmed_hud_never_sends_an_action_and_long_suspend_requires_resync(self):
        policy = HayabusaPolicy()
        known = screen(grounded=True, board_id='p4_transition', phase_hint='P4',
                       board_confidence=.95, board_margin=.2, board_progress=.2,
                       board_locked=True, board_onset=True, board_epoch=7)
        policy.completed(policy.decide(known, [], 1.), 1.)
        self.assertIsNone(policy.decide(screen(active=False), [attack(seen=1.1)], 1.1))
        self.assertTrue(policy.recipe.mode_active)
        self.assertIsNone(policy.suspend(1.36))
        self.assertFalse(policy.recipe.mode_active)
        self.assertEqual(policy.recipe.mode_state, 'resync_required')
        self.assertEqual(policy.recipe.blocked_epoch, 7)
        self.assertIsNone(policy.recipe.pending)


if __name__ == '__main__':
    unittest.main()
