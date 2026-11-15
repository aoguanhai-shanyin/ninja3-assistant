from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from battle import BladeTrack
from boss_policy import BossAction
from hayabusa_board_policy import BoardDodgePolicy, RECIPES


def scene(**changes):
    values = dict(active=True, result=None, player_x=300., player_y=394., rope_y=430.,
                  side='above', airborne=False)
    values.update(changes)
    return SimpleNamespace(**values)


def match(board='p2_fork_sword', **changes):
    values = dict(board_id=board, phase=RECIPES[board].phase if board in RECIPES else None,
                  confidence=.95, margin=.2, progress=.1, evidence='synthetic attack geometry')
    values.update(changes)
    return SimpleNamespace(**values)


def attack(**changes):
    values = dict(id=1, x=650., y=385., width=70, height=65, vx=-800., vy=0.,
                  seen=1., hits=4, kind='slash')
    values.update(changes)
    return BladeTrack(**values)


def locked_match(board='p2_fork_sword', **changes):
    values = dict(locked=True, onset=True, ended=False, epoch=7)
    values.update(changes)
    return match(board, **values)


def prime(policy, *, screen=None, tracks=None, recognition=None, jumps=0):
    screen = screen or scene()
    tracks = tracks or [attack()]
    recognition = recognition or match()
    first = policy.decide(screen, tracks, recognition, 1., jumps)
    assert first is None
    return policy.decide(screen, [replace(item, seen=1.05) for item in tracks], recognition, 1.05, jumps)


class BoardDodgePolicyTest(unittest.TestCase):
    def test_every_observed_catalog_entry_has_an_explicit_phase_and_recipe(self):
        catalog = Path(__file__).resolve().parents[1]/'data/hayabusa-video-study/board-catalog.json'
        if not catalog.is_file():
            self.skipTest('local study catalog unavailable')
        rows = json.loads(catalog.read_text(encoding='utf-8'))['boards']
        self.assertEqual({item['id'] for item in rows}, set(RECIPES))
        for row in rows:
            recipe = RECIPES[row['id']]
            self.assertEqual(recipe.phase, row['phase'])
            self.assertTrue(recipe.evidence)
            self.assertTrue(recipe.kinds)
        self.assertEqual({item.phase for item in RECIPES.values()}, {'P1', 'P2', 'P3', 'P4'})

    def test_single_frame_ambiguous_unknown_or_wrong_phase_never_starts_recipe(self):
        for recognition in (match('unknown'), match(confidence=.4), match(margin=.01), match(phase='P4')):
            policy = BoardDodgePolicy()
            for time in (1., 1.05, 1.1):
                self.assertIsNone(policy.decide(scene(), [attack(seen=time)], recognition, time))
        self.assertIsNone(BoardDodgePolicy().decide(scene(), [attack()], match(), 1.))

    def test_recognition_alone_does_not_play_a_timed_input_sequence(self):
        policy = BoardDodgePolicy()
        for time in (1., 1.05, 1.1, 1.5, 1.9):
            self.assertIsNone(policy.decide(scene(), [], match(), time))
        self.assertEqual(policy.stage, 0)

    def test_menu_missing_player_or_unknown_side_never_proposes_movement(self):
        for changes in ({'active': False}, {'result': 'victory'}, {'player_x': None},
                        {'player_y': None}, {'rope_y': None}, {'side': 'unknown'}):
            self.assertIsNone(prime(BoardDodgePolicy(), screen=scene(**changes)))

    def test_known_board_proposes_down_but_rejected_proposal_does_not_advance(self):
        policy = BoardDodgePolicy()
        first = prime(policy)
        self.assertEqual(first.kind, 'down')
        second = policy.decide(scene(), [attack(seen=1.15)], match(), 1.15)
        self.assertEqual(second.kind, 'down')
        self.assertEqual(policy.stage, 0)
        self.assertIsNone(policy.pending)

    def test_completed_flip_waits_for_visual_confirmation(self):
        policy = BoardDodgePolicy()
        action = prime(policy)
        policy.completed(action, 1.05)
        self.assertIsNone(policy.decide(scene(), [attack(seen=1.2)], match(), 1.2))
        self.assertEqual(policy.stage, 0)
        next_action = policy.decide(scene(side='below', player_y=464), [attack(seen=1.25)], match(), 1.25)
        self.assertEqual(policy.stage, 1)
        self.assertEqual(next_action.kind, 'up')

    def test_airborne_down_intent_first_drops_then_flips_after_landing(self):
        policy = BoardDodgePolicy()
        high = scene(airborne=True, player_y=280)
        dropping = prime(policy, screen=high, tracks=[attack(y=270)])
        self.assertEqual(dropping.kind, 'drop')
        policy.completed(dropping, 1.05)
        next_action = policy.decide(scene(), [attack(seen=1.25)], match(), 1.25)
        self.assertEqual(next_action.kind, 'down')
        self.assertEqual(policy.stage, 0)

    def test_unconfirmed_input_does_not_silently_advance_after_timeout(self):
        policy = BoardDodgePolicy()
        policy.completed(prime(policy), 1.05)
        for time in (1.3, 1.55):
            self.assertIsNone(policy.decide(scene(), [attack(seen=time)], match(), time))
        self.assertIsNone(policy.decide(scene(), [attack(seen=1.75)], match(), 1.75))
        self.assertIsNone(policy.pending)
        self.assertIsNone(policy.recipe_id)

    def test_tracking_short_board_baits_with_jump_and_waits_before_close_flip(self):
        policy = BoardDodgePolicy()
        recognition = match('p2_blackfire_tracking_short')
        fire = attack(kind='blackfire')
        jump = prime(policy, tracks=[fire], recognition=recognition)
        self.assertEqual(jump.kind, 'jump')
        policy.completed(jump, 1.05)
        high = scene(player_y=370, airborne=True)
        distant = replace(fire, y=360, seen=1.25)
        self.assertIsNone(policy.decide(high, [distant], recognition, 1.25, jumps=1))
        self.assertTrue(policy.hold_for_approach)
        close = replace(fire, x=480, y=360, seen=1.3)
        dodge = policy.decide(high, [close], recognition, 1.3, jumps=1)
        self.assertEqual(dodge.kind, 'drop')
        self.assertFalse(policy.hold_for_approach)

    def test_tracking_hold_is_cleared_by_unknown_match_and_reset(self):
        policy = BoardDodgePolicy()
        recognition = match('p2_blackfire_tracking_short')
        fire = attack(kind='blackfire')
        policy.completed(prime(policy, tracks=[fire], recognition=recognition), 1.05)
        high = scene(player_y=370, airborne=True)
        policy.decide(high, [replace(fire, y=360, seen=1.25)], recognition, 1.25, jumps=1)
        self.assertTrue(policy.hold_for_approach)
        policy.decide(high, [replace(fire, y=360, seen=1.3)], match('unknown'), 1.3, jumps=1)
        self.assertFalse(policy.hold_for_approach)
        policy.hold_for_approach = True
        policy.reset()
        self.assertFalse(policy.hold_for_approach)

    def test_four_jumps_are_allowed_but_no_fifth_is_proposed(self):
        recognition = match('p4_clone_fire')
        for jumps in (0, 1, 2, 3):
            policy = BoardDodgePolicy()
            action = prime(policy, recognition=recognition, tracks=[attack(kind='blackfire')], jumps=jumps)
            self.assertEqual(action.kind, 'jump')
        self.assertIsNone(prime(BoardDodgePolicy(), recognition=recognition,
                                tracks=[attack(kind='blackfire')], jumps=4))

    def test_depth_board_ignores_tiny_background_and_huge_foreground(self):
        recognition = match('p2_illusion_shuriken_1')
        for size in ((8, 5), (240, 120)):
            self.assertIsNone(prime(BoardDodgePolicy(), recognition=recognition,
                tracks=[attack(kind='shuriken', width=size[0], height=size[1])]))
        self.assertIsNotNone(prime(BoardDodgePolicy(), recognition=recognition,
            tracks=[attack(kind='shuriken', width=40, height=25)]))

    def test_phase_four_uses_separate_single_jumps_and_confirms_landing(self):
        policy = BoardDodgePolicy()
        recognition = match('p4_transition')
        jump = prime(policy, recognition=recognition)
        self.assertEqual(jump.kind, 'jump')
        policy.completed(jump, 1.05)
        high = scene(player_y=350, airborne=True)
        drop = policy.decide(high, [attack(y=350, seen=1.25)], recognition, 1.25, jumps=1)
        self.assertEqual(drop.kind, 'drop')
        policy.completed(drop, 1.25)
        second_jump = policy.decide(scene(), [attack(seen=1.45)], recognition, 1.45)
        self.assertEqual(second_jump.kind, 'jump')
        policy.completed(second_jump, 1.45)
        drop = policy.decide(high, [attack(y=350, seen=1.65)], recognition, 1.65, jumps=1)
        policy.completed(drop, 1.65)
        self.assertIsNone(policy.decide(scene(), [attack(seen=1.85)], recognition, 1.85))
        self.assertTrue(policy.done)

    def test_entering_phase_four_midair_does_not_turn_single_jump_into_multijump(self):
        action = prime(BoardDodgePolicy(), screen=scene(player_y=320, airborne=True),
                       tracks=[attack(y=320)], recognition=match('p4_transition'), jumps=2)
        self.assertEqual(action.kind, 'drop')

    def test_near_rope_height_does_not_confirm_unknown_or_airborne_controls(self):
        self.assertTrue(BoardDodgePolicy._grounded(scene(grounded=True)))
        for grounded in (False, None):
            self.assertFalse(BoardDodgePolicy._grounded(scene(grounded=grounded)))
        self.assertTrue(BoardDodgePolicy._grounded(scene()))  # Old lightweight caller compatibility.

    def test_phase_four_does_not_start_a_single_jump_with_unknown_controls(self):
        policy = BoardDodgePolicy()
        action = prime(policy, screen=scene(grounded=None), recognition=match('p4_transition'))
        self.assertIsNone(action)
        self.assertEqual(policy.stage, 0)

    def test_unknown_landing_cannot_advance_to_phase_four_second_jump(self):
        policy = BoardDodgePolicy()
        recognition = match('p4_transition')
        jump = prime(policy, screen=scene(grounded=True), recognition=recognition)
        policy.completed(jump, 1.05)
        high = scene(player_y=350, airborne=True, grounded=False)
        drop = policy.decide(high, [attack(y=350, seen=1.25)], recognition, 1.25, jumps=1)
        self.assertEqual(drop.kind, 'drop')
        policy.completed(drop, 1.25)
        self.assertIsNone(policy.decide(scene(grounded=None), [attack(seen=1.45)], recognition, 1.45, jumps=1))
        self.assertEqual(policy.stage, 1)
        self.assertIsNotNone(policy.pending)
        second = policy.decide(scene(grounded=True), [attack(seen=1.5)], recognition, 1.5)
        self.assertEqual(second.kind, 'jump')
        self.assertEqual(policy.stage, 2)

    def test_low_air_drop_confirms_fresh_landing_without_arbitrary_pixel_displacement(self):
        policy = BoardDodgePolicy()
        recognition = match('p4_transition')
        prime(policy, screen=scene(grounded=True), recognition=recognition)
        policy.stage = 1
        low_air = scene(player_y=392, airborne=True, grounded=False)
        drop = policy.decide(low_air, [attack(y=392, seen=1.1)], recognition, 1.1, jumps=1)
        self.assertEqual(drop.kind, 'drop')
        policy.completed(drop, 1.1)
        landed = scene(player_y=394, grounded=True)
        self.assertIsNone(policy.decide(landed, [attack(seen=1.11)], recognition, 1.11, jumps=1))
        self.assertEqual(policy.stage, 1)
        self.assertIsNotNone(policy.pending)
        second = policy.decide(landed, [attack(seen=1.3)], recognition, 1.3)
        self.assertEqual(second.kind, 'jump')
        self.assertEqual(policy.stage, 2)

    def test_roll_requires_confirmed_below_rope_position(self):
        policy = BoardDodgePolicy()
        recognition = match('p4_tracking_compound')
        prime(policy, recognition=recognition, tracks=[attack(kind='blackfire')])
        policy.stage = 2
        airborne = policy.decide(scene(player_y=300, airborne=True),
            [attack(kind='blackfire', x=450, y=300, seen=1.1)], recognition, 1.1)
        self.assertEqual(airborne.kind, 'drop')
        below = policy.decide(scene(side='below', player_y=464),
            [attack(kind='blackfire', x=450, y=460, seen=1.2)], recognition, 1.2)
        self.assertEqual(below.kind, 'roll')

    def test_resetting_board_state_cannot_change_external_jump_budget(self):
        policy = BoardDodgePolicy()
        policy.reset()
        action = prime(policy, recognition=match('p4_clone_fire'),
                       tracks=[attack(kind='blackfire')], jumps=4)
        self.assertIsNone(action)

    def test_foreign_fallback_completion_cannot_advance_recipe(self):
        policy = BoardDodgePolicy()
        prime(policy)
        policy.completed(BossAction('jump', (1090, 630)), 1.05)
        self.assertIsNone(policy.pending)
        self.assertEqual(policy.stage, 0)

    def test_fire_three_jump_sequence_waits_for_third_group_and_descending_flames(self):
        policy = BoardDodgePolicy()
        recognition = match('p1_fire_1')
        prime(policy, recognition=recognition, tracks=[attack(kind='fire')])
        self.assertEqual(policy.wave, 1)
        policy.decide(scene(), [], recognition, 1.1)
        policy.decide(scene(), [attack(id=2, kind='fire', seen=1.15)], recognition, 1.15)
        self.assertEqual(policy.wave, 2)
        self.assertEqual(policy.stage, 0)
        policy.decide(scene(), [], recognition, 1.2)
        third = [attack(id=3, kind='fire', seen=1.25),
                 attack(id=4, kind='fire', y=455, vy=100, seen=1.25),
                 attack(id=5, kind='fire', y=465, vy=100, seen=1.25)]
        jump = policy.decide(scene(), third, recognition, 1.25)
        self.assertEqual(policy.wave, 3)
        self.assertEqual(jump.kind, 'jump')
        policy.completed(jump, 1.25)
        second = policy.decide(scene(player_y=370, airborne=True),
            [attack(id=3, kind='fire', y=360, seen=1.45)], recognition, 1.45, jumps=1)
        self.assertEqual(policy.stage, 1)
        self.assertEqual(second.kind, 'jump')

    def test_phase_two_transition_focuses_middle_swords_not_extreme_ones(self):
        policy = BoardDodgePolicy()
        tracks = [attack(id=1, x=480, y=310), attack(id=2, x=600, y=385),
                  attack(id=3, x=620, y=440)]
        dodge = prime(policy, recognition=match('p2_transition'), tracks=tracks)
        self.assertEqual(dodge.kind, 'down')
        self.assertIn(dodge.threat_id, (2, 3))

    def test_tracking_clone_fire_does_not_keep_hovering_after_bait_jump(self):
        policy = BoardDodgePolicy()
        recognition = match('p4_clone_fire')
        jump = prime(policy, recognition=recognition, tracks=[attack(kind='blackfire')])
        policy.completed(jump, 1.05)
        dodge = policy.decide(scene(player_y=350, airborne=True),
            [attack(kind='blackfire', x=450, y=350, seen=1.25)], recognition, 1.25, jumps=1)
        self.assertEqual(dodge.kind, 'drop')

    def test_unknown_match_immediately_discards_old_pending_recipe(self):
        policy = BoardDodgePolicy()
        policy.completed(prime(policy), 1.05)
        self.assertIsNone(policy.decide(scene(), [attack(seen=1.1)], match('unknown'), 1.1))
        self.assertIsNone(policy.pending)
        self.assertIsNone(policy.recipe_id)

    def test_time_reverse_and_long_gap_discard_pending_recipe(self):
        for time in (.9, 2.):
            policy = BoardDodgePolicy()
            policy.completed(prime(policy), 1.05)
            self.assertIsNone(policy.decide(scene(), [attack(seen=time)], match(), time))
            self.assertIsNone(policy.pending)

    def test_locked_visual_onset_starts_mode_without_an_extra_label_vote(self):
        policy = BoardDodgePolicy()
        action = policy.decide(scene(grounded=True), [], locked_match(), 1.)
        self.assertEqual(action.kind, 'down')
        self.assertTrue(policy.mode_active)
        self.assertEqual(policy.mode_epoch, 7)

    def test_locked_label_without_onset_or_attack_does_not_start_inputs(self):
        policy = BoardDodgePolicy()
        recognition = locked_match(onset=False)
        for time in (1., 1.1, 1.2):
            self.assertIsNone(policy.decide(scene(grounded=True), [], recognition, time))
            self.assertFalse(policy.mode_active)
        action = policy.decide(scene(grounded=True), [attack(seen=1.3)], recognition, 1.3)
        self.assertEqual(action.kind, 'down')
        self.assertTrue(policy.mode_active)

    def test_same_epoch_ambiguity_keeps_step_and_continues_confirmed_connector(self):
        policy = BoardDodgePolicy()
        first = policy.decide(scene(grounded=True), [], locked_match(), 1.)
        policy.completed(first, 1.)
        ambiguous = locked_match('unknown', locked=False, onset=False, confidence=0., margin=0.)
        follow = policy.decide(scene(side='below', player_y=464, grounded=None), [], ambiguous, 1.2)
        self.assertEqual(follow.kind, 'up')
        self.assertEqual(policy.stage, 1)
        self.assertEqual(policy.recipe_id, 'p2_fork_sword')
        self.assertEqual(policy.mode_epoch, 7)

    def test_epoch_change_discards_old_pending_and_enters_new_attack(self):
        policy = BoardDodgePolicy()
        policy.completed(policy.decide(scene(grounded=True), [], locked_match(), 1.), 1.)
        new_attack = locked_match('p4_transition', epoch=8)
        action = policy.decide(scene(grounded=True), [], new_attack, 1.2)
        self.assertEqual(action.kind, 'jump')
        self.assertEqual(policy.mode_epoch, 8)
        self.assertEqual(policy.recipe_id, 'p4_transition')
        self.assertEqual(policy.stage, 0)
        self.assertIsNone(policy.pending)

    def test_explicit_attack_end_exits_even_when_old_movement_is_pending(self):
        policy = BoardDodgePolicy()
        policy.completed(policy.decide(scene(grounded=True), [], locked_match(), 1.), 1.)
        self.assertIsNone(policy.decide(scene(), [], locked_match(ended=True, onset=False), 1.1))
        self.assertFalse(policy.mode_active)
        self.assertEqual(policy.mode_state, 'ended')
        self.assertIsNone(policy.pending)

    def test_fixed_flow_connectors_continue_without_new_threat_detections(self):
        policy = BoardDodgePolicy()
        recognition = locked_match('p1_shuriken_3')
        start = policy.decide(scene(grounded=True), [], recognition, 1.)
        self.assertEqual(start.kind, 'jump')
        policy.completed(start, 1.)
        continuation = locked_match('p1_shuriken_3', onset=False)
        drop = policy.decide(scene(player_y=350, airborne=True, grounded=False), [], continuation, 1.2, jumps=1)
        self.assertEqual(drop.kind, 'drop')
        policy.completed(drop, 1.2)
        down = policy.decide(scene(grounded=True), [], continuation, 1.4)
        self.assertEqual(down.kind, 'down')
        policy.completed(down, 1.4)
        roll = policy.decide(scene(side='below', player_y=464, grounded=None), [], continuation, 1.6)
        self.assertEqual(roll.kind, 'roll')
        policy.completed(roll, 1.6)
        up = policy.decide(scene(side='below', player_x=308, player_y=464, grounded=None), [], continuation, 1.8)
        self.assertEqual(up.kind, 'up')
        policy.completed(up, 1.8)
        self.assertIsNone(policy.decide(scene(grounded=True), [], continuation, 2.))
        self.assertTrue(policy.mode_active)
        self.assertEqual(policy.mode_state, 'waiting_attack')
        self.assertIsNone(policy.decide(scene(grounded=True), [], continuation, 2.2))
        new_wave = policy.decide(scene(grounded=True), [attack(kind='shuriken', seen=2.3)], continuation, 2.3)
        self.assertEqual(new_wave.kind, 'jump')

    def test_tracking_mode_still_waits_for_actual_approach_between_connectors(self):
        policy = BoardDodgePolicy()
        recognition = locked_match('p2_blackfire_tracking_short')
        jump = policy.decide(scene(grounded=True), [], recognition, 1.)
        policy.completed(jump, 1.)
        high = scene(player_y=350, airborne=True, grounded=False)
        continuation = locked_match('p2_blackfire_tracking_short', onset=False)
        self.assertIsNone(policy.decide(high, [], continuation, 1.2, jumps=1))
        self.assertTrue(policy.hold_for_approach)
        self.assertTrue(policy.mode_active)
        self.assertEqual(policy.stage, 1)
        dodge = policy.decide(high, [attack(kind='blackfire', x=450, y=350, seen=1.3)], continuation, 1.3, jumps=1)
        self.assertEqual(dodge.kind, 'drop')
        self.assertFalse(policy.hold_for_approach)

    def test_short_player_occlusion_preserves_mode_without_stale_landing_confirmation(self):
        policy = BoardDodgePolicy()
        recognition = locked_match('p4_transition')
        policy.completed(policy.decide(scene(grounded=True), [], recognition, 1.), 1.)
        continuation = locked_match('p4_transition', onset=False)
        drop = policy.decide(scene(player_y=350, airborne=True, grounded=False), [], continuation, 1.2, jumps=1)
        policy.completed(drop, 1.2)
        hidden = scene(player_x=None, player_y=None, grounded=True)
        self.assertIsNone(policy.decide(hidden, [], continuation, 1.3, jumps=1))
        self.assertTrue(policy.mode_active)
        self.assertEqual(policy.mode_state, 'occluded')
        self.assertEqual(policy.stage, 1)
        self.assertIsNotNone(policy.pending)
        self.assertIsNone(policy.decide(scene(grounded=None), [], continuation, 1.4, jumps=1))
        self.assertEqual(policy.stage, 1)
        second_jump = policy.decide(scene(grounded=True), [], continuation, 1.45)
        self.assertEqual(second_jump.kind, 'jump')
        self.assertEqual(policy.stage, 2)

    def test_public_suspend_expires_after_last_visible_player_window(self):
        policy = BoardDodgePolicy()
        policy.completed(policy.decide(scene(grounded=True), [], locked_match(), 1.), 1.)
        policy.suspend(1.2)
        self.assertTrue(policy.mode_active)
        self.assertIsNotNone(policy.pending)
        policy.suspend(1.36)
        self.assertFalse(policy.mode_active)
        self.assertIsNone(policy.pending)
        self.assertEqual(policy.mode_state, 'resync_required')

    def test_late_first_confirmation_does_not_blindly_start_the_fixed_jump_sequence(self):
        policy = BoardDodgePolicy()
        late = locked_match('p1_sword_3', progress=.78)
        self.assertIsNone(policy.decide(scene(grounded=True), [], late, 1.))
        self.assertTrue(policy.mode_active)
        self.assertTrue(policy.late_entry)
        self.assertEqual(policy.mode_state, 'late_entry')
        self.assertEqual(policy.stage, 0)
        early_looking_frame = locked_match('p1_sword_3', onset=False, progress=.1)
        self.assertIsNone(policy.decide(scene(grounded=True), [], early_looking_frame, 1.1))
        self.assertTrue(policy.late_entry)

    def test_late_entry_uses_current_gap_not_the_first_expert_step(self):
        policy = BoardDodgePolicy()
        late = locked_match('p1_sword_3', progress=.78)
        action = policy.decide(scene(grounded=True), [attack(y=300)], late, 1.)
        self.assertEqual(action.kind, 'down')  # Expert step zero would have been a jump.
        self.assertTrue(policy.late_entry)
        self.assertEqual(policy.mode_state, 'late_entry_response')
        policy.completed(action, 1.)
        policy.decide(scene(side='below', player_y=464, grounded=None), [],
                      locked_match('p1_sword_3', onset=False, progress=.85), 1.2)
        self.assertEqual(policy.stage, 0)
        self.assertTrue(policy.late_entry)

    def test_new_epoch_early_confirmation_can_restore_fixed_flow(self):
        policy = BoardDodgePolicy()
        self.assertIsNone(policy.decide(scene(grounded=True), [],
            locked_match('p1_sword_3', progress=.78), 1.))
        action = policy.decide(scene(grounded=True), [],
            locked_match('p1_sword_3', epoch=8, progress=.1), 1.2)
        self.assertEqual(action.kind, 'jump')
        self.assertFalse(policy.late_entry)
        self.assertEqual(policy.mode_epoch, 8)

    def test_visible_clear_and_new_early_formation_can_resync_same_epoch(self):
        policy = BoardDodgePolicy()
        policy.decide(scene(grounded=True), [], locked_match('p1_sword_3', progress=.78), 1.)
        policy.decide(scene(grounded=True), [], locked_match('p1_sword_3', onset=False, progress=.8), 1.2)
        action = policy.decide(scene(grounded=True), [attack(id=8, y=300, seen=1.3)],
            locked_match('p1_sword_3', onset=False, progress=.1), 1.3)
        self.assertEqual(action.kind, 'jump')
        self.assertFalse(policy.late_entry)
        self.assertEqual(policy.mode_epoch, 7)

    def test_new_projectile_alone_does_not_turn_late_entry_into_a_new_round(self):
        policy = BoardDodgePolicy()
        first = policy.decide(scene(grounded=True), [attack(y=300)],
            locked_match('p1_sword_3', progress=.78), 1.)
        self.assertEqual(first.kind, 'down')
        action = policy.decide(scene(grounded=True), [attack(id=8, y=300, seen=1.1)],
            locked_match('p1_sword_3', onset=False, progress=.1), 1.1)
        self.assertEqual(action.kind, 'down')
        self.assertTrue(policy.late_entry)

    def test_ambiguous_early_progress_cannot_resync_late_mode(self):
        policy = BoardDodgePolicy()
        policy.decide(scene(grounded=True), [], locked_match('p1_sword_3', progress=.78), 1.)
        uncertain = locked_match('unknown', locked=False, onset=False, confidence=0., margin=0., progress=.1)
        policy.decide(scene(grounded=True), [], uncertain, 1.2)
        policy.decide(scene(grounded=True), [attack(id=8, y=300, seen=1.3)], uncertain, 1.3)
        self.assertTrue(policy.late_entry)
        self.assertEqual(policy.recipe_id, 'p1_sword_3')

    def test_short_occlusion_does_not_clear_late_entry_state(self):
        policy = BoardDodgePolicy()
        recognition = locked_match('p1_sword_3', progress=.78)
        policy.decide(scene(grounded=True), [], recognition, 1.)
        policy.suspend(1.1)
        self.assertTrue(policy.late_entry)
        self.assertEqual(policy.mode_state, 'late_entry_occluded')
        self.assertIsNone(policy.decide(scene(grounded=True), [],
            locked_match('p1_sword_3', onset=False, progress=.8), 1.2))
        self.assertTrue(policy.late_entry)

    def test_failed_second_jump_cannot_restart_whole_transition_in_the_same_epoch(self):
        policy = BoardDodgePolicy()
        onset = locked_match('p4_transition', progress=.2)
        continuation = locked_match('p4_transition', onset=False, progress=.2)
        first = policy.decide(scene(grounded=True), [], onset, 1.)
        policy.completed(first, 1.)
        drop = policy.decide(scene(player_y=350, airborne=True, grounded=False), [], continuation, 1.2, jumps=1)
        policy.completed(drop, 1.2)
        second = policy.decide(scene(grounded=True), [], continuation, 1.4)
        self.assertEqual(second.kind, 'jump')
        policy.completed(second, 1.4)
        for now in (1.65, 1.9, 2.1):
            self.assertIsNone(policy.decide(scene(grounded=True), [], continuation, now))
        self.assertFalse(policy.mode_active)
        self.assertEqual(policy.blocked_epoch, 7)
        self.assertEqual(policy.blocked_board, 'p4_transition')
        response = policy.decide(scene(grounded=True), [attack(y=300, seen=2.15)], continuation, 2.15)
        self.assertEqual(response.kind, 'down')  # Current gap response, not transition step zero's jump.
        self.assertTrue(policy.late_entry)
        self.assertEqual(policy.stage, 0)
        self.assertEqual(policy.mode_state, 'late_entry_blocked_response')
        # A detector gap plus another projectile and the stale early progress
        # cannot turn this same failed epoch into a new complete transition.
        policy.decide(scene(grounded=True), [], continuation, 2.35)
        policy.decide(scene(grounded=True), [], continuation, 2.5)
        response = policy.decide(scene(grounded=True), [attack(id=55, y=300, seen=2.6)], continuation, 2.6)
        self.assertEqual(response.kind, 'down')
        self.assertTrue(policy.late_entry)
        self.assertEqual(policy.blocked_epoch, 7)
        recovered = policy.decide(scene(grounded=True), [],
            locked_match('p4_transition', epoch=8, progress=.2), 2.7)
        self.assertEqual(recovered.kind, 'jump')
        self.assertFalse(policy.late_entry)
        self.assertIsNone(policy.blocked_epoch)

    def test_long_occlusion_retains_blocked_epoch_through_reset_and_repeated_onset(self):
        policy = BoardDodgePolicy()
        policy.completed(policy.decide(scene(grounded=True), [],
            locked_match('p4_transition', progress=.2), 1.), 1.)
        policy.suspend(1.1)
        policy.suspend(1.36)
        self.assertEqual(policy.blocked_epoch, 7)
        self.assertFalse(policy.mode_active)
        policy.reset()
        self.assertEqual(policy.blocked_epoch, 7)
        response = policy.decide(scene(grounded=True), [attack(y=300, seen=1.4)],
            locked_match('p4_transition', onset=False, progress=.2), 1.4)
        self.assertEqual(response.kind, 'down')
        self.assertTrue(policy.late_entry)
        policy.reset()
        response = policy.decide(scene(grounded=True), [attack(y=300, seen=1.5)],
            locked_match('p4_transition', onset=True, progress=.2), 1.5)
        self.assertEqual(response.kind, 'down')
        self.assertTrue(policy.late_entry)

    def test_failed_epoch_can_resync_only_after_later_progress_and_real_new_round_evidence(self):
        policy = BoardDodgePolicy()
        policy.completed(policy.decide(scene(grounded=True), [],
            locked_match('p4_transition', progress=.2), 1.), 1.)
        policy.suspend(1.36)
        policy.decide(scene(grounded=True), [attack(y=300, seen=1.4)],
            locked_match('p4_transition', onset=False, progress=.2), 1.4)
        self.assertTrue(policy.late_entry)
        later = locked_match('p4_transition', onset=False, progress=.8)
        policy.decide(scene(grounded=True), [], later, 1.6)
        policy.decide(scene(grounded=True), [], later, 1.75)
        start_again = policy.decide(scene(grounded=True), [attack(id=55, y=300, seen=1.85)],
            locked_match('p4_transition', onset=False, progress=.2), 1.85)
        self.assertEqual(start_again.kind, 'jump')
        self.assertFalse(policy.late_entry)
        self.assertIsNone(policy.blocked_epoch)


if __name__ == '__main__':
    unittest.main()
