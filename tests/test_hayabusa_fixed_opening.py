import unittest
from types import SimpleNamespace

from hayabusa_fixed_opening import FixedOpening


def screen(**values):
    return SimpleNamespace(**(dict(active=True, result=None, right_control='jump',
                                    left_control='down', grounded=True) | values))


class FixedOpeningTests(unittest.TestCase):
    def setUp(self):
        self.policy = FixedOpening(max_start_age=.5, ack_timeout=.4)

    def test_unknown_family_and_late_event_never_act(self):
        self.assertFalse(self.policy.start(1, None, 1, 1))
        self.assertFalse(self.policy.start(2, 'fire', 1, 1))
        self.assertFalse(self.policy.start(3, 'p1_shuriken_2', 1, 1.51))
        self.assertIsNone(self.policy.decide(screen(), 1.51, 1.51))

    def test_one_input_requires_post_press_ui_ack(self):
        self.assertTrue(self.policy.start(1, 'p1_shuriken_2', 1, 1))
        action = self.policy.decide(screen(), 1.1, 1.1)
        self.assertEqual((action.kind, action.point), ('down', (190, 630)))
        self.assertIsNone(self.policy.decide(screen(), 1.11, 1.11))
        self.assertTrue(self.policy.completed(action, 1.12))
        self.assertIsNone(self.policy.decide(screen(right_control='up'), 1.13, 1.10))
        self.assertEqual(self.policy.state, 'waiting_ack')
        self.assertIsNone(self.policy.decide(screen(right_control='up'), 1.2, 1.2))
        self.assertEqual(self.policy.state, 'complete')
        self.assertFalse(self.policy.start(1, 'p1_shuriken_2', 1.3, 1.3))

    def test_control_mode_overrules_bbox_and_no_repeat_when_below(self):
        self.policy.start(1, 'p1_shuriken_4', 1, 1)
        self.assertIsNone(self.policy.decide(screen(right_control='up', grounded=False, side='above'), 1.1, 1.1))
        self.assertEqual(self.policy.state, 'complete')

    def test_stale_frame_and_interrupt_invalidate_event(self):
        self.policy.start(1, 'p1_shuriken_2', 1, 1)
        self.assertIsNone(self.policy.decide(screen(), 1.3, 1.0))
        self.assertFalse(self.policy.start(1, 'p1_shuriken_2', 1.3, 1.3))
        self.policy.start(2, 'p1_shuriken_2', 2, 2)
        self.assertIsNone(self.policy.decide(screen(active=False), 2.1, 2.1))
        self.assertFalse(self.policy.start(2, 'p1_shuriken_2', 2.1, 2.1))

    def test_unknown_control_and_missing_ack_never_spam(self):
        self.policy.start(1, 'p1_shuriken_2', 1, 1)
        self.assertIsNone(self.policy.decide(screen(right_control='unknown'), 1.1, 1.1))
        action = self.policy.decide(screen(), 1.2, 1.2)
        self.policy.completed(action, 1.2)
        self.assertIsNone(self.policy.decide(screen(), 1.61, 1.61))
        self.assertEqual(self.policy.state, 'idle')
        self.assertFalse(self.policy.start(1, 'p1_shuriken_2', 1.7, 1.7))

    def test_fire_probe_is_explicit_opt_in_and_waits_for_ascent(self):
        self.assertFalse(self.policy.start(1, 'fire_probe', 1, 1))
        probe = FixedOpening(max_start_age=.8, ack_timeout=.4, fire_probe_delay=.2)
        self.assertTrue(probe.start(1, 'fire_probe', 1, 1))
        self.assertIsNone(probe.decide(screen(player_y=400), 1.1, 1.1))
        action = probe.decide(screen(player_y=400), 1.21, 1.21)
        self.assertEqual(action.kind, 'jump')
        probe.completed(action, 1.22)
        self.assertIsNone(probe.decide(screen(player_y=399), 1.25, 1.25))
        self.assertEqual(probe.state, 'waiting_ack')
        self.assertIsNone(probe.decide(screen(player_y=380), 1.3, 1.3))
        self.assertEqual(probe.state, 'complete')
        self.assertFalse(probe.start(1, 'fire_probe', 1.4, 1.4))

    def _air_probe(self):
        probe = FixedOpening(max_start_age=.8, ack_timeout=.4, fire_probe_delay=.2,
                             fire_air_jump_delay=.2, fire_air_jump_deadline=.5)
        self.assertTrue(probe.start(1, 'fire_probe', 1, 1))
        first = probe.decide(screen(player_y=400), 1.21, 1.21)
        self.assertTrue(probe.completed(first, 1.22))
        airborne = screen(player_y=380, grounded=False, airborne=True)
        self.assertIsNone(probe.decide(airborne, 1.3, 1.3))
        self.assertEqual(probe.state, 'waiting_air_jump')
        return probe, airborne

    def test_opt_in_second_jump_requires_ascent_delay_and_real_callback(self):
        probe, airborne = self._air_probe()
        self.assertIsNone(probe.decide(airborne, 1.4, 1.4))
        second = probe.decide(airborne, 1.43, 1.43)
        self.assertEqual(second.kind, 'jump')
        self.assertEqual(probe.state, 'proposed_air_jump')
        self.assertIsNone(probe.decide(airborne, 1.44, 1.44))
        self.assertTrue(probe.completed(second, 1.45))
        self.assertEqual(probe.state, 'complete')
        self.assertEqual(probe.reason, 'air jump pressed; movement effect unverified')
        self.assertFalse(probe.completed(second, 1.46))
        self.assertIsNone(probe.decide(airborne, 1.5, 1.5))
        self.assertFalse(probe.start(1, 'fire_probe', 2, 2))

    def test_second_jump_aborts_on_landing_stale_or_pause(self):
        for event in ('landed', 'stale', 'pause'):
            with self.subTest(event=event):
                probe, airborne = self._air_probe()
                if event == 'pause':
                    probe.abort('pause')
                elif event == 'landed':
                    self.assertIsNone(probe.decide(screen(player_y=400, grounded=True), 1.43, 1.43))
                else:
                    self.assertIsNone(probe.decide(airborne, 1.55, 1.30))
                self.assertEqual(probe.state, 'idle')
                self.assertIsNone(probe.decide(airborne, 1.6, 1.6))

    def test_second_jump_requires_valid_controls_and_expires(self):
        probe, airborne = self._air_probe()
        self.assertIsNone(probe.decide(screen(player_y=350, grounded=False, airborne=False), 1.43, 1.43))
        self.assertIsNone(probe.decide(screen(player_y=350, grounded=False, airborne=True, right_control='up'), 1.50, 1.50))
        self.assertIsNone(probe.decide(airborne, 1.73, 1.73))
        self.assertEqual(probe.reason, 'air jump window expired')

    def test_second_jump_option_requires_explicit_bounded_window(self):
        for args in ({'fire_air_jump_delay':.2},
                     {'fire_air_jump_delay':.2, 'fire_air_jump_deadline':.1},
                     {'fire_air_jump_delay':.2, 'fire_air_jump_deadline':.5}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                FixedOpening(max_start_age=.8, ack_timeout=.4, **args)

    def test_repeated_first_press_callback_cannot_ack_second_action(self):
        probe = FixedOpening(max_start_age=.8, ack_timeout=.4, fire_probe_delay=.2,
                             fire_air_jump_delay=.2, fire_air_jump_deadline=.5)
        probe.start(1, 'fire_probe', 1, 1)
        first = probe.decide(screen(player_y=400), 1.21, 1.21)
        self.assertTrue(probe.completed(first, 1.22))
        airborne = screen(player_y=380, grounded=False, airborne=True)
        probe.decide(airborne, 1.3, 1.3)
        second = probe.decide(airborne, 1.43, 1.43)
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertFalse(probe.completed(first, 1.44))
        self.assertEqual(probe.state, 'proposed_air_jump')
        self.assertTrue(probe.completed(second, 1.45))


if __name__ == '__main__':
    unittest.main()
