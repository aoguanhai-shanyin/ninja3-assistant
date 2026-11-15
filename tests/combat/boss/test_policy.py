import unittest

from ninja3_assistant.combat.battle import BladeTrack
from ninja3_assistant.combat.boss.policy import BossPolicy
from ninja3_assistant.combat.boss.screen import BossScreen


class BossPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = BossPolicy()
        self.above = BossScreen(target_boss=True, rope_y=480, side='above', player_x=300)

    def attack(self, *, kind='spin', x=650, y=400, width=100, height=100, vx=-500, ident=1):
        return BladeTrack(ident, x, y, width, height, vx, 0, 1.0, 8, kind)

    def test_does_not_act_without_target_or_player(self):
        self.assertIsNone(self.policy.decide(BossScreen(), [self.attack()], 1))
        self.assertIsNone(self.policy.decide(BossScreen(target_boss=True, rope_y=480, side='above'),
                                             [self.attack()], 1))

    def test_uses_recent_player_position_only_briefly_when_occluded(self):
        self.policy.decide(self.above, [], 1)
        obscured = BossScreen(target_boss=True, rope_y=480, side='above')
        current_attack = BladeTrack(1, 650, 400, 100, 100, -500, 0, 1.15, 8, 'spin')
        self.assertEqual(self.policy.decide(obscured, [current_attack], 1.15).kind, 'down')
        stale_attack = BladeTrack(1, 650, 400, 100, 100, -500, 0, 1.5, 8, 'spin')
        self.assertIsNone(self.policy.decide(obscured, [stale_attack], 1.5))

    def test_flips_under_rope_for_upper_attack(self):
        action = self.policy.decide(self.above, [self.attack()], 1)
        self.assertEqual((action.kind, action.point, action.threat_id), ('down', (190, 630), 1))
        self.policy.completed(action, 1)
        self.assertIsNone(self.policy.decide(self.above, [self.attack()], 1.7))

    def test_jumps_over_low_thin_blade(self):
        action = self.policy.decide(self.above, [self.attack(kind='thin', y=455, height=60)], 1)
        self.assertEqual((action.kind, action.point), ('jump', (1090, 630)))

    def test_flips_up_for_attack_below_the_rope(self):
        below = BossScreen(target_boss=True, rope_y=480, side='below', player_x=300)
        action = self.policy.decide(below, [self.attack(y=555, height=80)], 1)
        self.assertEqual((action.kind, action.point), ('up', (1090, 630)))

    def test_ignores_attack_moving_away_or_too_early(self):
        self.assertIsNone(self.policy.decide(self.above, [self.attack(vx=500)], 1))
        self.assertIsNone(self.policy.decide(self.above, [self.attack(x=1100, vx=-200)], 1))

    def test_does_not_flip_into_an_attack_below(self):
        lower = self.attack(y=580, height=230, ident=2)
        self.assertIsNone(self.policy.decide(self.above, [lower], 1))
        self.assertIsNone(self.policy.decide(self.above, [self.attack(), lower], 1))

    def test_fast_orb_needs_two_observations_but_not_four(self):
        orb = BladeTrack(5, 820, 400, 32, 32, -800, 0, 1.0, 2, 'orb')
        self.assertEqual(self.policy.decide(self.above, [orb], 1).kind, 'down')
        one_frame = BladeTrack(5, 820, 400, 32, 32, -800, 0, 1.0, 1, 'orb')
        self.assertIsNone(self.policy.decide(self.above, [one_frame], 1))

    def test_predicts_orb_that_will_cross_below_the_rope(self):
        below = BossScreen(target_boss=True, rope_y=421, side='below', player_x=281)
        orb = BladeTrack(5, 558, 358, 32, 32, -746, 290, 1.0, 6, 'orb')
        self.assertEqual(self.policy.decide(below, [orb], 1).kind, 'up')


    def test_dodge_takes_priority_over_ready_skills(self):
        ready = BossScreen(target_boss=True, rope_y=480, side='above', player_x=300,
                           weapon_ready=True)
        self.assertEqual(self.policy.decide(ready, [self.attack()], 1).kind, 'down')
        skill = self.policy.decide(ready, [], 2)
        self.assertEqual((skill.kind, skill.point), ('weapon', (380, 630)))
        self.policy.completed(skill, 2)
        self.assertIsNone(self.policy.decide(ready, [], 3))


if __name__ == '__main__':
    unittest.main()
