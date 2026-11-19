import unittest
from unittest.mock import Mock

import numpy as np

from ninja3_assistant.activities.bounty.combat import BountyBattleState, BountyCombat, BountyCombatVision
from ninja3_assistant.runtime.realtime_battle import RealtimeBattleKernel
from ninja3_assistant.runtime.runner import RunControl


class BountyCombatTest(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.control = RunControl()
        self.kernel = RealtimeBattleKernel(self.control,clock=lambda:self.now)
        self.driver = Mock()
        self.driver.movement_remaining.return_value = 0.
        self.driver.schedule_movement.return_value = 1
        self.driver.plan.return_value = 1
        self.kernel.attach(self.driver)
        self.control.set_release_input(self.driver.stop_all)
        self.kernel.capture(Mock())
        self.combat = BountyCombat(self.kernel)

    def next_frame(self, seconds=.01):
        self.now += seconds
        self.kernel.capture(Mock())

    def test_ground_starts_bounded_skating_without_using_skills(self):
        self.combat.update(BountyBattleState(active=True,ground=True),self.now)
        self.driver.schedule_movement.assert_called_once_with(3.,captured_at=10.)
        self.driver.plan.assert_not_called()
        self.assertTrue(self.combat.moving)

    def test_skating_continues_across_frames_without_restarting_each_cycle(self):
        state = BountyBattleState(active=True,ground=True)
        self.combat.update(state,self.now)
        self.driver.movement_remaining.return_value = 2.
        self.next_frame()
        self.combat.update(state,self.now)
        self.assertEqual(self.driver.schedule_movement.call_count,1)
        self.driver.release.assert_not_called()

    def test_rope_cancels_ground_skating_and_never_flips_or_jumps(self):
        self.combat.update(BountyBattleState(active=True,ground=True),self.now)
        self.next_frame()
        rope = BountyBattleState(active=True,on_rope=True)
        self.combat.update(rope,self.now)
        self.combat.update(rope,self.now)
        self.driver.release.assert_called_once()
        self.driver.plan.assert_not_called()
        self.driver.tap.assert_not_called()
        self.driver.priority_tap.assert_not_called()

    def test_boss_health_stops_skating_and_plans_skill_and_both_weapons_independently(self):
        self.combat.update(BountyBattleState(active=True,ground=True),self.now)
        self.next_frame()
        state = BountyBattleState(True,True,False,True,(55,60),((380,630),(900,630)))
        self.combat.update(state,self.now)
        self.driver.release.assert_called_once()
        plans = {call.args[0]:call.args[1] for call in self.driver.plan.call_args_list}
        self.assertEqual([a.point for a in plans[3]],[(55,60)]*3)
        self.assertEqual([a.point for a in plans[4]],[(380,630),(900,630)])
        self.assertLess(max(a.offset_ms+a.hold_ms for p in plans.values() for a in p),200)

    def test_boss_skills_on_rope_do_not_send_movement_commands(self):
        self.combat.update(BountyBattleState(True,False,True,True,(70,330)),self.now)
        self.driver.schedule_movement.assert_not_called()
        self.driver.priority_tap.assert_not_called()
        self.assertEqual(self.driver.plan.call_args.args[0],3)

    def test_first_boss_ninjutsu_starts_immediately_after_actual_recognition_cost(self):
        self.now += .008
        self.combat.update(BountyBattleState(True,True,False,True,(55,60)),self.now)
        first = self.driver.plan.call_args.args[1][0]
        dispatch_at = self.kernel.captured_at+first.offset_ms/1000
        self.assertGreaterEqual(dispatch_at,self.now)
        self.assertLessEqual(dispatch_at-self.now,.003)
        self.assertEqual(first.point,(55,60))

    def test_boss_skill_plans_repeat_only_when_due_and_menu_cancels_them(self):
        state = BountyBattleState(True,False,True,True,(70,330),((380,630),(900,630)))
        self.combat.update(state,self.now)
        self.next_frame(.1)
        self.combat.update(state,self.now)
        self.assertEqual(self.driver.plan.call_count,2)
        self.next_frame(.11)
        self.combat.update(state,self.now)
        self.assertEqual(self.driver.plan.call_count,3)
        self.next_frame()
        self.combat.update(BountyBattleState(),self.now)
        self.assertEqual({c.args[0] for c in self.driver.cancel_plan.call_args_list},{3,4})

    def test_confirmed_change_replaces_the_old_ninjutsu_point(self):
        self.combat.update(BountyBattleState(True,False,True,True,(70,330)),self.now)
        self.next_frame()
        self.combat.update(BountyBattleState(True,False,True,True),self.now)
        self.driver.cancel_plan.assert_not_called()
        self.next_frame()
        self.combat.update(BountyBattleState(True,False,True,True,(55,60)),self.now)
        self.driver.cancel_plan.assert_called_once_with(3)
        self.assertTrue(all(action.point == (55,60) for action in self.driver.plan.call_args.args[1]))

    def test_stale_capture_pause_and_stop_cannot_dispatch_more_actions(self):
        state = BountyBattleState(True,True,False,True,(55,60))
        self.now += .3
        self.combat.update(state,self.now)
        self.driver.plan.assert_not_called()
        self.next_frame()
        self.control.pause()
        self.combat.update(state,self.now)
        self.driver.plan.assert_not_called()
        self.driver.stop_all.assert_called_once()
        self.control.resume()
        self.combat.update(state,self.now)
        self.driver.plan.assert_not_called()
        self.next_frame()
        self.control.request_stop()
        self.combat.update(state,self.now)
        self.driver.plan.assert_not_called()

    def test_saturated_short_icons_and_grey_menu_lines_are_not_boss_health(self):
        frame = np.zeros((720,1280,3),np.uint8)
        frame[65:80,430:810] = 200
        self.assertFalse(BountyCombatVision.boss_bar(frame))
        frame[65:80,430:470] = (245,165,0)
        self.assertFalse(BountyCombatVision.boss_bar(frame))
        frame[40:50,450:530] = 240
        frame[65:80,430:810] = (245,165,0)
        self.assertTrue(BountyCombatVision.boss_bar(frame))


if __name__ == '__main__':
    unittest.main()
