"""Room grade provenance when a verified invitation has been accepted."""
import unittest
from types import SimpleNamespace

import numpy as np

from ninja3_assistant.activities.bounty.daily_flow import DailyBountyFlow
from ninja3_assistant.activities.bounty.daily_vision import DailyBountyScreen


def screen(page, **fields):
    return DailyBountyScreen(np.zeros((720,1280,3),np.uint8),{'words':[]},page,**fields)


class InvitationRoomGradeTest(unittest.TestCase):
    def accepted(self, grade):
        flow = DailyBountyFlow('bounty_s' if grade == 'S' else 'bounty',
                               'D' if grade == 'S' else grade)
        invite = screen('invite',point=(470,525),invitation=SimpleNamespace(
            source='family',grade=grade,accept=(810,525)))
        decision = flow.step(invite,None)
        self.assertEqual(decision.point,(810,525))
        flow.completed(decision,invite)
        return flow

    def test_verified_invitation_supplies_missing_b_c_and_s_badges(self):
        for grade in ('B','C','S'):
            with self.subTest(grade=grade):
                flow = self.accepted(grade)
                room = screen('bounty_room',grade=None,ready=(1100,660))
                self.assertEqual(flow.step(room,None).point,(1100,660))

    def test_observed_conflict_clears_invitation_trust(self):
        flow = self.accepted('B')
        self.assertEqual(flow.step(screen('bounty_room',grade='A'),None).action,'unavailable')
        self.assertEqual(flow.step(screen('bounty_room',grade=None,ready=(1100,660)),None).action,'unavailable')

    def test_stable_navigation_after_failed_invitation_clears_trust(self):
        flow = self.accepted('C')
        flow.step(screen('home'),None)
        room = screen('bounty_room',grade=None,ready=(1100,660))
        self.assertEqual(flow.step(room,None).action,'unavailable')

    def test_leaving_an_invited_room_does_not_trust_a_later_unknown_room(self):
        flow = self.accepted('C')
        room = screen('bounty_room',grade=None,ready=(1100,660))
        self.assertEqual(flow.step(room,None).point,(1100,660))
        flow.step(screen('home'),None)
        self.assertEqual(flow.step(room,None).action,'unavailable')

    def test_unexecuted_accept_does_not_authorize_a_room(self):
        flow = DailyBountyFlow('bounty','B')
        invite = screen('invite',point=(470,525),invitation=SimpleNamespace(
            source='friend',grade='B',accept=(810,525)))
        self.assertEqual(flow.step(invite,None).point,(810,525))
        room = screen('bounty_room',grade=None,ready=(1100,660))
        self.assertEqual(flow.step(room,None).action,'unavailable')


if __name__ == '__main__':
    unittest.main()
