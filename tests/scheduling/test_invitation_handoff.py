import unittest
from unittest.mock import Mock

import numpy as np

from ninja3_assistant.runtime.runner import RunControl
from ninja3_assistant.scheduling.invitation_handoff import bounty_handoff, clear_handled_invitation, InvitationHandoff


class InvitationHandoffTest(unittest.TestCase):
    def test_only_an_enabled_grade_at_a_safe_checkpoint_is_handed_off(self):
        frame = np.zeros((720,1280,3),np.uint8)
        control = RunControl()
        control.bounty_invite_targets = {'A':'daily_bounty'}
        vision = Mock()
        vision.bounty_invitation.return_value = Mock(grade='A')
        ocr = Mock()
        ocr.read.return_value = {'words':[]}
        self.assertIsNone(bounty_handoff(frame,vision,ocr,control,safe=False))
        ocr.read.assert_not_called()
        self.assertEqual(bounty_handoff(frame,vision,ocr,control,safe=True),
                         InvitationHandoff('daily_bounty'))
        vision.bounty_invitation.return_value.grade = 'S'
        self.assertIsNone(bounty_handoff(frame,vision,ocr,control,safe=True))
        vision.bounty_invitation.return_value = None
        self.assertIsNone(bounty_handoff(frame,vision,ocr,control,safe=True))

    def test_stop_during_ocr_never_hands_off(self):
        frame = np.zeros((720,1280,3),np.uint8)
        control = RunControl()
        control.bounty_invite_targets = {'A':'daily_bounty'}
        ocr,vision = Mock(),Mock()
        def read(*args,**kwargs):
            control.request_stop()
            return {'words':[]}
        ocr.read.side_effect = read
        self.assertIsNone(bounty_handoff(frame,vision,ocr,control,safe=True))
        vision.bounty_invitation.assert_not_called()

    def test_a_new_invitation_is_allowed_after_a_confirmed_safe_page(self):
        frame = np.zeros((720,1280,3),np.uint8)
        control = RunControl()
        control.bounty_invite_targets = {'A':'daily_bounty'}
        vision,ocr = Mock(),Mock()
        vision.bounty_invitation.return_value.grade = 'A'
        ocr.read.return_value = {'words':[]}
        self.assertEqual(bounty_handoff(frame,vision,ocr,control,safe=True),InvitationHandoff('daily_bounty'))
        control.bounty_invite_suppressed = True
        self.assertIsNone(bounty_handoff(frame,vision,ocr,control,safe=True))
        clear_handled_invitation(control)
        self.assertEqual(bounty_handoff(frame,vision,ocr,control,safe=True),InvitationHandoff('daily_bounty'))


if __name__ == '__main__':
    unittest.main()
