from ninja3_assistant.paths import project_root
import unittest

import cv2
import numpy as np

from ninja3_assistant.vision.invitation import InvitationVision
from ninja3_assistant.activities.intelligence.vision import ActivityVision
from ninja3_assistant.activities.intelligence.flow import ActivityFlow
from ninja3_assistant.activities.weekend.runner import RaceVision

ROOT = project_root()
EVIDENCE = ROOT.parent/'android-icon-clicker/build/evidence/bounty'


class InvitationTest(unittest.TestCase):
    def test_family_invitation_uses_same_reject_action(self):
        path=ROOT/'data/evidence/2026-11-15/invitations/family.png'
        image=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        point=InvitationVision().detect(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY))
        self.assertIsNotNone(point)
        self.assertTrue(360<=point[0]<=580)
    def test_blank_frame_never_produces_reject(self):
        self.assertIsNone(InvitationVision().detect(np.zeros((720,1280),np.uint8)))

    def test_real_invite_is_rejected_across_activity_and_race(self):
        for name in ('05-current-after-retry.png','24-perf-screen.png'):
            image = cv2.cvtColor(cv2.imread(str(EVIDENCE/name)),cv2.COLOR_BGR2RGB)
            point = InvitationVision().detect(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY))
            self.assertIsNotNone(point)
            self.assertTrue(360 <= point[0] <= 580 and 490 <= point[1] <= 565)
            screen = ActivityVision().detect(image)
            self.assertEqual(screen.kind,'invite')
            decision = ActivityFlow().step(screen,0)
            self.assertEqual(decision.action,'invite')
            self.assertEqual(decision.point,point)
            self.assertEqual(RaceVision().detect(image),'invite')
