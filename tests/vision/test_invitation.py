from ninja3_assistant.paths import project_root
import json
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
    def test_real_bounty_invitation_with_ocr_omission(self):
        evidence = ROOT/'data/evidence/2026-11-16/daily-exploration/175313-trial-guide-next-1'
        gray = cv2.imread(str(evidence.with_suffix('.png')),cv2.IMREAD_GRAYSCALE)
        ocr = json.loads(evidence.with_suffix('.json').read_text(encoding='utf-8'))['ocr']
        invitation = InvitationVision().bounty_invitation(gray,ocr)
        self.assertIsNotNone(invitation)
        self.assertEqual((invitation.source,invitation.grade),('friend','S'))
        self.assertTrue(700 <= invitation.accept[0] <= 920 and 490 <= invitation.accept[1] <= 565)

    def test_left_aligned_friend_sender_keeps_first_ocr_character(self):
        evidence = ROOT/'data/evidence/2026-11-16/daily-exploration/175313-trial-guide-next-1.png'
        gray = cv2.imread(str(evidence),cv2.IMREAD_GRAYSCALE)
        words = [{'text':text,'bbox':box} for text,box in (
            ('来',[104,213,21,24]), ('自',[131,213,17,23]),
            ('好',[152,214,23,23]), ('友',[178,213,21,23]),
            ('A级每日赏',[566,195,161,29]),
        )]
        invitation = InvitationVision().bounty_invitation(gray,{'words':words})
        self.assertIsNotNone(invitation)
        self.assertEqual((invitation.source,invitation.grade),('friend','A'))

    def test_bounty_invitation_requires_sender_and_unambiguous_title(self):
        evidence = ROOT/'data/evidence/2026-11-16/daily-exploration/175313-trial-guide-next-1.png'
        gray = cv2.imread(str(evidence),cv2.IMREAD_GRAYSCALE)
        vision = InvitationVision()
        def read(sender, title):
            return vision.bounty_invitation(gray,{'words':[
                {'text':sender,'bbox':[140,210,100,25]},
                {'text':title,'bbox':[560,195,160,25]},
            ]})
        for source,sender in (('friend','来自好友'),('family','来自族员')):
            for grade in 'DCBAS':
                invite = read(sender,grade+'级每日悬赏')
                self.assertEqual((invite.source,invite.grade),(source,grade))
        for sender,title in (('来自族员','家族BOSS组队邀请'),('来自好友','SS级召令悬赏'),
                             ('来自好友','A级竞技场'),('来自玩家','A级每日悬赏'),
                             ('来自好友','每日悬赏')):
            self.assertIsNone(read(sender,title))
        self.assertIsNone(vision.bounty_invitation(np.zeros((720,1280),np.uint8),{'words':[]}))

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
