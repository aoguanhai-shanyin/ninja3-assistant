"""Replay recorded clan invitations without controlling the game."""
import json
import unittest

import cv2

from ninja3_assistant.activities.bounty.daily_flow import DailyBountyFlow
from ninja3_assistant.activities.bounty.daily_vision import DailyBountyVision
from ninja3_assistant.paths import project_root


class ClanRecruitmentFramesTest(unittest.TestCase):
    def test_152_opens_invites_without_recognizing_the_caption(self):
        folder = project_root()/'data/runtime/daily-cleanup/runs/20261119-134910-075600-bounty-A'
        path = folder/'0019-bounty_room-wait.png'
        if not path.is_file():
            self.skipTest('requires the local 152 bounty replay')
        vision = DailyBountyVision(grade='A')
        frame = cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        screen = vision.observe(frame,{'words':[]})
        self.assertEqual((screen.page,screen.minimum_players),('bounty_room',True))
        flow = DailyBountyFlow('bounty','A')
        flow.room_owned = flow.room_trusted = True  # The immediately preceding action created the A room.
        decision = flow.step(screen,vision)
        self.assertEqual(decision.status,'打开悬赏队伍邀请列表')
        self.assertTrue(vision.current_action(screen,decision,frame))
        unavailable = frame.copy()
        unavailable[300:376,595:685] = 0
        self.assertIsNone(vision.observe(unavailable,{'words':[]}).invite_open)
        self.assertFalse(vision.current_action(screen,decision,unavailable))

    def test_send_invitation_once_and_wait_for_teammate(self):
        folder = project_root()/'data/evidence/2026-11-17/daily-exploration'
        vision = DailyBountyVision()

        def read(stem):
            record = json.loads((folder/(stem+'.json')).read_text(encoding='utf-8'))
            frame = cv2.cvtColor(cv2.imread(str(folder/(stem+'.png'))),cv2.COLOR_BGR2RGB)
            return vision.observe(frame,record['ocr'])

        panel = read('112739-bounty-recruit-clan')
        self.assertEqual((panel.page,panel.recruit_tab),('bounty_recruit','family'))
        points = [point for _,point in panel.recruit_candidates]
        self.assertIn((1211,232),points)
        self.assertNotIn((1211,133),points)  # The first member is in battle.

        flow = DailyBountyFlow('bounty','A')
        flow.room_owned = flow.room_trusted = True
        flow.recruit_stage = 'family'
        decision = flow.step(panel,vision)
        self.assertEqual(decision.point,(1211,232))
        self.assertTrue(vision.current_action(panel,decision,panel.frame))
        flow.completed(decision,panel)

        sent = read('112809-bounty-recruit-clan-send')
        self.assertIn('邀请已发送',sent.text())
        after = flow.step(sent,vision)
        self.assertEqual(after.point,(750,350))
        self.assertIn('等待队友加入',after.status)


if __name__ == '__main__':
    unittest.main()
