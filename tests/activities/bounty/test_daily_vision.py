"""Local saved-frame bounty regressions; no emulator or game input."""
import json
import unittest

import cv2
import numpy as np

from ninja3_assistant.activities.bounty.daily_flow import DailyBountyFlow
from ninja3_assistant.activities.bounty.daily_vision import DailyBountyVision
from ninja3_assistant.paths import project_root


DATA=project_root()/'data/evidence/2026-11-16/daily-exploration'


class DailyBountyVisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision=DailyBountyVision()

    def read(self, stem):
        record=json.loads((DATA/(stem+'.json')).read_text(encoding='utf-8'))
        frame=cv2.cvtColor(cv2.imread(str(DATA/(stem+'.png'))),cv2.COLOR_BGR2RGB)
        return self.vision.observe(frame,record['ocr'])

    def test_daily_quota_does_not_include_the_plus_icon_ocr_zero(self):
        s=self.read('133910-bounty-daily-ready')
        self.assertEqual(s.page,'bounty_daily')
        self.assertEqual(s.remaining,3)
        self.assertEqual(s.grade,'A')

    def test_free_dialog_is_recognized_before_its_background(self):
        s=self.read('133930-bounty-free-dialog')
        self.assertEqual(self.vision.navigation(s.frame)[0],'bounty_free')
        self.assertEqual(s.free_claim,(640,480))
        decision=DailyBountyFlow().step(s,self.vision)
        self.assertTrue(self.vision.current_action(s,decision,s.frame))
        self.assertFalse(self.vision.current_action(s,decision,np.zeros_like(s.frame)))

    def test_grade_dialog_selects_only_target_then_confirms_the_highlight(self):
        flow=DailyBountyFlow('bounty','D')
        a=self.read('134030-bounty-grade-dialog')
        self.assertEqual(self.vision.navigation(a.frame)[0],'bounty_grade')
        self.assertEqual(flow.step(a,self.vision).point,(640,246))
        d=self.read('134107-bounty-grade-d-selected')
        self.assertEqual(d.grade,'D')
        self.assertEqual(flow.step(d,self.vision).point,(640,490))

    def test_empty_room_and_leave_modal_are_bounty_screens_without_battle_hud(self):
        room=self.read('134331-bounty-created-room')
        self.assertEqual(self.vision.navigation(room.frame)[0],'bounty_room')
        self.assertEqual(room.grade,'D')
        self.assertTrue(room.minimum_players)
        self.assertIsNone(room.start)
        leave=self.read('134425-bounty-leave-room')
        self.assertEqual(leave.page,'bounty_leave')
        self.assertEqual(self.vision.navigation(leave.frame)[0],'bounty_leave')

    def test_other_grade_team_is_not_a_d_team(self):
        s=self.read('134614-bounty-quickjoin-d')
        self.assertEqual(s.grade,'D')
        self.assertTrue(s.join_failed)
        self.assertIsNone(s.join)
        self.assertIsNotNone(self.vision._join(s.frame,'A'))

    def test_no_owned_orders_is_a_search_page_not_a_shop(self):
        s=self.read('135141-bounty-my-orders')
        self.assertEqual(s.page,'bounty_own_empty')
        self.assertEqual(self.vision.navigation(s.frame)[0],'bounty_own_empty')
        self.assertEqual(DailyBountyFlow('bounty_s').step(s,self.vision).point,(1115,670))

    def test_bounty_rewards_can_close_on_blank_space_without_pressing_an_item(self):
        s=self.read('135654-bounty-after-first-battle')
        self.assertEqual(s.page,'reward')
        self.assertEqual(self.vision.navigation(s.frame),('reward',(1150,685)))
        self.assertEqual(DailyBountyFlow().step(s,self.vision).point,(1150,685))
        # The two visible labels also identify the menu without running OCR.
        self.assertEqual(self.vision.observe(s.frame,{'words':[]}).confirm,(1150,685))

    def test_d_result_title_resumes_after_reward_recovery_without_ocr_title(self):
        s=self.read('140316-bounty-result-dismiss-footer')
        self.assertEqual(s.page,'result')
        self.assertEqual(s.grade,'D')
        self.assertEqual(self.vision.navigation(s.frame),('result',(640,680)))
        self.assertEqual(DailyBountyFlow('bounty','D').step(s,self.vision).point,(640,680))
        self.assertEqual(DailyBountyFlow('bounty','A').step(s,self.vision).action,'unavailable')
        s=self.vision.observe(s.frame,{'words':[]})
        self.assertEqual(s.grade,'D')

    def test_live_battle_is_not_a_blank_space_reward_menu(self):
        folder=project_root()/'data/runtime/daily-cleanup/runs/20261116-135300-314700-bounty-D'
        for name in ('0022-battle.png','0023-battle.png','0024-battle.png','0025-battle.png'):
            with self.subTest(name=name):
                frame=cv2.cvtColor(cv2.imread(str(folder/name)),cv2.COLOR_BGR2RGB)
                self.assertIsNone(self.vision.navigation(frame)[0])
                self.assertNotEqual(self.vision.observe(frame,{'words':[]}).page,'reward')

    def test_locked_s_screen_is_not_an_unknown_or_empty_s_list(self):
        folder=project_root()/'data/runtime/daily-cleanup/runs/20261116-134932-550600-bounty-S'
        record=next(json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()
                    if json.loads(line).get('frame')=='0010-bounty_order-wait.png')
        frame=cv2.cvtColor(cv2.imread(str(folder/record['frame'])),cv2.COLOR_BGR2RGB)
        screen=self.vision.observe(frame,record['ocr'])
        self.assertEqual(screen.page,'bounty_locked')
        self.assertEqual(self.vision.navigation(frame)[0],'bounty_locked')
        self.assertEqual(DailyBountyFlow('bounty_s').step(screen,self.vision).action,'unavailable')

    def test_stale_rank_back_is_rejected_after_home_arrives(self):
        rank=self.read('104526-blocked-state')
        home=self.read('104405-rank-back')
        decision=DailyBountyFlow().step(rank,self.vision)
        self.assertEqual(decision.point,(75,50))
        self.assertTrue(self.vision.current_action(rank,decision,rank.frame))
        self.assertFalse(self.vision.current_action(rank,decision,home.frame))


if __name__ == '__main__':
    unittest.main()
