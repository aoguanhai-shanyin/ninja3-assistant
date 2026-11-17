"""Local saved-frame bounty regressions; no emulator or game input."""
import json
import unittest
from unittest.mock import patch

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

    def test_real_bounty_task_panel_including_claimed_ocr_error(self):
        folder=project_root()/'data/evidence/2026-11-17/daily-exploration'
        record=json.loads((folder/'112544-bounty-task-panel.json').read_text(encoding='utf-8'))
        frame=cv2.cvtColor(cv2.imread(str(folder/'112544-bounty-task-panel.png')),cv2.COLOR_BGR2RGB)
        screen=self.vision.observe(frame,record['ocr'])
        self.assertEqual(screen.page,'bounty_tasks')
        self.assertEqual(screen.task_states,{'friend':'claimed','family':'claimed'})
        self.assertIsNone(screen.task_claim)
        self.assertEqual(self.vision.navigation(frame)[0],'bounty_tasks')
        without_ocr=self.vision.observe(frame,{'words':[]})
        self.assertEqual(without_ocr.page,'bounty_tasks')
        self.assertEqual(without_ocr.task_states,{'friend':'unknown','family':'unknown'})
        decision=DailyBountyFlow().step(screen,self.vision)
        self.assertEqual(decision.point,(1025,155))
        self.assertTrue(self.vision.current_action(screen,decision,frame))

    def test_reward_overlay_takes_precedence_over_task_text(self):
        folder=project_root()/'data/evidence/2026-11-17/daily-exploration'
        record=json.loads((folder/'112544-bounty-task-panel.json').read_text(encoding='utf-8'))
        frame=cv2.cvtColor(cv2.imread(str(folder/'112544-bounty-task-panel.png')),cv2.COLOR_BGR2RGB)
        record['ocr']['words'].extend([
            {'text':'恭喜获得','bbox':[560,120,140,30]},
            {'text':'点击空白处关闭','bbox':[540,630,200,30]}])
        self.assertEqual(self.vision.observe(frame,record['ocr']).page,'reward')

    def test_task_statuses_are_limited_to_the_corresponding_row(self):
        folder=project_root()/'data/evidence/2026-11-17/daily-exploration'
        record=json.loads((folder/'112544-bounty-task-panel.json').read_text(encoding='utf-8'))
        frame=cv2.cvtColor(cv2.imread(str(folder/'112544-bounty-task-panel.png')),cv2.COLOR_BGR2RGB)
        ocr=record['ocr']
        ocr['words']=[w for w in ocr['words'] if not (800 <= w['bbox'][0] < 970 and 235 <= w['bbox'][1] < 315)]
        ocr['words'].append({'text':'领取','bbox':[860,264,50,22]})
        screen=self.vision.observe(frame,ocr)
        self.assertEqual(screen.task_states,{'friend':'claimable','family':'claimed'})
        self.assertEqual(screen.task_claim,(885,275))
        for label in ('未完成','去完成','前往','去完成任务','0/1'):
            ocr['words'][-1]['text']=label
            screen=self.vision.observe(frame,ocr)
            self.assertEqual(screen.task_states,{'friend':'pending','family':'claimed'})
            self.assertIsNone(screen.task_claim)

    def test_full_a_room_departure_label_is_a_start_button(self):
        run=project_root()/'data/runtime/daily-cleanup/runs/20261117-101056-253400-bounty-A'
        records=[json.loads(line) for line in (run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        record=next(r for r in records if r.get('event')=='decision' and r.get('page')=='bounty_room')
        frame=cv2.cvtColor(cv2.imread(str(run/record['frame'])),cv2.COLOR_BGR2RGB)
        screen=self.vision.observe(frame,record['ocr'])
        self.assertEqual(screen.grade,'A')
        self.assertFalse(screen.minimum_players)
        self.assertIsNotNone(screen.start)
        decision=DailyBountyFlow('bounty','A').step(screen,self.vision)
        self.assertEqual(decision.action,'tap')
        self.assertEqual(decision.point,screen.start)
        self.assertIn('开始目标等级悬赏',decision.status)

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

    def test_recorded_friend_panel_invites_only_ocr_confirmed_online_button(self):
        folder=project_root()/'data/evidence/2026-11-17/daily-exploration'
        record=json.loads((folder/'100722-invite-panel-current.json').read_text(encoding='utf-8'))
        frame=cv2.cvtColor(cv2.imread(str(folder/'100722-invite-panel-current.png')),cv2.COLOR_BGR2RGB)
        panel=self.vision.observe(frame,record['ocr'])
        self.assertEqual(panel.page,'bounty_recruit')
        self.assertEqual(panel.recruit_tab,'friend')
        self.assertEqual([point for _,point in panel.recruit_candidates],[(1211,133)])
        self.assertTrue(self.vision.navigation(frame)[0]=='bounty_recruit')
        # A color rectangle without the invite label or player name is not a candidate.
        self.assertEqual(self.vision._recruit_candidates(type(panel)(frame,{'words':[]})),[])

    def test_chat_tab_is_never_treated_as_an_invite_list(self):
        folder=project_root()/'data/evidence/2026-11-17/daily-exploration'
        record=json.loads((folder/'100747-recruit-third-tab.json').read_text(encoding='utf-8'))
        frame=cv2.cvtColor(cv2.imread(str(folder/'100747-recruit-third-tab.png')),cv2.COLOR_BGR2RGB)
        panel=self.vision.observe(frame,record['ocr'])
        self.assertEqual(panel.page,'bounty_recruit')
        self.assertEqual(panel.recruit_tab,'chat')
        self.assertEqual(panel.recruit_candidates,[])

    def test_a_room_grade_is_confirmed_without_ocr_or_reward_attempts(self):
        path=project_root()/'data/experiments/2026-11-17/bounty-validation/team-prep/'
        frame=cv2.cvtColor(cv2.imread(str(path/'095948-122600/0009-bounty_room-unavailable.png')),
                           cv2.COLOR_BGR2RGB)
        screen=self.vision.observe(frame,{'words':[]})
        self.assertEqual(screen.page,'bounty_room')
        self.assertEqual(screen.grade,'A')
        self.assertTrue(screen.minimum_players)
        self.assertNotEqual(DailyBountyFlow('bounty','A').step(screen,self.vision).action,'unavailable')
        self.assertEqual(DailyBountyFlow('bounty','D').step(screen,self.vision).action,'unavailable')

    def test_a_room_badge_does_not_match_d_s_or_a_missing_badge(self):
        room=self.read('134331-bounty-created-room')
        self.assertEqual(room.grade,'D')
        self.assertIsNone(self.vision.match(room.frame,'room-grade-A',(280,0,360,85),.96))
        path=project_root()/'data/experiments/2026-11-17/bounty-validation/team-prep/'
        frame=cv2.cvtColor(cv2.imread(str(path/'095948-122600/0009-bounty_room-unavailable.png')),
                           cv2.COLOR_BGR2RGB)
        frame[:85,280:360]=0
        self.assertIsNone(self.vision.observe(frame,{'words':[]}).grade)
        rank_s=cv2.imread(str(project_root()/'assets/rank-s.png'))
        frame[18:62,305:335]=cv2.cvtColor(cv2.resize(rank_s,(30,44)),cv2.COLOR_BGR2RGB)
        self.assertNotEqual(self.vision.observe(frame,{'words':[]}).grade,'A')

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

    def test_unlocked_s_order_uses_the_enlarged_s_counter_when_full_ocr_misses_it(self):
        folder=project_root()/'data/runtime/daily-cleanup/runs/20261116-174444-675600-bounty-S'
        record=next(json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()
                    if json.loads(line).get('frame')=='0004-bounty_order-wait.png')
        frame=cv2.cvtColor(cv2.imread(str(folder/record['frame'])),cv2.COLOR_BGR2RGB)
        self.assertEqual(self.vision.navigation(frame)[0],'bounty_order')
        self.assertIsNone(self.vision.observe(frame,record['ocr']).remaining)
        result={**record['ocr'],'bounty_order_counter':{'text':'7 剩 余 奖 励 次 数 ： s 悬 赏 ： 0 丿'}}
        screen=self.vision.observe(frame,result)
        self.assertEqual(screen.remaining,0)
        flow=DailyBountyFlow('bounty_s')
        flow.tasks_checked=True
        flow.task_states={'friend':'claimed','family':'claimed'}
        self.assertEqual(flow.step(screen,self.vision).action,'done')
        result['bounty_order_counter']['text']='SS+悬赏:0'
        self.assertIsNone(self.vision.observe(frame,result).remaining)

    def test_s_counter_metadata_never_completes_a_locked_order_page_or_hub(self):
        folder=project_root()/'data/runtime/daily-cleanup/runs/20261116-134932-550600-bounty-S'
        record=next(json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()
                    if json.loads(line).get('frame')=='0010-bounty_order-wait.png')
        frame=cv2.cvtColor(cv2.imread(str(folder/record['frame'])),cv2.COLOR_BGR2RGB)
        result={**record['ocr'],'bounty_order_counter':{'text':'剩余奖励次数:S悬赏:0'}}
        screen=self.vision.observe(frame,result)
        self.assertEqual(screen.page,'bounty_locked')
        self.assertEqual(DailyBountyFlow('bounty_s').step(screen,self.vision).action,'unavailable')
        hub=DailyBountyVision()
        # A visible hub's S0 is not sufficient evidence that orders are unlocked.
        with patch.object(hub,'navigation',return_value=('bounty_hub',(65,50))), \
                patch.object(hub,'match',return_value=None):
            screen=hub.observe(np.zeros_like(frame),{'words':[],
                'bounty_order_counter':{'text':'剩余奖励次数:S悬赏:0'}})
        self.assertEqual(screen.page,'bounty_hub')
        self.assertIsNone(screen.remaining)


if __name__ == '__main__':
    unittest.main()
