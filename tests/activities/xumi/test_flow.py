import unittest
from dataclasses import replace
from unittest.mock import Mock

from ninja3_assistant.activities.daily.flow import DailyDecision
from ninja3_assistant.activities.xumi.runner import XumiFlow
from ninja3_assistant.activities.xumi.tasks import XumiTaskRow
from ninja3_assistant.activities.xumi.vision import XumiScreen


class XumiFlowTest(unittest.TestCase):
    def test_preserves_team_score_when_the_game_asks_to_reset_it(self):
        decision = XumiFlow().step(None,XumiScreen('score_confirm'))
        self.assertEqual(decision.action,'unavailable')
        self.assertIsNone(decision.point)

    def test_first_and_second_results_have_distinct_actions(self):
        flow = XumiFlow()
        self.assertEqual(flow.step(None,XumiScreen('result_next',point=(976,635))).action,'continue')
        self.assertEqual(flow.step(None,XumiScreen('result_done',point=(890,635))).action,'settle')

    def test_daqian_requires_weapon_check_before_start(self):
        flow = XumiFlow()
        flow.mode = 'daqian'
        flow.teams = Mock()
        flow.teams.step.return_value = DailyDecision('start',(1090,638))
        setup = XumiScreen('daqian_setup',back=(47,40))
        self.assertEqual(flow.step(None,setup).action,'start')
        flow.gear_required = True
        check = flow.step(None,setup)
        self.assertEqual(check.action,'check_gear')
        flow.completed(check)
        self.assertEqual(flow.step(None,XumiScreen('team',team_number=1)).point,(1250,363))
        flow.completed(DailyDecision('gear_done',(1115,678)))
        self.assertEqual(flow.step(None,XumiScreen('equipment')).action,'wait')
        flow.step(None,XumiScreen('team',team_number=1))
        check = flow.step(None,setup)
        self.assertEqual(check.point,(563,647))
        flow.completed(check)
        flow.completed(DailyDecision('gear_done',(1115,678)))
        flow.step(None,XumiScreen('team',team_number=2))
        self.assertEqual(flow.step(None,setup).action,'start')

    def test_daily_progress_requires_both_modes_and_all_claims(self):
        scene = XumiScreen('tasks', back=(1078,118))
        flow = XumiFlow()
        rows = tuple(XumiTaskRow(mode,kind,0,goal) for mode in ('fangcun','daqian')
                     for kind,goal in (('match',1),('score',50000)))
        self.assertEqual(flow.step(None,scene,rows[:3]).action,'wait')
        self.assertEqual(flow.step(None,scene,rows).point, scene.back)
        self.assertEqual(flow.mode, 'fangcun')
        fangcun = tuple(XumiTaskRow(row.mode,row.kind,row.goal,row.goal,claimed=True) for row in rows[:2])
        self.assertEqual(flow.step(None,scene,fangcun+rows[2:]).point, scene.back)
        self.assertEqual(flow.mode, 'daqian')
        rewards = tuple(XumiTaskRow(row.mode,row.kind,row.goal,row.goal,claim_point=(990,310)) for row in rows[2:])
        self.assertEqual(flow.step(None,scene,fangcun+rewards).point,(990,310))
        self.assertEqual(flow.step(None,scene,fangcun+tuple(
            XumiTaskRow(row.mode,row.kind,row.goal,row.goal,claimed=True) for row in rows[2:])).action,'done')

    def test_alternates_progress_and_defers_trials_that_stop_scoring(self):
        scene = XumiScreen('tasks',back=(1078,118))
        flow = XumiFlow()
        def rows(fangcun,daqian):
            return tuple(row for mode,points in (('fangcun',fangcun),('daqian',daqian)) for row in (
                XumiTaskRow(mode,'match',None,1,claimed=True),XumiTaskRow(mode,'score',points,50000)))

        for turn,(expected,points) in enumerate((
                ('fangcun',(100,200)),('daqian',(110,200)),('fangcun',(110,200)),
                ('daqian',(110,200)),('fangcun',(110,200)))):
            if turn:
                self.assertEqual(flow.step(None,scene,rows(*points)).action,'checkpoint')
                flow = XumiFlow(flow.progress)
            self.assertEqual(flow.step(None,scene,rows(*points)).action,'tap')
            self.assertEqual(flow.mode,expected)
            flow.completed(DailyDecision('settle'))

        self.assertEqual(flow.step(None,scene,rows(110,200)).action,'defer')

    def test_round_checkpoint_waits_for_verified_progress_and_pending_reward(self):
        flow = XumiFlow()
        scene = XumiScreen('tasks',back=(1078,118))
        rows = tuple(XumiTaskRow(mode,kind,0,goal) for mode in ('fangcun','daqian')
                     for kind,goal in (('match',1),('score',50000)))
        flow.step(None,scene,rows)
        first = flow.step(None,XumiScreen('result_next',point=(976,635)))
        flow.completed(first)
        self.assertIsNone(flow.progress.settled_mode)

        second = flow.step(None,XumiScreen('result_done',point=(890,635)))
        flow.completed(second)
        self.assertEqual(flow.step(None,scene,rows[:3]).action,'wait')
        reached = replace(rows[0],current=1)
        self.assertEqual(flow.step(None,scene,(reached,*rows[1:])).action,'wait')
        claim = replace(reached,claim_point=(990,310))
        self.assertEqual(flow.step(None,scene,(claim,*rows[1:])).point,(990,310))
        claimed = replace(reached,claimed=True)
        self.assertEqual(flow.step(None,scene,(claimed,*rows[1:])).action,'checkpoint')
        self.assertIsNone(flow.mode)
