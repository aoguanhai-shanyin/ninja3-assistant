import unittest
from dataclasses import replace

from ninja3_assistant.activities.daily.flow import DailyDecision
from ninja3_assistant.activities.xumi.runner import XumiFlow
from ninja3_assistant.activities.xumi.tasks import XumiTaskRow
from ninja3_assistant.activities.xumi.vision import XumiScreen
from ninja3_assistant.game_ui.xumi import TrialSetup


class XumiFlowTest(unittest.TestCase):
    def test_preserves_team_score_when_the_game_asks_to_reset_it(self):
        decision = XumiFlow().step(None,XumiScreen('score_confirm'))
        self.assertEqual(decision.action,'unavailable')
        self.assertIsNone(decision.point)

    def test_first_and_second_results_have_distinct_actions(self):
        flow = XumiFlow()
        self.assertEqual(flow.step(None,XumiScreen('result_next',point=(976,635))).action,'continue')
        self.assertEqual(flow.step(None,XumiScreen('result_done',point=(890,635))).action,'settle')

    def test_unknown_lead_waits_and_missing_weapon_skips_without_editing(self):
        flow = XumiFlow()
        flow.mode = 'daqian'
        setup = XumiScreen('daqian_setup',back=(47,40),setup=TrialSetup(('filled','unknown'),1,None))
        self.assertEqual(flow.step(None,setup).action,'wait')
        self.assertFalse(flow.progress.blocked_modes)
        editor = flow.step(None,XumiScreen('equipment',back=(47,40)))
        self.assertEqual(editor.action,'defer')
        self.assertIsNone(editor.point)
        skip = flow.step(None,replace(setup,notice='missing_weapon'))
        self.assertEqual((skip.operation,skip.point),('back',(47,40)))
        self.assertIn('daqian',flow.progress.blocked_modes)

    def test_manual_empty_team_skips_only_that_mode_and_survives_checkpoints(self):
        flow = XumiFlow()
        rows = tuple(XumiTaskRow(mode,kind,0,goal) for mode in ('fangcun','daqian')
                     for kind,goal in (('match',1),('score',50000)))
        tasks = XumiScreen('tasks',back=(1078,118))
        flow.step(None,tasks,rows)
        self.assertEqual(flow.mode,'fangcun')
        setup = XumiScreen('fangcun_setup',back=(47,40),setup=TrialSetup(('filled','empty'),1,None))
        self.assertEqual(flow.step(None,setup).operation,'back')
        resumed = XumiFlow(flow.progress)
        resumed.step(None,tasks,rows)
        self.assertEqual(resumed.mode,'daqian')
        claimed = tuple(replace(row,current=row.goal,claimed=True) for row in rows[2:])
        self.assertEqual(resumed.step(None,tasks,rows[:2]+claimed).action,'defer')

    def test_replay_checks_current_difficulty_and_does_not_tap_progress_markers(self):
        flow = XumiFlow(trial='daqian',difficulty=2)
        setup = XumiScreen('daqian_setup',setup=TrialSetup(('filled','filled'),3,(1092,634)))
        skipped = flow.step(None,setup)
        self.assertEqual(skipped.action,'defer')
        self.assertIsNone(skipped.point)
        ready = replace(setup,setup=TrialSetup(('filled','filled'),2,(1092,634)))
        for flow in (XumiFlow(trial='daqian',difficulty=2), XumiFlow()):
            flow.mode = 'daqian'
            start = flow.step(None,ready)
            self.assertEqual((start.action,start.point,start.value),('start',(1092,634),2))

    def test_daily_trials_leave_third_difficulty_for_manual_play(self):
        flow = XumiFlow()
        flow.mode = 'daqian'
        setup = XumiScreen('daqian_setup',back=(47,40),setup=TrialSetup(('filled','filled'),3,(1092,634)))
        skipped = flow.step(None,setup)
        self.assertEqual((skipped.operation,skipped.point),('back',(47,40)))
        self.assertIn('daqian',flow.progress.blocked_modes)

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
