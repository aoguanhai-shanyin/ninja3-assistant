import unittest
from unittest.mock import Mock

import cv2

from ninja3_assistant.activities.bounty.combat import BountyBattleState
from ninja3_assistant.activities.bounty.daily_flow import DailyBountyFlow
from ninja3_assistant.activities.bounty.daily_runner import _recognize_frame
from ninja3_assistant.activities.bounty.daily_vision import DailyBountyVision
from ninja3_assistant.paths import project_root


SETTLEMENT = project_root()/'data/evidence/2026-11-19/bounty-152-continuation/004-s-current.png'
DEFEAT = project_root()/'data/evidence/2026-11-20/bounty-defeat-recovery/01-defeat.png'


@unittest.skipUnless(SETTLEMENT.is_file(),'local S bounty settlement screenshot unavailable')
class OrderSettlementTest(unittest.TestCase):
    def test_live_s_order_results_confirm_with_hud_or_after_a_restart(self):
        frame = cv2.cvtColor(cv2.imread(str(SETTLEMENT)),cv2.COLOR_BGR2RGB)
        vision = DailyBountyVision(grade='S')
        screen = vision.observe(frame,{'words':[]})
        self.assertEqual((screen.page,screen.grade,screen.confirm),('result','S',(649,678)))
        self.assertEqual(vision.navigation(frame),('result',screen.confirm))

        battle = Mock()
        battle.detect.return_value = BountyBattleState(active=True,ground=True)
        page,state = _recognize_frame(frame,vision,battle)
        self.assertEqual(page,'result')
        self.assertFalse(state.active)

        decision = DailyBountyFlow('bounty_s').step(screen,vision)
        self.assertEqual((decision.action,decision.point),('tap',screen.confirm))
        self.assertTrue(vision.current_action(screen,decision,frame))
        for x1,y1,x2,y2 in ((570,49,825,123),(526,630,773,719)):
            changed = frame.copy()
            changed[y1:y2,x1:x2] = 0
            self.assertFalse(vision.current_action(screen,decision,changed))


@unittest.skipUnless(DEFEAT.is_file(),'local bounty defeat screenshot unavailable')
class BountyDefeatTest(unittest.TestCase):
    def test_defeat_can_be_dismissed_after_restart_and_rechecks_the_button(self):
        frame = cv2.cvtColor(cv2.imread(str(DEFEAT)),cv2.COLOR_BGR2RGB)
        vision = DailyBountyVision(grade='A')
        screen = vision.observe(frame,{'words':[]})
        self.assertEqual((screen.page,screen.grade,screen.confirm),('result',None,(640,543)))
        self.assertEqual(vision.navigation(frame),('result',screen.confirm))

        battle = Mock()
        battle.detect.return_value = BountyBattleState(active=True,ground=True)
        page,state = _recognize_frame(frame,vision,battle)
        self.assertEqual(page,'result')
        self.assertFalse(state.active)

        flow = DailyBountyFlow('bounty',grade='A')
        flow.progress.regular_remaining = 2
        decision = flow.step(screen,vision)
        self.assertEqual((decision.operation,decision.point),('confirm_settlement',screen.confirm))
        self.assertTrue(vision.current_action(screen,decision,frame))
        flow.completed(decision,screen)
        self.assertEqual(flow.progress.regular_remaining,2)
        for x1,y1,x2,y2 in ((508,36,764,172),(535,522,745,564)):
            changed = frame.copy()
            changed[y1:y2,x1:x2] = 0
            self.assertFalse(vision.current_action(screen,decision,changed))

        training = project_root()/'tests/fixtures/hayabusa/live-shadow-defeat.png'
        if training.is_file():
            self.assertIsNone(vision.defeat_confirm(cv2.cvtColor(cv2.imread(str(training)),cv2.COLOR_BGR2RGB)))
