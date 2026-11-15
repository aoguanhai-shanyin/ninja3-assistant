from dataclasses import replace
from datetime import datetime
from pathlib import Path
import unittest

import cv2

from activity_flow import ActivityFlow,CHINA
from activity_tasks import TaskVision
from activity_vision import ActivityVision

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'data/evidence/2026-11-15/intelligence-daily'


class DailyRowsTest(unittest.TestCase):
    def screens(self, completed=False):
        vision=TaskVision()
        top=cv2.cvtColor(cv2.imread(str(EVIDENCE/'top-incomplete.png')),cv2.COLOR_BGR2RGB)
        bottom=cv2.cvtColor(cv2.imread(str(EVIDENCE/'bottom-claimed.png')),cv2.COLOR_BGR2RGB)
        observation=vision.detect(top,{'words':[{'text':'4/5','bbox':[1030,180,40,25]}]})
        if completed:
            observation=replace(observation,rows=(replace(observation.rows[0],state='claimed',progress=None),
                                                  observation.rows[1]))
        return ActivityVision.task_screen(observation),ActivityVision.task_screen(vision.detect(bottom))

    def scan(self,top,bottom):
        flow=ActivityFlow()
        wall=datetime(2026,11,15,12,tzinfo=CHINA)
        decision=flow.step(top,0,wall);flow.completed(decision,0)
        decision=flow.step(top,2,wall)
        self.assertEqual(decision.action,'scroll_tasks')
        flow.completed(decision,2)
        return flow,flow.step(bottom,4,wall),wall

    def test_any_unfinished_progress_continues_play_without_task_type(self):
        top,bottom=self.screens()
        self.assertEqual(top.task.rows[0].progress,(4,5))
        flow,decision,_=self.scan(top,bottom)
        self.assertEqual(decision.action,'close_tasks')
        self.assertFalse(flow.daily_complete)

    def test_three_received_rows_across_scroll_finish_daily(self):
        top,bottom=self.screens(completed=True)
        flow,decision,_=self.scan(top,bottom)
        self.assertEqual(decision.action,'done')
        self.assertTrue(flow.daily_complete)

    def test_failed_scroll_cannot_count_the_same_two_rows_as_three(self):
        top,_=self.screens(completed=True)
        flow,decision,wall=self.scan(top,top)
        self.assertEqual(decision.action,'scroll_tasks')
        flow.completed(decision,4)
        self.assertEqual(flow.step(top,6,wall).action,'retry')
        self.assertFalse(flow.daily_complete)
