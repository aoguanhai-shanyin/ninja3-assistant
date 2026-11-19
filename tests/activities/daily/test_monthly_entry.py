import json
import unittest

import cv2

from ninja3_assistant.activities.daily.flow import DailyFlow
from ninja3_assistant.activities.daily.vision import DailyVision
from ninja3_assistant.paths import project_root


RUN = project_root()/'data/runtime/daily-cleanup/runs/20261119-133036-472700-monthly'
FRAME = '0001-home-tap.png'


class MonthlyEntryReplayTest(unittest.TestCase):
    @unittest.skipUnless((RUN/FRAME).is_file() and (RUN/'events.jsonl').is_file(),
                         'requires the local daily-sign-in failure capture')
    def test_entry_taps_gate_and_never_the_ocr_label(self):
        with (RUN/'events.jsonl').open(encoding='utf-8') as log:
            record = next(row for line in log if (row := json.loads(line)).get('frame') == FRAME)
        frame = cv2.cvtColor(cv2.imread(str(RUN/FRAME)),cv2.COLOR_BGR2RGB)
        vision,flow = DailyVision(),DailyFlow('monthly')
        screen = vision.observe(frame,record['ocr'])
        self.assertEqual(screen.hits('活动',(0,150,100,360))[0].point,(52,324))
        decision = flow.step(screen,vision)
        self.assertEqual(decision.point,(53,274))
        self.assertTrue(vision.current_action(screen,decision,frame))

        hidden = frame.copy()
        hidden[250:301,23:83] = 0
        self.assertFalse(vision.current_action(screen,decision,hidden))
        self.assertIsNone(flow.step(vision.observe(hidden,record['ocr']),vision).point)
