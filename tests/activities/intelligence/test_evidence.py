from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import cv2
import numpy as np

from ninja3_assistant.activities.intelligence.battle_vision import BattleObservation, SHOP_BOXES
from ninja3_assistant.activities.intelligence.evidence import UnknownCardRecorder


class UnknownCardEvidenceTest(unittest.TestCase):
    def test_unknown_card_keeps_source_crop_position_and_candidates(self):
        frame = np.zeros((720,1280,3),np.uint8)
        frame[:,:,0] = 200
        observation = BattleObservation('placement',round_number=2,coins=4,unknown_shop=(1,),
            card_candidates={'shop:1':[{'card':'c007','score':.8}]})
        with TemporaryDirectory() as directory:
            output = Path(directory)
            recorder = UnknownCardRecorder(output)
            record = recorder.save(frame,observation)
            slot = record['slots'][0]
            self.assertEqual(slot['box'],list(SHOP_BOXES[1]))
            self.assertEqual(slot['candidates'],[{'card':'c007','score':.8}])
            crop = cv2.imread(str(output/slot['crop']))
            x1,y1,x2,y2 = SHOP_BOXES[1]
            self.assertEqual(crop.shape,(y2-y1,x2-x1,3))
            self.assertEqual(crop[0,0].tolist(),[0,0,200])
            self.assertEqual(cv2.imread(str(output/record['frame'])).shape,frame.shape)
            self.assertIsNone(recorder.save(frame,observation))
            saved = json.loads((output/'unknown-cards/index.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(saved,record)

    def test_animation_is_bounded_and_known_or_combat_frames_are_not_saved(self):
        with TemporaryDirectory() as directory:
            recorder = UnknownCardRecorder(Path(directory))
            unknown = BattleObservation('placement',round_number=1,unknown_board=(0,))
            for value in (0,40,80,120,160):
                recorder.save(np.full((720,1280,3),value,np.uint8),unknown)
            self.assertEqual(len(recorder.seen),3)
            self.assertIsNone(recorder.save(np.zeros((720,1280,3),np.uint8),BattleObservation('combat')))
            self.assertIsNone(recorder.save(np.zeros((720,1280,3),np.uint8),BattleObservation('placement')))
