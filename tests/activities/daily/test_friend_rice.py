import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from ninja3_assistant.activities.daily.flow import DailyFlow
from ninja3_assistant.activities.daily.friend_rice import FriendRice
from ninja3_assistant.activities.daily.vision import DailyVision
from ninja3_assistant.paths import project_root


EVIDENCE = project_root()/'data/evidence/2026-11-19'
FRIENDS = EVIDENCE/'friend-rice-152/001-friends-list.png'
INBOX = EVIDENCE/'friend-rice-152/002-receive-panel.png'
GUIDE = EVIDENCE/'xumi-152/064-live-stuck.png'
EMPTY_RUN = project_root()/'data/runtime/daily-cleanup/runs/20261119-134109-169300-friend_rice'
EMPTY_INBOX = EMPTY_RUN/'0003-friend-rice-inbox-wait.png'


class FriendRiceBudgetTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.budget = Path(temporary.name)/'instance-152'/'friend-rice.json'
        self.day = date(2026,11,19)

    def task(self):
        return FriendRice(self.budget,day=lambda: self.day)

    def test_batch_reservation_survives_a_restart_and_daily_reset(self):
        self.assertEqual(self.task().remaining(),15)
        self.assertTrue(self.task().reserve(4))
        restarted = self.task()
        self.assertEqual(restarted.remaining(),11)
        self.assertFalse(restarted.reserve(12))
        self.assertTrue(restarted.reserve(11))
        self.assertFalse(self.task().reserve(1))
        self.assertEqual(self.task().remaining(),0)

        self.day = date(2026,11,20)
        self.assertEqual(self.task().remaining(),15)
        self.assertTrue(self.task().reserve(1))
        self.assertEqual(self.task().remaining(),14)

    def test_invalid_state_does_not_reset_the_daily_gift_cap(self):
        self.assertTrue(self.task().reserve(4))
        invalid_fields = ((self.day.isoformat(),16),(self.day.isoformat(),True),
                          (None,4),('invalid-date',4),('2026-11-20',4))
        invalid_states = ['{', '{}', *(json.dumps({'day':day,'sent_upper_bound':sent})
                                       for day,sent in invalid_fields)]
        for invalid in invalid_states:
            with self.subTest(invalid=invalid):
                self.budget.write_text(invalid,encoding='utf-8')
                self.assertEqual(self.task().remaining(),0)
                self.assertFalse(self.task().reserve(1))
                self.assertEqual(self.budget.read_text(encoding='utf-8'),invalid)


@unittest.skipUnless(all(path.is_file() for path in (FRIENDS,INBOX,GUIDE)),
                     'requires locally retained 152 friend-menu captures')
class FriendRiceReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = DailyVision()

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.task = FriendRice(Path(temporary.name)/'friend-rice.json')
        self.flow = DailyFlow('friend_rice',task_flow=self.task)

    def screen(self, path, ocr=None):
        frame = cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        return self.vision.observe(frame,ocr if ocr is not None else {'words':[]})

    def test_prefers_batch_return_for_four_visible_receipts(self):
        inbox = self.screen(INBOX)
        decision = self.flow.step(inbox,self.vision)
        self.assertEqual((decision.point,decision.rice_gifts),((425,650),4))
        self.assertTrue(self.vision.current_action(inbox,decision,inbox.frame))

        updated = inbox.frame.copy()
        updated[375:458,1039:1114] = 0
        self.assertFalse(self.vision.current_action(inbox,decision,updated))
        self.assertFalse(self.vision.current_action(inbox,decision,self.screen(FRIENDS).frame))

    def test_exhausted_gift_cap_keeps_collection_available(self):
        self.assertTrue(self.task.reserve(15))
        inbox = self.screen(INBOX)
        decision = self.flow.step(inbox,self.vision)
        self.assertEqual((decision.point,decision.rice_gifts),((908,650),0))
        self.assertTrue(self.vision.current_action(inbox,decision,inbox.frame))

    def test_batch_larger_than_remaining_quota_uses_a_single_return(self):
        self.assertTrue(self.task.reserve(12))
        inbox = self.screen(INBOX)
        decision = self.flow.step(inbox,self.vision)
        self.assertEqual((decision.point,decision.rice_gifts),((968,130),1))
        self.assertTrue(self.vision.current_action(inbox,decision,inbox.frame))

    @unittest.skipUnless(EMPTY_INBOX.is_file() and (EMPTY_RUN/'events.jsonl').is_file(),
                         'requires the local empty rice inbox capture and OCR')
    def test_confirmed_empty_inbox_after_receipt_returns_to_gifting(self):
        loading = self.screen(EMPTY_INBOX)
        pending = self.flow.step(loading,self.vision)
        self.assertEqual((pending.action,pending.point),('wait',None))
        self.assertFalse(self.task.inbox_complete)

        with (EMPTY_RUN/'events.jsonl').open(encoding='utf-8') as log:
            record = next(row for line in log if (row := json.loads(line)).get('frame') == EMPTY_INBOX.name)
        empty = self.screen(EMPTY_INBOX,record['ocr'])
        close = self.flow.step(empty,self.vision)
        self.assertEqual((close.action,close.point,close.rice_gifts),('tap',(640,700),0))
        self.assertTrue(self.vision.current_action(empty,close,empty.frame))
        self.assertTrue(self.task.inbox_complete)

        gift = self.flow.step(self.screen(FRIENDS),self.vision)
        self.assertEqual((gift.point,gift.rice_gifts),((1204,236),1))

    def test_other_daily_tasks_can_dismiss_friend_guide_and_return_from_list(self):
        monthly = DailyFlow('monthly')
        guide = self.screen(GUIDE)
        friends = self.screen(FRIENDS)
        self.assertEqual(self.vision.navigation(guide.frame),('friend-guide',(640,331)))
        close = monthly.step(guide,self.vision)
        self.assertEqual(close.point,(640,331))
        self.assertTrue(self.vision.current_action(guide,close,guide.frame))
        self.assertFalse(self.vision.current_action(guide,close,friends.frame))

        self.assertEqual(self.vision.navigation(friends.frame),('friend-list',(55,46)))
        self.assertEqual(monthly.step(friends,self.vision).point,(55,46))
        open_inbox = self.flow.step(friends,self.vision)
        self.assertEqual(open_inbox.point,(1217,49))
