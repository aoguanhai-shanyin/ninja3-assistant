"""Replay of saved daily-menu frames. Never connects to a device."""
import json
import unittest
import cv2

from ninja3_assistant.paths import project_root
from ninja3_assistant.activities.daily.flow import DailyFlow
from ninja3_assistant.activities.daily.vision import DailyVision


DATA=project_root()/'data/evidence/2026-11-16/daily-exploration'


class DailyMenuReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision=DailyVision()

    def read(self,stem):
        record=json.loads((DATA/(stem+'.json')).read_text(encoding='utf-8'))
        image=cv2.cvtColor(cv2.imread(str(DATA/(stem+'.png'))),cv2.COLOR_BGR2RGB)
        return self.vision.observe(image,record['ocr'])

    def test_free_and_paid_screens_are_distinct_for_all_three_free_operations(self):
        for kind,before,after in (
            ('curse','090157-curse-free','090243-curse-done'),
            ('weapon','084640-weapons-free','084716-weapon-done'),
            ('blessing','090013-family-blessing','090104-blessing-done')):
            with self.subTest(kind=kind):
                free,paid=self.read(before),self.read(after)
                self.assertTrue(self.vision.free(free,kind))
                self.assertFalse(self.vision.free(paid,kind))
                decision=DailyFlow(kind).step(free,self.vision)
                self.assertEqual(decision.free_kind,kind)
                self.assertTrue(self.vision.current_action(free,decision,free.frame))
                self.assertFalse(self.vision.current_action(free,decision,paid.frame))
                self.assertEqual(DailyFlow(kind).step(paid,self.vision).action,'done')

    def test_monthly_unclaimed_and_already_claimed_have_different_decisions(self):
        before=self.read('084337-login-check')
        after=self.read('085540-activities')
        claim=DailyFlow('monthly').step(before,self.vision)
        self.assertIn('第7天',claim.status)
        self.assertEqual(claim.action,'tap')
        self.assertEqual(DailyFlow('monthly').step(after,self.vision).action,'done')

    def test_current_limited_signup_targets_the_unclaimed_second_day(self):
        s=self.read('084412-monthly-confirm')
        self.assertEqual(s.page,'ten-signin')
        decision=DailyFlow('limited').step(s,self.vision)
        self.assertEqual(decision.point,(450,413))
        s=self.read('084451-promotion')
        self.assertEqual(DailyFlow('limited').step(s,self.vision).point,(402,410))

    def test_store_ninja_coin_icons_do_not_match_jade_or_food_prices(self):
        s=self.read('085216-mystery-shop')
        self.assertFalse(self.vision.coin(s,(55,310,235,360)))
        self.assertTrue(self.vision.coin(s,(292,310,472,360)))
        self.assertFalse(self.vision.coin(s,(292,580,472,630)))
        flow=DailyFlow('bounty_shop')
        shop=self.read('085114-bounty-shop')
        first=flow.step(shop,self.vision)
        self.assertEqual(first.action,'swipe')
        flow.completed(first,shop)
        self.assertEqual(flow.step(shop,self.vision).status,'选择D级悬赏奖册')

    def test_home_words_do_not_turn_the_home_screen_into_a_house(self):
        s=self.read('104405-rank-back')
        s.ocr['words'].extend([{'text':'小屋好友管理','bbox':[130,170,100,25]}])
        self.assertEqual(self.vision.observe(s.frame,s.ocr).page,'home')
        house=self.read('100048-house-resumed')
        back=DailyFlow('monthly').step(house,self.vision)
        self.assertFalse(self.vision.current_action(house,back,s.frame))

    def runtime_screen(self,run,image):
        root=project_root()/'data/runtime/daily-cleanup/runs'/run
        record=next(json.loads(line) for line in (root/'events.jsonl').read_text(encoding='utf-8').splitlines()
                    if json.loads(line).get('frame')==image)
        frame=cv2.cvtColor(cv2.imread(str(root/image)),cv2.COLOR_BGR2RGB)
        return self.vision.observe(frame,record['ocr'])

    def test_bounty_card_moves_left_after_purchase_and_dialog_accepts_lowercase_c(self):
        s=self.runtime_screen('20261116-104628-915800-bounty_shop','0019-limited-shop-tap.png')
        flow=DailyFlow('bounty_shop');flow.bounty_at_start=True
        decision=flow.step(s,self.vision)
        self.assertEqual(decision.status,'选择C级悬赏奖册')
        self.assertLess(decision.point[0],250)
        flow.completed(decision,s)
        dialog=self.runtime_screen('20261116-104828-833900-bounty_shop','0002-shop-tap.png')
        self.assertEqual(dialog.page,'bounty_dialog')
        self.assertEqual(flow.step(dialog,self.vision).point,(640,668))

    def test_daily_store_uses_the_same_book_dialog_and_buys_the_remaining_coin_item(self):
        flow=DailyFlow('daily_shop')
        shop=self.runtime_screen('20261116-104839-405400-daily_shop','0040-daily_shop-tap.png')
        choose=flow.step(shop,self.vision);flow.completed(choose,shop)
        self.assertEqual(flow.pending_item,'D级')
        dialog=self.runtime_screen('20261116-104839-405400-daily_shop','0041-bounty_dialog-tap.png')
        self.assertEqual(flow.step(dialog,self.vision).point,(640,668))
        remaining=self.runtime_screen('20261116-105658-073000-daily_shop','0004-daily_shop-wait.png')
        self.assertEqual(flow.step(remaining,self.vision).point,(610,510))

    def test_monthly_scan_scrolls_back_towards_the_current_day(self):
        s=self.runtime_screen('20261116-105209-568400-monthly','0008-monthly-unavailable.png')
        decision=DailyFlow('monthly').step(s,self.vision)
        self.assertGreater(decision.end[1],decision.point[1])
        self.assertEqual(DailyFlow('weapon').step(s,self.vision).point,(1238,48))
        self.assertEqual(self.vision.navigation(s.frame),('monthly',(1238,48)))

    def test_single_digit_food_is_recognized_when_ocr_omits_it(self):
        s=self.runtime_screen('20261116-110440-423200-village','0008-trial-wait.png')
        self.assertEqual(self.vision.food(s),9)
        s=self.read('085932-trial-first-done')
        self.assertEqual(self.vision.food(s),25)
        s=self.runtime_screen('20261116-110925-891600-village','0004-trial-wait.png')
        self.assertEqual(self.vision.food(s),4)
        self.assertEqual(DailyFlow('village').step(s,self.vision).action,'unavailable')

    def test_house_receipt_continue_and_countdown_are_distinct(self):
        for stem,action in (('095937-house-loaded','tap'),('100018-house-reward-done','tap'),
                            ('100048-house-resumed','done')):
            with self.subTest(stem=stem):
                self.assertEqual(DailyFlow('house').step(self.read(stem),self.vision).action,action)

    def test_trial_chooses_the_rightmost_cleared_stage_and_never_launches_battle(self):
        flow=DailyFlow('village')
        decision=flow.step(self.read('085820-trial-guide'),self.vision)
        self.assertIn('第28关',decision.status)
        self.assertEqual(decision.point,(697,360))
