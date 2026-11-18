"""Replay of saved daily-menu frames. Never connects to a device."""
import json
import unittest
import cv2

from ninja3_assistant.paths import project_root
from ninja3_assistant.activities.daily.flow import DailyFlow
from ninja3_assistant.activities.daily.vision import DailyVision
from ninja3_assistant.activities.daily.resources import remaining_seconds


DATA=project_root()/'data/evidence/2026-11-16/daily-exploration'


class DailyMenuReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision=DailyVision()

    def read(self,stem):
        record=json.loads((DATA/(stem+'.json')).read_text(encoding='utf-8'))
        image=cv2.cvtColor(cv2.imread(str(DATA/(stem+'.png'))),cv2.COLOR_BGR2RGB)
        return self.vision.observe(image,record['ocr'])

    def test_login_recovers_missing_switch_caption_without_guessing_other_pages(self):
        s=self.runtime_screen('20261117-094029-108500-limited','0001-unknown-wait.png')
        self.assertNotIn('点击切换',s.text())
        self.assertEqual(s.page,'login')
        self.assertEqual(self.vision.navigation(s.frame),('login',(640,515)))
        decision=DailyFlow('limited').step(s,self.vision)
        self.assertEqual(decision.point,(640,515))
        self.assertTrue(self.vision.current_action(s,decision,s.frame))
        for box in ((500,470,800,560),(680,395,840,465)):
            changed=s.frame.copy();x1,y1,x2,y2=box;changed[y1:y2,x1:x2]=0
            self.assertIsNone(self.vision.login(changed))
        for stem in ('165731-daily-interference-current','173943-shop-guide-dismiss',
                     '095937-house-loaded','090157-curse-free'):
            with self.subTest(other=stem):
                frame=self.read(stem).frame
                self.assertIsNone(self.vision.login(frame))
                self.assertFalse(self.vision.current_action(s,decision,frame))

    def test_thanks_icon_locates_old_shortcut_when_ocr_omits_its_label(self):
        s=self.read('104405-rank-back')
        self.assertFalse(s.hits(r'答[谢榭]礼',(80,130,330,600)))
        point=DailyFlow('limited').step(s,self.vision).point
        self.assertIsNotNone(point)
        self.assertLess(abs(point[0]-230),10)
        self.assertLess(abs(point[1]-200),10)

    def test_family_entry_banner_is_dismissed_before_the_courtyard(self):
        s=self.runtime_screen('20261116-172753-527900-blessing','0006-unknown-wait.png')
        normal=self.read('173014-family-welcome-dismiss')
        self.assertEqual(s.page,'family-welcome')
        self.assertEqual(self.vision.navigation(s.frame),('family-welcome',(640,650)))
        decision=DailyFlow('blessing').step(s,self.vision)
        self.assertEqual(decision.point,(640,650))
        self.assertTrue(self.vision.current_action(s,decision,s.frame))
        self.assertFalse(self.vision.current_action(s,decision,normal.frame))

    def test_shop_guide_is_closed_before_selecting_a_coin_item(self):
        s=self.runtime_screen('20261116-173657-493700-bounty_shop','0008-unknown-wait.png')
        normal=self.read('173943-shop-guide-dismiss')
        self.assertEqual(s.page,'menu-guide')
        self.assertEqual(self.vision.navigation(s.frame),('menu-guide',(640,100)))
        decision=DailyFlow('bounty_shop').step(s,self.vision)
        self.assertEqual(decision.point,(640,100))
        self.assertTrue(self.vision.current_action(s,decision,s.frame))
        self.assertFalse(self.vision.current_action(s,decision,normal.frame))

    def test_limited_shop_welcome_is_dismissed_before_either_shop_task(self):
        normal=self.read('173943-shop-guide-dismiss')
        self.assertEqual(self.vision.navigation(normal.frame),('limited-shop',(75,45)))
        for kind,run,image,tab in (
                ('bounty_shop','20261118-094619-490400-bounty_shop','0005-unknown-wait.png',(1190,378)),
                ('daily_shop','20261118-094652-707300-daily_shop','0001-unknown-wait.png',(1000,690))):
            with self.subTest(kind=kind):
                guide=self.runtime_screen(run,image)
                self.assertEqual(guide.page,'menu-guide')
                self.assertEqual(self.vision.observe(guide.frame,{'words':[]}).page,'menu-guide')
                self.assertEqual(self.vision.navigation(guide.frame),('menu-guide',(640,100)))
                flow=DailyFlow(kind)
                dismiss=flow.step(guide,self.vision)
                self.assertEqual(dismiss.point,(640,100))
                self.assertTrue(self.vision.current_action(guide,dismiss,guide.frame))
                self.assertFalse(self.vision.current_action(guide,dismiss,normal.frame))
                enter=flow.step(normal,self.vision)
                self.assertEqual(enter.point,tab)
                self.assertFalse(self.vision.current_action(normal,enter,guide.frame))

    def test_second_shop_stone_guide_is_dismissed_before_any_product(self):
        s=self.runtime_screen('20261117-095019-052000-bounty_shop','0005-unknown-wait.png')
        normal=self.read('173943-shop-guide-dismiss')
        self.assertEqual(s.page,'menu-guide')
        self.assertEqual(self.vision.navigation(s.frame),('menu-guide',(640,100)))
        # The fixed text template also works when OCR omits the explanation.
        self.assertEqual(self.vision.observe(s.frame,{'words':[]}).page,'menu-guide')
        self.assertNotEqual(self.vision.navigation(normal.frame)[0],'menu-guide')
        for kind in ('bounty_shop','daily_shop'):
            with self.subTest(kind=kind):
                decision=DailyFlow(kind).step(s,self.vision)
                self.assertEqual(decision.point,(640,100))
                self.assertTrue(self.vision.current_action(s,decision,s.frame))
                self.assertFalse(self.vision.current_action(s,decision,normal.frame))
        from ninja3_assistant.activities.daily.flow import DailyDecision
        old=DailyDecision('tap',(1200,375),'进入悬赏商店')
        self.assertFalse(self.vision.current_action(normal,old,s.frame))

    def test_paged_trial_guide_and_interrupting_invitation_have_clear_priority(self):
        guide=self.runtime_screen('20261116-174621-239500-village','0007-unknown-wait.png')
        self.assertEqual(guide.page,'paged-guide')
        self.assertEqual(self.vision.navigation(guide.frame)[0],'paged-guide')
        decision=DailyFlow('village').step(guide,self.vision)
        self.assertGreater(decision.point[0],950)
        invite=self.read('175313-trial-guide-next-1')
        self.assertEqual(invite.page,'invite')
        self.assertFalse(self.vision.current_action(guide,decision,invite.frame))
        self.assertEqual(DailyFlow('village').step(invite,self.vision).point,(471,526))
        last=self.runtime_screen('20261116-175643-206300-village','0003-unknown-wait.png')
        self.assertEqual(last.page,'paged-guide')
        self.assertEqual(DailyFlow('village').step(last,self.vision).point,(640,100))

    def test_new_trial_reward_blocks_an_old_click_from_its_background(self):
        background=self.read('165731-daily-interference-current')
        from ninja3_assistant.activities.daily.flow import DailyDecision
        decision=DailyDecision('tap',(678,687),'进入每日清理项目')
        for name in ('0002-reward-tap.png','0003-reward-tap.png'):
            with self.subTest(name=name):
                s=self.runtime_screen('20261116-154029-909600-village',name)
                self.assertEqual(self.vision.navigation(s.frame)[0],'reward')
                self.assertFalse(self.vision.current_action(background,decision,s.frame))

    def test_family_drum_recognition_tolerates_courtyard_lighting(self):
        for stem in ('085955-family-loaded','173014-family-welcome-dismiss'):
            with self.subTest(stem=stem):
                s=self.read(stem)
                self.assertEqual(s.page,'family')
                self.assertEqual(self.vision.navigation(s.frame)[0],'family')
                point=DailyFlow('blessing').step(s,self.vision).point
                self.assertLess(abs(point[0]-405),20)
                self.assertLess(abs(point[1]-465),20)
        for stem in ('104405-rank-back','165731-daily-interference-current',
                     '172320-summon-guide-dismiss','085932-trial-first-done'):
            with self.subTest(other=stem):
                self.assertIsNone(self.vision.family_entry(self.read(stem).frame))

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

    def test_blessing_recognizes_enlarged_free_label_and_rechecks_it_before_tap(self):
        run='20261118-094520-945100-blessing'
        opening=self.runtime_screen(run,'0003-prayer-wait.png')
        free=self.runtime_screen(run,'0004-prayer-wait.png')
        paid=self.read('090104-blessing-done')
        self.assertEqual(DailyFlow('blessing').step(opening,self.vision).action,'wait')
        decision=DailyFlow('blessing').step(free,self.vision)
        self.assertEqual((decision.point,decision.free_kind),((310,651),'blessing'))
        self.assertTrue(self.vision.current_action(free,decision,free.frame))
        self.assertFalse(self.vision.current_action(free,decision,paid.frame))

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

    def test_summon_up_mode_guide_is_dismissed_before_any_draw_or_shop_entry(self):
        guide=self.runtime_screen('20261116-172041-810200-limited','0005-unknown-wait.png')
        self.assertEqual(guide.page,'summon-guide')
        normal=self.read('172320-summon-guide-dismiss')
        self.assertEqual(normal.page,'summon')
        for kind in ('limited','weapon','curse'):
            with self.subTest(kind=kind):
                decision=DailyFlow(kind).step(guide,self.vision)
                self.assertEqual(decision.point,(640,100))
                self.assertIsNotNone(decision.guard)
                self.assertTrue(self.vision.current_action(guide,decision,guide.frame))
                self.assertFalse(self.vision.current_action(guide,decision,normal.frame))

    def test_ten_day_signin_entry_tracks_unlocked_bottom_options(self):
        points=[]
        for stem in ('084625-summon','172320-summon-guide-dismiss'):
            with self.subTest(stem=stem):
                s=self.read(stem)
                self.assertEqual(s.page,'summon')
                flow=DailyFlow('limited');flow.limited_checked.add('anniversary')
                decision=flow.step(s,self.vision)
                self.assertEqual(decision.action,'tap')
                self.assertGreater(decision.point[1],615)
                self.assertLess(decision.point[1],695)
                points.append(decision.point)
        self.assertGreater(points[1][0]-points[0][0],85)
        self.assertGreater(points[1][0],235)

    def test_ten_day_signin_icon_recovers_when_caption_ocr_is_missing(self):
        for stem,x in (('084625-summon',168),('172320-summon-guide-dismiss',267)):
            with self.subTest(stem=stem):
                s=self.read(stem)
                s.ocr={'words':[]}
                flow=DailyFlow('limited');flow.limited_checked.add('anniversary')
                decision=flow.step(s,self.vision)
                self.assertEqual(decision.action,'tap')
                self.assertLessEqual(abs(decision.point[0]-x),3)
                self.assertGreater(decision.point[1],615)
                self.assertLess(decision.point[1],695)

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

    def test_unlocked_home_and_toast_do_not_block_navigation_or_allow_stale_back(self):
        bad=self.runtime_screen('20261116-165546-388500-monthly','0001-unknown-wait.png')
        current=self.read('165731-daily-interference-current')
        old=self.read('104405-rank-back')
        back=DailyFlow('monthly').step(self.read('104526-blocked-state'),self.vision)
        for s,point in ((bad,(678,687)),(current,(678,687)),(old,(582,676))):
            with self.subTest(summon=point):
                self.assertEqual(s.page,'home')
                self.assertEqual(self.vision.navigation(s.frame),('home',None))
                self.assertEqual(s.home_points['summon'],point)
                self.assertFalse(self.vision.current_action(s,back,s.frame))

    def test_unlocked_home_uses_shifted_summon_shop_and_limited_entry_points(self):
        for s in (self.runtime_screen('20261116-165546-388500-monthly','0001-unknown-wait.png'),
                  self.read('165731-daily-interference-current')):
            for kind,point in (('weapon',(678,687)),('curse',(678,687)),
                               ('bounty_shop',(596,682)),('daily_shop',(596,682)),
                               ('limited',(141,474))):
                with self.subTest(kind=kind):
                    self.assertEqual(DailyFlow(kind).step(s,self.vision).point,point)
            limited=DailyFlow('limited');limited.limited_checked.add('anniversary')
            self.assertEqual(limited.step(s,self.vision).point,(678,687))

    def test_original_home_still_uses_its_actual_bottom_button_positions(self):
        s=self.read('104405-rank-back')
        for kind,point in (('weapon',(582,676)),('curse',(582,676)),
                           ('bounty_shop',(478,676)),('daily_shop',(478,676))):
            with self.subTest(kind=kind):
                self.assertEqual(DailyFlow(kind).step(s,self.vision).point,point)

    def test_rank_daily_and_preview_pages_are_both_recognized_for_handoff(self):
        for stem in ('104338-hidden-state','104526-blocked-state'):
            with self.subTest(stem=stem):
                s=self.read(stem)
                self.assertEqual(s.page,'rank')
                self.assertEqual(self.vision.navigation(s.frame),('rank',(75,50)))
                self.assertEqual(DailyFlow('monthly').step(s,self.vision).point,(75,50))
        self.assertEqual(self.vision.navigation(self.read('104405-rank-back').frame),('home',None))

    def test_rank_claim_ignores_highlight_animation_and_excludes_mainline_rewards(self):
        before=self.read('104338-hidden-state')
        animated=self.read('144911-rank-before')
        claim=DailyFlow('rank').step(before,self.vision)
        self.assertEqual(claim.point,(1145,340))
        self.assertTrue(self.vision.current_action(before,claim,animated.frame))
        self.assertFalse(self.vision.current_action(before,claim,self.read('104405-rank-back').frame))
        after=self.read('145556-rank-next-tier')
        flow=DailyFlow('rank')
        self.assertEqual(flow.step(after,self.vision).point,(135,685))
        self.assertEqual(flow.step(self.read('152918-rank-weekly-done'),self.vision).action,'done')

    def test_full_rank_preview_list_can_return_to_daily_and_use_current_tier(self):
        s=self.runtime_screen('20261116-154043-795700-rank','0005-unknown-wait.png')
        self.assertEqual(s.page,'rank-preview')
        self.assertEqual(self.vision.navigation(s.frame),('rank-preview',(75,50)))
        self.assertEqual(DailyFlow('rank').step(s,self.vision).point,(75,50))

    def test_rank_chest_reward_has_a_navigation_exit_and_preview_leads_to_tasks(self):
        reward=self.read('145142-rank-after-claim')
        self.assertEqual(reward.page,'reward_continue')
        self.assertEqual(self.vision.observe(reward.frame,{'words':[]}).page,'reward_continue')
        self.assertEqual(self.vision.navigation(reward.frame),('reward_continue',(1130,590)))
        self.assertEqual(DailyFlow('rank').step(reward,self.vision).point,(1130,590))
        preview=self.read('145537-rank-preview-settled')
        flow=DailyFlow('rank')
        claim=flow.step(preview,self.vision)
        self.assertIn('每周随机道具',claim.status)
        self.assertTrue(self.vision.current_action(preview,claim,preview.frame))
        self.assertFalse(self.vision.current_action(preview,claim,self.read('104405-rank-back').frame))
        point=flow.step(self.read('152918-rank-weekly-done'),self.vision).point
        self.assertGreater(point[0],1030)
        self.assertGreater(point[1],660)

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

    def test_house_bad_hour_ocr_does_not_stall_or_shorten_the_training_deadline(self):
        s=self.runtime_screen('20261116-150734-344900-house','0003-house-wait.png')
        self.assertEqual(remaining_seconds(s.text((245,590,460,715))),17*3600+53*60)
        self.assertEqual(DailyFlow('house').step(s,self.vision).action,'done')

    def test_trial_chooses_the_rightmost_cleared_stage_and_never_launches_battle(self):
        flow=DailyFlow('village')
        decision=flow.step(self.read('085820-trial-guide'),self.vision)
        self.assertIn('第28关',decision.status)
        self.assertEqual(decision.point,(697,360))

    def test_trial_ocr_missing_count_label_character_scans_left_instead_of_waiting(self):
        for name in ('0001-trial-wait.png','0002-trial-wait.png','0003-trial-wait.png'):
            with self.subTest(frame=name):
                s=self.runtime_screen('20261116-175919-475000-village',name)
                flow=DailyFlow('village')
                decision=flow.step(s,self.vision)
                self.assertEqual(s.page,'trial')
                self.assertEqual(decision.action,'swipe')
                self.assertGreater(decision.end[0],decision.point[0])
                self.assertIn(50,flow.visited_stages)
                self.assertNotIn(45,flow.visited_stages)
