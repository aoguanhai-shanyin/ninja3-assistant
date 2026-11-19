import unittest
from unittest.mock import Mock
import numpy as np

from ninja3_assistant.activities.daily.flow import DailyFlow
from ninja3_assistant.activities.daily.shop_stock import recover_coin_stock
from ninja3_assistant.activities.daily.vision import DailyScreen


def screen(page, words):
    return DailyScreen(np.zeros((720,1280,3),np.uint8),
                       {'words':[{'text':text,'bbox':box} for text,box in words]},page)


class DailyFlowTest(unittest.TestCase):
    def test_login_uses_the_recognized_button_not_unlocated_ocr_words(self):
        current=screen('login',[]);current.point=(640,515)
        decision=DailyFlow('monthly').step(current,self.vision())
        self.assertEqual(decision.point,(640,515))
        self.assertEqual(decision.status,'登录当前已选账号')
        words=[('登录',[600,490,80,50]),('点击切换',[700,418,102,29])]
        self.assertEqual(DailyFlow('monthly').step(screen('unknown',words),self.vision()).action,'wait')

    def test_home_routes_follow_unlocked_shortcuts_and_shifted_bottom_buttons(self):
        s=screen('home',[])
        s.home_points={'summon':(690,676),'shop':(586,676),'limited':(145,475)}
        for kind,target in (('weapon','summon'),('curse','summon'),
                            ('bounty_shop','shop'),('daily_shop','shop'),('limited','limited')):
            with self.subTest(kind=kind):
                self.assertEqual(DailyFlow(kind).step(s,self.vision()).point,s.home_points[target])
        flow=DailyFlow('limited');flow.limited_checked.add('anniversary')
        self.assertEqual(flow.step(s,self.vision()).point,s.home_points['summon'])

    def test_missing_movable_home_entry_does_not_tap_its_old_position(self):
        for kind in ('weapon','curse','bounty_shop','daily_shop','limited'):
            with self.subTest(kind=kind):
                decision=DailyFlow(kind).step(screen('home',[]),self.vision())
                self.assertEqual(decision.action,'wait')
                self.assertIsNone(decision.point)
                self.assertEqual(decision.defer_on_timeout,kind == 'limited')

    def test_summon_signup_entry_follows_its_label_after_unlocking_recycle(self):
        flow=DailyFlow('limited');flow.limited_checked.add('anniversary')
        s=screen('summon',[('十日签',[240,670,60,22])])
        self.assertEqual(flow.step(s,self.vision()).point,(270,681))
        missing=flow.step(screen('summon',[]),self.vision())
        self.assertEqual(missing.action,'wait')
        self.assertFalse(missing.defer_on_timeout)

    def test_known_summon_guide_is_closed_outside_all_draw_buttons(self):
        decision=DailyFlow('weapon').step(screen('summon-guide',[]),self.vision())
        self.assertEqual(decision.point,(640,100))
        self.assertIsNone(decision.free_kind)

    def vision(self, *, free=False, coin=True):
        v=Mock();v.free.return_value=free;v.coin.return_value=coin;v.match.return_value=None
        v.food.side_effect=lambda s:int(s.hits(r'\d+',(1160,15,1225,75))[0].text) if s.hits(r'\d+',(1160,15,1225,75)) else None
        return v

    def test_free_draw_is_requested_only_once_while_waiting_for_result(self):
        for kind,page in (('curse','curse'),('weapon','weapon'),('blessing','prayer')):
            with self.subTest(kind=kind):
                flow=DailyFlow(kind);s=screen(page,[]);v=self.vision(free=True)
                first=flow.step(s,v)
                self.assertEqual(first.free_kind,kind)
                flow.completed(first,s)
                self.assertEqual(flow.step(s,v).action,'wait')

    def test_curse_cooldown_is_not_a_free_draw(self):
        flow=DailyFlow('curse')
        s=screen('curse',[('下次免费:19小时57分60炼化1次',[310,680,230,20])])
        decision=flow.step(s,self.vision())
        self.assertEqual(decision.action,'done')
        self.assertIsNone(decision.point)

    def test_silver_vault_accepts_observed_cooldown_ocr_error_without_tapping(self):
        flow=DailyFlow('weapon')
        s=screen('weapon',[('下次晚费:]7小时09分',[674,579,210,28])])
        decision=flow.step(s,self.vision())
        self.assertEqual(decision.action,'done')
        self.assertIsNone(decision.point)

    def test_reward_confirmation_does_not_press_the_paid_repeat_button(self):
        s=screen('reward',[('确定',[425,653,60,30]),('60再炼化1次',[710,650,230,35])])
        decision=DailyFlow('curse').step(s,self.vision())
        self.assertEqual(decision.point,(455,668))
        self.assertLess(decision.point[0],600)

    def test_premium_currency_shop_dialog_is_never_confirmed(self):
        flow=DailyFlow('daily_shop');flow.pending_item='ninbi'
        decision=flow.step(screen('shop_dialog',[]),self.vision(coin=False))
        self.assertEqual(decision.action,'unavailable')
        self.assertIsNone(decision.point)

    def test_unknown_stock_does_not_count_as_a_completed_purchase(self):
        s=screen('daily_shop',[('神龙之血',[530,110,100,30])])
        self.assertEqual(DailyFlow('daily_shop').step(s,self.vision()).action,'wait')

    def test_bare_positive_stock_or_a_zero_outside_the_stock_line_is_unconfirmed(self):
        v=self.vision();v.coin.side_effect=lambda _,box:box == (292,580,472,630)
        for words in ([('剩余次数:1',[310,427,119,19])],
                      [('剩余次数:',[310,427,94,19]),('0',[404,590,11,15])]):
            with self.subTest(words=words):
                decision=DailyFlow('daily_shop').step(screen('daily_shop',words),v)
                self.assertEqual(decision.action,'wait')

    def test_zoomed_stock_recovers_the_coin_slot_from_merged_digits(self):
        s=screen('daily_shop',[('火锤剩余次数:忉',[310,427,119,19]),
                               ('碎片剩余次数:1/1',[758,425,120,19])])
        v=self.vision();v.coin.side_effect=lambda _,box:box == (292,580,472,630)
        engine=Mock();engine.read.return_value={'text':'剩余次数：1 / 1'}
        self.assertTrue(recover_coin_stock(s,v,engine,None))
        engine.read.assert_called_once()
        self.assertEqual(engine.read.call_args.args[0].shape,(198,710,3))
        self.assertEqual(DailyFlow('daily_shop').step(s,v).point,(372,510))

    def test_zoomed_stock_without_a_valid_fraction_keeps_waiting(self):
        v=self.vision();v.coin.side_effect=lambda _,box:box == (292,580,472,630)
        for text,action in (('剩余次数:忉','wait'),('剩余次数:2/1','wait'),('剩余次数:0/1','done')):
            with self.subTest(text=text):
                s=screen('daily_shop',[('火锤剩余次数:忉',[310,427,119,19])])
                engine=Mock();engine.read.return_value={'text':text}
                self.assertTrue(recover_coin_stock(s,v,engine,None))
                self.assertEqual(DailyFlow('daily_shop').step(s,v).action,action)

    def test_monthly_signin_stops_at_twenty_five(self):
        s=screen('monthly',[('本月累计签到25次',[500,240,230,30])])
        self.assertEqual(DailyFlow('monthly').step(s,self.vision()).action,'done')

    def test_house_only_continues_the_known_five_food_request(self):
        flow=DailyFlow('house')
        s=screen('house_continue',[('是否消耗5饭团继续进行修炼?',[400,350,470,50])])
        self.assertEqual(flow.step(s,self.vision()).point,(810,530))
        s=screen('house_continue',[('消耗60勾玉修炼',[400,350,470,50])])
        self.assertEqual(flow.step(s,self.vision()).action,'unavailable')

    def test_house_waiting_for_training_uses_the_same_countdown_parser(self):
        flow=DailyFlow('house')
        for text in ('17刁；时53分','53分','59秒'):
            with self.subTest(text=text):
                s=screen('house',[(text,[275,680,100,25])])
                self.assertEqual(flow.step(s,self.vision()).action,'done')
        for text in ('17?时53分','无法识别','0秒'):
            with self.subTest(text=text):
                s=screen('house',[(text,[275,680,100,25])])
                self.assertEqual(flow.step(s,self.vision()).action,'wait')

    def test_food_reservation_stops_sweeping_before_house_is_starved(self):
        s=screen('trial',[('5',[1180,30,30,30]),('第28关',[660,220,75,25]),
                          ('通关奖励剩余挑战次数:2',[610,350,180,190])])
        flow=DailyFlow('village',clock=lambda:0)
        self.assertEqual(flow.step(s,self.vision()).action,'unavailable')
        flow=DailyFlow('village',training_finishes_at=10800,clock=lambda:0)
        choose=flow.step(s,self.vision());flow.completed(choose,s)
        self.assertIn('第28关',choose.status)
        sweep=flow.step(s,self.vision())
        self.assertEqual(sweep.point,(895,649))
        self.assertNotIn('出发',sweep.status)

    def test_unknown_food_or_scan_limit_does_not_mark_sweeps_completed(self):
        flow=DailyFlow('village')
        unknown=flow.step(screen('trial',[]),self.vision())
        self.assertEqual(unknown.action,'wait')
        self.assertTrue(unknown.defer_on_timeout)
        flow.scans=7
        self.assertEqual(flow.step(screen('trial',[('20',[1180,30,30,30])]),self.vision()).action,'unavailable')

    def test_unreadable_rightmost_counter_does_not_skip_to_a_lower_stage(self):
        s=screen('trial',[('20',[1180,30,30,30]),('第28关',[880,220,75,25]),
                          ('通关奖励',[860,350,120,30]),('第27关',[660,220,75,25]),
                          ('通关奖励挑战次数:2',[610,350,180,190])])
        flow=DailyFlow('village')
        choose=flow.step(s,self.vision());flow.completed(choose,s)
        self.assertIn('第28关',choose.status)
        unreadable=flow.step(s,self.vision())
        self.assertEqual(unreadable.action,'wait')
        self.assertTrue(unreadable.defer_on_timeout)

    def test_trial_counter_missing_the_last_label_character_keeps_right_to_left_order(self):
        s=screen('trial',[('20',[1180,30,30,30]),('第50关',[1100,220,75,25]),
                          ('通关奖励剩余挑战次00',[1080,350,180,190]),
                          ('第49关',[860,220,75,25]),
                          ('通关奖励剩余挑战次数:2',[820,350,180,190])])
        flow=DailyFlow('village')
        decision=flow.step(s,self.vision())
        self.assertIn('第49关',decision.status)
        self.assertEqual(flow.visited_stages,{50})

    def test_selected_trial_counter_accepts_a_missing_label_character_but_requires_a_value(self):
        flow=DailyFlow('village');flow.stage=50
        base=[('20',[1180,30,30,30]),('第50关',[1100,220,75,25]),
              ('通关奖励x4',[1080,350,180,190])]
        unreadable=screen('trial',base+[('剩余挑战次:',[1080,585,180,25])])
        self.assertEqual(flow.step(unreadable,self.vision()).action,'wait')
        readable=screen('trial',base+[('剩余挑战次:2',[1080,585,180,25])])
        self.assertEqual(flow.step(readable,self.vision()).point,(895,649))

    def rank_screen(self, buttons):
        return screen('rank',[('每日任务',[625,240,90,25]),
                              ('主线任务',[625,540,90,25]),
                              *[(text,[1100,y,95,25]) for text,y in buttons]])

    def test_rank_claims_only_daily_rewards_and_does_not_follow_unfinished_tasks(self):
        flow=DailyFlow('rank')
        s=self.rank_screen([('点击领取',330),('点击领取',414),
                            ('点击前往',502),('点击领取',626)])
        claim=flow.step(s,self.vision())
        self.assertEqual(claim.point,(1148,342))
        self.assertIn('每日任务',claim.status)
        # Claimed rows are removed, so completion cannot require three remaining rows.
        remaining=self.rank_screen([('点击前往',330),('点击领取',626)])
        flow.rank_weekly_checked=True
        self.assertEqual(flow.step(remaining,self.vision()).action,'done')
        empty=screen('rank',[('每日任务',[625,240,90,25]),('主线任务',[625,280,90,25]),
                             ('点击领取',[1100,330,95,25])])
        self.assertEqual(flow.step(empty,self.vision()).action,'done')

    def test_rank_incomplete_ocr_and_missing_section_are_not_completed(self):
        flow=DailyFlow('rank')
        self.assertEqual(flow.step(self.rank_screen([]),self.vision()).action,'wait')
        unreadable=self.rank_screen([('点击前往',502)])
        unreadable.frame[280:360,1060:1235]=(160,200,40)
        self.assertEqual(flow.step(unreadable,self.vision()).action,'wait')
        partial=screen('rank',[('每日任务',[625,240,90,25]),('点击领取',[1100,600,95,25])])
        self.assertEqual(flow.step(partial,self.vision()).action,'swipe')

    def test_rank_preview_claims_weekly_reward_then_enters_daily_tasks(self):
        preview=screen('rank',[('特殊思阶',[1100,50,120,30]),
                               ('点击领取每周随机道具',[670,455,310,30]),
                               ('下一忍阶',[1050,680,160,30])])
        flow=DailyFlow('rank')
        decision=flow.step(preview,self.vision())
        self.assertEqual(decision.point,(825,470))
        claimed=screen('rank',[('特殊思阶',[1100,50,120,30]),
                              ('每周随机获得道具8个',[630,405,280,30]),
                              ('下一忍阶',[1050,680,160,30])])
        self.assertEqual(flow.step(claimed,self.vision()).point,(1130,695))
        self.assertEqual(DailyFlow('monthly').step(preview,self.vision()).point,(75,50))
        self.assertEqual(flow.step(self.rank_screen([('点击前往',330)]),self.vision()).action,'done')

    def test_rank_daily_completion_checks_weekly_before_finishing(self):
        flow=DailyFlow('rank')
        decision=flow.step(self.rank_screen([('点击前往',330)]),self.vision())
        self.assertEqual(decision.point,(135,685))
        claimed=screen('rank',[('特殊思阶',[1100,50,120,30]),
                              ('每周随机获得道具8个',[630,405,280,30])])
        self.assertEqual(flow.step(claimed,self.vision()).action,'done')
        self.assertEqual(DailyFlow('rank').step(screen('rank',[
            ('特殊思阶',[1100,50,120,30])]),self.vision()).action,'wait')
        self.assertEqual(flow.step(screen('rank-preview',[]),self.vision()).point,(75,50))

    def test_rank_chest_reward_closes_outside_the_bottom_prompt(self):
        reward=screen('reward_continue',[])
        self.assertEqual(DailyFlow('rank').step(reward,self.vision()).point,(1130,590))
