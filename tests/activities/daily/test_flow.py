import unittest
from unittest.mock import Mock
import numpy as np

from ninja3_assistant.activities.daily.flow import DailyFlow
from ninja3_assistant.activities.daily.vision import DailyScreen


def screen(page, words):
    return DailyScreen(np.zeros((720,1280,3),np.uint8),
                       {'words':[{'text':text,'bbox':box} for text,box in words]},page)


class DailyFlowTest(unittest.TestCase):
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
        self.assertEqual(flow.step(screen('trial',[]),self.vision()).action,'wait')
        flow.scans=7
        self.assertEqual(flow.step(screen('trial',[('20',[1180,30,30,30])]),self.vision()).action,'unavailable')

    def test_unreadable_rightmost_counter_does_not_skip_to_a_lower_stage(self):
        s=screen('trial',[('20',[1180,30,30,30]),('第28关',[880,220,75,25]),
                          ('通关奖励',[860,350,120,30]),('第27关',[660,220,75,25]),
                          ('通关奖励挑战次数:2',[610,350,180,190])])
        flow=DailyFlow('village')
        choose=flow.step(s,self.vision());flow.completed(choose,s)
        self.assertIn('第28关',choose.status)
        self.assertEqual(flow.step(s,self.vision()).action,'wait')

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
