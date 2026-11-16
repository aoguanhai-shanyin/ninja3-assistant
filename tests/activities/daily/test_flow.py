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
