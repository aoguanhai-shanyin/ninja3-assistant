import unittest

import numpy as np

from ninja3_assistant.activities.bounty.daily_flow import DailyBountyFlow
from ninja3_assistant.activities.bounty.daily_vision import DailyBountyScreen, DailyBountyVision


def screen(page, **fields):
    return DailyBountyScreen(np.zeros((720,1280,3),np.uint8),{'words':[]},page,**fields)


class DailyBountyFlowTest(unittest.TestCase):
    def test_invalid_kind_and_grade_are_rejected(self):
        for kind,grade in (('bounty','S'),('reverse','D'),('bounty','SS+')):
            with self.assertRaises(ValueError):
                DailyBountyFlow(kind,grade)

    def test_only_the_selected_grade_is_joined(self):
        flow=DailyBountyFlow('bounty','B')
        flow.free_checked=True;flow.confirmed_grade='B'
        wrong=screen('bounty_daily',remaining=3,grade='D',join=(390,300))
        self.assertIn('设置',flow.step(wrong,None).status)
        target=screen('bounty_daily',remaining=3,grade='B',join=(390,300))
        choice=flow.step(target,None)
        self.assertEqual(choice.point,(390,300))
        self.assertIn('加入B级',choice.status)

    def test_no_target_team_creates_one_at_the_selected_grade(self):
        flow=DailyBountyFlow('bounty','C')
        flow.free_checked=True;flow.confirmed_grade='C'
        decision=flow.step(screen('bounty_daily',remaining=3,grade='C'),None)
        self.assertIn('创建C级',decision.status)
        self.assertIsNone(flow.step(screen('bounty_create',grade='A',create=(500,500)),None).point)

    def test_zero_regular_quota_still_checks_the_free_purchase(self):
        flow=DailyBountyFlow()
        decision=flow.step(screen('bounty_daily',remaining=0),None)
        self.assertEqual(decision.action,'tap')
        self.assertIn('免费',decision.status)

    def test_paid_purchase_is_closed_and_zero_quota_then_finishes(self):
        flow=DailyBountyFlow()
        paid=screen('bounty_free',free_exhausted=True,close=(900,200))
        self.assertEqual(flow.step(paid,None).point,(900,200))
        self.assertEqual(flow.step(screen('bounty_daily',remaining=0),None).action,'done')

    def test_free_request_is_sent_once_and_does_not_claim_success(self):
        flow=DailyBountyFlow()
        listing=screen('bounty_daily',remaining=0)
        check=flow.step(listing,None);flow.completed(check,listing)
        free=screen('bounty_free',free_claim=(500,500),free_guard=(400,450,600,550))
        request=flow.step(free,None)
        self.assertEqual(request.free_kind,'bounty')
        flow.completed(request,free)
        self.assertEqual(flow.step(free,None).action,'wait')
        self.assertFalse(flow.free_checked)
        self.assertNotEqual(flow.step(screen('bounty_daily',remaining=0),None).action,'done')
        flow.step(screen('bounty_daily',remaining=1,grade='D'),None)
        self.assertTrue(flow.free_checked)

    def test_unknown_quotas_are_not_completed(self):
        for kind,page in (('bounty','bounty_daily'),('bounty_s','bounty_order')):
            self.assertEqual(DailyBountyFlow(kind).step(screen(page),None).action,'wait')

    def test_s_task_clears_the_observed_quota_and_never_enables_ss(self):
        flow=DailyBountyFlow('bounty_s','D')
        s=screen('bounty_order',remaining=7,s_filter='off',ss_filter='on')
        self.assertEqual(flow.step(s,None).point,(876,171))
        s.s_filter='on'
        self.assertEqual(flow.step(s,None).point,(978,171))
        s.ss_filter='off';s.join=(390,300)
        self.assertEqual(flow.step(s,None).point,(390,300))
        s.remaining=0
        self.assertEqual(flow.step(s,None).action,'done')

    def test_room_must_belong_to_a_target_grade_action(self):
        flow=DailyBountyFlow('bounty','A')
        flow.free_checked=True;flow.confirmed_grade='A'
        room=screen('bounty_room',ready=(1100,660))
        self.assertEqual(flow.step(room,None).action,'unavailable')
        listing=screen('bounty_daily',remaining=2,grade='A',join=(390,300))
        join=flow.step(listing,None);flow.completed(join,listing)
        self.assertEqual(flow.step(room,None).point,(1100,660))

    def test_combat_and_settlement_do_not_invent_a_quota_decrement(self):
        flow=DailyBountyFlow()
        self.assertEqual(flow.step(screen('battle'),None).action,'battle')
        result=flow.step(screen('result',confirm=(640,670)),None)
        self.assertEqual(result.point,(640,670))
        self.assertEqual(flow.step(screen('bounty_daily',remaining=None),None).action,'wait')

    def test_matching_grade_result_can_resume_after_a_worker_restart(self):
        flow=DailyBountyFlow('bounty','D')
        result=screen('result',grade='D',confirm=(640,670))
        self.assertEqual(flow.step(result,None).point,(640,670))
        result.grade='A'
        self.assertEqual(flow.step(result,None).action,'unavailable')

    def test_counter_reader_separates_s_and_ss_plus(self):
        self.assertEqual(DailyBountyVision._count('今日奖励剩余次数:3','daily'),3)
        self.assertEqual(DailyBountyVision._count('剩余奖励次数:S悬赏:7/SS+悬赏:1','order'),7)
        self.assertEqual(DailyBountyVision._count('S懣赏．0/SS+Æ赏:1','order'),0)
        self.assertIsNone(DailyBountyVision._count('SS+悬赏:7','order'))

    def test_s_creation_uses_only_a_matching_single_s_title_and_same_card_button(self):
        s=screen('unknown')
        s.ocr={'words':[{'text':'S级悬赏令','bbox':[450,200,180,30]},
                        {'text':'创建队伍','bbox':[475,430,130,35]}]}
        self.assertEqual(DailyBountyVision._s_creation(s),(540,448))
        s.ocr['words'][0]['text']='SS+级悬赏令'
        self.assertIsNone(DailyBountyVision._s_creation(s))
        s.ocr['words'][0]['text']='S级悬赏令'
        s.ocr['words'][1]['bbox']=[900,430,130,35]
        self.assertIsNone(DailyBountyVision._s_creation(s))

    def test_s_creation_does_not_infer_remaining_quota_from_holding_a_book(self):
        flow=DailyBountyFlow('bounty_s')
        create=screen('bounty_create',grade='S',create=(540,448))
        self.assertNotEqual(flow.step(create,None).point,(540,448))
        flow.step(screen('bounty_order',remaining=3,s_filter='on',ss_filter='off'),None)
        self.assertEqual(flow.step(create,None).point,(540,448))

    def test_locked_s_mode_is_not_reported_as_zero_stock_or_completed(self):
        decision=DailyBountyFlow('bounty_s').step(screen('bounty_locked'),None)
        self.assertEqual(decision.action,'unavailable')
        self.assertIn('尚未通过等级测试',decision.status)

    def test_empty_owned_orders_return_to_search_then_retry_without_buying(self):
        flow=DailyBountyFlow('bounty_s')
        owned=screen('bounty_own_empty')
        back=flow.step(owned,None);flow.completed(back,owned)
        self.assertEqual(back.point,(1115,670))
        listing=screen('bounty_order',remaining=3,s_filter='on',ss_filter='off')
        self.assertEqual(flow.step(listing,None).action,'retry')

    def test_failed_quick_join_is_retried_by_the_planner(self):
        flow=DailyBountyFlow('bounty','D')
        decision=flow.step(screen('bounty_daily',remaining=4,grade='D',join_failed=True),None)
        self.assertEqual(decision.action,'retry')
        self.assertIsNone(decision.point)

    def test_quick_join_is_not_resent_while_waiting_for_the_room(self):
        now=[0]
        flow=DailyBountyFlow('bounty','D',clock=lambda:now[0])
        flow.free_checked=True;flow.confirmed_grade='D';flow.try_quick_join=True
        listing=screen('bounty_daily',remaining=4,grade='D')
        first=flow.step(listing,None);flow.completed(first,listing)
        now[0]=1
        self.assertEqual(flow.step(listing,None).action,'wait')
        now[0]=6
        self.assertEqual(flow.step(listing,None).action,'retry')

    def test_empty_created_room_waits_then_leaves_without_inviting_friends(self):
        now=[0]
        flow=DailyBountyFlow('bounty','D',clock=lambda:now[0])
        flow.free_checked=True;flow.confirmed_grade='D'
        listing=screen('bounty_daily',remaining=3,grade='D')
        create=flow.step(listing,None);flow.completed(create,listing)
        room=screen('bounty_room',grade='D',minimum_players=True)
        self.assertEqual(flow.step(room,None).action,'wait')
        now[0]=21
        leave=flow.step(room,None)
        self.assertEqual(leave.point,(65,50))
        confirm=flow.step(screen('bounty_leave'),None);flow.completed(confirm,screen('bounty_leave'))
        self.assertEqual(confirm.point,(810,525))
        self.assertIn('快捷加入D级',flow.step(listing,None).status)

    def test_grade_popup_confirmation_allows_a_missing_list_badge(self):
        flow=DailyBountyFlow('bounty','B');flow.free_checked=True
        listing=screen('bounty_daily',remaining=3,grade=None)
        self.assertIn('设置',flow.step(listing,None).status)
        popup=screen('bounty_grade',grade='B',confirm=(640,490),grade_choices={'B':(640,365)})
        decision=flow.step(popup,None);flow.completed(decision,popup)
        self.assertIn('创建B级',flow.step(listing,None).status)

    def test_free_request_requires_observed_quota_increase_from_initial_three(self):
        flow=DailyBountyFlow()
        listing=screen('bounty_daily',remaining=3)
        check=flow.step(listing,None);flow.completed(check,listing)
        free=screen('bounty_free',free_claim=(500,500),free_guard=(400,450,600,550))
        request=flow.step(free,None);flow.completed(request,free)
        flow.step(listing,None)
        self.assertFalse(flow.free_checked)
        listing.remaining=4
        flow.step(listing,None)
        self.assertTrue(flow.free_checked)


if __name__ == '__main__':
    unittest.main()
