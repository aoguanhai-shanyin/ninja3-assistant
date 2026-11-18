import unittest
from types import SimpleNamespace

import numpy as np

from ninja3_assistant.activities.bounty.daily_flow import DailyBountyFlow
from ninja3_assistant.activities.bounty.daily_vision import DailyBountyScreen, DailyBountyVision


def screen(page, **fields):
    return DailyBountyScreen(np.zeros((720,1280,3),np.uint8),{'words':[]},page,**fields)


def checked_flow(*args, **kwargs):
    flow = DailyBountyFlow(*args, **kwargs)
    flow.tasks_checked = True
    flow.task_states = {'friend':'claimed','family':'claimed'}
    return flow


class DailyBountyFlowTest(unittest.TestCase):
    def test_daily_tasks_are_checked_and_only_completed_when_both_are_claimed(self):
        flow=DailyBountyFlow('bounty','A')
        self.assertEqual(flow.step(screen('bounty_hub'),None).point,(155,675))
        self.assertEqual(flow.step(screen('bounty_daily',remaining=0),None).point,(330,670))
        tasks=screen('bounty_tasks',task_states={'friend':'claimed','family':'unknown'})
        self.assertEqual(flow.step(tasks,None).point,(1025,155))
        flow.free_checked=True
        self.assertEqual(flow.step(screen('bounty_daily',remaining=0),None).action,'retry')
        flow.task_states={'friend':'claimed'}
        self.assertEqual(flow.step(screen('bounty_daily',remaining=0),None).action,'retry')
        tasks.task_states={'friend':'claimed','family':'claimed'}
        flow.step(tasks,None)
        self.assertEqual(flow.step(screen('bounty_daily',remaining=0),None).action,'done')
        result=screen('result',grade='A',confirm=(640,670))
        flow.completed(flow.step(result,None),result)
        self.assertEqual(flow.step(screen('bounty_daily',remaining=0),None).point,(330,670))

    def test_reward_overlay_requires_the_tasks_to_be_rechecked(self):
        flow=checked_flow()
        reward=screen('reward',confirm=(1150,685))
        self.assertEqual(flow.step(reward,None).point,(1150,685))
        self.assertEqual(flow.step(screen('bounty_daily',remaining=0),None).point,(330,670))

    def test_claim_and_unfinished_task_are_not_reported_as_done(self):
        flow=DailyBountyFlow('bounty','A')
        tasks=screen('bounty_tasks',task_states={'friend':'claimable','family':'pending'},task_claim=(880,275))
        self.assertEqual(flow.step(tasks,None).point,(880,275))
        self.assertFalse(flow.tasks_checked)
        tasks.task_claim=None
        tasks.task_states={'friend':'claimed','family':'pending'}
        flow.step(tasks,None)
        flow.free_checked=True
        flow.confirmed_grade='A'
        self.assertIn('创建A级',flow.step(screen('bounty_daily',remaining=0,grade='A'),None).status)

    def test_invalid_kind_and_grade_are_rejected(self):
        for kind,grade in (('bounty','S'),('reverse','D'),('bounty','SS+')):
            with self.assertRaises(ValueError):
                checked_flow(kind,grade)

    def test_only_the_selected_grade_is_joined(self):
        flow=checked_flow('bounty','B')
        flow.free_checked=True;flow.confirmed_grade='B'
        wrong=screen('bounty_daily',remaining=3,grade='D',join=(390,300))
        self.assertIn('设置',flow.step(wrong,None).status)
        target=screen('bounty_daily',remaining=3,grade='B',join=(390,300))
        choice=flow.step(target,None)
        self.assertEqual(choice.point,(390,300))
        self.assertIn('加入B级',choice.status)

    def test_no_target_team_creates_one_at_the_selected_grade(self):
        flow=checked_flow('bounty','C')
        flow.free_checked=True;flow.confirmed_grade='C'
        decision=flow.step(screen('bounty_daily',remaining=3,grade='C'),None)
        self.assertIn('创建C级',decision.status)
        self.assertIsNone(flow.step(screen('bounty_create',grade='A',create=(500,500)),None).point)

    def test_zero_regular_quota_still_checks_the_free_purchase(self):
        flow=checked_flow()
        decision=flow.step(screen('bounty_daily',remaining=0),None)
        self.assertEqual(decision.action,'tap')
        self.assertIn('免费',decision.status)

    def test_paid_purchase_is_closed_and_zero_quota_then_finishes(self):
        flow=checked_flow()
        paid=screen('bounty_free',free_exhausted=True,close=(900,200))
        self.assertEqual(flow.step(paid,None).point,(900,200))
        self.assertEqual(flow.step(screen('bounty_daily',remaining=0),None).action,'done')

    def test_free_request_is_sent_once_and_does_not_claim_success(self):
        flow=checked_flow()
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
            self.assertEqual(checked_flow(kind).step(screen(page),None).action,'wait')

    def test_s_task_clears_the_observed_quota_and_never_enables_ss(self):
        flow=checked_flow('bounty_s','D')
        s=screen('bounty_order',remaining=7,s_filter='off',ss_filter='on')
        self.assertEqual(flow.step(s,None).point,(876,171))
        s.s_filter='on'
        self.assertEqual(flow.step(s,None).point,(978,171))
        s.ss_filter='off';s.join=(390,300)
        self.assertEqual(flow.step(s,None).point,(390,300))
        s.remaining=0
        self.assertEqual(flow.step(s,None).action,'done')

    def test_room_must_belong_to_a_target_grade_action(self):
        flow=checked_flow('bounty','A')
        flow.free_checked=True;flow.confirmed_grade='A'
        room=screen('bounty_room',ready=(1100,660))
        self.assertEqual(flow.step(room,None).action,'unavailable')
        listing=screen('bounty_daily',remaining=2,grade='A',join=(390,300))
        join=flow.step(listing,None);flow.completed(join,listing)
        self.assertEqual(flow.step(room,None).point,(1100,660))

    def test_only_safe_matching_friend_or_family_bounty_invites_are_accepted(self):
        flow=checked_flow('bounty','A')
        invitation=SimpleNamespace(source='friend',grade='A',accept=(810,525))
        dialog=screen('invite',point=(470,525),invitation=invitation)
        decision=flow.step(dialog,None)
        self.assertEqual(decision.point,(810,525))
        flow.completed(decision,dialog)
        self.assertEqual(flow.step(dialog,None).point,(470,525))
        self.assertEqual(flow.step(screen('bounty_room',grade='B'),None).action,'unavailable')
        self.assertEqual(flow.step(screen('bounty_room',grade='A',ready=(1100,660)),None).point,(1100,660))
        for source,grade in (('friend','B'),('family','S'),('unknown','A')):
            other=checked_flow('bounty','A')
            dialog.invitation=SimpleNamespace(source=source,grade=grade,accept=(810,525))
            self.assertEqual(other.step(dialog,None).point,(470,525))
        family=checked_flow('bounty_s')
        dialog.invitation=SimpleNamespace(source='family',grade='S',accept=(810,525))
        self.assertEqual(family.step(dialog,None).point,(810,525))
        family.battle_seen=True
        self.assertEqual(family.step(dialog,None).point,(470,525))

    def test_no_remaining_rewards_does_not_forbid_assisting_family(self):
        flow=checked_flow('bounty','D')
        flow.free_checked=True
        flow.step(screen('bounty_daily',remaining=0),None)
        invite=SimpleNamespace(source='family',grade='D',accept=(810,525))
        self.assertEqual(flow.step(screen('invite',point=(470,525),invitation=invite),None).point,(810,525))

    def test_active_and_passive_invitation_switches_are_independent(self):
        for kind,grade in (('bounty','A'),('bounty_s','S')):
            for active,passive in ((False,False),(True,False),(False,True),(True,True)):
                with self.subTest(grade=grade,active=active,passive=passive):
                    flow=checked_flow(kind,'D' if grade=='S' else grade,
                                      active_invite=active,passive_invite=passive)
                    for source in ('friend','family'):
                        invite=SimpleNamespace(source=source,grade=grade,accept=(810,525))
                        dialog=screen('invite',point=(470,525),invitation=invite)
                        self.assertEqual(flow.step(dialog,None).point,(810,525) if passive else (470,525))

                    flow.room_owned=True
                    room=screen('bounty_room',grade=grade,minimum_players=True,invite_open=(640,383))
                    self.assertEqual(flow.step(room,None).point,(640,383) if active else None)

                    flow.recruit_stage='friends'
                    panel=screen('bounty_recruit',recruit_tab='friend',
                                 recruit_candidates=[('好友甲',(1210,135))])
                    self.assertEqual(flow.step(panel,None).point,(1210,135) if active else (750,350))

                    ready=screen('bounty_room',grade=grade,ready=(1100,660))
                    self.assertEqual(flow.step(ready,None).point,(1100,660))

    def test_combat_and_settlement_do_not_invent_a_quota_decrement(self):
        flow=checked_flow()
        self.assertEqual(flow.step(screen('battle'),None).action,'battle')
        result=flow.step(screen('result',confirm=(640,670)),None)
        self.assertEqual(result.point,(640,670))
        self.assertEqual(flow.step(screen('bounty_daily',remaining=None),None).action,'wait')

    def test_matching_grade_result_can_resume_after_a_worker_restart(self):
        flow=checked_flow('bounty','D')
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
        flow=checked_flow('bounty_s')
        create=screen('bounty_create',grade='S',create=(540,448))
        self.assertNotEqual(flow.step(create,None).point,(540,448))
        flow.step(screen('bounty_order',remaining=3,s_filter='on',ss_filter='off'),None)
        self.assertEqual(flow.step(create,None).point,(540,448))

    def test_locked_s_mode_is_not_reported_as_zero_stock_or_completed(self):
        decision=checked_flow('bounty_s').step(screen('bounty_locked'),None)
        self.assertEqual(decision.action,'unavailable')
        self.assertIn('尚未通过等级测试',decision.status)

    def test_empty_owned_orders_return_to_search_then_retry_without_buying(self):
        flow=checked_flow('bounty_s')
        owned=screen('bounty_own_empty')
        back=flow.step(owned,None);flow.completed(back,owned)
        self.assertEqual(back.point,(1115,670))
        listing=screen('bounty_order',remaining=3,s_filter='on',ss_filter='off')
        self.assertEqual(flow.step(listing,None).action,'retry')

    def test_failed_quick_join_is_retried_by_the_planner(self):
        flow=checked_flow('bounty','D')
        decision=flow.step(screen('bounty_daily',remaining=4,grade='D',join_failed=True),None)
        self.assertEqual(decision.action,'retry')
        self.assertIsNone(decision.point)

    def test_quick_join_is_not_resent_while_waiting_for_the_room(self):
        now=[0]
        flow=checked_flow('bounty','D',clock=lambda:now[0])
        flow.free_checked=True;flow.confirmed_grade='D';flow.try_quick_join=True
        listing=screen('bounty_daily',remaining=4,grade='D')
        first=flow.step(listing,None);flow.completed(first,listing)
        now[0]=1
        self.assertEqual(flow.step(listing,None).action,'wait')
        now[0]=6
        self.assertEqual(flow.step(listing,None).action,'retry')

    def test_empty_created_room_waits_then_leaves_without_inviting_friends(self):
        now=[0]
        flow=checked_flow('bounty','D',clock=lambda:now[0])
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

    def test_owned_room_invites_once_per_list_and_waits_for_actual_join(self):
        now=[0]
        flow=checked_flow('bounty','A',clock=lambda:now[0])
        listing=screen('bounty_daily',remaining=3,grade='A')
        flow.free_checked=True;flow.confirmed_grade='A'
        create=flow.step(listing,None);flow.completed(create,listing)
        room=screen('bounty_room',grade='A',minimum_players=True,invite_open=(640,383))
        opened=flow.step(room,None);flow.completed(opened,room)
        self.assertIn('打开',opened.status)
        friends=screen('bounty_recruit',recruit_tab='friend',recruit_candidates=[('好友甲',(1210,135))])
        invite=flow.step(friends,None);flow.completed(invite,friends)
        self.assertEqual(invite.point,(1210,135))
        change=flow.step(friends,None);flow.completed(change,friends)
        self.assertEqual(change.point,(1040,60))
        self.assertEqual(flow.step(friends,None).action,'wait')
        family=screen('bounty_recruit',recruit_tab='family',recruit_candidates=[('好友甲',(1210,135)),('族员乙',(1210,235))])
        invite=flow.step(family,None);flow.completed(invite,family)
        self.assertEqual(invite.point,(1210,235))
        close=flow.step(family,None);flow.completed(close,family)
        self.assertIn('等待队友加入',close.status)
        self.assertEqual(flow.step(room,None).action,'wait')
        now[0]=21
        self.assertEqual(flow.step(room,None).point,(65,50))

    def test_joined_room_does_not_send_outgoing_invitations(self):
        flow=checked_flow('bounty','A')
        flow.room_trusted=True
        room=screen('bounty_room',grade='A',minimum_players=True,invite_open=(640,383))
        self.assertEqual(flow.step(room,None).action,'wait')
        panel=screen('bounty_recruit',recruit_tab='friend',recruit_candidates=[('好友甲',(1210,135))])
        self.assertEqual(flow.step(panel,None).action,'wait')

    def test_recruit_open_timeout_does_not_repeat_the_tap(self):
        now=[0]
        flow=checked_flow('bounty','A',clock=lambda:now[0])
        flow.room_owned=True;flow.room_trusted=True
        room=screen('bounty_room',grade='A',minimum_players=True,invite_open=(640,383))
        first=flow.step(room,None);flow.completed(first,room)
        self.assertEqual(flow.step(room,None).action,'wait')
        now[0]=4
        self.assertEqual(flow.step(room,None).action,'wait')
        self.assertEqual(flow.recruit_stage,'done')

    def test_grade_popup_confirmation_allows_a_missing_list_badge(self):
        flow=checked_flow('bounty','B');flow.free_checked=True
        listing=screen('bounty_daily',remaining=3,grade=None)
        self.assertIn('设置',flow.step(listing,None).status)
        popup=screen('bounty_grade',grade='B',confirm=(640,490),grade_choices={'B':(640,365)})
        decision=flow.step(popup,None);flow.completed(decision,popup)
        self.assertIn('创建B级',flow.step(listing,None).status)

    def test_free_request_requires_observed_quota_increase_from_initial_three(self):
        flow=checked_flow()
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
