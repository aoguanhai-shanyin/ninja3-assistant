from dataclasses import replace
from datetime import datetime
import unittest
from unittest.mock import patch

from ninja3_assistant.activities.intelligence.battle_vision import BattleObservation, SHOP_POINTS
from ninja3_assistant.activities.intelligence.flow import CHINA, ActivityFlow, ActivityScreen, daily_key
from ninja3_assistant.activities.intelligence.policy import Card
from ninja3_assistant.activities.intelligence.tasks import TaskObservation


WALL_TIME = datetime(2026, 11, 13, 19, tzinfo=CHINA)
CARD = Card('chosen', 3, 2)


def placement(**changes):
    values = dict(board=(None,)*9, shop=(CARD,)+(None,)*4, coins=5,
                  round_number=1, place_points=(None,)*9, ready_point=(1118, 654))
    values.update(changes)
    return BattleObservation('placement', **values)


def step(flow, battle, now):
    return step_after_scan(flow, ActivityScreen('battle', battle=battle), now, WALL_TIME)


def step_after_scan(flow, screen, now, wall_time):
    decision = flow.step(screen, now, wall_time)
    while decision.action in ('reset_tasks_scroll', 'scroll_tasks'):
        flow.completed(decision, now)
        now += 1.01
        decision = flow.step(screen, now, wall_time)
    return decision


class ActivityBattleFlowTest(unittest.TestCase):
    def test_extra_refreshes_are_used_only_after_the_counter_decreases(self):
        flow = ActivityFlow()
        current = placement(shop=(None,)*5,refresh_point=(1115,535),refreshes=3)
        for now,count in ((0,3),(1,2),(2,1)):
            decision = step(flow,replace(current,refreshes=count),now)
            self.assertEqual(decision.action,'refresh')
            flow.completed(decision,now)
            self.assertEqual(step(flow,replace(current,refreshes=count),now+.5).action,'prepare')
        self.assertEqual(step(flow,replace(current,refreshes=0,refresh_point=None),3).action,'prepare')

    def test_special_victory_early_finish_does_not_keep_refresh_state_in_the_next_match(self):
        flow = ActivityFlow()
        current = placement(shop=(None,)*5,refresh_point=(1115,535),refreshes=1)
        decision = step(flow,current,0)
        flow.completed(decision,0)
        step(flow,BattleObservation('result',result_point=(640,541)),2)
        self.assertEqual(step(flow,current,4).action,'refresh')

    def test_selected_plan_explains_budget_and_preserves_unknown_occupancy(self):
        flow = ActivityFlow()
        result = step(flow, placement(unknown_board=(0,)),0)
        self.assertIn('预算5',result.status)
        self.assertIn('未识别卡',result.status)
        self.assertEqual(result.selection['unknown_board'],[0])
        self.assertEqual(result.selection['spent'],2)
        self.assertNotEqual(result.selection['plan'][0]['cell'],0)

    def test_unreadable_shop_retries_briefly_then_refreshes_once(self):
        flow = ActivityFlow()
        unknown = placement(shop=(None,)*5,unknown_shop=(0,1),refresh_point=(1115,535))
        self.assertEqual(step(flow,unknown,0).action,'wait')
        self.assertEqual(step(flow,unknown,.9).action,'wait')
        refresh = step(flow,unknown,1)
        self.assertEqual(refresh.action,'refresh')
        flow.completed(refresh,1)
        self.assertEqual(step(flow,unknown,2).action,'prepare')

    def selected(self):
        flow = ActivityFlow()
        decision = step(flow, placement(), 0)
        self.assertEqual((decision.action, decision.point), ('select_card', SHOP_POINTS[0]))
        flow.completed(decision, 0)
        return flow

    def test_selection_requires_both_the_selected_shop_and_target_plus(self):
        for selected_shop, points in ((None, (None,)*9), (0, (None,)*9),
                                     (1, ((140, 232),)+(None,)*8)):
            with self.subTest(selected_shop=selected_shop, points=points):
                flow = self.selected()
                result = step(flow, placement(selected_shop=selected_shop, place_points=points), 1)
                self.assertEqual(result.action, 'wait')
                self.assertIsNone(result.point)
        flow = self.selected()
        result = step(flow, placement(selected_shop=0, place_points=((140, 232),)+(None,)*8), 1)
        self.assertEqual((result.action, result.point), ('place', (140, 232)))

    def test_selection_does_not_place_a_different_card_in_the_same_shop_slot(self):
        flow = self.selected()
        changed = placement(shop=(Card('different', 1, 1),)+(None,)*4,
                            selected_shop=0, place_points=((140, 232),)+(None,)*8)
        result = step(flow, changed, 1)
        self.assertNotEqual(result.action, 'place')
        self.assertIsNone(result.point)

    def test_selection_does_not_place_using_changed_coin_evidence(self):
        flow = self.selected()
        changed = placement(coins=1, selected_shop=0, place_points=((140, 232),)+(None,)*8)
        result = step(flow, changed, 1)
        self.assertNotEqual(result.action, 'place')
        self.assertIsNone(result.point)

    def test_successful_place_continues_without_waiting_for_receipts(self):
        flow = self.selected()
        drop = step(flow, placement(selected_shop=0, place_points=((140, 232),)+(None,)*8), 1)
        flow.completed(drop, 1)
        self.assertIsNone(flow.pending)
        current = placement(board=(CARD,)+(None,)*8, shop=(None,)*5, coins=3)
        with patch('ninja3_assistant.activities.intelligence.flow.choose_purchase', return_value=None) as choose:
            result = step(flow, current, 1.5)
        choose.assert_called_once()
        self.assertEqual(result.action, 'prepare')

    def test_selected_highlight_can_obscure_card_identity(self):
        flow = self.selected()
        observation = placement(shop=(None,)*5, unknown_shop=(0,), selected_shop=0,
                                place_points=((140, 232),)+(None,)*8)
        self.assertEqual(step(flow, observation, 1).action, 'place')
        self.assertNotEqual(step(self.selected(), replace(observation, unknown_shop=()), 1).action, 'place')

    def test_missing_selection_feedback_retries_from_current_state(self):
        flow = self.selected()
        result = step(flow, placement(), 5.01)
        self.assertEqual(result.action, 'select_card')
        self.assertIsNone(flow.pending)

    def test_combat_and_new_round_clear_pending_inputs(self):
        for observation in (BattleObservation('combat', round_number=1), placement(round_number=2)):
            flow = self.selected()
            with patch('ninja3_assistant.activities.intelligence.flow.choose_purchase', return_value=None):
                result = step(flow, observation, 2)
            self.assertIsNone(flow.pending)
            self.assertNotEqual(result.action, 'place')

    def test_unknown_board_does_not_become_a_confirmed_empty_slot(self):
        flow = ActivityFlow()
        result = step(flow, placement(unknown_board=(0,)), 0)
        self.assertEqual(result.action, 'select_card')
        self.assertNotEqual(flow.purchase[1], 0)
        full = placement(unknown_board=tuple(range(9)))
        self.assertEqual(step(ActivityFlow(), full, 0).action, 'prepare')

    def test_purchase_can_be_confirmed_when_only_placed_portrait_is_obscured(self):
        flow = self.selected()
        drop = step(flow, placement(selected_shop=0, place_points=((140, 232),)+(None,)*8), 1)
        flow.completed(drop, 1)
        confirmed = placement(unknown_board=(0,), shop=(None,)*5, coins=3)
        self.assertEqual(step(flow, confirmed, 2).action, 'prepare')
        self.assertIsNone(flow.pending)

    def test_empty_shop_animation_does_not_block_confirmed_card_and_payment(self):
        flow = self.selected()
        drop = step(flow, placement(selected_shop=0, place_points=((140, 232),)+(None,)*8), 1)
        flow.completed(drop, 1)
        confirmed = placement(board=(CARD,)+(None,)*8, shop=(None,)*5, coins=3, unknown_shop=(0,))
        self.assertEqual(step(flow, confirmed, 2).action, 'wait')
        self.assertEqual(step(flow, confirmed, 3).action, 'prepare')
        self.assertIsNone(flow.pending)

    def test_result_requires_completed_confirmation_and_return_home_to_count(self):
        flow = ActivityFlow()
        result = BattleObservation('result', result_point=(640, 540))
        decision = step(flow, result, 0)
        self.assertEqual(decision.action, 'result_confirm')
        self.assertEqual(flow.completed_matches, 0)
        step_after_scan(flow, ActivityScreen('event_home'), 4, WALL_TIME)
        self.assertEqual(flow.completed_matches, 0)
        flow.completed(decision, 2)
        step(flow, result, 4)
        self.assertEqual(flow.completed_matches, 0)
        home = step_after_scan(flow, ActivityScreen('event_home'), 4, WALL_TIME)
        self.assertEqual(flow.completed_matches, 1)
        self.assertEqual(home.action, 'tasks')
        step_after_scan(flow, ActivityScreen('event_home'), 6, WALL_TIME)
        self.assertEqual(flow.completed_matches, 1)

    def test_daily_completion_stops_daily_mode_but_continuous_can_match(self):
        for mode, task_action, home_action in (('daily', 'done', 'done'),
                                               ('continuous', 'close_tasks', 'match')):
            with self.subTest(mode=mode):
                flow = ActivityFlow(mode)
                result = step_after_scan(flow, ActivityScreen('tasks', daily=('claimed', 'claimed', 'claimed'), task_ids=('games', 'damage', 'characters')), 0, WALL_TIME)
                self.assertEqual(result.action, task_action)
                self.assertEqual(step_after_scan(flow, ActivityScreen('event_home'), 4, WALL_TIME).action, home_action)

    def test_missing_daily_rows_need_their_own_confirmed_claim_evidence(self):
        task = TaskObservation('tasks', present=())
        missing = ActivityScreen('tasks', daily=('unknown', 'unknown'), task=task)
        flow = ActivityFlow()
        self.assertEqual(step_after_scan(flow, missing, 0, WALL_TIME).action, 'wait')
        self.assertFalse(flow.daily_complete)
        flow.confirmed_claims.add('games')
        self.assertEqual(step_after_scan(flow, missing, 2, WALL_TIME).action, 'wait')
        self.assertFalse(flow.daily_complete)

    def test_character_daily_claim_then_next_visible_damage_daily(self):
        flow = ActivityFlow()
        task = TaskObservation('tasks', present=('games', 'characters'))
        current = ActivityScreen('tasks', daily=('incomplete', 'unknown', 'claimable'),
                                 claim_points=(None, None, (980, 205)), task=task,
                                 task_ids=('games', 'damage', 'characters'))
        claim = step_after_scan(flow, current, 0, WALL_TIME)
        self.assertEqual(claim.action, 'claim')
        flow.completed(claim, 0)
        self.assertEqual(flow.claiming, 'characters')
        popup = step_after_scan(flow, ActivityScreen('reward', point=(640, 644)), 2, WALL_TIME)
        flow.completed(popup, 2)
        next_task = TaskObservation('tasks', present=('games', 'damage'))
        next_screen = ActivityScreen('tasks', daily=('incomplete', 'incomplete'),
                                     claim_points=(None, None), task=next_task)
        self.assertEqual(step_after_scan(flow, next_screen, 4, WALL_TIME).action, 'close_tasks')
        self.assertFalse(flow.daily_complete)
        self.assertEqual(flow.task_states['characters'], 'claimed')

    def test_known_claim_can_be_received_while_another_daily_row_is_unknown(self):
        for mode in ('daily', 'continuous'):
            with self.subTest(mode=mode):
                flow = ActivityFlow(mode)
                task = TaskObservation('tasks', daily_states={'games': 'unknown', 'damage': 'claimable'},
                                       claim_points={'games': None, 'damage': (970, 302)},
                                       present=('damage',))
                screen = ActivityScreen('tasks', daily=('unknown', 'claimable'),
                                        claim_points=(None, (970, 302)), task=task)
                claim = step_after_scan(flow, screen, 0, WALL_TIME)
                self.assertEqual((claim.action, claim.point), ('claim', (970, 302)))
                flow.completed(claim, 0)
                self.assertEqual(flow.claiming, 'damage')
                self.assertFalse(flow.daily_complete)

    def test_continuous_can_leave_confirmed_tasks_with_unknown_progress_without_claiming_completion(self):
        screen = ActivityScreen('tasks', daily=('unknown', 'incomplete'),
                                claim_points=(None, None),
                                task=TaskObservation('tasks', present=('damage',)))
        flow = ActivityFlow('continuous')
        result = step_after_scan(flow, screen, 0, WALL_TIME)
        self.assertEqual((result.action, result.point), ('close_tasks', (1120, 111)))
        self.assertEqual(flow.checked_day, daily_key(WALL_TIME))
        self.assertFalse(flow.daily_complete)
        self.assertEqual(step_after_scan(flow, ActivityScreen('event_home'), 4, WALL_TIME).action, 'match')

        daily = ActivityFlow('daily')
        result = step_after_scan(daily, screen, 0, WALL_TIME)
        self.assertEqual(result.action, 'close_tasks')
        self.assertEqual(result.point, (1120, 111))
        self.assertEqual(daily.checked_day, daily_key(WALL_TIME))
        self.assertFalse(daily.daily_complete)

    def test_claimable_without_a_confirmed_point_is_not_clicked_or_skipped(self):
        screen = ActivityScreen('tasks', daily=('unknown', 'claimable'),
                                claim_points=(None, None),
                                task=TaskObservation('tasks', present=('damage',)))
        for mode in ('daily', 'continuous'):
            with self.subTest(mode=mode):
                flow = ActivityFlow(mode)
                result = step_after_scan(flow, screen, 0, WALL_TIME)
                self.assertEqual(result.action, 'wait')
                self.assertIsNone(result.point)
                self.assertIsNone(flow.checked_day)
                self.assertFalse(flow.daily_complete)

    def test_unknown_progress_never_retains_a_previous_completion_flag(self):
        screen = ActivityScreen('tasks', daily=('unknown', 'unknown'),
                                task=TaskObservation('tasks', present=()))
        for mode, expected_action in (('daily', 'wait'), ('continuous', 'close_tasks')):
            with self.subTest(mode=mode):
                flow = ActivityFlow(mode)
                flow.claim_day = flow.checked_day = daily_key(WALL_TIME)
                flow.daily_complete = True
                self.assertEqual(step_after_scan(flow, screen, 0, WALL_TIME).action, expected_action)
                self.assertFalse(flow.daily_complete)

    def test_continuous_still_waits_when_task_page_or_shape_is_unconfirmed(self):
        for screen in (ActivityScreen(), ActivityScreen('tasks')):
            with self.subTest(screen=screen):
                flow = ActivityFlow('continuous')
                result = step_after_scan(flow, screen, 0, WALL_TIME)
                self.assertEqual(result.action, 'wait')
                self.assertIsNone(result.point)
                self.assertIsNone(flow.checked_day)

    def test_claim_popup_confirmation_then_row_disappearance_completes_daily(self):
        flow = ActivityFlow()
        for index, task_id in enumerate(('games', 'damage', 'characters')):
            states = ['claimed' if key in flow.confirmed_claims else 'claimable'
                      for key in ('games', 'damage', 'characters')]
            points = [None if key in flow.confirmed_claims else (970, 200+102*i)
                      for i, key in enumerate(('games', 'damage', 'characters'))]
            tasks = ActivityScreen('tasks', daily=tuple(states), claim_points=tuple(points), task_ids=('games', 'damage', 'characters'))
            claim = step_after_scan(flow, tasks, index*4, WALL_TIME)
            self.assertEqual(claim.action, 'claim')
            flow.completed(claim, index*4)
            self.assertNotIn(task_id, flow.confirmed_claims)
            popup = step_after_scan(flow, ActivityScreen('reward', point=(640, 644)), index*4+2, WALL_TIME)
            self.assertEqual(popup.action, 'confirm_reward')
            self.assertNotIn(task_id, flow.confirmed_claims)
            flow.completed(popup, index*4+2)
            self.assertIn(task_id, flow.confirmed_claims)
        missing = ActivityScreen('tasks', daily=('unknown', 'unknown'),
                                 task=TaskObservation('tasks', present=()))
        self.assertEqual(step_after_scan(flow, missing, 14, WALL_TIME).action, 'done')

    def test_day_reset_does_not_reuse_yesterdays_disappearing_claims(self):
        flow = ActivityFlow()
        flow.claim_day = daily_key(WALL_TIME)
        flow.confirmed_claims.update(('games', 'damage'))
        missing = ActivityScreen('tasks', daily=('unknown', 'unknown'),
                                 task=TaskObservation('tasks', present=()))
        tomorrow = WALL_TIME.replace(day=14)
        self.assertEqual(step_after_scan(flow, missing, 0, tomorrow).action, 'wait')
        self.assertEqual(flow.confirmed_claims, set())


if __name__ == '__main__':
    unittest.main()
