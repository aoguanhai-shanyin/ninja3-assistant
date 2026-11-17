"""Multi-card execution regressions; simulated observations, no game access."""
from dataclasses import replace
from datetime import datetime
import unittest
from unittest.mock import patch

from ninja3_assistant.activities.intelligence.battle_vision import (
    BOARD_CENTERS, BattleObservation, SHOP_POINTS,
)
from ninja3_assistant.activities.intelligence.flow import ActivityFlow, ActivityScreen, CHINA
from ninja3_assistant.activities.intelligence.policy import Card


WALL_TIME = datetime(2026,11,16,19,tzinfo=CHINA)
CHOOSE = 'ninja3_assistant.activities.intelligence.flow.choose_purchase'
ANCHOR = Card('c020',3,2,'wind','character',
              '若上场3种[鬼斩之铠]，直接胜利','鬼斩·紫原')
BUFF = Card('c024',1,2,'thunder','character',
            '同列其他[雷系]卡攻击+3，持续2回合','月晖·小夜')
TARGET = Card('c007',1,1,'thunder','character',
              '忍村5虎。上场2虎后自身攻击+2','雷兽·阿力')
CARDS = (ANCHOR,BUFF,TARGET)


def observation(**changes):
    values = dict(board=(None,)*9,shop=CARDS+(None,None),coins=5,
                  round_number=1,place_points=(None,)*9,ready_point=(1118,654))
    values.update(changes)
    return BattleObservation('placement',**values)


def step(flow,battle,now):
    return flow.step(ActivityScreen('battle',battle=battle),now,WALL_TIME)


def planner(board,shop,coins,round_number,*,explanation):
    # Reproduce the observed 18:14 regression: re-evaluating after the buff
    # forgets its same-column benefit and redirects the third card to cell 1.
    if board[0] is None:
        moves=((0,0),(1,3),(2,6))
    elif board[3] is None:
        moves=((1,3),(2,6))
    else:
        moves=((2,1),)
    explanation.update(budget=coins,spent=sum(CARDS[i].cost for i,_ in moves),
        remaining_coins=coins-sum(CARDS[i].cost for i,_ in moves),
        estimated_gain=37,synergy_gain=12,future_income=0,income_per_turn=0,
        granted_refreshes=0,special_victory=False,
        plan=[{'shop':i,'cell':cell,'card':CARDS[i].id,'name':CARDS[i].name,
               'cost':CARDS[i].cost,'effect':CARDS[i].effect} for i,cell in moves])
    return moves[0]


class ComboExecutionTest(unittest.TestCase):
    def select_first(self):
        flow=ActivityFlow('continuous')
        current=observation()
        with patch(CHOOSE,side_effect=planner):
            decision=step(flow,current,0)
        self.assertEqual((decision.action,decision.point),('select_card',SHOP_POINTS[0]))
        flow.completed(decision,0)
        return flow,current

    def place(self,flow,current,shop,cell,now):
        points=[None]*9
        points[cell]=BOARD_CENTERS[cell]
        selected=replace(current,selected_shop=shop,place_points=tuple(points))
        decision=step(flow,selected,now)
        self.assertEqual((decision.action,decision.point),('place',BOARD_CENTERS[cell]))
        flow.completed(decision,now)
        board=list(current.board); board[cell]=current.shop[shop]
        cards=list(current.shop); cards[shop]=None
        return replace(current,board=tuple(board),shop=tuple(cards),
                       coins=current.coins-current.shop[shop].cost)

    def after_first(self):
        flow,current=self.select_first()
        current=self.place(flow,current,0,0,1)
        return flow,current

    def test_confirmed_multi_purchase_keeps_the_original_same_column_combination(self):
        flow,current=self.after_first()
        with patch(CHOOSE,side_effect=planner):
            second=step(flow,current,2)
            self.assertEqual((second.action,second.point),('select_card',SHOP_POINTS[1]))
            flow.completed(second,2)
            current=self.place(flow,current,1,3,3)
            third=step(flow,current,4)
            self.assertEqual((third.action,third.point),('select_card',SHOP_POINTS[2]))
            flow.completed(third,4)
            current=self.place(flow,current,2,6,5)
        self.assertEqual([c.id if c else None for c in current.board],
                         ['c020',None,None,'c024',None,None,'c007',None,None])

    def test_continuation_logs_remaining_budget_without_reusing_the_original_gain(self):
        flow,current=self.after_first()
        decision=step(flow,current,2)
        selection=decision.selection
        self.assertTrue(selection['continued_combo'])
        self.assertEqual((selection['budget'],selection['spent'],selection['remaining_coins']),
                         (3,3,0))
        self.assertEqual([(item['shop'],item['cell']) for item in selection['plan']],
                         [(1,3),(2,6)])
        self.assertNotIn('estimated_gain',selection)
        self.assertEqual(selection['original_selection']['estimated_gain'],37)
        self.assertIn('原组合预计增益37',decision.status)

    def assert_no_old_tail(self,change):
        flow,current=self.after_first()
        changed=change(current)
        with patch(CHOOSE,return_value=None):
            decision=step(flow,changed,2)
        self.assertNotIn(decision.action,('select_card','place'))

    def test_changed_coin_balance_discards_the_old_tail(self):
        self.assert_no_old_tail(lambda b:replace(b,coins=b.coins+1))

    def test_temporarily_hidden_ready_button_does_not_break_the_pending_combo(self):
        flow,current=self.after_first()
        with patch(CHOOSE,side_effect=planner):
            second=step(flow,current,2)
            flow.completed(second,2)
            # Vision reports prepared whenever the ready button is absent,
            # including an intermediate selection animation.
            waiting=step(flow,BattleObservation('prepared',round_number=1),3)
            self.assertEqual(waiting.action,'wait')
            self.assertIsNotNone(flow.pending)
            current=self.place(flow,current,1,3,4)
            third=step(flow,current,5)
            flow.completed(third,5)
            self.place(flow,current,2,6,6)

    def test_refreshed_shop_discards_the_old_tail(self):
        different=Card('replacement',2,2)
        self.assert_no_old_tail(lambda b:replace(b,shop=(None,different,TARGET,None,None)))

    def test_changed_later_combo_card_discards_the_whole_tail(self):
        different=Card('replacement',2,1)
        self.assert_no_old_tail(lambda b:replace(b,shop=(None,BUFF,different,None,None)))

    def test_occupied_destination_discards_the_old_tail(self):
        self.assert_no_old_tail(lambda b:replace(b,board=b.board[:3]+(TARGET,)+b.board[4:]))

    def test_unknown_destination_is_not_treated_as_empty_for_a_cached_plan(self):
        self.assert_no_old_tail(lambda b:replace(b,unknown_board=(3,)))

    def test_unconfirmed_previous_landing_does_not_advance_to_the_next_card(self):
        flow,current=self.select_first()
        self.place(flow,current,0,0,1)
        with patch(CHOOSE,return_value=None):
            decision=step(flow,current,2)
        self.assertNotIn(decision.action,('select_card','place'))

    def test_unconfirmed_card_identity_does_not_advance_the_old_tail(self):
        self.assert_no_old_tail(lambda b:replace(b,board=(None,)+b.board[1:],unknown_board=(0,)))

    def test_new_round_does_not_reuse_the_previous_round_combination(self):
        self.assert_no_old_tail(lambda b:replace(b,round_number=2))

    def test_pause_reset_discards_the_combo_without_reusing_old_inputs(self):
        flow,current=self.after_first()
        flow.reset_inputs()
        with patch(CHOOSE,return_value=None):
            decision=step(flow,current,2)
        self.assertNotIn(decision.action,('select_card','place'))

    def test_combat_and_result_discard_the_previous_combination(self):
        for transition in (BattleObservation('combat',round_number=1),
                           BattleObservation('result',result_point=(640,541))):
            with self.subTest(transition=transition.kind):
                flow,current=self.after_first()
                step(flow,transition,2)
                with patch(CHOOSE,return_value=None):
                    decision=step(flow,current,4)
                self.assertNotIn(decision.action,('select_card','place'))


if __name__ == '__main__':
    unittest.main()
