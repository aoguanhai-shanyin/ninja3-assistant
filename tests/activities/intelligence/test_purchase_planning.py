import unittest

from ninja3_assistant.activities.intelligence.policy import Card, choose_purchase


class BudgetPlanningTest(unittest.TestCase):
    def test_income_card_pays_back_early_but_not_after_the_final_round(self):
        cards = [Card('plain',2,1),Card('income',1,1,effect='每回合结束，金币+1')]
        first,early = self.plan(cards,coins=1)
        self.assertEqual(first[0],1)
        self.assertEqual(early['future_income'],4)
        self.assertGreater(early['economic_gain'],0)
        last,late = self.plan(cards,coins=1,round_number=5)
        self.assertEqual(last[0],0)
        self.assertEqual(late['future_income'],0)

    def test_diverse_attributes_increase_existing_economic_card_income(self):
        board = [Card('income',2,3,'fire',effect='每回合结束每有1种属性卡，金币+1')]+[None]*8
        cards = [Card('same',1,1,'fire'),Card('new',1,1,'water')]
        first,details = self.plan(cards,board=board,coins=1,round_number=3)
        self.assertEqual(first[0],1)
        self.assertEqual(details['income_per_turn'],2)
        self.assertEqual(details['future_income'],4)

    def test_income_cannot_pay_for_purchases_in_the_current_round(self):
        _,details = self.plan([Card('income',1,1,effect='每回合结束，金币+1'),
                              Card('unaffordable',20,2)],coins=1)
        self.assertEqual([p['card'] for p in details['plan']],['income'])
        self.assertEqual(details['spent'],1)

    def test_full_board_can_save_economic_income_for_final_cat_damage(self):
        board = [Card('cat',1,2,effect='第5回合造成剩余金币数×4的伤害'),
                 Card('income',1,1,effect='每回合结束金币+1')]+[Card('plain',1,1)]*7
        choice,details = self.plan([],board=board,coins=2,round_number=3)
        self.assertIsNone(choice)
        self.assertEqual(details['future_income'],2)
        self.assertEqual(details['income_per_turn'],1)

    def test_last_space_can_save_current_coins_for_later_purchases(self):
        income = Card('income',1,1,effect='每回合结束金币+1')
        board = [income]+[Card('occupied',3,1)]*7+[None]
        cards = [Card('weak',1,1),Card('reference',3,2)]
        choice,details = self.plan(cards,board=board,coins=1,round_number=4)
        self.assertIsNone(choice)
        self.assertEqual(details['remaining_coins'],1)
        self.assertEqual(details['empty_cells'],1)
        self.assertEqual(details['held_coin_value'],1.5)
        self.assertEqual(details['future_income'],1)

        # Plentiful room does not incur the same cost for this weak purchase.
        choice,roomy = self.plan(cards,board=[income]+[None]*8,coins=1,round_number=4)
        self.assertEqual(choice[0],0)
        self.assertEqual(roomy['held_coin_value_before'],0)

    def test_last_space_is_not_reserved_against_a_strong_immediate_purchase(self):
        board = [Card('income',1,1,effect='每回合结束金币+1')]+[Card('occupied',3,1)]*7+[None]
        choice,details = self.plan([Card('strong',6,1),Card('reference',3,2)],
                                   board=board,coins=1,round_number=4)
        self.assertEqual(choice,(0,8))
        self.assertEqual(details['empty_cells'],0)

    def test_final_round_never_reserves_space_for_future_purchases(self):
        board = [Card('income',1,1,effect='每回合结束金币+1')]+[Card('occupied',3,1)]*7+[None]
        choice,details = self.plan([Card('weak',1,1),Card('reference',3,2)],
                                   board=board,coins=1,round_number=5)
        self.assertEqual(choice,(0,8))
        self.assertEqual(details['held_coin_value_before'],0)
        self.assertEqual(details['future_income'],0)

    def test_saved_coins_are_not_also_invested_when_counting_cat_damage(self):
        board = [Card('cat',1,2,effect='第5回合造成剩余金币数×4的伤害')]+[Card('occupied',3,1)]*7+[None]
        choice,details = self.plan([Card('strong',2,1)],board=board,coins=2,round_number=3)
        self.assertEqual(choice,(0,8))
        self.assertEqual(details['held_coin_value_before'],0)
        self.assertEqual(details['economic_gain'],-4)
        self.assertEqual(details['estimated_gain'],2)

    def test_refresh_card_has_future_value_but_no_last_round_income(self):
        cards = [Card('plain',2,2),Card('ring',2,2,effect='上场回合结束时，刷新+2')]
        first,early = self.plan(cards,coins=2)
        self.assertEqual(first[0],1)
        self.assertEqual(early['granted_refreshes'],2)
        self.assertEqual(self.plan(cards,coins=2,round_number=5)[0][0],0)

    def test_visible_special_victory_outweighs_a_high_damage_card(self):
        cards = [Card('plain',100,5),Card('trigger',3,2,effect='若上场3种[鬼斩之铠]，直接胜利'),
                 Card('chest',3,1,effect='鬼斩之铠'),Card('arm',2,1,effect='鬼斩之铠'),
                 Card('head',2,1,effect='鬼斩之铠')]
        _,details = self.plan(cards,coins=5,round_number=5)
        self.assertTrue(details['special_victory'])
        self.assertEqual(set(details['armor_types']),{'chest','arm','head'})
        self.assertNotIn('plain',[p['card'] for p in details['plan']])

    def test_duplicate_armor_and_a_missing_trigger_do_not_claim_victory(self):
        for board in ([Card('trigger',3,2,effect='若上场3种鬼斩之铠直接胜利')]+
                      [Card('same',2,1,effect='鬼斩之铠')]*3+[None]*5,
                      [Card(i,2,1,effect='鬼斩之铠') for i in ('a','b','c')]+[None]*6):
            _,details = self.plan([],board=board,round_number=5)
            self.assertFalse(details['special_victory'])

    def test_complete_special_victory_preserves_coins_and_existing_formation(self):
        board = ([Card('trigger',3,2,effect='若上场3种鬼斩之铠直接胜利')]+
                 [Card(i,2,1,effect='鬼斩之铠') for i in ('a','b','c')]+[None]*5)
        choice,details = self.plan([Card('plain',100,1)],board=board,coins=5)
        self.assertIsNone(choice)
        self.assertTrue(details['special_victory'])
        self.assertEqual(details['remaining_coins'],5)

    def plan(self, cards, *, board=None, coins=5, round_number=1):
        details = {}
        choice = choose_purchase(board or [None]*9,cards+[None]*(5-len(cards)),coins,
                                 round_number,explanation=details)
        return choice,details

    def test_budget_combination_beats_the_largest_single_purchase(self):
        cards = [Card('large',7,5),Card('first',4,2),Card('second',4,3)]
        choice,details = self.plan(cards,coins=5,round_number=5)
        self.assertEqual(choice[0],1)
        self.assertEqual([item['card'] for item in details['plan']],['first','second'])
        self.assertEqual(details['spent'],5)
        self.assertEqual(details['estimated_gain'],8)

    def test_two_water_purchases_complete_the_existing_row_synergy(self):
        board = [Card('water-core',4,2,'water',effect='若同行有3张水系卡自身攻击×2')]+[None]*8
        cards = [Card('plain',5,4),Card('water-a',1,2,'water'),Card('water-b',1,2,'water')]
        choice,details = self.plan(cards,board=board,coins=4,round_number=5)
        self.assertEqual(choice,(1,1))
        self.assertEqual([(p['shop'],p['cell']) for p in details['plan']],[(1,1),(2,2)])
        self.assertEqual(details['synergy_gain'],4)

    def test_tiger_pair_outvalues_a_single_plain_card(self):
        cards = [Card('plain',5,2),Card('tiger-a',1,1,effect='忍村5虎。上场2虎后自身攻击+2'),
                 Card('tiger-b',1,1,effect='忍村5虎。上场2虎后自身攻击+2')]
        _,details = self.plan(cards,coins=2,round_number=5)
        self.assertEqual([p['card'] for p in details['plan']],['tiger-a','tiger-b'])
        self.assertEqual(details['estimated_gain'],6)

    def test_adjacency_bonus_prefers_a_cell_touching_two_matching_cards(self):
        board = [None,Card('fire-a',1,1,'fire'),None,Card('fire-b',1,1,'fire')]+[None]*5
        aura = Card('mask',1,2,effect='在场时相邻卡攻击+1，[火系]卡额外+1')
        choice,details = self.plan([aura],board=board,coins=2,round_number=5)
        self.assertEqual(choice,(0,0))
        self.assertEqual(details['estimated_gain'],5)

    def test_plan_never_reuses_a_shop_slot_or_spends_beyond_budget(self):
        cards = [Card('same',2,1)]*5
        _,details = self.plan(cards,coins=3)
        self.assertEqual(len(details['plan']),3)
        self.assertEqual(len({p['shop'] for p in details['plan']}),3)
        self.assertEqual(len({p['cell'] for p in details['plan']}),3)
        self.assertEqual(details['remaining_coins'],0)

    def test_order_places_attackers_before_a_column_entry_damage_card(self):
        cards = [Card('burst',1,2,effect='上场回合造成本列卡攻击和×2的伤害'),Card('attacker',4,2)]
        _,details = self.plan(cards,coins=4,round_number=5)
        self.assertEqual([p['card'] for p in details['plan']],['attacker','burst'])
        self.assertEqual(details['plan'][0]['cell']%3,details['plan'][1]['cell']%3)
        self.assertEqual(details['placement_damage'],10)
