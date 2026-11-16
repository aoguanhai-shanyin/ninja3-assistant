from ninja3_assistant.paths import project_root
import json
import unittest

from ninja3_assistant.activities.intelligence.policy import Card, choose_purchase


class ActivityPolicyTest(unittest.TestCase):
    def choose(self, cards, *, board=None, coins=5, round_number=1):
        return choose_purchase(board or [None]*9, cards + [None]*(5-len(cards)), coins, round_number)

    def test_only_known_affordable_shop_cards_and_empty_cells_are_eligible(self):
        known = Card('known', 2, 2)
        board = [Card('occupied', 3, 1)]*8 + [None]
        self.assertEqual(self.choose([None, Card('expensive', 20, 6), known], board=board), (2, 8))
        self.assertIsNone(self.choose([Card('', 20, 1), known], coins=1))
        self.assertIsNone(self.choose([known], board=[known]*9))
        self.assertIsNone(self.choose([None]))

    def test_full_budget_values_total_damage_instead_of_single_card_efficiency(self):
        cards = [Card('efficient', 3, 1), Card('large', 5, 5)]
        self.assertEqual(self.choose(cards, round_number=1), (1, 0))
        self.assertEqual(self.choose(cards, round_number=5), (1, 0))

    def test_static_row_water_synergy_places_third_water_in_matching_row(self):
        board = [Card('water-synergy', 3, 2, 'water', 'character',
                      '若同行有3张[水系]卡，自身攻击×2'),
                 Card('water', 1, 1, 'water'), None] + [None]*6
        self.assertEqual(self.choose([Card('third', 1, 1, 'water')], board=board), (0, 2))

    def test_new_row_buff_uses_matching_row_and_excludes_itself(self):
        board = [Card('one', 1, 1), Card('two', 1, 1), None] + [None]*6
        buff = Card('buff', 1, 2, 'water', 'character', '同行其他卡攻击+2，持续2回合')
        self.assertEqual(self.choose([Card('plain', 3, 2), buff], board=board, coins=2,
                                     round_number=4), (1, 2))

    def test_new_element_column_buff_chooses_column_with_matching_cards(self):
        board = [Card('fire', 1, 1, 'fire'), None, None,
                 Card('fire-2', 1, 1, 'fire'), None, None, None, None, None]
        buff = Card('buff', 1, 2, 'fire', 'character', '同列其他[火系]卡攻击+3，持续2回合')
        self.assertEqual(self.choose([buff], board=board), (0, 6))

    def test_existing_water_aura_values_new_card_in_its_row(self):
        board = [Card('fox', 1, 4, 'water', 'beast', '在场时同行其他[水系]卡攻击+4'),
                 None, None] + [None]*6
        self.assertEqual(self.choose([Card('water', 1, 1, 'water')], board=board), (0, 1))

    def test_existing_weapon_synergy_can_make_a_weapon_more_valuable(self):
        board = [Card('knight', 2, 2, 'fire', 'character', '每上场1张武器卡牌，攻击+2')] + [None]*8
        self.assertEqual(self.choose([Card('plain', 2, 1), Card('weapon', 1, 1, profession='weapon')],
                                     board=board), (1, 1))

    def test_known_direct_damage_is_included_on_last_round(self):
        direct = Card('weapon', 1, 2, 'water', 'weapon', '上场回合造成7点伤害')
        self.assertEqual(self.choose([Card('plain', 5, 2), direct], round_number=5), (1, 0))

    def test_catalogue_effects_choose_actual_matching_column_and_direct_damage(self):
        raw = json.loads((project_root()/'assets/intelligence-cards.json').read_text(encoding='utf-8'))
        cards = {item['id']: Card(**{field: item[field] for field in Card.__dataclass_fields__})
                 for item in raw['cards']}
        board = [cards['c001'], None, None, cards['c009'], None, None, None, None, None]
        self.assertEqual(self.choose([cards['c021']], board=board), (0, 6))
        self.assertEqual(self.choose([cards['c002'], cards['c031']], round_number=5), (1, 0))

    def test_late_round_damage_can_make_a_low_base_attack_card_the_best_purchase(self):
        late = Card('late', 1, 3, 'water', 'character', '第4/5回合，造成自身攻击×3的伤害')
        self.assertEqual(self.choose([Card('plain', 3, 3), late], round_number=5), (1, 0))

    def test_actual_black_dragon_fire_count_can_outdamage_an_immediate_weapon(self):
        raw = json.loads((project_root()/'assets/intelligence-cards.json').read_text(encoding='utf-8'))
        cards = {item['id']: Card(**{field: item[field] for field in Card.__dataclass_fields__})
                 for item in raw['cards']}
        board = [cards['c001'], cards['c009'], cards['c021']] + [None]*6
        self.assertEqual(self.choose([cards['c031'], cards['c047']], board=board,
                                     coins=4, round_number=5), (1, 3))

    def test_actual_thunder_dragon_uses_a_full_row_only_after_five_thunder_cards(self):
        raw = json.loads((project_root()/'assets/intelligence-cards.json').read_text(encoding='utf-8'))
        cards = {item['id']: Card(**{field: item[field] for field in Card.__dataclass_fields__})
                 for item in raw['cards']}
        board = [cards['c007'], cards['c008'], None,
                 cards['c024'], cards['c027'], None, None, None, None]
        self.assertEqual(self.choose([cards['c031'], cards['c050']], board=board,
                                     coins=4, round_number=5), (1, 2))
        board[4] = None
        self.assertEqual(self.choose([cards['c031'], cards['c050']], board=board,
                                     coins=4, round_number=5), (0, 2))

    def test_actual_cat_boss_uses_remaining_coins_on_the_last_round(self):
        raw = json.loads((project_root()/'assets/intelligence-cards.json').read_text(encoding='utf-8'))
        cards = {item['id']: Card(**{field: item[field] for field in Card.__dataclass_fields__})
                 for item in raw['cards']}
        self.assertEqual(self.choose([cards['c031'], cards['c018']], coins=8,
                                     round_number=5), (1, 0))
        self.assertEqual(self.choose([cards['c031'], cards['c018']], coins=3,
                                     round_number=5), (0, 0))

    def test_last_round_can_save_coins_when_purchasing_would_reduce_total_damage(self):
        cat = Card('cat', 1, 2, 'wind', 'character', '第5回合，造成剩余金币数×4的伤害')
        board = [cat]+[None]*8
        self.assertIsNone(self.choose([Card('weak', 2, 1)], board=board,
                                     coins=1, round_number=5))
        self.assertEqual(self.choose([Card('strong', 5, 1)], board=board,
                                     coins=1, round_number=5), (0, 1))

    def test_unknown_effect_is_not_interpreted_as_a_bonus(self):
        self.assertEqual(self.choose([Card('unknown-effect', 1, 1, effect='攻击变成999'),
                                      Card('known', 2, 1)]), (1, 0))

    def test_decision_does_not_mutate_inputs_and_ties_are_stable(self):
        board, shop = [None]*9, [Card('same', 1, 1)]*5
        self.assertEqual(choose_purchase(board, shop, 5, 1), (0, 0))
        self.assertEqual(board, [None]*9)
        self.assertEqual(shop, [Card('same', 1, 1)]*5)

    def test_invalid_observation_is_rejected_before_a_decision(self):
        for board, shop, coins, round_number in (([None]*8, [None]*5, 5, 1),
                                                 ([None]*9, [None]*4, 5, 1),
                                                 ([None]*9, [None]*5, 5, 6),
                                                 ([Card('', 1, 1)]+[None]*8, [None]*5, 5, 1),
                                                 (['unknown']+[None]*8, [None]*5, 5, 1)):
            with self.subTest(board=board, shop=shop, round=round_number), self.assertRaises(ValueError):
                choose_purchase(board, shop, coins, round_number)


if __name__ == '__main__':
    unittest.main()
