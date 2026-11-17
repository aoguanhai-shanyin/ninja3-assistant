"""Replay the saved 18:14 placement sequence; local evidence is not published."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from ninja3_assistant.activities.intelligence.battle_vision import (
    BOARD_CENTERS, BattleObservation,
)
from ninja3_assistant.activities.intelligence.cards import load_catalog
from ninja3_assistant.activities.intelligence.flow import ActivityFlow
from ninja3_assistant.paths import project_root

from tests.activities.intelligence.test_combo_execution import CHOOSE, step


ROOT=project_root()
LOG=ROOT/'data/runtime/intelligence-simulation/runs/20261116-181439-683900/events.jsonl'
CATALOG=ROOT/'assets/intelligence-cards.json'


@unittest.skipUnless(LOG.is_file() and CATALOG.is_file(),'requires local placement evidence and card catalog')
class ComboReplayTest(unittest.TestCase):
    def test_recorded_small_night_combo_keeps_its_thunder_card_in_the_same_column(self):
        records=[json.loads(line) for line in LOG.read_text(encoding='utf-8').splitlines()]
        cards={card.id:card for card in load_catalog(CATALOG)}

        def observed(line):
            b=records[line-1]['battle']
            board=tuple(cards[c] if c else None for c in b['board'])
            points=tuple(BOARD_CENTERS[i] if c is None and i not in b['unknown_board'] else None
                         for i,c in enumerate(board)) if b['selected_shop'] is not None else (None,)*9
            return BattleObservation('placement',board=board,
                shop=tuple(cards[c] if c else None for c in b['shop']),
                coins=b['coins'],round_number=b['round'],selected_shop=b['selected_shop'],
                unknown_board=tuple(b['unknown_board']),unknown_shop=tuple(b['unknown_shop']),
                place_points=points,ready_point=(1118,654))

        def recorded_planner(board,shop,coins,round_number,*,explanation):
            line={5:11,3:15,1:19}[coins]
            explanation.update(deepcopy(records[line-1]['selection']))
            first=explanation['plan'][0]
            return first['shop'],first['cell']

        self.assertEqual(records[10]['selection']['plan'][-1]['cell'],6)
        self.assertEqual(records[18]['selection']['plan'][0]['cell'],1)
        flow=ActivityFlow('continuous')
        with patch(CHOOSE,side_effect=recorded_planner):
            for now,line in enumerate((11,12,15,16,19,20)):
                decision=step(flow,observed(line),now)
                if line==20:
                    self.assertEqual((decision.action,decision.point),('place',BOARD_CENTERS[6]))
                flow.completed(decision,now)


if __name__=='__main__':
    unittest.main()
