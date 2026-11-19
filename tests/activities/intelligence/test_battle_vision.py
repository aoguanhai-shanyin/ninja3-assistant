"""Regression on actual live frames, not a claim of unseen-screen generalization."""
from ninja3_assistant.paths import project_root
from concurrent.futures import Future
from dataclasses import replace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ninja3_assistant.activities.intelligence.battle_vision import BattleVision
from ninja3_assistant.activities.intelligence.flow import ActivityFlow

LIVE = project_root()/'data/intelligence-simulation/live'


def image(name):
    return cv2.cvtColor(cv2.imread(str(LIVE/name)), cv2.COLOR_BGR2RGB)


class BattleVisionTest(unittest.TestCase):
    def test_actual_highlight_recovers_c036_without_blocking_placement(self):
        root = LIVE.parents[1]/'evidence/2026-11-15/intelligence-selection'
        frames = [cv2.cvtColor(cv2.imread(str(root/name)), cv2.COLOR_BGR2RGB)
                  for name in ('before.png', 'selected.png')]
        before, selected = [replace(self.vision.detect(frame), coins=4) for frame in frames]
        self.assertEqual(before.shop[2].id, 'c036')
        self.assertEqual(selected.selected_shop, 2)
        self.assertEqual(selected.shop[2].id,'c036')
        self.assertNotIn(2,selected.unknown_shop)
        # Isolate the selection/occlusion protocol from the purchase policy.
        before = replace(before, shop=tuple(card if i == 2 else None for i,card in enumerate(before.shop)))
        flow = ActivityFlow()
        choose = flow.battle_step(before, 0)
        self.assertEqual(choose.action, 'select_card')
        self.assertEqual(flow.purchase[0], 2)
        flow.completed(choose, 0)
        place = flow.battle_step(selected, 1)
        self.assertEqual(place.action, 'place')
        self.assertIsNotNone(place.point)
        flow.completed(place, 1)
        self.assertIsNone(flow.pending)

    def test_current_dark_empty_backs_preserve_the_three_occupied_cards(self):
        path = project_root()/'data/runtime/intelligence-simulation/runs/20261116-180405-083200/'
        frame = cv2.cvtColor(cv2.imread(str(path/'unknown-cards/frame-0006.png')),cv2.COLOR_BGR2RGB)
        observation = self.vision.detect(frame)
        self.assertEqual(observation.unknown_board,())
        self.assertEqual(tuple(c.id if c else None for c in observation.board),
                         ('c003','c006','c035',None,None,None,None,None,None))
        self.assertFalse(any(observation.place_points))

    def test_current_green_backs_still_require_a_visible_plus_and_selected_shop(self):
        path = project_root()/'data/runtime/intelligence-simulation/runs/20261116-184005-135300/'
        frame = cv2.cvtColor(cv2.imread(str(path/'unknown-cards/frame-0009.png')),cv2.COLOR_BGR2RGB)
        observation = self.vision.detect(frame)
        self.assertEqual(observation.selected_shop,3)
        self.assertEqual(observation.shop[3].id,'c052')
        self.assertEqual(observation.place_points[2],(480,232))
        self.assertEqual(observation.unknown_board,(8,))  # Obscured edge remains unknown.
        obscured = frame.copy()
        obscured[211:253,459:501] = 0
        self.assertIsNone(self.vision.detect(obscured).place_points[2])
        # A plus pasted onto a real card is insufficient to make that cell empty.
        occupied = frame.copy()
        occupied[138:310,407:552] = frame[138:310,67:212]
        occupied[217:248,465:496] = frame[217:248,295:326]
        check = self.vision.detect(occupied)
        self.assertIsNone(check.place_points[2])
        self.assertTrue(check.board[2] is not None or 2 in check.unknown_board)

    @classmethod
    def setUpClass(cls):
        cls.vision = BattleVision()

    def test_first_round_all_slots_are_known(self):
        observation = self.vision.detect(image('20261113-181458-match-search-01-end.png'))
        self.assertEqual(observation.kind, 'placement')
        self.assertEqual(observation.round_number, 1)
        self.assertEqual(observation.board, (None,)*9)
        self.assertEqual(observation.unknown_board, ())
        self.assertEqual(observation.unknown_shop, ())
        self.assertEqual([card.id for card in observation.shop], ['c050', 'c030', 'c016', 'c020', 'c019'])
        self.assertEqual(observation.refreshes, 1)
        self.assertIsNone(observation.selected_shop)
        self.assertFalse(any(observation.place_points))

    def test_winml_runtime_error_preserves_opencv_recognition(self):
        frame = image('20261113-181458-match-search-01-end.png')
        expected = self.vision.detect(frame)
        vision = BattleVision(use_winml=True)
        module = 'ninja3_assistant.activities.intelligence.battle_vision'
        ready = Future()
        ready.set_result((object(), 32.))
        with patch(module+'.warm_portrait_session', return_value=ready) as session, \
                patch(module+'.portrait_scores', side_effect=RuntimeError('GPU unavailable')) as scores, \
                self.assertLogs(module, level='WARNING'):
            self.assertEqual(vision.detect(frame), expected)
            self.assertEqual(vision.detect(frame), expected)
        session.assert_called_once()
        scores.assert_called_once()
        self.assertEqual(vision.backend, 'CPU: OpenCV')

    def test_winml_scores_near_an_acceptance_threshold_are_rechecked(self):
        frame = image('20261113-181458-match-search-01-end.png')
        expected = self.vision.detect(frame)
        vision = BattleVision(use_winml=True)
        module = 'ninja3_assistant.activities.intelligence.battle_vision'
        ready = Future()

        def borderline(_session, faces):
            scores = np.zeros((len(faces), len(vision.cards)), np.float32)
            scores[:, 0] = .8795
            upper_scores = scores.copy()
            upper_scores[:, 0] = .82
            return scores, upper_scores

        with patch(module+'.warm_portrait_session', return_value=ready) as session, \
                patch(module+'.portrait_scores', side_effect=borderline) as scores:
            self.assertEqual(vision.detect(frame), expected)
            self.assertEqual(vision.backend, 'CPU: OpenCV')
            scores.assert_not_called()
            ready.set_result((object(), 32.))
            self.assertEqual(vision.detect(frame), expected)
        self.assertIn('Windows ML', vision.backend)
        session.assert_called_once()
        scores.assert_called_once()

    def test_live_empty_board_sparkles_preserve_occupied_cards(self):
        observation = self.vision.detect(image('20261114-round1-check.png'))
        self.assertEqual(observation.unknown_board, ())
        self.assertEqual(observation.board[0].id, 'c034')
        self.assertEqual(observation.board[3].id, 'c025')
        self.assertTrue(all(card is None for i, card in enumerate(observation.board) if i not in (0, 3)))

    def test_live_empty_shop_sparkle_is_not_an_unknown_card(self):
        path = LIVE.parent/'runs/20261114-132033-551800/discard-battle_state_changed-after.png'
        observation = self.vision.detect(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
        self.assertIsNone(observation.shop[2])
        self.assertNotIn(2, observation.unknown_shop)

    def test_damage_overlay_uses_upper_portrait_and_name(self):
        for index in range(5):
            observation = self.vision.detect(image(f'20261114-stability-{index}.png'))
            self.assertEqual(observation.board[4].id, 'c032')
            self.assertFalse(observation.unknown_board)
        changed = image('20261114-stability-0.png')
        changed[475:495, 255:372] = 0
        self.assertIn(4, self.vision.detect(changed).unknown_board)

    def test_selected_shop_requires_green_border_and_visible_empty_plus(self):
        observation = self.vision.detect(image('20261113-181534-match1-select-purple.png'))
        self.assertEqual(observation.selected_shop, 3)
        self.assertEqual(observation.shop[3].id, 'c020')
        self.assertEqual(len([point for point in observation.place_points if point]), 9)

    def test_purchase_changes_one_board_and_one_shop_slot(self):
        observation = self.vision.detect(image('20261113-181543-match1-drop-purple.png'))
        self.assertEqual(observation.board[0].id, 'c020')
        self.assertEqual(observation.board[1:], (None,)*8)
        self.assertIsNone(observation.shop[3])
        self.assertEqual(observation.unknown_board, ())
        self.assertEqual(observation.unknown_shop, ())
        self.assertIsNone(observation.selected_shop)

    def test_actual_c012_green_selection_regression(self):
        path = LIVE.parent/'runs/20261113-183109-143500/000008-battle-stop.png'
        frame = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        observation = self.vision.detect(frame)
        self.assertEqual(observation.kind, 'placement')
        self.assertEqual(observation.selected_shop, 4)
        self.assertEqual(observation.shop[4].id, 'c012')
        self.assertEqual(observation.unknown_shop, ())

    def test_c029_selection_is_confirmed_by_portrait_and_name(self):
        path = LIVE.parent/'runs/20261114-133227-224600/000021-battle-stop.png'
        observation = self.vision.detect(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
        self.assertEqual(observation.selected_shop, 1)
        self.assertEqual(observation.shop[1].id, 'c029')

    def test_later_round_can_read_three_persistent_cards(self):
        observation = self.vision.detect(image('20261113-181936-match1-round5.png'))
        self.assertEqual(observation.kind, 'placement')
        self.assertEqual(observation.round_number, 5)
        self.assertEqual([card.id for card in observation.board[:3]], ['c020', 'c046', 'c044'])
        self.assertEqual(observation.board[3:], (None,)*6)
        self.assertFalse(observation.unknown_board)
        self.assertFalse(observation.unknown_shop)

    def test_matching_combat_and_result_have_no_purchase_controls(self):
        cases = [('20261113-181458-match-search-01.png', 'matching'),
                 ('20261113-181556-match1-select-fireweapon.png', 'combat'),
                 ('20261113-182012-match1-result.png', 'result')]
        for name, kind in cases:
            with self.subTest(name=name):
                observation = self.vision.detect(image(name))
                self.assertEqual(observation.kind, kind)
                self.assertIsNone(observation.ready_point)
                self.assertFalse(any(observation.place_points))
        self.assertEqual(self.vision.detect(image(cases[-1][0])).result_point, (640, 541))

    def test_result_requires_both_title_and_confirm_button(self):
        frame = image('20261113-182012-match1-result.png')
        for region in ((105, 175, 515, 720), (510, 575, 585, 700)):
            changed = frame.copy()
            y1, y2, x1, x2 = region
            changed[y1:y2, x1:x2] = 0
            self.assertEqual(self.vision.detect(changed).kind, 'unknown')

    def test_shifted_result_title_with_twenty_points(self):
        frame = image('20261114-observation-stopped.png')
        observation = self.vision.detect(frame)
        self.assertEqual(observation.kind, 'result')
        self.assertEqual(observation.result_point, (640, 541))
        frame[510:575, 585:700] = 0
        self.assertEqual(self.vision.detect(frame).kind, 'unknown')

    def test_unrelated_and_obscured_pages_do_not_claim_battle(self):
        for name in ('20261113-140318-guide-close.png', '20261113-140331-tasks-first.png',
                     '20261113-140100-event-entry.png', '20261113-181319-evening-initial.png'):
            self.assertEqual(self.vision.detect(image(name)).kind, 'unknown')
        frame = image('20261113-181458-match-search-01-end.png')
        self.assertEqual(self.vision.detect((frame*.35).astype(np.uint8)).kind, 'unknown')
        self.assertEqual(self.vision.detect(np.zeros_like(frame)).kind, 'unknown')

    def test_ocr_parsers_keep_missing_digits_unknown(self):
        self.assertEqual(self.vision.parse_numbers(None), (None, None))
        result = {'words': [
            {'text': '金币：17', 'bbox': [30, 30, 90, 25]},
            {'text': '剩余刷新次数：0', 'bbox': [30, 150, 180, 25]},
        ]}
        self.assertEqual(self.vision.parse_numbers(result), (17, 0))
        result = {'words': [{'text': '当前积分：4本局+0', 'bbox': [0, 0, 200, 50]}]}
        self.assertEqual(self.vision.parse_result_numbers(result), (4, 0))


if __name__ == '__main__':
    unittest.main()
