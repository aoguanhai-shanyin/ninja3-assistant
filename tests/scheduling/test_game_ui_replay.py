import unittest
from unittest.mock import Mock

import cv2

from ninja3_assistant.game_ui.pages import MenuVision
from ninja3_assistant.paths import project_root
from tests.runtime.test_game_ui import ScriptedMenu


DATA = project_root() / 'data'
FRAME_PATHS = {
    'intelligence_tasks': DATA / 'intelligence-simulation/live/20261113-140331-tasks-first.png',
    'intelligence_lobby': DATA / 'intelligence-simulation/live/20261113-140347-tasks-close.png',
    'hall_left': DATA / 'intelligence-simulation/live/20261113-141926-navigation-start-hall.png',
    'hall_middle': DATA / 'weekend-race/live/hall-right.png',
    'hall_right': DATA / 'weekend-race/live/hall-far-right.png',
    'weekend_lobby': DATA / 'weekend-race/live/20261114-143223-entry.png',
    'weekend_tasks': DATA / 'weekend-race/live/20261114-143242-tasks.png',
}
XUMI = DATA / 'evidence/2026-11-19/xumi-152'
RANK_FRAME = DATA / 'evidence/2026-11-16/daily-exploration/104338-hidden-state.png'
XUMI_STEPS = (
    ('back', ('tap', (75, 50)), XUMI / '000-home.png'),
    ('open_hall', ('tap', (864, 612)), XUMI / '001-entry.png'),
    ('open_xumi', ('tap', (725, 400)), XUMI / '002-event.png'),
    ('advance_guide', ('tap', (1083, 595)), XUMI / '003-guide-2.png'),
    ('advance_guide', ('tap', (1083, 595)), XUMI / '004-guide-3.png'),
    ('advance_guide', ('tap', (1083, 595)), XUMI / '005-guide-4.png'),
    ('advance_guide', ('tap', (1083, 595)), XUMI / '006-guide-5.png'),
    ('advance_guide', ('tap', (640, 100)), XUMI / '008-lobby.png'),
)


@unittest.skipUnless(all(path.is_file() for path in FRAME_PATHS.values()), 'requires local game captures')
class GameUINavigationReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = {
            name: cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            for name, path in FRAME_PATHS.items()
        }

    def test_intelligence_and_weekend_task_pages_connect_both_directions(self):
        cases = (
            ('intelligence_tasks', 'weekend.tasks', (
                ('back', ('tap', (1120, 111)), 'intelligence_lobby'),
                ('back', ('tap', (50, 45)), 'hall_left'),
                ('find_weekend', ('swipe', (1150, 410), (300, 410)), 'hall_middle'),
                ('find_weekend', ('swipe', (1150, 410), (300, 410)), 'hall_right'),
                ('open_weekend', ('tap', (1130, 390)), 'weekend_lobby'),
                ('open_tasks', ('tap', (885, 47)), 'weekend_tasks'),
            )),
            ('weekend_tasks', 'intelligence.tasks', (
                ('back', ('tap', (1190, 400)), 'weekend_lobby'),
                ('back', ('tap', (65, 45)), 'hall_right'),
                ('find_intelligence', ('swipe', (300, 410), (1150, 410)), 'hall_middle'),
                ('find_intelligence', ('swipe', (300, 410), (1150, 410)), 'hall_left'),
                ('open_intelligence', ('tap', (373, 389)), 'intelligence_lobby'),
                ('open_tasks', ('tap', (866, 51)), 'intelligence_tasks'),
            )),
        )
        for start, target, steps in cases:
            with self.subTest(start=start, target=target):
                menu = ScriptedMenu(self.frames[start], MenuVision(),
                                    [(command, self.frames[following]) for _, command, following in steps])

                result = menu.ui.navigate(target)

                self.assertEqual((result.status, result.page), ('arrived', target))
                self.assertEqual([step.action for step in result.steps], [action for action, _, _ in steps])
                self.assertEqual(menu.inputs, [command for _, command, _ in steps])
                self.assertFalse(menu.changes)

    def test_hall_click_requires_entrance_evidence_even_if_tap_area_did_not_change(self):
        hall = self.frames['hall_right']
        menu = ScriptedMenu(hall, MenuVision())
        observed = menu.ui.observe()
        self.assertIn('open_weekend', observed.controls)

        no_entrance = hall.copy()
        no_entrance[110:230, 900:1280] = 0
        menu.frame = no_entrance
        rechecked = menu.ui.observe()
        self.assertEqual(rechecked.page, 'game.hall')
        self.assertNotIn('open_weekend', rechecked.controls)

        self.assertIsNone(menu.ui.perform(observed, 'open_weekend'))
        self.assertEqual(menu.inputs, [])


@unittest.skipUnless(RANK_FRAME.is_file() and all(path.is_file() for _, _, path in XUMI_STEPS),
                     'requires local rank and Xumi captures')
class RankToXumiReplayTest(unittest.TestCase):
    def test_rank_navigates_through_home_and_all_five_xumi_guide_pages(self):
        def frame(path):
            return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

        vision = MenuVision()
        vision.ocr = Mock()
        vision.ocr.read.return_value = {}
        menu = ScriptedMenu(frame(RANK_FRAME), vision,
                            [(command, frame(path)) for _, command, path in XUMI_STEPS])
        self.assertEqual(menu.ui.observe().page, 'daily.rank')

        result = menu.ui.navigate('xumi.lobby')

        self.assertEqual((result.status, result.page), ('arrived', 'xumi.lobby'))
        self.assertEqual([step.action for step in result.steps], [name for name, _, _ in XUMI_STEPS])
        self.assertEqual(menu.inputs, [command for _, command, _ in XUMI_STEPS])
        self.assertFalse(menu.changes)
