from collections import deque
from dataclasses import replace
import unittest
from unittest.mock import patch

from ninja3_assistant.game_ui import GamePage, GameUI, NavigationError, PageAction, PageObservation
from ninja3_assistant.game_ui.pages import ROUTES
from ninja3_assistant.runtime.runner import RunControl


class ScriptedMenu:
    """Replay page changes only when the expected input is sent."""

    def __init__(self, frame, recognize, changes=()):
        self.frame = frame
        self.changes = deque(changes)
        self.inputs = []
        self.captures = 0
        self.now = 0.
        self.control = RunControl(clock=self.clock)
        self.control.stop.wait = self.wait
        self.ui = GameUI(self.control, self.capture, self, recognize=recognize,
                         routes=ROUTES, swipe=self.swipe, clock=self.clock)

    def clock(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds
        return self.control.stop.is_set()

    def capture(self):
        self.captures += 1
        return self.frame

    def _send(self, action):
        self.inputs.append(action)
        if self.changes:
            expected, following = self.changes.popleft()
            if expected != action:
                raise AssertionError(f'expected {expected}, sent {action}')
            self.frame = following

    def priority_tap(self, point, *, captured_at, **_):
        self._send(('tap', point))
        return captured_at, captured_at

    def swipe(self, start, end):
        self._send(('swipe', start, end))
        return True


def page(name, *controls, blocking=False):
    return PageObservation(name, object(), 0, 0.,
                           {control.name: control for control in controls}, blocking=blocking)


def recognize(observation, generation, captured_at):
    return replace(observation, generation=generation, captured_at=captured_at)


class GameUINavigationTest(unittest.TestCase):
    def test_typed_navigation_returns_the_observed_page_and_its_named_actions(self):
        class House(GamePage):
            page_id = 'daily.house'
            related_pages = ('daily.house_continue',)

            @property
            def training_remaining(self):
                return self.observation.detail['remaining']

            def collect_fragments(self):
                return self._perform('collect_fragments')

        home = page('game.home', PageAction('open_house', (143, 195)))
        house = replace(page('daily.house', PageAction('collect_fragments', (320, 650))), detail={'remaining': 0})
        confirm = page('daily.house_continue', blocking=True)
        menu = ScriptedMenu(home, recognize, ((('tap', (143, 195)), house), (('tap', (320, 650)), confirm)))

        view = menu.ui.navigate(House)
        self.assertIsInstance(view, House)
        self.assertEqual(view.training_remaining, 0)
        self.assertEqual(menu.inputs, [('tap', (143, 195))])
        self.assertIsNotNone(view.collect_fragments())
        self.assertEqual(menu.ui.read_page(House).observation.page, 'daily.house_continue')
        with self.assertRaises(NavigationError) as failure:
            menu.ui.navigate(House)
        self.assertEqual(failure.exception.result.status, 'blocked')

    def test_replans_after_an_unexpected_recognized_page(self):
        home = page('game.home', PageAction('open_hall', (864, 612)))
        weekend = page('weekend.lobby', PageAction('back', (65, 45)))
        hall = page('game.hall', PageAction('open_intelligence', (373, 389)))
        lobby = page('intelligence.lobby', PageAction('open_tasks', (866, 51)))
        tasks = page('intelligence.tasks')
        changes = (
            (('tap', (864, 612)), weekend),
            (('tap', (65, 45)), hall),
            (('tap', (373, 389)), lobby),
            (('tap', (866, 51)), tasks),
        )
        menu = ScriptedMenu(home, recognize, changes)

        result = menu.ui.navigate('intelligence.tasks')

        self.assertEqual((result.status, result.page), ('arrived', 'intelligence.tasks'))
        self.assertEqual([(step.source, step.expected) for step in result.steps], [
            ('game.home', 'game.hall'), ('weekend.lobby', 'game.hall'),
            ('game.hall', 'intelligence.lobby'), ('intelligence.lobby', 'intelligence.tasks'),
        ])
        self.assertEqual(menu.inputs, [command for command, _ in changes])
        self.assertFalse(menu.changes)

    def test_hall_search_has_a_bounded_attempt_count(self):
        hall = page('game.hall', PageAction('find_weekend', (1150, 410), (300, 410)))
        menu = ScriptedMenu(hall, recognize)

        result = menu.ui.navigate('weekend.lobby')

        self.assertEqual((result.status, result.page), ('unreachable', 'game.hall'))
        self.assertEqual(menu.inputs, [('swipe', (1150, 410), (300, 410))] * 3)
        self.assertEqual([step.action for step in result.steps], ['find_weekend'] * 3)

    def test_bidirectional_hall_search_switches_after_three_attempts(self):
        hall = page('game.hall', PageAction('find_bounty', (1080, 400), (280, 400)),
                    PageAction('find_bounty_left', (280, 400), (1080, 400)))
        menu = ScriptedMenu(hall, recognize)

        result = menu.ui.navigate('bounty.hub')

        self.assertEqual((result.status, result.page), ('unreachable', 'game.hall'))
        self.assertEqual(menu.inputs, [('swipe', (1080, 400), (280, 400))] * 3 +
                         [('swipe', (280, 400), (1080, 400))] * 3)

    def test_missing_control_times_out_without_touch(self):
        menu = ScriptedMenu(page('game.home'), recognize)

        result = menu.ui.navigate('game.hall')

        self.assertEqual((result.status, result.page, result.steps), ('timeout', 'game.home', ()))
        self.assertEqual(menu.inputs, [])
        self.assertEqual(menu.clock(), 20.)

    def test_acknowledged_tap_does_not_establish_arrival(self):
        lobby = page('weekend.lobby', PageAction('open_tasks', (885, 47)))
        menu = ScriptedMenu(lobby, recognize)

        result = menu.ui.navigate('weekend.tasks')

        self.assertEqual((result.status, result.page), ('timeout', 'weekend.lobby'))
        self.assertEqual(menu.inputs, [('tap', (885, 47))] * 2)

    def test_recognition_after_navigation_deadline_cannot_establish_arrival(self):
        def slow_recognition(observation, generation, captured_at):
            menu.wait(20.1)
            return recognize(observation, generation, captured_at)

        menu = ScriptedMenu(page('game.hall'), slow_recognition)

        result = menu.ui.navigate('game.hall')

        self.assertEqual((result.status, result.page), ('timeout', 'game.hall'))
        self.assertEqual(menu.inputs, [])

    def test_blocking_overlay_preempts_even_the_destination(self):
        tasks = page('weekend.tasks', PageAction('back', (1190, 400)), blocking=True)
        menu = ScriptedMenu(tasks, recognize)

        result = menu.ui.navigate('weekend.tasks')

        self.assertEqual((result.status, result.page), ('blocked', 'weekend.tasks'))
        self.assertEqual(menu.inputs, [])
        self.assertEqual(menu.captures, 1)

    def test_action_checks_the_refreshed_frame_before_touch(self):
        reference, changed = object(), object()
        action = PageAction('open_tasks', (885, 47), guard=lambda frame: frame is reference)
        menu = ScriptedMenu(changed, recognize)
        observed = PageObservation('weekend.lobby', reference, menu.control.generation(),
                                   menu.clock(), {action.name: action})

        self.assertIsNone(menu.ui.perform(observed, action.name))
        self.assertEqual(menu.inputs, [])
        self.assertEqual(menu.captures, 1)

    def test_pause_generation_rejects_old_observation(self):
        menu = ScriptedMenu(page('weekend.lobby', PageAction('open_tasks', (885, 47))), recognize)
        observed = menu.ui.observe()
        menu.control.pause()
        menu.control.resume()

        self.assertIsNone(menu.ui.perform(observed, 'open_tasks'))
        self.assertEqual(menu.inputs, [])
        self.assertEqual(menu.captures, 1)

    def test_expired_frame_during_preparation_is_rejected(self):
        menu = ScriptedMenu(page('weekend.lobby', PageAction('open_tasks', (885, 47))), recognize)
        observed = menu.ui.observe()

        self.assertIsNone(menu.ui.perform(observed, 'open_tasks', before_send=lambda: menu.wait(.21)))
        self.assertEqual(menu.inputs, [])
        self.assertEqual(menu.captures, 2)

    def test_direct_action_preserves_explicit_capture_deadline_and_hold(self):
        menu = ScriptedMenu(page('intelligence.battle'), recognize)
        observed = menu.ui.observe()
        action = PageAction('place_card', (640, 550), max_age=.5, hold_seconds=.07)
        menu.wait(.1)

        with patch.object(menu.ui, 'tap', wraps=menu.ui.tap) as tap:
            self.assertEqual(menu.ui.perform(observed, action, refresh=False), (0., 0.))

        tap.assert_called_once_with((640, 550), captured_at=0., deadline=.5,
                                    hold_seconds=.07, wait_seconds=.2)
        self.assertEqual(menu.captures, 1)

    def test_stop_during_recognition_cancels_route(self):
        menu = ScriptedMenu(page('game.home', PageAction('open_hall', (864, 612))), recognize)

        def stop_during_capture():
            menu.control.request_stop()
            return menu.capture()

        with patch.object(menu.ui, 'capture', side_effect=stop_during_capture):
            result = menu.ui.navigate('game.hall')

        self.assertEqual((result.status, result.page), ('stopped', 'game.home'))
        self.assertEqual(menu.inputs, [])
