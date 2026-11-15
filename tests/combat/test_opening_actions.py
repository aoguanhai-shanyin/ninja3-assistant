from ninja3_assistant.paths import project_root
import threading
import unittest
import os
from types import SimpleNamespace
from unittest.mock import patch

import cv2

from ninja3_assistant.device.mumu import MuMu
from ninja3_assistant.combat.opening_actions import OpeningActions, execute_opening
from ninja3_assistant.runtime.runner import RunControl


ROOT = project_root()
FRAMES = ROOT/'data/hayabusa-windup-v1/live-calibration-first'
SKILLS = ROOT.parent/'android-icon-clicker/build/evidence/battle-inspect'
LIVE = ROOT/'data/hayabusa-windup-v1/live-calibration-third'


def frame(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


@unittest.skipUnless((FRAMES/'frame-00280.jpg').is_file(), 'captured ground image unavailable')
class OpeningImageTests(unittest.TestCase):
    def test_ground_action_only_after_two_ground_frames_and_never_on_boss(self):
        actions = OpeningActions()
        ground = frame(FRAMES/'frame-00280.jpg')
        skill = actions.decide(ground, 1)
        self.assertEqual((skill.kind, skill.point), ('ninjutsu', (55, 60)))
        actions.completed(skill, 1)
        self.assertEqual(actions.decide(ground, 1.1).kind, 'ground_jump_slide')
        actions.completed(actions.decide(ground, 1.1), 1.1)
        self.assertEqual(actions.decide(ground, 1.13).kind, 'ground_jump_slide')
        actions.completed(actions.decide(ground, 1.13), 1.13)
        self.assertEqual(actions.decide(ground, 1.14).kind, 'ground_jump_slide')
        skill = actions.decide(ground, 1.34)
        self.assertEqual(skill.kind, 'ninjutsu')
        actions.completed(skill, 1.34)
        self.assertEqual(actions.decide(ground, 1.36).kind, 'ground_jump_slide')
        skill = actions.decide(ground, 2, boss_active=True)
        self.assertEqual((skill.kind, skill.point), ('ninjutsu', (55, 60)))
        actions.completed(skill, 2)
        self.assertIsNone(actions.decide(frame(FRAMES/'frame-00353.jpg'), 2.1, boss_active=True))

    def test_due_skill_and_skate_alternate_even_at_low_frame_rate(self):
        actions = OpeningActions()
        ground = frame(FRAMES/'frame-00280.jpg')
        skill = actions.decide(ground, 1)
        actions.completed(skill, 1)
        skate = actions.decide(ground, 2)
        self.assertEqual(skate.kind, 'ground_jump_slide')
        actions.completed(skate, 2)
        self.assertEqual(actions.decide(ground, 2.01).kind, 'ninjutsu')

    @unittest.skipUnless((SKILLS/'ninjutsu-5.png').is_file(), 'battle sample unavailable')
    def test_skill_repeats_without_readiness_prompt_and_stops_outside_battle(self):
        actions = OpeningActions()
        full = frame(SKILLS/'ninjutsu-5.png')
        no_hand = frame(SKILLS/'ninjutsu-8.png')
        action = actions.decide(full, 1, boss_active=True)
        self.assertEqual(action.kind, 'ninjutsu')
        self.assertEqual(action.point, (70,330))
        actions.completed(action, 1)
        self.assertIsNone(actions.decide(full, 1.1, boss_active=True))
        self.assertIsNone(actions.decide(no_hand, 1.3, boss_active=True, battle_allowed=False))
        again = actions.decide(no_hand, 1.3, boss_active=True)
        self.assertEqual((again.kind, again.point), ('ninjutsu', (70,330)))
        actions.completed(again, 1.3)
        self.assertIsNone(actions.decide(no_hand, 1.4, boss_active=True))

    @unittest.skipUnless((LIVE/'frame-00344.jpg').is_file(), 'current battle images unavailable')
    def test_current_battle_taps_without_waiting_for_prompt(self):
        actions = OpeningActions()
        action = actions.decide(frame(LIVE/'frame-00343.jpg'), 1, boss_active=True)
        self.assertEqual(action.kind, 'ninjutsu')
        self.assertEqual(action.point, (70,330))
        actions.completed(action, 1)
        self.assertIsNone(actions.decide(frame(LIVE/'frame-00346.jpg'), 1.2, boss_active=True))
        self.assertEqual(actions.decide(frame(LIVE/'frame-00344.jpg'), 1.3, boss_active=True).kind, 'ninjutsu')

    @unittest.skipUnless((ROOT/'data/skate-video-study/live-easy-02/progress2.png').is_file(), 'live boss screenshot unavailable')
    def test_live_boss_hud_button_can_be_at_top(self):
        actions = OpeningActions()
        image = frame(ROOT/'data/skate-video-study/live-easy-02/progress2.png')
        action = actions.decide(image, 1, boss_active=True)
        self.assertEqual((action.kind, action.point), ('ninjutsu', (55, 60)))
        hidden = frame(ROOT/'data/skate-video-study/live-easy-02/progress.png')
        self.assertIsNone(OpeningActions().decide(hidden, 1, boss_active=True))


class SwipeTests(unittest.TestCase):
    def device(self, down, up):
        device = MuMu.__new__(MuMu)
        device.handle, device.display = 1, 0
        device.dll = SimpleNamespace(nemu_input_event_touch_down=down, nemu_input_event_touch_up=up)
        return device

    def test_swipe_is_continuous_contact_then_release(self):
        calls = []
        device = self.device(lambda *x: calls.append(('down',x[-2:])) or 0,
                             lambda *x: calls.append(('up',)) or 0)
        self.assertTrue(device.swipe((190,620),(190,700),threading.Event(),step_seconds=0,steps=4))
        self.assertEqual(calls, [('down',(190,620)),('down',(190,640)),('down',(190,660)),
                                 ('down',(190,680)),('down',(190,700)),('up',)])

    def test_release_occurs_when_stopped_during_swipe(self):
        calls, stop = [], threading.Event()
        def down(*args):
            calls.append(('down',args[-2:]))
            stop.set()
            return 0
        device = self.device(down, lambda *x: calls.append(('up',)) or 0)
        self.assertFalse(device.swipe((190,620),(190,700),stop,step_seconds=0))
        self.assertEqual(calls, [('down',(190,620)),('up',)])

    @patch.dict('os.environ', {'NINJA_SKATE_MULTITOUCH': '0'})
    def test_slide_jump_slide_are_separate_ordered_contacts(self):
        events, control = [], RunControl()
        class Device:
            def tap(self, point, stopped, **kwargs):
                kwargs['on_pressed']()
                return True
            def swipe(self, start, end, stopped, **kwargs):
                kwargs['on_pressed']()
                return True
        action = SimpleNamespace(kind='ground_jump_slide')
        self.assertTrue(execute_opening(action, Device(), control, lambda k,v: events.append(v)))
        self.assertEqual([e['kind'] for e in events],
                         ['ground_slide_start', 'ground_jump', 'ground_slide_cancel'])

    @patch.dict('os.environ', {'NINJA_SKATE_MULTITOUCH': '0'})
    def test_sequential_sdk_contacts_are_released_between_actions(self):
        calls, control = [], RunControl()
        device = self.device(lambda *x: calls.append(('down', x[-2:])) or 0,
                             lambda *x: calls.append(('up',)) or 0)
        self.assertTrue(execute_opening(SimpleNamespace(kind='ground_jump_slide'),
                                        device, control, lambda *_: None))
        self.assertEqual([i for i, c in enumerate(calls) if c[0] == 'up'], [5, 7, 13])
        self.assertEqual(calls[0], ('down', (190, 620)))
        self.assertEqual(calls[6], ('down', (1090, 630)))
        self.assertEqual(calls[8], ('down', (190, 620)))

    @patch.dict('os.environ', {'NINJA_SKATE_MULTITOUCH': '0'})
    def test_stop_between_jump_and_final_slide_prevents_swipe(self):
        events, control = [], RunControl()
        class Device:
            def tap(self, point, stopped, **kwargs):
                kwargs['on_pressed']()
                stopped.set()
                return True
            def swipe(self, *args, **kwargs):
                kwargs['on_pressed']()
                return True
        action = SimpleNamespace(kind='ground_jump_slide')
        self.assertFalse(execute_opening(action, Device(), control, lambda k,v: events.append(v)))
        self.assertEqual([e['kind'] for e in events], ['ground_slide_start', 'ground_jump'])

    @patch.dict('os.environ', {'NINJA_SKATE_MULTITOUCH': '0'})
    def test_interrupted_first_slide_does_not_send_jump(self):
        events, control = [], RunControl()
        class Device:
            def tap(self, *args, **kwargs):
                raise AssertionError('jump must not follow an interrupted slide')
            def swipe(self, start, end, stopped, **kwargs):
                kwargs['on_pressed']()
                stopped.set()
                return False
        action = SimpleNamespace(kind='ground_jump_slide')
        self.assertFalse(execute_opening(action, Device(), control, lambda k,v: events.append(v)))
        self.assertEqual([e['kind'] for e in events], ['ground_slide_start'])

    def test_multitouch_path_is_default_and_can_be_selected_explicitly(self):
        control, device, events = RunControl(), object(), []
        for setting in (None, '1'):
            with self.subTest(setting=setting), patch.dict('os.environ'), \
                    patch('ninja3_assistant.device.mumu_multitouch.ExperimentalMultiTouch') as gesture:
                if setting is None:
                    os.environ.pop('NINJA_SKATE_MULTITOUCH', None)
                else:
                    os.environ['NINJA_SKATE_MULTITOUCH'] = setting
                gesture.return_value.run.return_value = True
                action = SimpleNamespace(kind='ground_jump_slide')
                self.assertTrue(execute_opening(action, device, control, lambda k,v: events.append((k,v))))
                gesture.assert_called_once_with(device)
                gesture.return_value.run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
