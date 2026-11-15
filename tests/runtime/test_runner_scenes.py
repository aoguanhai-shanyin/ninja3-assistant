import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ninja3_assistant.combat.boss.policy import BossAction
from ninja3_assistant.activities.bounty.flow import Decision
from ninja3_assistant.combat.hayabusa.scene import SceneDecision
from ninja3_assistant.combat.opening_actions import OpeningAction
from ninja3_assistant.runtime.runner import RunControl, run_flow
from ninja3_assistant.vision.vision import Screen


class SceneRig:
    """One final capture stops the runner without advancing either workflow."""
    def __init__(self, screens, decisions, *, control=None):
        self.control = control or RunControl()
        self.screens = screens
        self.decisions = decisions
        self.events = []
        self.order = []
        self.frames = []
        self.detected = []
        self.scene_frames = []
        self.flow_screens = []
        self.taps = []
        self.completions = []
        self.touch_errors = []
        self.capture_durations = []
        self.input_durations = []
        self.resets = self.releases = 0
        self.closed = self.scene_closed = False
        self.tap_hook = None
        self.detect_hook = None
        self.flow_hook = None
        self.scene_clock = None
        self.native = None

    def capture(self):
        frame = SimpleNamespace(index=len(self.frames))
        self.frames.append(frame)
        self.order.append(('capture', frame.index))
        return frame

    def detect(self, frame):
        self.detected.append(frame)
        self.order.append(('detect', frame.index))
        if self.detect_hook is not None:
            self.detect_hook(frame)
        if frame.index == len(self.screens):
            self.control.stop.set()
            return Screen()
        if frame.index > len(self.screens):
            raise AssertionError('runner did not stop at final capture')
        return self.screens[frame.index]

    def scene_step(self, frame, now):
        self.scene_frames.append(frame)
        self.order.append(('scene', frame.index))
        decision = self.decisions.get(frame.index)
        return decision(frame, now) if callable(decision) else decision

    def tap(self, point, stopped, hold_seconds=.07, on_pressed=None):
        self.taps.append((point, hold_seconds, self.frames[-1]))
        if on_pressed is not None:
            on_pressed()
        if self.tap_hook is not None:
            return self.tap_hook(stopped)
        return True

    def run(self, emit=None):
        rig = self

        class Scene:
            def __init__(self, *, clock):
                rig.scene_clock = clock
            step = staticmethod(rig.scene_step)
            def reset(self):
                rig.resets += 1
                rig.order.append(('reset', len(rig.frames)))
            def release(self):
                rig.releases += 1
            def completed(self, action, now):
                rig.completions.append((action, now))
            def failed_touch(self, error):
                rig.touch_errors.append(str(error))
            def frame_age(self, seconds):
                pass
            def capture_duration(self, seconds):
                rig.capture_durations.append(seconds)
            def input_duration(self, seconds):
                rig.input_durations.append(seconds)
            def close(self):
                rig.scene_closed = True

        class Detector:
            backend = 'S detector'
            detect = staticmethod(rig.detect)

        class Device:
            capture = staticmethod(rig.capture)
            tap = staticmethod(rig.tap)
            def reopen_game(self, stopped):
                raise AssertionError('unexpected reopen')
            def close(self):
                rig.closed = True

        class Flow:
            successful_kills = 0
            def step(self, screen, now):
                rig.flow_screens.append(screen)
                if rig.flow_hook is not None:
                    return rig.flow_hook(screen, now)
                return Decision('wait', status='旧任务等待')
            def completed(self, decision, now):
                pass

        def collect(kind, value):
            rig.events.append((kind, value))
            if emit is not None:
                emit(kind, value)

        with patch('ninja3_assistant.runtime.runner.prepare_game', return_value=False), \
                patch('ninja3_assistant.runtime.runner.create_native_touch', return_value=self.native), \
                patch('ninja3_assistant.runtime.runner.Vision', Detector), patch('ninja3_assistant.runtime.runner.MuMu', return_value=Device()), \
                patch('ninja3_assistant.runtime.runner.Flow', Flow), patch('ninja3_assistant.runtime.runner.HayabusaScene', Scene):
            run_flow('test', 0, self.control, collect)


def battle(action=None, *, active=True, result=None, stalled=False):
    return SceneDecision(active=active, result=result, action=action,
                         status='隼白测试', backend='scene GPU', stalled=stalled)


class RunnerScenesTest(unittest.TestCase):
    def assert_clean(self, rig):
        self.assertTrue(rig.closed)
        self.assertTrue(rig.scene_closed)
        self.assertEqual(rig.events[-1], ('finished', ''))
        self.assertFalse([value for kind, value in rig.events
                          if kind == 'status' and value.startswith('已停止：')], rig.events)

    def test_unrecognized_scene_keeps_the_existing_flow(self):
        screen = Screen()
        rig = SceneRig([screen], {})
        rig.run()
        self.assertEqual(rig.flow_screens, [screen])
        self.assertEqual(rig.scene_frames, rig.frames[:1])
        self.assertEqual(rig.detected, rig.frames)
        self.assertEqual(rig.releases, 0)
        self.assertEqual(rig.scene_clock.__self__, rig.control)
        self.assert_clean(rig)

    def test_active_scene_owns_the_same_capture_and_actual_touch_callback(self):
        action = BossAction('jump', (1090, 630))
        rig = SceneRig([Screen()], {0: battle(action)})
        rig.run()
        self.assertEqual(rig.flow_screens, [])
        self.assertEqual(rig.scene_frames, rig.frames[:1])
        self.assertEqual(rig.detected, rig.frames)
        self.assertEqual(rig.taps, [(action.point, .025, rig.frames[0])])
        self.assertEqual([item[0] for item in rig.completions], [action])
        self.assertEqual(len(rig.capture_durations), 1)
        self.assertEqual(len(rig.input_durations), 1)
        self.assertIn(('backend', 'scene GPU'), rig.events)
        self.assert_clean(rig)

    def test_scene_result_is_exclusive_even_when_old_detector_sees_a_result(self):
        rig = SceneRig([Screen(result=(100, 200))], {0: battle(active=False, result='victory')})
        rig.run()
        self.assertEqual(rig.flow_screens, [])
        self.assertEqual(rig.releases, 0)
        self.assertEqual(rig.taps, [])
        self.assert_clean(rig)

    def test_owned_unknown_and_generic_battle_frames_never_reopen_or_release(self):
        rig = SceneRig([Screen(), Screen(), Screen(battle_active=True), Screen(battle_loading=True)],
                       {0: battle(), 1: battle(active=False), 2: battle(active=False), 3: battle(active=False)})
        rig.run()
        self.assertEqual(rig.flow_screens, [])
        self.assertEqual(rig.releases, 0)
        self.assertEqual(len(rig.scene_frames), 4)
        self.assert_clean(rig)

    def test_unknown_scene_cannot_touch_an_unconfirmed_action(self):
        action = BossAction('jump', (1090, 630))
        rig = SceneRig([Screen()], {0: battle(action, active=False)})
        rig.run()
        self.assertEqual(rig.taps, [])
        self.assertEqual(rig.completions, [])
        self.assert_clean(rig)

    def test_stalled_scene_skips_opening_until_fresh_frame(self):
        rig = SceneRig([Screen(), Screen()], {0: battle(stalled=True), 1: battle()})
        with patch('ninja3_assistant.runtime.runner.OpeningActions') as opening:
            opening.return_value.decide.return_value = OpeningAction('ninjutsu', (70, 330))
            rig.run()
        self.assertEqual(opening.return_value.decide.call_count, 1)
        self.assertEqual([(tap[0], tap[2].index) for tap in rig.taps], [((70, 330), 1)])
        self.assert_clean(rig)

    def test_native_ground_macro_allows_skill_but_stops_before_navigation(self):
        class Macro:
            def __init__(self): self.held = False; self.events = []
            def renew(self): self.held = True; self.events.append('renew'); return True
            def release(self): self.held = False; self.events.append('release')
            def close(self): self.release(); self.events.append('close')
        class Opening:
            ground_frames = 2
            def decide(self, frame, *args, **kwargs):
                return OpeningAction('ninjutsu', (55, 60)) if frame.index == 1 else OpeningAction('ground_jump_slide')
            def completed(self, *args): pass
            def reset(self): pass
        macro = Macro()
        rig = SceneRig([Screen(battle_active=True), Screen(battle_active=True), Screen(board=True)], {})
        def skill_tap(stopped):
            self.assertTrue(macro.held)
            return True
        rig.tap_hook = skill_tap
        def flow(screen, now):
            self.assertFalse(macro.held)
            return Decision('wait')
        rig.flow_hook = flow
        with patch.dict('os.environ', {'NINJA_USE_MUMU_MACRO': '1'}), \
                patch('ninja3_assistant.runtime.runner.create_skate_macro', return_value=macro), patch('ninja3_assistant.runtime.runner.OpeningActions', Opening):
            rig.run()
        self.assertEqual([tap[0] for tap in rig.taps], [(55, 60)])
        self.assertEqual(macro.events[:2], ['renew', 'renew'])
        self.assertIn('release', macro.events)
        self.assertEqual(macro.events[-1], 'close')
        self.assert_clean(rig)

    def test_native_ground_macro_stops_before_boss_touch(self):
        class Macro:
            def __init__(self): self.held = False
            def renew(self): self.held = True; return True
            def release(self): self.held = False
            def close(self): self.release()
        class Opening:
            ground_frames = 2
            def decide(self, *args, **kwargs): return OpeningAction('ground_jump_slide')
            def completed(self, *args): pass
            def reset(self): pass
        macro = Macro()
        action = BossAction('jump', (1090, 630))
        rig = SceneRig([Screen(battle_active=True), Screen()], {1: battle(action)})
        def boss_tap(stopped):
            self.assertFalse(macro.held)
            return True
        rig.tap_hook = boss_tap
        with patch.dict('os.environ', {'NINJA_USE_MUMU_MACRO': '1'}), \
                patch('ninja3_assistant.runtime.runner.create_skate_macro', return_value=macro), patch('ninja3_assistant.runtime.runner.OpeningActions', Opening):
            rig.run()
        self.assertEqual([tap[0] for tap in rig.taps], [action.point])
        self.assertEqual([item[0] for item in rig.completions], [action])
        self.assert_clean(rig)

    def test_cpp_input_owns_skate_and_skill_and_quiesces_before_boss(self):
        class Native:
            def __init__(self): self.held = False; self.events = []
            def note_frame(self, at): pass
            def renew(self): self.held = True; self.events.append('renew'); return True
            def release(self): self.held = False; self.events.append('release')
            def stop_all(self): self.held = False; self.events.append('stop_all')
            def quiesce(self, **kwargs): self.held = False; self.events.append('quiesce')
            def tap(self, point, *, deadline=None): self.events.append(('tap', point)); return True
            def priority_tap(self, point, *, captured_at, deadline, hold_seconds=.025):
                self.held = False
                self.events.append(('priority_tap', point))
                return (captured_at+.01,captured_at+.011)
            def close(self): self.held = False; self.events.append('close')
        class Opening:
            ground_frames = 2
            def decide(self, frame, *args, **kwargs):
                return OpeningAction('ninjutsu', (55,60)) if frame.index == 1 else OpeningAction('ground_jump_slide')
            def completed(self, *args): pass
            def scheduled(self, *args): pass
            def reset(self): pass
        action = BossAction('jump', (1090, 630))
        rig = SceneRig([Screen(battle_active=True), Screen(battle_active=True), Screen()], {2:battle(action)})
        rig.native = Native()
        rig.tap_hook = lambda _stop: self.assertFalse(rig.native.held) or True
        with patch('ninja3_assistant.runtime.runner.OpeningActions', Opening), patch('ninja3_assistant.runtime.runner.create_skate_macro') as old_macro, \
                patch.dict('os.environ', {'NINJA_USE_MUMU_MACRO':'1'}):
            rig.run()
        old_macro.assert_not_called()
        self.assertIn(('tap',(55,60)),rig.native.events)
        self.assertEqual(rig.taps,[])
        self.assertIn(('priority_tap',action.point),rig.native.events)
        self.assertEqual([item[0] for item in rig.completions],[action])
        self.assertEqual(rig.native.events[-1],'close')
        self.assert_clean(rig)

    def test_cpp_boss_without_press_ack_is_not_completed(self):
        class Native:
            def note_frame(self, _): pass
            def renew(self): return True
            def release(self): pass
            def stop_all(self): pass
            def quiesce(self, **kwargs): pass
            def priority_tap(self, point, **kwargs): return None
            def close(self): pass
        action = BossAction('jump',(1090,630))
        rig = SceneRig([Screen()],{0:battle(action)})
        rig.native = Native()
        rig.run()
        self.assertEqual(rig.completions,[])
        self.assertEqual(rig.taps,[])
        self.assertEqual(rig.touch_errors,['native press expired or cancelled'])
        self.assert_clean(rig)

    def test_navigation_and_popups_override_active_scene_on_the_same_frame(self):
        screens = [Screen(board=True), Screen(menu=(1, 2)), Screen(ready=(1, 2)),
                   Screen(prepared=True), Screen(home=(1, 2)), Screen(login=(1, 2)),
                   Screen(hall=(1, 2)), Screen(network_error=True), Screen(invite=(1, 2))]
        for screen in screens:
            with self.subTest(screen=screen):
                action = BossAction('jump', (1090, 630))
                rig = SceneRig([Screen(), screen], {0: battle(), 1: battle(action)})
                rig.run()
                self.assertEqual(rig.scene_frames, rig.frames[:1])
                self.assertEqual(rig.flow_screens, [screen])
                self.assertEqual(rig.releases, 1)
                self.assertEqual(rig.taps, [])
                self.assertEqual([value for kind, value in rig.events if kind == 'backend'],
                                 ['S detector', 'scene GPU', 'S detector'])
                self.assert_clean(rig)

    def test_generic_result_releases_inactive_owned_scene(self):
        screen = Screen(result=(100, 200))
        rig = SceneRig([Screen(), screen], {0: battle(), 1: battle(active=False)})
        rig.run()
        self.assertEqual(rig.flow_screens, [screen])
        self.assertEqual(rig.releases, 1)
        self.assert_clean(rig)

    def test_stop_after_recognition_prevents_touch(self):
        rig = SceneRig([Screen()], {})
        def stop(frame, now):
            rig.control.stop.set()
            return battle(BossAction('jump', (1090, 630)))
        rig.decisions[0] = stop
        rig.run()
        self.assertEqual(rig.taps, [])
        self.assertEqual(rig.completions, [])
        self.assert_clean(rig)

    def test_pause_generation_discards_old_action_and_resets_before_new_capture(self):
        stale = BossAction('jump', (100, 200))
        fresh = BossAction('down', (200, 300))
        rig = SceneRig([Screen(), Screen()], {1: battle(fresh)})
        def pause_resume(frame, now):
            rig.control.pause()
            rig.control.resume()
            return battle(stale)
        rig.decisions[0] = pause_resume
        rig.run()
        self.assertEqual([item[0] for item in rig.taps], [fresh.point])
        self.assertEqual([item[0] for item in rig.completions], [fresh])
        self.assertEqual(rig.resets, 1)
        self.assertLess(rig.order.index(('reset', 1)), rig.order.index(('capture', 1)))
        self.assert_clean(rig)

    def test_frame_older_than_point_two_seconds_is_not_pressed(self):
        precise = [100.0]
        rig = SceneRig([Screen()], {})
        def slow(frame, now):
            precise[0] += .201
            return battle(BossAction('jump', (1090, 630)))
        rig.decisions[0] = slow
        with patch('ninja3_assistant.runtime.runner.time.perf_counter', side_effect=lambda: precise[0]):
            rig.run()
        self.assertEqual(rig.taps, [])
        self.assertEqual(rig.completions, [])
        self.assert_clean(rig)

    def test_pause_before_dispatch_prevents_touch_and_recaptures(self):
        action = BossAction('jump', (1090, 630))
        rig = SceneRig([Screen(), Screen()], {0: battle(action), 1: battle(action)})
        def emit(kind, value):
            if (kind, value) == ('backend', 'scene GPU'):
                rig.control.pause()
                rig.control.resume()
        rig.run(emit)
        self.assertEqual([item[2].index for item in rig.taps], [1])
        self.assertEqual(rig.resets, 1)
        self.assert_clean(rig)

    def test_press_before_stop_during_hold_is_still_completed(self):
        action = BossAction('jump', (1090, 630))
        rig = SceneRig([Screen()], {0: battle(action)})
        def stop_during_hold(stopped):
            stopped.set()
            return False
        rig.tap_hook = stop_during_hold
        rig.run()
        self.assertEqual([item[0] for item in rig.completions], [action])
        self.assertEqual(len(rig.taps), 1)
        self.assert_clean(rig)

    def test_metrics_count_actual_presses_even_when_tap_returns_false(self):
        clock = [100.0]
        action = BossAction('jump', (1090, 630))
        rig = SceneRig([Screen(), Screen(), Screen()], {},
                       control=RunControl(clock=lambda: clock[0]))
        def recognized(frame, now):
            clock[0] += 1.1
            return battle(action)
        rig.decisions = dict.fromkeys(range(3), recognized)
        rig.tap_hook = lambda stopped: False
        rig.run()
        metrics = [value for kind, value in rig.events if kind == 'metrics']
        self.assertEqual([item['clicks'] for item in metrics], [0, 1, 2])
        self.assertEqual(len(rig.completions), 3)
        self.assert_clean(rig)

    def test_touch_exception_records_error_after_successful_down(self):
        action = BossAction('jump', (1090, 630))
        rig = SceneRig([Screen()], {0: battle(action)})
        def failed_up(stopped):
            raise RuntimeError('touch up failed')
        rig.tap_hook = failed_up
        rig.run()
        self.assertEqual([item[0] for item in rig.completions], [action])
        self.assertEqual(rig.touch_errors, ['touch up failed'])
        self.assertEqual(len(rig.input_durations), 1)
        self.assertIn(('status', '已停止：touch up failed'), rig.events)
        self.assertTrue(rig.closed)
        self.assertTrue(rig.scene_closed)
        self.assertEqual(rig.events[-1], ('finished', ''))

    def test_scene_exception_stops_and_closes_both_resources(self):
        rig = SceneRig([Screen()], {})
        def failed(frame, now):
            raise RuntimeError('scene inference failed')
        rig.decisions[0] = failed
        rig.run()
        self.assertTrue(rig.closed)
        self.assertTrue(rig.scene_closed)
        self.assertIn(('status', '已停止：scene inference failed'), rig.events)
        self.assertEqual(rig.events[-1], ('finished', ''))
        self.assertEqual(rig.taps, [])


if __name__ == '__main__':
    unittest.main()
