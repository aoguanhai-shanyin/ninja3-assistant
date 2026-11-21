from ninja3_assistant.paths import project_root
from pathlib import Path
import subprocess
import unittest
import cv2
from ninja3_assistant.activities.weekend.runner import RaceVision, RaceAvoider, RaceSkills, remaining_matches
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from ninja3_assistant.runtime.runner import RunControl
from ninja3_assistant.activities.weekend.runner import run_weekend
from ninja3_assistant.scheduling.task_outcome import TaskOutcome

ROOT = project_root()

def frame(name):
    return cv2.cvtColor(cv2.imread(str(ROOT/'data/weekend-race/live'/name)),cv2.COLOR_BGR2RGB)

class WeekendRaceTest(unittest.TestCase):
    def test_unexpected_home_after_battle_cancels_input_and_yields_retry(self):
        control=RunControl()
        native,device,vision=Mock(),Mock(),Mock()
        native.movement_remaining.return_value=0.
        vision.detect.side_effect=['race','home']
        vision.confirmed_race=True
        vision.obscured=False
        vision.on_rope=False
        vision.opening.ground_frames=1
        device.capture.return_value=np.zeros((720,1280,3),np.uint8)
        with tempfile.TemporaryDirectory() as directory, \
             patch('ninja3_assistant.activities.weekend.runner.ROOT',Path(directory)), \
             patch('ninja3_assistant.activities.weekend.runner.prepare_game',return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RaceVision',return_value=vision), \
             patch('ninja3_assistant.activities.weekend.runner.ActivityVision'),patch('ninja3_assistant.activities.weekend.runner.WindowsOcr'), \
             patch('ninja3_assistant.activities.weekend.runner.RealtimeBattleKernel.recognition_due',return_value=True), \
             patch('ninja3_assistant.activities.weekend.runner.MuMu',return_value=device), \
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch',return_value=native):
            outcome=run_weekend('folder',1,control,lambda *a:None,mode='25',checkpoint=True)
        self.assertIs(outcome,TaskOutcome.ERROR)
        native.quiesce.assert_called()
        native.priority_tap.assert_not_called()
        self.assertFalse(control.stop.is_set())
    def test_resume_requires_actual_race_marker_and_never_skates_on_loading_portraits(self):
        for confirmed in (False,True):
            control = RunControl()
            image = np.zeros((720,1280,3),np.uint8)
            native,device,vision = Mock(),Mock(),Mock()
            native.movement_remaining.return_value=0.
            vision.detect.return_value = 'race'
            vision.confirmed_race = confirmed
            vision.obscured = False
            vision.on_rope = False
            vision.opening.ground_frames = 1
            captures = 0
            def capture():
                nonlocal captures
                captures += 1
                if captures == 2:
                    control.request_stop()
                return image
            device.capture.side_effect = capture
            with tempfile.TemporaryDirectory() as directory, \
                 patch('ninja3_assistant.activities.weekend.runner.ROOT',Path(directory)), \
                 patch('ninja3_assistant.activities.weekend.runner.prepare_game',return_value=False), \
                 patch('ninja3_assistant.activities.weekend.runner.RaceVision',return_value=vision), \
                 patch('ninja3_assistant.activities.weekend.runner.ActivityVision'), patch('ninja3_assistant.activities.weekend.runner.WindowsOcr'), \
                 patch('ninja3_assistant.activities.weekend.runner.MuMu',return_value=device), \
                 patch('ninja3_assistant.activities.weekend.runner.create_native_touch',return_value=native):
                run_weekend('folder',1,control,lambda *a:None)
            self.assertEqual(native.schedule_movement.call_count,1 if confirmed else 0)

    def test_respawn_recovery_jumps_before_resuming_slide(self):
        control = RunControl()
        image = np.zeros((720,1280,3),np.uint8)
        native,device,vision = Mock(),Mock(),Mock()
        native.movement_remaining.return_value=0.
        vision.detect.side_effect = ['ready','ready','race','respawn','race','unknown']
        vision.opening.ground_frames = 1
        vision.obscured = False
        vision.on_rope = False
        captures = 0
        def capture():
            nonlocal captures
            captures += 1
            if captures == 6:
                control.request_stop()
            return image
        device.capture.side_effect = capture
        with tempfile.TemporaryDirectory() as directory, \
             patch('ninja3_assistant.activities.weekend.runner.ROOT',Path(directory)), \
             patch('ninja3_assistant.activities.weekend.runner.prepare_game',return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RealtimeBattleKernel.recognition_due',return_value=True), \
             patch('ninja3_assistant.activities.weekend.runner.RaceVision',return_value=vision), \
             patch('ninja3_assistant.activities.weekend.runner.ActivityVision'), \
             patch('ninja3_assistant.activities.weekend.runner.WindowsOcr'), \
             patch('ninja3_assistant.activities.weekend.runner.MuMu',return_value=device), \
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch',return_value=native) as create_touch:
            run_weekend('folder',1,control,lambda *a:None)
        self.assertEqual(create_touch.call_args.kwargs['movement'],'jump-two-slide')
        native.quiesce.assert_any_call(reason='respawn')
        jumps = [c for c in native.priority_tap.call_args_list if c.args[0] == (1090,630)]
        self.assertEqual(len(jumps),1)
        self.assertTrue(jumps[0].kwargs['preserve_slide'])
        self.assertEqual(native.schedule_movement.call_count,1)
        native.release.assert_not_called()

    def test_rope_jumps_on_its_own_finger_without_ground_slide(self):
        control = RunControl(clock=Mock(side_effect=range(100)))
        image = np.zeros((720,1280,3),np.uint8)
        native, device, vision = Mock(), Mock(), Mock()
        native.movement_remaining.return_value=0.
        vision.detect.side_effect = ['ready','ready','race','race','race','unknown']
        vision.opening.ground_frames = 0
        vision.on_rope = True
        vision.obscured = False
        captures = 0
        def capture():
            nonlocal captures
            captures += 1
            if captures == 6:
                control.request_stop()
            return image
        device.capture.side_effect = capture
        with tempfile.TemporaryDirectory() as directory, \
             patch('ninja3_assistant.activities.weekend.runner.ROOT',Path(directory)), \
             patch('ninja3_assistant.activities.weekend.runner.prepare_game',return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RealtimeBattleKernel.recognition_due',return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RaceVision',return_value=vision), \
             patch('ninja3_assistant.activities.weekend.runner.ActivityVision'), patch('ninja3_assistant.activities.weekend.runner.WindowsOcr'), \
             patch('ninja3_assistant.activities.weekend.runner.MuMu',return_value=device), \
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch',return_value=native):
            run_weekend('folder',1,control,lambda *a:None)
        native.schedule_movement.assert_not_called()
        self.assertEqual(native.release.call_count,1)
        jumps = [call for call in native.plan.call_args_list if call.args[0] == 3]
        self.assertTrue(jumps)
        self.assertTrue(all(action.point == (1090,630) for call in jumps for action in call.args[1]))
        self.assertTrue(any(call.args[0] == 4 for call in native.plan.call_args_list))
        self.assertFalse(any(c.args[0] == (1090,630) for c in native.priority_tap.call_args_list))

    def test_leaving_rope_cancels_rope_jumps_before_resuming_ground_movement(self):
        control = RunControl()
        native,device,vision = Mock(),Mock(),Mock()
        native.movement_remaining.return_value = 0.
        image = np.zeros((720,1280,3),np.uint8)
        captures = 0
        def capture():
            nonlocal captures
            captures += 1
            if captures == 3:
                control.request_stop()
            return image
        def detect(*_):
            vision.on_rope = captures == 1
            vision.opening.ground_frames = 0 if vision.on_rope else 1
            return 'race'
        device.capture.side_effect = capture
        vision.detect.side_effect = detect
        vision.confirmed_race = True
        vision.obscured = False
        with tempfile.TemporaryDirectory() as directory, \
             patch('ninja3_assistant.activities.weekend.runner.ROOT',Path(directory)), \
             patch('ninja3_assistant.activities.weekend.runner.prepare_game',return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RealtimeBattleKernel.recognition_due',return_value=True), \
             patch('ninja3_assistant.activities.weekend.runner.RaceVision',return_value=vision), \
             patch('ninja3_assistant.activities.weekend.runner.ActivityVision'),patch('ninja3_assistant.activities.weekend.runner.WindowsOcr'), \
             patch('ninja3_assistant.activities.weekend.runner.RaceRopeTraversal.upper_platform',return_value=(550,350,160,25)), \
             patch('ninja3_assistant.activities.weekend.runner.MuMu',return_value=device), \
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch',return_value=native):
            run_weekend('folder',1,control,lambda *args:None)
        jumps = [call for call in native.plan.call_args_list if call.args[0] == 3]
        self.assertEqual(len(jumps[0].args[1]),2)
        native.cancel_plan.assert_called_once_with(3)
        native.schedule_movement.assert_called_once()
        names = [call[0] for call in native.method_calls]
        self.assertLess(names.index('cancel_plan'),names.index('schedule_movement'))

    def test_both_rope_directions_disable_ground_slide(self):
        vision = RaceVision()
        for arrow in (vision.rope_down,vision.rope_up):
            image = np.zeros((720,1280,3),np.uint8)
            h,w = arrow.shape
            image[590:590+h,165:165+w] = arrow[:,:,None]
            vision.detect(image)
            self.assertTrue(vision.on_rope)
            self.assertEqual(vision.opening.ground_frames,0)

    def test_hall_entry_requires_the_race_card_on_the_refreshed_frame(self):
        control = RunControl()
        device, native, vision = Mock(), Mock(), Mock()
        hall = np.zeros((720, 1280, 3), np.uint8)
        different_hall_view = np.full_like(hall, 255)
        vision.detect.return_value = 'hall'
        vision.match.side_effect = lambda frame, *_: frame is hall

        captures = iter((hall, different_hall_view))

        def capture():
            frame = next(captures, None)
            if frame is not None:
                return frame
            control.request_stop()
            return different_hall_view

        device.capture.side_effect = capture
        with tempfile.TemporaryDirectory() as directory, \
             patch('ninja3_assistant.activities.weekend.runner.ROOT', Path(directory)), \
             patch('ninja3_assistant.activities.weekend.runner.prepare_game', return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RaceVision', return_value=vision), \
             patch('ninja3_assistant.activities.weekend.runner.ActivityVision'), \
             patch('ninja3_assistant.activities.weekend.runner.WindowsOcr'), \
             patch('ninja3_assistant.activities.weekend.runner.MuMu', return_value=device), \
             patch('ninja3_assistant.activities.weekend.runner.subprocess.run') as swipe, \
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch', return_value=native):
            outcome = run_weekend('folder', 1, control, lambda *_: None)

        self.assertIs(outcome, TaskOutcome.STOPPED)
        self.assertIs(vision.match.call_args_list[1].args[0], different_hall_view)
        native.priority_tap.assert_not_called()
        swipe.assert_not_called()

    def test_zero_remaining_stops_before_sending_match(self):
        vision = Mock()
        vision.detect.side_effect = ['tasks','tasks','home']
        vision.claimable.return_value = False
        vision.claimed.return_value = True
        vision.match.return_value = True
        ocr = Mock()
        ocr.read.return_value = {'text':'剩余0次'}
        device,native = Mock(),Mock()
        device.capture.return_value = np.zeros((720,1280,3),np.uint8)
        events = []
        with tempfile.TemporaryDirectory() as directory, \
             patch('ninja3_assistant.activities.weekend.runner.ROOT',Path(directory)), \
             patch('ninja3_assistant.activities.weekend.runner.prepare_game',return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RaceVision',return_value=vision), \
             patch('ninja3_assistant.activities.weekend.runner.ActivityVision'), \
             patch('ninja3_assistant.activities.weekend.runner.WindowsOcr',return_value=ocr), \
             patch('ninja3_assistant.activities.weekend.runner.MuMu',return_value=device), \
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch',return_value=native):
            outcome = run_weekend('folder',1,RunControl(clock=Mock(side_effect=range(0,40,2))),lambda k,v:events.append((k,v)),'25')
        self.assertIs(outcome, TaskOutcome.LIMIT_REACHED)
        self.assertIn(('status','今日25场积分比赛已用完，任务结束'),events)
        self.assertTrue(all(call.args[0] != (1140,677) for call in native.priority_tap.call_args_list))
        native.close.assert_called_once()
    def terrain(self):
        image = np.full((720,1280,3),180,np.uint8)
        image[540:600] = 50
        image[550:580:4] = 100
        return image

    def test_stationary_wall_is_not_missed_when_motion_stops(self):
        image = self.terrain()
        image[440:540,650:700] = 0
        avoider = RaceAvoider()
        self.assertIsNone(avoider.decide(image,0))
        self.assertEqual(avoider.decide(image,.02),(1090,630))
        self.assertTrue(avoider.blocked)
        self.assertGreater(avoider.air_until,.1)
        self.assertIsNone(avoider.decide(image,.4))
        self.assertEqual(avoider.decide(image,.6),(1090,630))

    def test_forward_wall_is_detected_before_contact(self):
        image = self.terrain()
        image[440:540,780:830] = 0
        avoider = RaceAvoider()
        self.assertIsNone(avoider.decide(image,0))
        self.assertEqual(avoider.decide(image,.02),(1090,630))

    def test_avoidance_tracks_local_health_bar_when_player_falls_behind(self):
        image = self.terrain()
        image[399:404,385:417] = 20
        image[400:403,386:410] = (238,198,52)
        image[440:540,580:630] = 0
        avoider = RaceAvoider()
        avoider.decide(image,0)
        self.assertEqual(avoider.player_x,400)
        self.assertEqual(avoider.decide(image,.02),(1090,630))

    def test_yellow_scenery_does_not_move_local_player_anchor(self):
        image = self.terrain()
        image[400:403,240:264] = (238,198,52)
        avoider = RaceAvoider()
        avoider.decide(image,0)
        self.assertEqual(avoider.player_x,600)

    def test_overhead_platform_is_not_used_as_the_players_floor(self):
        image = self.terrain()
        image[350:430] = 0
        image[449:454,585:617] = 20
        image[450:453,586:610] = (238,198,52)
        avoider = RaceAvoider()
        avoider.decide(image,0)
        self.assertTrue(avoider.player_tracked)
        self.assertGreater(avoider.ground,500)

    def test_actual_health_bar_accepts_light_outline(self):
        avoider = RaceAvoider()
        avoider.decide(frame('20261114-170712-recovery-kernel.png'),0)
        self.assertEqual((avoider.player_x,avoider.player_y),(608,458))

    def test_pit_stops_slide_before_jump_and_flat_road_allows_slide(self):
        image = self.terrain()
        avoider = RaceAvoider()
        self.assertIsNone(avoider.decide(image,0))
        self.assertTrue(avoider.road_confirmed)
        image[530:600,700:940] = 180
        self.assertIsNone(avoider.decide(image,1))
        self.assertEqual(avoider.decide(image,1.02),(1090,630))
        self.assertEqual(avoider.hazard,'pit')
        self.assertGreater(avoider.air_until,1)

    def test_both_weapons_are_scheduled_with_cards(self):
        skills = RaceSkills()
        self.assertTrue(skills.due(0))
        skills.scheduled(0)
        self.assertFalse(skills.due(.5))
        self.assertTrue(skills.due(.6))
        points = [a.point for a in skills.actions]
        self.assertIn((380,630),points)
        self.assertIn((900,630),points)
        self.assertIn((740,630),points)
        self.assertEqual(len(set(points)),5)

    def test_unobserved_road_under_effect_does_not_invent_a_pit(self):
        avoider = RaceAvoider()
        avoider.decide(self.terrain(),0)
        gap = np.full((720,1280,3),180,np.uint8)
        self.assertIsNone(avoider.decide(gap,.1))
        self.assertIsNone(avoider.decide(gap,.12))
        self.assertIsNone(avoider.hazard)

    def test_actual_players_on_flat_road_are_not_a_wall(self):
        path = ROOT/'data/evidence/2026-11-15/weekend-race/flat-road-players.png'
        image = cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        avoider = RaceAvoider()
        avoider.decide(image,0)
        self.assertIsNone(avoider.decide(image,.02))
        self.assertIsNone(avoider.hazard)

    def test_standings_animation_is_never_a_battle(self):
        path = ROOT/'data/evidence/2026-11-15/weekend-race/standings-animation.png'
        image = cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        self.assertEqual(RaceVision().detect(image),'standings')

    def test_inventory_portraits_never_authorize_battle_input(self):
        path = ROOT/'data/evidence/2026-11-15/weekend-race/inventory.png'
        image = cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        vision = RaceVision()
        self.assertEqual(vision.detect(image),'unknown')
        self.assertFalse(vision.confirmed_race)

    def test_smoke_card_is_not_death_and_keeps_battle_owned(self):
        path=ROOT/'data/evidence/2026-11-15/weekend-race/smoke.png'
        image=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        vision=RaceVision()
        self.assertEqual(vision.detect(image),'race')
        self.assertTrue(vision.obscured)

    def test_counter_is_daily_reward_allowance_not_session_count(self):
        for text, expected in [('可获得积分场次剩余21次',21),('剩余0次',0),('剩余25次',25),('剩余26次',None),('未知',None)]:
            self.assertEqual(remaining_matches(text),expected)

    def test_actual_daily_requires_all_five_claimed(self):
        vision = RaceVision()
        self.assertFalse(vision.claimed(frame('20261114-143242-tasks.png')))
        image = frame('20261114-143320-claimed.png')
        self.assertTrue(vision.claimed(image))
        self.assertFalse(vision.claimable(image))
        self.assertTrue(vision.claimable(frame('20261114-143242-tasks.png')))
        image[525:580,925:1070] = 0
        self.assertFalse(vision.claimed(image))

    def test_rotation_does_not_match_other_activity(self):
        vision = RaceVision()
        image = cv2.cvtColor(cv2.imread(str(ROOT/'data/intelligence-simulation/live/20261113-140318-guide-close.png')),cv2.COLOR_BGR2RGB)
        self.assertEqual(vision.detect(image),'unknown')
        self.assertEqual(vision.detect(frame('20261114-143223-entry.png')),'home')
        self.assertEqual(vision.detect(frame('runtime-home.png')),'home')
        self.assertTrue(vision.match(frame('runtime-home.png'),'match',(1030,655,230,55)))
        self.assertEqual(vision.detect(frame('20261114-144041-result-detail.png')),'result')

    def test_skating_jump_icon_pose_change_remains_battle(self):
        vision = RaceVision()
        self.assertEqual(vision.detect(frame('current-fix-check.png')),'race')
        self.assertEqual(vision.detect(frame('sample-94.png')),'race')

    def test_super_start_button_is_recognized_through_countdown(self):
        vision = RaceVision()
        folder = ROOT/'data/weekend-race/runs/20261114-164702'
        for name in ('1794646088523972500-startup.png','1794646090232204500-startup.png'):
            image = cv2.cvtColor(cv2.imread(str(folder/name)),cv2.COLOR_BGR2RGB)
            self.assertEqual(vision.detect(image),'dash')

    def test_death_is_distinguished_from_running_when_movement_buttons_disappear(self):
        path = ROOT/'data/weekend-race/runs/20261114-165630/1794646780212204400-unknown.png'
        image = cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        self.assertEqual(RaceVision().detect(image),'respawn')

    def test_real_flat_road_with_skill_effects_does_not_invent_pits(self):
        for name in ('20261114-160545-latest.png',
                     '20261114-160046-glide-check.png'):
            with self.subTest(name=name):
                avoider = RaceAvoider()
                image = frame(name)
                avoider.decide(image,0)
                self.assertIsNone(avoider.decide(image,.02))
                self.assertIsNone(avoider.hazard)
                self.assertTrue(avoider.road_confirmed)

    def test_native_navigation_swipe_releases_and_validates_bounds(self):
        executable = ROOT/'build/ninja-touch/NinjaTouch.exe'
        run = subprocess.run([str(executable),'--mock','--swipe','1150','410','300','410'],capture_output=True,text=True,timeout=5)
        self.assertEqual(run.returncode,0)
        self.assertIn('SWIPED',run.stdout)
        run = subprocess.run([str(executable),'--mock','--swipe','1280','410','300','410'],capture_output=True,timeout=5)
        self.assertEqual(run.returncode,2)
