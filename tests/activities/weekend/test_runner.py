from ninja3_assistant.paths import project_root
from pathlib import Path
import subprocess
import unittest
import cv2
from ninja3_assistant.activities.weekend.runner import RaceVision, RaceAvoider, RaceSkills, remaining_matches, WeekendProgress
from ninja3_assistant.activities.weekend import runner as weekend_runner
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from ninja3_assistant.runtime.runner import RunControl
from ninja3_assistant.activities.weekend.runner import run_weekend
from ninja3_assistant.scheduling.task_outcome import TaskOutcome
from ninja3_assistant.scheduling.invitation_handoff import InvitationHandoff
from ninja3_assistant.vision.invitation import BountyInvitation

ROOT = project_root()

def frame(name):
    return cv2.cvtColor(cv2.imread(str(ROOT/'data/weekend-race/live'/name)),cv2.COLOR_BGR2RGB)


def current_frame(name):
    path = ROOT/'data/evidence/2026-11-21/weekend-activity'/name
    return cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)


def no_obstacles():
    """FrameScores stand-in: a race frame with nothing to jump over."""
    scores = Mock()
    scores.obstacles.return_value = ()
    scores.matched.return_value = False
    scores.score.return_value = 0.
    return scores

class WeekendRaceTest(unittest.TestCase):
    def test_unexpected_home_after_battle_cancels_input_and_yields_retry(self):
        control=RunControl()
        native,device,vision=Mock(),Mock(),Mock()
        native.movement_remaining.return_value=0.
        vision.detect.side_effect=['race','home']
        vision.confirmed_race=True
        vision.obscured=False
        vision.ninjutsu_ready=False
        vision.observe.return_value=no_obstacles()
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
            vision.observe.return_value = no_obstacles()
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

    def test_same_screen_revival_hops_out_of_the_hazard(self):
        """同屏竞速死亡后会自己复活，复活瞬间先跳出危险区。"""
        control = RunControl()
        image = np.zeros((720,1280,3),np.uint8)
        native,device,vision = Mock(),Mock(),Mock()
        native.movement_remaining.return_value=0.
        vision.detect.side_effect = ['ready','ready','race','respawn','race','unknown']
        vision.opening.ground_frames = 1
        vision.obscured = False
        vision.observe.return_value = no_obstacles()
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
            run_weekend('folder',1,control,lambda *a:None,activity='samescreen')
        self.assertEqual(create_touch.call_args.kwargs['movement'],'jump-two-slide')
        native.quiesce.assert_any_call(reason='respawn')
        jumps = [c for c in native.priority_tap.call_args_list if c.args[0] == (1090,630)]
        self.assertEqual(len(jumps),1)
        self.assertTrue(jumps[0].kwargs['preserve_slide'])
        self.assertGreaterEqual(native.schedule_movement.call_count,1)

    def test_race_plans_skills_on_one_finger_and_keeps_movement_in_the_kernel(self):
        control = RunControl(clock=Mock(side_effect=range(100)))
        image = np.zeros((720,1280,3),np.uint8)
        native, device, vision = Mock(), Mock(), Mock()
        native.movement_remaining.return_value=0.
        vision.detect.side_effect = ['ready','ready','race','race','race','unknown']
        vision.opening.ground_frames = 0
        vision.on_rope = True
        vision.obscured = False
        vision.ninjutsu_ready = False
        vision.observe.return_value = no_obstacles()
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
        # Every race frame keeps the ground slide lease alive through the
        # kernel; a jump is a bounded tap, never a second movement finger.
        self.assertGreaterEqual(native.schedule_movement.call_count,1)
        self.assertEqual({call.args[0] for call in native.plan.call_args_list},{4})
        self.assertTrue(all(action.offset_ms < 200
                            for call in native.plan.call_args_list
                            for action in call.args[1]))
        self.assertFalse(any(c.args[0] == (1090,630) for c in native.priority_tap.call_args_list))

    def test_always_visible_slide_arrow_cannot_confirm_a_race(self):
        vision = RaceVision()
        arrow = cv2.imread(str(ROOT/'assets/weekend-race/rope-down.png'), cv2.IMREAD_GRAYSCALE)
        image = np.zeros((720,1280,3),np.uint8)
        height, width = arrow.shape
        image[590:590+height,165:165+width] = arrow[:,:,None]
        self.assertEqual(vision.detect(image),'unknown')
        self.assertFalse(vision.confirmed_race)

    def test_race_start_skates_without_waiting_for_a_super_start(self):
        """The rotation has no super start: a confirmed race starts skating at once."""
        control = RunControl()
        image = np.zeros((720,1280,3),np.uint8)
        native,device,vision = Mock(),Mock(),Mock()
        native.movement_remaining.return_value=0.
        vision.detect.side_effect = ['ready','respawn','race','race']
        vision.confirmed_race = True
        vision.obscured = False
        vision.observe.return_value = no_obstacles()
        vision.on_rope = False
        vision.opening.ground_frames = 1
        captures = 0
        def capture():
            nonlocal captures
            captures += 1
            if captures == 4:
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
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch',return_value=native):
            run_weekend('folder',1,control,lambda *a:None)
        self.assertFalse(any(c.kwargs.get('reason') == 'respawn'
                             for c in native.quiesce.call_args_list))
        self.assertGreaterEqual(native.schedule_movement.call_count,1)

    def test_hall_entry_requires_the_race_card_on_the_refreshed_frame(self):
        control = RunControl()
        device, native, vision = Mock(), Mock(), Mock()
        hall = np.zeros((720, 1280, 3), np.uint8)
        different_hall_view = np.full_like(hall, 255)
        vision.detect.return_value = 'hall'
        vision.match.side_effect = lambda frame, *_: frame is hall
        vision.match_any.side_effect = lambda frame, *_: frame is hall

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
        self.assertIs(vision.match_any.call_args_list[1].args[0], different_hall_view)
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
        self.assertIn(('status','今日积分场次已用完，任务结束'),events)
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

    def test_water_surface_does_not_invent_pits(self):
        """The water map has no continuous road, so a missing row is not a pit."""
        avoider = RaceAvoider()
        image = self.terrain()
        self.assertIsNone(avoider.decide(image,0))
        image[530:600,700:940] = 180
        self.assertIsNone(avoider.decide(image,1))
        self.assertIsNone(avoider.hazard)

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

    def test_weapon_and_cards_share_one_bounded_plan(self):
        skills = RaceSkills()
        self.assertTrue(skills.due(0))
        skills.scheduled(0)
        self.assertFalse(skills.due(.5))
        self.assertTrue(skills.due(.6))
        cooling = [a.point for a in RaceSkills.actions(False)]
        armed = RaceSkills.actions(True)
        self.assertEqual(cooling,[(900,630),(520,630),(635,630),(740,630)])
        self.assertEqual([a.point for a in armed],[(380,630),*cooling])
        self.assertTrue(all(action.offset_ms < 200 for action in armed))

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
        for text, expected in [('可获得积分场次剩余21次',21),('剩余0次',0),('剩余25次',25),('剩余26次',26),('未知',None)]:
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

    def test_countdown_frames_keep_the_race_owned_without_a_super_start_state(self):
        vision = RaceVision()
        folder = ROOT/'data/weekend-race/runs/20261114-164702'
        for name in ('1794646088523972500-startup.png','1794646090232204500-startup.png'):
            image = cv2.cvtColor(cv2.imread(str(folder/name)),cv2.COLOR_BGR2RGB)
            self.assertIn(vision.detect(image),('race','respawn'))
            self.assertTrue(vision.confirmed_race)

    def test_death_is_distinguished_from_running_when_movement_buttons_disappear(self):
        """死亡时跳跃图标消失；要连续两帧，避免屋顶遮挡的镜头被当成倒地。"""
        path = ROOT/'data/weekend-race/runs/20261114-165630/1794646780212204400-unknown.png'
        image = cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        vision = RaceVision()
        self.assertEqual(vision.detect(image),'race')
        self.assertEqual(vision.detect(image),'respawn')
        self.assertGreaterEqual(vision.hidden_frames,2)
        running = frame('current-fix-check.png')
        steady = RaceVision()
        self.assertEqual(steady.detect(running),'race')
        self.assertEqual(steady.detect(running),'race')
        self.assertEqual(steady.hidden_frames,0)

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

    def test_recognizes_entry_intro_home_tasks_and_ready(self):
        vision = RaceVision()
        hall = current_frame('15-hall-two-swipes.png')
        self.assertEqual(vision.detect(hall), 'hall')
        self.assertTrue(vision.match_any(hall, ('entry', 'entry-current'), weekend_runner.HALL_ENTRY_REGION))

        intro = current_frame('17-enter-activity.png')
        self.assertEqual(vision.detect(intro), 'home')
        self.assertTrue(vision.intro)
        self.assertEqual(vision.layout, 'intro')

        home = current_frame('18-activity-home.png')
        self.assertEqual(vision.detect(home), 'home')
        self.assertFalse(vision.intro)
        self.assertEqual(vision.layout, 'current')

        tasks = current_frame('19-daily-tasks.png')
        self.assertEqual(vision.detect(tasks), 'tasks')
        self.assertEqual(vision.layout, 'current')

        ready = current_frame('24-start-match.png')
        self.assertEqual(vision.detect(ready), 'ready')
        self.assertEqual(vision.detect(current_frame('result-flow-01/005.png')), 'result')
        self.assertEqual(vision.detect(current_frame('26-restart-scene.png')), 'summary')
        self.assertEqual(vision.detect(current_frame('restart-return/005.png')), 'result')
        self.assertEqual(vision.detect(current_frame('tasks-claim/007.png')), 'reward')

    def test_settlement_ranking_view_is_recognized_instead_of_unknown(self):
        """结算页关闭后停在观战排行榜：必须有模板，否则 15 秒 unknown 超时。"""
        vision = RaceVision()
        image = current_frame('40-after-result.png')
        self.assertEqual(vision.detect(image),'ranking')
        self.assertEqual(vision.layout,'current')

    def test_settlement_is_recorded_even_when_the_confirm_tap_misses(self):
        """结算页约 1 秒自动关闭：看到结算页就应记一场，而不是依赖点击成功。"""
        control = RunControl()
        image = np.zeros((720,1280,3),np.uint8)
        native,device,vision = Mock(),Mock(),Mock()
        native.movement_remaining.return_value = 0.
        vision.detect.side_effect = ['ready','race','result','home']
        vision.confirmed_race = True
        vision.obscured = False
        vision.observe.return_value = no_obstacles()
        device.capture.return_value = image
        ui = Mock()
        ui.perform.return_value = False
        progress = WeekendProgress(day='2026-11-21',matches=0)
        with tempfile.TemporaryDirectory() as directory, \
             patch('ninja3_assistant.activities.weekend.runner.ROOT',Path(directory)), \
             patch('ninja3_assistant.activities.weekend.runner.prepare_game',return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RaceVision',return_value=vision), \
             patch('ninja3_assistant.activities.weekend.runner.ActivityVision'), \
             patch('ninja3_assistant.activities.weekend.runner.WindowsOcr'), \
             patch('ninja3_assistant.activities.weekend.runner.GameUI',return_value=ui), \
             patch('ninja3_assistant.activities.weekend.runner.MuMu',return_value=device), \
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch',return_value=native):
            outcome = run_weekend('folder',1,control,lambda *a:None,'25',
                                  checkpoint=True,progress=progress)
        self.assertIs(outcome,TaskOutcome.MATCH_DONE)
        self.assertEqual(progress.matches,1)

    def test_settlement_is_recorded_when_the_board_never_shows_a_result_page(self):
        """实机结算走 finish -> standings -> home，整条链路没有 result 页。

        只认 result 页记账会让每局都漏记，25 场计数因此一直是 0。
        """
        control = RunControl()
        image = np.zeros((720,1280,3),np.uint8)
        native,device,vision = Mock(),Mock(),Mock()
        native.movement_remaining.return_value = 0.
        vision.detect.side_effect = ['ready','race','finish','standings','home']
        vision.confirmed_race = True
        vision.obscured = False
        vision.observe.return_value = no_obstacles()
        device.capture.return_value = image
        ui = Mock()
        ui.perform.return_value = False
        progress = WeekendProgress(day='2026-11-21',matches=0)
        with tempfile.TemporaryDirectory() as directory, \
             patch('ninja3_assistant.activities.weekend.runner.ROOT',Path(directory)), \
             patch('ninja3_assistant.activities.weekend.runner.prepare_game',return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RaceVision',return_value=vision), \
             patch('ninja3_assistant.activities.weekend.runner.ActivityVision'), \
             patch('ninja3_assistant.activities.weekend.runner.WindowsOcr'), \
             patch('ninja3_assistant.activities.weekend.runner.GameUI',return_value=ui), \
             patch('ninja3_assistant.activities.weekend.runner.MuMu',return_value=device), \
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch',return_value=native):
            outcome = run_weekend('folder',1,control,lambda *a:None,'25',
                                  checkpoint=True,progress=progress)
        self.assertIs(outcome,TaskOutcome.MATCH_DONE)
        self.assertEqual(progress.matches,1)

    def test_spectate_home_is_driven_through_its_own_button(self):
        """观战态由「进入观战」驱动：点它进结算页，确认后回到可开始的主页。

        退回集会所会陷入 hall→活动 死循环；实测干等 2~8 分钟且有时不恢复。
        """
        control = RunControl(clock=Mock(side_effect=range(0,60,2)))
        image = np.zeros((720,1280,3),np.uint8)
        native,device,vision = Mock(),Mock(),Mock()
        native.movement_remaining.return_value = 0.
        vision.detect.side_effect = ['ready','race','ranking','ranking','home']
        vision.confirmed_race = True
        vision.obscured = False
        vision.observe.return_value = no_obstacles()
        device.capture.return_value = image
        ui = Mock()
        ui.perform.return_value = False
        progress = WeekendProgress(day='2026-11-21',matches=0)
        with tempfile.TemporaryDirectory() as directory, \
             patch('ninja3_assistant.activities.weekend.runner.ROOT',Path(directory)), \
             patch('ninja3_assistant.activities.weekend.runner.prepare_game',return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RaceVision',return_value=vision), \
             patch('ninja3_assistant.activities.weekend.runner.ActivityVision'), \
             patch('ninja3_assistant.activities.weekend.runner.WindowsOcr'), \
             patch('ninja3_assistant.activities.weekend.runner.GameUI',return_value=ui), \
             patch('ninja3_assistant.activities.weekend.runner.MuMu',return_value=device), \
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch',return_value=native):
            outcome = run_weekend('folder',1,control,lambda *a:None,'25',
                                  checkpoint=True,progress=progress)
        self.assertIs(outcome,TaskOutcome.MATCH_DONE)
        # The spectate state also proves the match ended, so it must count.
        self.assertEqual(progress.matches,1)
        taps = [entry.args[1] for entry in ui.perform.call_args_list if len(entry.args) > 1]
        self.assertIn('enter_spectate',taps)
        self.assertNotIn('leave_ranking',taps)

    def test_spectator_trace_page_uses_its_own_back_button(self):
        """观战轨迹页的返回在 (540,570)，赛后轨迹页的 (640,667) 点不中它。"""
        control = RunControl(clock=Mock(side_effect=range(0,80,2)))
        image = np.zeros((720,1280,3),np.uint8)
        native,device,vision = Mock(),Mock(),Mock()
        native.movement_remaining.return_value = 0.
        vision.detect.side_effect = ['ready','race','ranking','spectate','finish','home']
        vision.confirmed_race = True
        vision.obscured = False
        vision.observe.return_value = no_obstacles()
        device.capture.return_value = image
        ui = Mock()
        ui.perform.return_value = False
        progress = WeekendProgress(day='2026-11-21',matches=0)
        with tempfile.TemporaryDirectory() as directory, \
             patch('ninja3_assistant.activities.weekend.runner.ROOT',Path(directory)), \
             patch('ninja3_assistant.activities.weekend.runner.prepare_game',return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RaceVision',return_value=vision), \
             patch('ninja3_assistant.activities.weekend.runner.ActivityVision'), \
             patch('ninja3_assistant.activities.weekend.runner.WindowsOcr'), \
             patch('ninja3_assistant.activities.weekend.runner.GameUI',return_value=ui), \
             patch('ninja3_assistant.activities.weekend.runner.MuMu',return_value=device), \
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch',return_value=native):
            outcome = run_weekend('folder',1,control,lambda *a:None,'25',
                                  checkpoint=True,progress=progress)
        self.assertIs(outcome,TaskOutcome.MATCH_DONE)
        self.assertEqual(progress.matches,1)
        taps = [entry.args[1] for entry in ui.perform.call_args_list if len(entry.args) > 1]
        self.assertIn('enter_spectate',taps)
        self.assertIn('spectate_back',taps)
        self.assertNotIn('confirm_summary',taps)

    def test_relay_panel_is_recognized_before_the_race(self):
        """死亡弹框压在竞速 HUD 上：必须先判接力，否则整局都在往弹框送操作。

        标题亮绿占比在两块地图是 0.30 / 0.23，九张页面和所有竞速帧都是
        0.0000。模板匹配做不到这件事——平坦横幅用 NCC 会假阳性 1.00。
        """
        vision = RaceVision()
        self.assertEqual(vision.detect(current_frame('58-relay-bamboo.png')), 'relay')
        self.assertTrue(vision.confirmed_race)
        self.assertEqual(RaceVision().detect(current_frame('59-relay-cave.png')), 'relay')
        self.assertEqual(RaceVision().detect(current_frame('18-activity-home.png')), 'home')
        self.assertEqual(RaceVision().detect(current_frame('19-daily-tasks.png')), 'tasks')
        self.assertEqual(RaceVision().detect(frame('current-fix-check.png')), 'race')

    def test_relay_panel_is_skipped_instead_of_joined(self):
        """用户规则：死亡出接力弹框，不接了就行——跳过并等本局结算。"""
        control = RunControl(clock=Mock(side_effect=range(0,60,2)))
        image = np.zeros((720,1280,3),np.uint8)
        native,device,vision = Mock(),Mock(),Mock()
        native.movement_remaining.return_value = 0.
        vision.detect.side_effect = ['ready','race','relay','relay','finish','home']
        vision.confirmed_race = True
        vision.obscured = False
        vision.observe.return_value = no_obstacles()
        device.capture.return_value = image
        ui = Mock()
        ui.perform.return_value = False
        progress = WeekendProgress(day='2026-11-21',matches=0)
        with tempfile.TemporaryDirectory() as directory, \
             patch('ninja3_assistant.activities.weekend.runner.ROOT',Path(directory)), \
             patch('ninja3_assistant.activities.weekend.runner.prepare_game',return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RaceVision',return_value=vision), \
             patch('ninja3_assistant.activities.weekend.runner.ActivityVision'), \
             patch('ninja3_assistant.activities.weekend.runner.WindowsOcr'), \
             patch('ninja3_assistant.activities.weekend.runner.GameUI',return_value=ui), \
             patch('ninja3_assistant.activities.weekend.runner.MuMu',return_value=device), \
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch',return_value=native):
            outcome = run_weekend('folder',1,control,lambda *a:None,'25',
                                  checkpoint=True,progress=progress)
        self.assertIs(outcome,TaskOutcome.MATCH_DONE)
        self.assertEqual(progress.matches,1)
        taps = [entry.args[1] for entry in ui.perform.call_args_list if len(entry.args) > 1]
        self.assertIn('skip_relay',taps)
        # Joining the relay would keep sending movement into the panel.
        self.assertFalse(any(call.args and call.args[0] == (1090,630)
                             for call in native.priority_tap.call_args_list))

    def test_match_goal_stops_the_daily_phase_and_updates_the_ui_counter(self):
        """选「完成 4 场」时第一阶段也要在第 4 场停；WinUI 计数不能恒为 0。"""
        control = RunControl(clock=Mock(side_effect=range(0,60,2)))
        image = np.zeros((720,1280,3),np.uint8)
        native,device,vision = Mock(),Mock(),Mock()
        native.movement_remaining.return_value = 0.
        vision.detect.side_effect = ['ready','race','finish','home']
        vision.confirmed_race = True
        vision.obscured = False
        vision.observe.return_value = no_obstacles()
        device.capture.return_value = image
        ui = Mock()
        ui.perform.return_value = False
        progress = WeekendProgress(day='2026-11-21',matches=3)
        events = []
        with tempfile.TemporaryDirectory() as directory, \
             patch('ninja3_assistant.activities.weekend.runner.ROOT',Path(directory)), \
             patch('ninja3_assistant.activities.weekend.runner.prepare_game',return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RaceVision',return_value=vision), \
             patch('ninja3_assistant.activities.weekend.runner.ActivityVision'), \
             patch('ninja3_assistant.activities.weekend.runner.WindowsOcr'), \
             patch('ninja3_assistant.activities.weekend.runner.GameUI',return_value=ui), \
             patch('ninja3_assistant.activities.weekend.runner.MuMu',return_value=device), \
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch',return_value=native):
            outcome = run_weekend('folder',1,control,lambda k,v:events.append((k,v)),'daily',
                                  checkpoint=True,progress=progress,goal=4)
        self.assertIs(outcome,TaskOutcome.LIMIT_REACHED)
        self.assertEqual(progress.matches,4)
        self.assertIn(('kill',1),events)

    def test_current_race_marker_confirms_battle_without_old_distance_board(self):
        vision = RaceVision()
        image = current_frame('race-02/100.png')
        self.assertEqual(vision.detect(image), 'race')
        self.assertTrue(vision.confirmed_race)

    def test_current_task_claim_states_use_one_click_and_claimed_labels(self):
        vision = RaceVision()
        claimable = current_frame('tasks-after-race/005.png')
        self.assertEqual(vision.detect(claimable), 'tasks')
        self.assertTrue(vision.claimable(claimable))
        self.assertFalse(vision.claimed(claimable))
        self.assertEqual(vision.current_claimed_count(claimable), 0)

        claimed = current_frame('tasks-claimed/007.png')
        self.assertEqual(vision.detect(claimed), 'tasks')
        self.assertFalse(vision.claimable(claimed))
        self.assertEqual(vision.current_claimed_count(claimed), 1)

    def test_five_current_claimed_labels_are_counted_separately(self):
        vision = RaceVision()
        template = cv2.imread(str(ROOT/'assets/weekend-race/claimed-current.png'),
                              cv2.IMREAD_GRAYSCALE)
        template = cv2.cvtColor(template, cv2.COLOR_GRAY2RGB)
        height, width = template.shape[:2]
        image = np.zeros((720, 1280, 3), np.uint8)
        for top in (140, 240, 340, 440, 540):
            image[top:top+height, 990:990+width] = template
        self.assertEqual(vision.count(image, 'claimed-current'), 5)

    def test_weekend_progress_resets_at_daily_boundary(self):
        progress = WeekendProgress()
        progress.refresh('2026-11-21')
        progress.record()
        progress.record()
        self.assertEqual(progress.matches, 2)
        progress.refresh('2026-11-22')
        self.assertEqual(progress.matches, 0)

    def test_twenty_five_confirmed_matches_stop_before_starting_another(self):
        vision = Mock()
        vision.detect.side_effect = ['tasks', 'tasks', 'home']
        vision.layout = 'current'
        vision.intro = False
        vision.current_tasks.return_value = True
        vision.claimable.return_value = False
        vision.claimed.return_value = False
        device, native = Mock(), Mock()
        device.capture.return_value = np.zeros((720, 1280, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory, \
             patch('ninja3_assistant.activities.weekend.runner.ROOT', Path(directory)), \
             patch('ninja3_assistant.activities.weekend.runner.prepare_game', return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RaceVision', return_value=vision), \
             patch('ninja3_assistant.activities.weekend.runner.ActivityVision'), \
             patch('ninja3_assistant.activities.weekend.runner.WindowsOcr'), \
             patch('ninja3_assistant.activities.weekend.runner.MuMu', return_value=device), \
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch', return_value=native):
            progress = WeekendProgress(day='2026-11-21', matches=25)
            outcome = run_weekend('folder', 1, RunControl(clock=Mock(side_effect=range(0, 40, 2))),
                                  lambda *_: None, '25', progress=progress)
        self.assertIs(outcome, TaskOutcome.LIMIT_REACHED)
        self.assertTrue(all(call.args[0] != weekend_runner.START_MATCH_CURRENT
                            for call in native.priority_tap.call_args_list))

    def _invite_scene(self, *, targets, grade, frames=('home', 'invite', 'home'),
                      confirmed_race=False):
        """Drive one invite frame through the real, shared bounty_handoff policy."""
        control = RunControl()
        if targets is not None:
            control.bounty_invite_targets = targets
        image = np.zeros((720, 1280, 3), np.uint8)
        native, device, vision = Mock(), Mock(), Mock()
        native.movement_remaining.return_value = 0.
        vision.detect.side_effect = list(frames)
        vision.confirmed_race = confirmed_race
        vision.obscured = False
        vision.observe.return_value = no_obstacles()
        vision.invitations.detect.return_value = (340, 520)
        vision.invitations.bounty_invitation.return_value = (
            BountyInvitation('friend', grade, (700, 520)) if grade else None)
        ui = Mock()
        ui.perform.return_value = False
        captures = 0

        def capture():
            nonlocal captures
            captures += 1
            if captures >= 3:
                control.request_stop()
            return image

        device.capture.side_effect = capture
        with tempfile.TemporaryDirectory() as directory, \
             patch('ninja3_assistant.activities.weekend.runner.ROOT', Path(directory)), \
             patch('ninja3_assistant.activities.weekend.runner.prepare_game', return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RaceVision', return_value=vision), \
             patch('ninja3_assistant.activities.weekend.runner.ActivityVision'), \
             patch('ninja3_assistant.activities.weekend.runner.WindowsOcr') as ocr_cls, \
             patch('ninja3_assistant.activities.weekend.runner.GameUI', return_value=ui), \
             patch('ninja3_assistant.activities.weekend.runner.MuMu', return_value=device), \
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch', return_value=native):
            ocr_cls.return_value.read.return_value = {'words': []}
            outcome = run_weekend('folder', 1, control, lambda *a: None)
        return outcome, ui, vision

    @staticmethod
    def _declined(ui):
        return any(call.args[1] == 'decline_invitation' for call in ui.perform.call_args_list)

    def test_configured_passive_invite_is_handed_off_instead_of_declined(self):
        """被动邀请开启且等级命中：把邀请交给悬赏任务去同意，不自己点拒绝。"""
        outcome, ui, vision = self._invite_scene(targets={'A': 'daily_bounty'}, grade='A')
        self.assertEqual(outcome, InvitationHandoff('daily_bounty'))
        self.assertTrue(vision.invitations.bounty_invitation.called)
        self.assertFalse(self._declined(ui))

    def test_invite_is_declined_when_passive_invite_is_off(self):
        """被动邀请关闭：现有策略不交接，周末 runner 按配置拒绝。"""
        outcome, ui, vision = self._invite_scene(targets={}, grade='A')
        self.assertIs(outcome, TaskOutcome.STOPPED)
        self.assertFalse(vision.invitations.bounty_invitation.called)
        self.assertTrue(self._declined(ui))

    def test_invite_is_declined_when_grade_does_not_match(self):
        """等级不在启用的悬赏任务里：按配置拒绝，不误同意。"""
        outcome, ui, _vision = self._invite_scene(targets={'A': 'daily_bounty'}, grade='S')
        self.assertIs(outcome, TaskOutcome.STOPPED)
        self.assertTrue(self._declined(ui))

    def test_live_race_never_hands_off_an_invitation(self):
        """竞速中不交接去打悬赏，按配置拒绝以免中断比赛。"""
        outcome, ui, vision = self._invite_scene(
            targets={'A': 'daily_bounty'}, grade='A',
            frames=('ready', 'invite', 'home'), confirmed_race=True)
        self.assertIs(outcome, TaskOutcome.STOPPED)
        self.assertFalse(vision.invitations.bounty_invitation.called)
        self.assertTrue(self._declined(ui))

    def _race_scene(self, *, frames, ninjutsu=False, confirmed_race=True):
        """Drive a scripted frame sequence through the battle loop with mocks.

        GameUI is a mock so ``detect`` only runs in the main loop; the capture
        count therefore matches ``len(frames)`` exactly.
        """
        control = RunControl()
        frames = list(frames)
        image = np.zeros((720, 1280, 3), np.uint8)
        native, device, vision = Mock(), Mock(), Mock()
        native.movement_remaining.return_value = 0.
        vision.detect.side_effect = frames
        vision.confirmed_race = confirmed_race
        vision.obscured = False
        vision.ninjutsu_ready = ninjutsu
        vision.weapon_ready = False
        vision.observe.return_value = no_obstacles()
        vision.on_rope = False
        vision.opening.ground_frames = 1
        ui = Mock()
        ui.perform.return_value = False
        captures = 0

        def capture():
            nonlocal captures
            captures += 1
            if captures >= len(frames):
                control.request_stop()
            return image

        device.capture.side_effect = capture
        with tempfile.TemporaryDirectory() as directory, \
             patch('ninja3_assistant.activities.weekend.runner.ROOT', Path(directory)), \
             patch('ninja3_assistant.activities.weekend.runner.prepare_game', return_value=False), \
             patch('ninja3_assistant.activities.weekend.runner.RealtimeBattleKernel.recognition_due', return_value=True), \
             patch('ninja3_assistant.activities.weekend.runner.RaceVision', return_value=vision), \
             patch('ninja3_assistant.activities.weekend.runner.ActivityVision'), \
             patch('ninja3_assistant.activities.weekend.runner.WindowsOcr'), \
             patch('ninja3_assistant.activities.weekend.runner.GameUI', return_value=ui), \
             patch('ninja3_assistant.activities.weekend.runner.MuMu', return_value=device), \
             patch('ninja3_assistant.activities.weekend.runner.create_native_touch', return_value=native):
            outcome = run_weekend('folder', 1, control, lambda *a: None)
        return outcome, native

    def test_ninjutsu_ready_threshold(self):
        """忍术圆环充满(整圈金色)才就绪；0.375 充电弧不就绪。"""
        ready_fn = weekend_runner.ninjutsu_ready
        x, y, r = weekend_runner.NINJUTSU_RING
        blank = np.zeros((720, 1280, 3), np.uint8)
        self.assertFalse(ready_fn(blank))
        full = blank.copy()
        cv2.circle(full, (x, y), r, (255, 200, 60), 10)
        self.assertTrue(ready_fn(full))
        charging = blank.copy()
        cv2.ellipse(charging, (x, y), (r, r), 0, 180, 315, (255, 200, 60), 10)
        self.assertFalse(ready_fn(charging))

    def test_ready_ninjutsu_is_fired_during_race(self):
        """忍术充满时立即点忍术图标(55,60)释放。"""
        _, native = self._race_scene(frames=['race', 'race', 'race'], ninjutsu=True)
        self.assertTrue(any(c.args and c.args[0] == weekend_runner.NINJUTSU_POINT
                            for c in native.priority_tap.call_args_list))

    def test_charging_ninjutsu_is_not_fired(self):
        """忍术充电中(圆环未满)不点忍术。"""
        _, native = self._race_scene(frames=['race', 'race', 'race'], ninjutsu=False)
        self.assertFalse(any(c.args and c.args[0] == weekend_runner.NINJUTSU_POINT
                             for c in native.priority_tap.call_args_list))

    def test_unknown_race_frame_skates_out_instead_of_freezing(self):
        """竞速中画面未识别：不停手，复用上跳2次+下滑脱困。"""
        _, native = self._race_scene(frames=['ready', 'race', 'unknown', 'unknown'])
        # 只有 ready 页 quiesce 一次；两个 unknown 脱困帧继续移动、不 quiesce。
        self.assertEqual(native.quiesce.call_count, 1)
        self.assertGreaterEqual(native.schedule_movement.call_count, 2)

    def test_unknown_frame_outside_race_still_quiesces(self):
        """未确认竞速的 unknown(菜单/加载)仍停手，不无脑脱困。"""
        _, native = self._race_scene(frames=['ready', 'unknown', 'unknown'])
        self.assertEqual(native.quiesce.call_count, 3)
        self.assertFalse(any(c.args and c.args[0] == weekend_runner.NINJUTSU_POINT
                             for c in native.priority_tap.call_args_list))

    @staticmethod
    def _weapon_icon(bright_share, colour=(255, 255, 255)):
        """A dark icon with a bright centre block covering ``bright_share``."""
        frame = np.zeros((720, 1280, 3), np.uint8)
        half = weekend_runner.WEAPON_DIGIT_BOX
        box = int(2 * half * np.sqrt(bright_share))
        box = min(box, 2 * half)
        frame[613 - box // 2:613 - box // 2 + box,
              380 - box // 2:380 - box // 2 + box] = colour
        return frame

    def test_weapon_readiness_reads_countdown_digit_not_brightness(self):
        """倒计时数字（绿或白）判冷却；无数字、武器图案占满中心判就绪。"""
        ready_fn = weekend_runner._weapon_icon_state
        # white digit covers only part of the centre -> cooling
        self.assertFalse(ready_fn(self._weapon_icon(.16, (255, 255, 255)), 380, 613))
        # green digit (cooldown reduction) also covers part -> cooling
        self.assertFalse(ready_fn(self._weapon_icon(.24, (60, 220, 60)), 380, 613))
        # ready weapon art + ring fill most of the centre -> armed
        self.assertTrue(ready_fn(self._weapon_icon(.50, (240, 180, 120)), 380, 613))

    def test_both_weapon_icons_are_either_ready_or_cooling(self):
        """主武器和副武器两个图标各自独立判定，就绪的给出点击点。"""
        frame = np.zeros((720, 1280, 3), np.uint8)
        half = weekend_runner.WEAPON_DIGIT_BOX
        # main weapon: ready (full bright centre); side weapon: digit (small)
        frame[613 - half:613 + half, 380 - half:380 + half] = (240, 180, 120)
        small = int(2 * half * np.sqrt(.2))
        frame[613 - small // 2:613 - small // 2 + small,
              900 - small // 2:900 - small // 2 + small] = (60, 220, 60)
        points = weekend_runner.ready_weapon_points(frame)
        self.assertEqual(points, ((380, 630),))
        # a fully cooling pair yields no points; cards still cycle
        cooling = np.zeros((720, 1280, 3), np.uint8)
        self.assertEqual(weekend_runner.ready_weapon_points(cooling), ())
        self.assertEqual([a.point for a in RaceSkills.actions(())],
                         [(900, 630), (520, 630), (635, 630), (740, 630)])
