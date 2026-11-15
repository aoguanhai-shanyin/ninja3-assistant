import threading
import unittest
from unittest.mock import Mock
from realtime_battle import RealtimeBattleKernel, RecognitionCadence
from native_touch import TimedTap
from runner import RunControl


class KernelTest(unittest.TestCase):
    def test_slow_frame_drops_new_decision_without_cutting_existing_movement(self):
        control,core,driver=self.rig()
        core.movement_horizon=3.
        driver.movement_remaining.return_value=2.
        self.now+=.4
        self.assertFalse(core.renew())
        driver.schedule_movement.assert_not_called()
        driver.release.assert_not_called()
        control.pause();control.resume()
        self.assertFalse(core.renew())
        driver.release.assert_called_once()
    def test_rolling_movement_renews_only_near_end_without_per_frame_dispatch(self):
        _,core,driver=self.rig()
        core.movement_horizon=3.
        driver.movement_remaining.return_value=2.
        core.renew()
        driver.schedule_movement.assert_not_called()
        driver.renew.assert_not_called()
        driver.movement_remaining.return_value=.2
        core.renew()
        driver.schedule_movement.assert_called_once_with(3.,captured_at=10.)
    def test_bounded_movement_shares_freshness_generation_and_cancellation(self):
        control,core,driver=self.rig()
        core.schedule_movement(3)
        driver.schedule_movement.assert_called_once_with(3,captured_at=10.)
        core.cancel_movement()
        driver.release.assert_called_once()
        control.pause();control.resume()
        self.assertIsNone(core.schedule_movement(3))
        self.assertEqual(driver.schedule_movement.call_count,1)

    def test_older_frame_cannot_replace_or_revive_newer_movement(self):
        _,core,driver=self.rig()
        old=core.stamp()
        self.now+=.01
        core.capture(Mock())
        core.schedule_movement(3)
        self.assertIsNone(core.schedule_movement(3,frame=old))
        core.cancel_movement()
        self.assertIsNone(core.schedule_movement(3,frame=old))
        self.assertEqual(driver.schedule_movement.call_count,1)
    def rig(self):
        self.now = 10.
        control = RunControl()
        core = RealtimeBattleKernel(control,clock=lambda:self.now)
        driver = Mock()
        core.attach(driver)
        core.capture(Mock())
        return control,core,driver

    def test_latest_capture_replaces_previous_frame_without_queue(self):
        _,core,_ = self.rig()
        camera = Mock()
        camera.capture.side_effect = ['first','latest']
        self.assertEqual(core.capture(camera),'first')
        self.now += .1
        self.assertEqual(core.capture(camera),'latest')
        self.assertEqual(core.captured_at,self.now)

    def test_pause_resume_generation_discards_pre_pause_actions(self):
        control,core,driver = self.rig()
        control.pause()
        control.resume()
        self.assertFalse(core.tap((380,630),deadline=self.now+.2))
        driver.tap.assert_not_called()
        core.capture(Mock())
        core.tap((380,630),deadline=self.now+.2)
        driver.tap.assert_called_once()

    def test_expired_frame_does_not_renew_slide_or_send_jump(self):
        _,core,driver = self.rig()
        self.now += .3
        self.assertFalse(core.renew())
        self.assertIsNone(core.priority_tap((1090,630),captured_at=10.,deadline=10.2))
        driver.renew.assert_not_called()
        driver.priority_tap.assert_not_called()
        driver.release.assert_called_once()

    def test_regular_weapon_tap_does_not_preempt_skating(self):
        _,core,driver = self.rig()
        core.renew()
        core.tap((900,630),deadline=10.2)
        driver.tap.assert_called_once()
        driver.priority_tap.assert_not_called()
        driver.release.assert_not_called()

    def test_future_command_deadline_does_not_refresh_old_frame(self):
        _,core,driver = self.rig()
        self.now += .3
        self.assertFalse(core.tap((900,630),deadline=self.now+.2))
        driver.tap.assert_not_called()

    def test_caller_cannot_replace_original_capture_timestamp(self):
        _,core,driver = self.rig()
        self.now = 11.
        self.assertIsNone(core.priority_tap((1090,630),captured_at=11.,deadline=11.2))
        driver.priority_tap.assert_not_called()

    def test_native_deadline_is_clamped_to_original_frame_expiry(self):
        _,core,driver = self.rig()
        self.now = 10.19
        core.tap((900,630),deadline=10.39)
        self.assertEqual(driver.tap.call_args.kwargs['deadline'],10.2)

    def test_newer_frame_plan_prevents_old_frame_replacing_it(self):
        _,core,driver = self.rig()
        driver.plan.return_value = 7
        old = core.stamp()
        self.now += .02
        core.capture(Mock())
        actions = (TimedTap(30,(900,630)),)
        self.assertEqual(core.plan(4,actions),7)
        self.assertIsNone(core.plan(4,actions,frame=old))
        self.assertEqual(driver.plan.call_count,1)

    def test_cancel_is_a_frame_barrier_without_releasing_movement(self):
        _,core,driver = self.rig()
        old = core.stamp()
        self.now += .02
        core.capture(Mock())
        core.cancel_plan(4)
        self.assertIsNone(core.plan(4,(TimedTap(30,(900,630)),),frame=old))
        driver.release.assert_not_called()
        driver.stop_all.assert_not_called()

    def test_pause_invalidates_cross_frame_plan(self):
        control,core,driver = self.rig()
        old = core.stamp()
        control.pause()
        control.resume()
        self.now += .02
        core.capture(Mock())
        self.assertIsNone(core.plan(4,(TimedTap(30,(900,630)),),frame=old))
        driver.plan.assert_not_called()


class CadenceTest(unittest.TestCase):
    def test_stable_scene_skips_one_capture_but_keeps_latest_frame(self):
        cadence = RecognitionCadence()
        self.assertTrue(cadence.due(0.,0))
        cadence.completed(0.,.005,stable=True)
        self.assertFalse(cadence.due(1/60,0))
        self.assertTrue(cadence.due(2/60,0))
        self.assertEqual(cadence.skipped,1)

    def test_urgency_and_generation_bypass_skip(self):
        cadence = RecognitionCadence()
        cadence.completed(0.,.005,stable=True)
        self.assertTrue(cadence.due(.01,1))
        self.assertTrue(cadence.due(.012,1,urgent=True))

    def test_high_recognition_cost_reduces_work_without_unbounded_skip(self):
        cadence = RecognitionCadence()
        self.assertTrue(cadence.due(0.,0))
        cadence.completed(0.,.014)
        self.assertFalse(cadence.due(1/60,0))
        self.assertTrue(cadence.due(2/60,0))
        self.assertLessEqual(cadence.next_at,1/30)
