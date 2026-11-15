import unittest
from pathlib import Path

import cv2
import numpy as np

from hayabusa_recognition import HayabusaBoardRecognizer
from hayabusa_vision import HayabusaScreen, HayabusaSegmenter, HayabusaVision, color_masks


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'data/hayabusa-video-study'
FIXTURES = ROOT/'tests/fixtures/hayabusa'


class HayabusaVisionTest(unittest.TestCase):
    def setUp(self):
        self.vision = HayabusaVision(use_gpu=False, recognize_boards=False, detector=False)

    def scene(self):
        frame = np.full((720, 1280, 3), (25, 35, 65), dtype=np.uint8)
        for name, (x, y) in {'name': (449, 29), 'portrait': (383, 21), 'down': (165, 615),
                             'face-2': (285, 365)}.items():
            image = cv2.imread(str(ROOT/'assets'/f'hayabusa-{name}.png'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            frame[y:y+h, x:x+w] = image
        cv2.line(frame, (0, 428), (1279, 428), (0, 0, 0), 6)
        return frame

    def test_tracks_attacks_approaching_from_both_sides_without_swapping_ids(self):
        ids = None
        for i in range(6):
            frame = self.scene()
            cv2.ellipse(frame, (650-i*12, 380), (24, 10), 0, 0, 360, (255, 180, 25), -1)
            cv2.ellipse(frame, (120+i*12, 380), (24, 10), 0, 0, 360, (255, 180, 25), -1)
            cv2.ellipse(frame, (650-i*12, 380), (19, 5), 0, 0, 360, (255, 245, 210), -1)
            cv2.ellipse(frame, (120+i*12, 380), (19, 5), 0, 0, 360, (255, 245, 210), -1)
            screen, tracks = self.vision.detect(frame, 1+i*.04)
            self.assertTrue(screen.active)
            self.assertAlmostEqual(screen.player_x, 300, delta=2)
            self.assertAlmostEqual(screen.rope_y, 428, delta=3)
            fire = sorted((x for x in tracks if x.kind == 'fire'), key=lambda x: x.x)
            self.assertEqual(len(fire), 2)
            if ids is None:
                ids = [x.id for x in fire]
            self.assertEqual([x.id for x in fire], ids)
        self.assertGreater(fire[0].vx, 200)
        self.assertLess(fire[1].vx, -200)
        self.assertEqual([x.hits for x in fire], [6, 6])

    def test_mask_excludes_blue_background_and_large_foreground_effect(self):
        frame = self.scene()
        cv2.rectangle(frame, (600, 280), (680, 390), (70, 140, 210), -1)
        cv2.circle(frame, (900, 350), 110, (35, 220, 255), -1)
        cv2.circle(frame, (720, 380), 10, (35, 220, 255), -1)
        cv2.circle(frame, (720, 380), 4, (10, 10, 10), -1)
        screen, tracks = self.vision.detect(frame, 1)
        self.assertTrue(screen.active)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].kind, 'shuriken')
        self.assertAlmostEqual(tracks[0].x, 720, delta=2)

    def test_black_fire_and_wide_flash_have_distinct_kinds(self):
        frame = self.scene()
        cv2.ellipse(frame, (720, 350), (40, 10), 0, 0, 360, (160, 20, 220), -1)
        cv2.rectangle(frame, (100, 470), (1100, 490), (30, 220, 255), -1)
        _, tracks = self.vision.detect(frame, 1)
        self.assertEqual({x.kind for x in tracks}, {'blackfire', 'flash'})

    def test_unknown_character_is_not_assigned_a_fixed_player_position(self):
        frame = self.scene()
        frame[350:415, 260:340] = (25, 35, 65)
        screen, _ = self.vision.detect(frame, 1)
        self.assertTrue(screen.active)
        self.assertIsNone(screen.player_x)
        self.assertIsNone(screen.player_y)
        self.assertEqual(screen.side, 'unknown')

    def test_dim_pause_overlay_and_invalid_resolution_never_remain_active(self):
        frame = self.scene()
        self.assertTrue(self.vision.detect(frame, 1)[0].active)
        dim = (frame.astype(np.float32)*.35).astype(np.uint8)
        self.assertFalse(self.vision.detect(dim, 1.04)[0].active)
        self.assertFalse(self.vision.detect(frame[:600], 1.08)[0].active)
        self.assertEqual(self.vision.tracks, {})

    def locked_recognizer(self):
        recognizer = HayabusaBoardRecognizer()
        recognizer.locked = next(index for index, board in enumerate(recognizer.boards)
                                 if board['id'] == 'p1_fire_1')
        recognizer.locked_confidence, recognizer.locked_margin = .95, .2
        recognizer.locked_progress = .2
        recognizer.last_time = recognizer.last_supported = 1.
        self.vision.recognizer = recognizer
        return recognizer

    def test_short_hud_occlusion_retains_epoch_without_claiming_an_active_screen(self):
        recognizer = self.locked_recognizer()
        epoch = recognizer.epoch
        frame = self.scene()
        hidden, tracks = self.vision.detect((frame*.35).astype(np.uint8), 1.04)
        self.assertFalse(hidden.active)
        self.assertIsNone(hidden.board_id)
        self.assertEqual(tracks, [])
        self.assertEqual(recognizer.epoch, epoch)
        restored, _ = self.vision.detect(frame, 1.18)
        self.assertTrue(restored.active)
        self.assertEqual(restored.board_id, 'p1_fire_1')
        self.assertEqual(restored.board_epoch, epoch)
        self.assertFalse(restored.board_onset)

    def test_long_hud_occlusion_clears_epoch_including_a_late_first_restored_frame(self):
        for still_hidden in (True, False):
            with self.subTest(still_hidden=still_hidden):
                self.vision.reset()
                recognizer = self.locked_recognizer()
                epoch = recognizer.epoch
                frame = self.scene()
                hidden = (frame*.35).astype(np.uint8)
                self.vision.detect(hidden, 1.04)
                state, _ = self.vision.detect(hidden if still_hidden else frame, 1.4)
                self.assertGreater(recognizer.epoch, epoch)
                self.assertIsNone(recognizer.locked)
                self.assertIsNone(state.board_id)

    def test_result_immediately_retires_a_recognized_attack_epoch(self):
        recognizer = self.locked_recognizer()
        epoch = recognizer.epoch
        state, tracks = self.vision._recognized(self.scene(), HayabusaScreen(result='victory'), [], 1.04)
        self.assertEqual(state.result, 'victory')
        self.assertEqual(tracks, [])
        self.assertGreater(recognizer.epoch, epoch)
        self.assertIsNone(recognizer.locked)

    def test_time_jump_and_reverse_reset_attack_velocity(self):
        frame = self.scene()
        cv2.circle(frame, (700, 380), 12, (30, 220, 255), -1)
        cv2.circle(frame, (700, 380), 5, (10, 10, 10), -1)
        self.vision.detect(frame, 1)
        second = self.vision.detect(frame, 1.05)[1]
        self.assertEqual(second[0].hits, 2)
        for timestamp in (4, 3):
            new = self.vision.detect(frame, timestamp)[1]
            self.assertEqual(new[0].hits, 1)
            self.assertEqual((new[0].vx, new[0].vy), (0, 0))


@unittest.skipUnless((STUDY/'overview/frame-0020.000.jpg').is_file(), 'local teaching frames unavailable')
class HayabusaRecordingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = HayabusaVision(use_gpu=False, recognize_boards=False, detector=False)

    def frame(self, folder, seconds):
        raw = cv2.imread(str(STUDY/folder/f'frame-{seconds:08.3f}.jpg'))
        self.assertIsNotNone(raw)
        return cv2.cvtColor(cv2.resize(raw, (1280, 720)), cv2.COLOR_BGR2RGB)

    def test_held_out_frames_locate_player_and_do_not_require_teaching_caption(self):
        # These frames were not used to make any player or HUD template.
        for seconds, side in ((40, 'above'), (50, 'above'), (180, 'below')):
            frame = self.frame('overview', seconds)
            frame[530:600] = 0
            screen, _ = self.vision.detect(frame, seconds)
            self.assertTrue(screen.active, seconds)
            self.assertIsNotNone(screen.player_x, seconds)
            self.assertEqual(screen.side, side, seconds)

    def test_uncovered_player_pose_stays_unknown(self):
        screen, _ = self.vision.detect(self.frame('overview', 150), 150)
        self.assertTrue(screen.active)
        self.assertIsNone(screen.player_x)
        self.assertIsNone(screen.player_y)

    def test_intro_and_end_card_are_inactive_and_success_is_recognized(self):
        for folder, seconds in (('overview', 0), ('tail', 660)):
            screen, tracks = self.vision.detect(self.frame(folder, seconds), seconds)
            self.assertFalse(screen.active)
            self.assertEqual(tracks, [])
        screen, tracks = self.vision.detect(self.frame('tail', 640), 640)
        self.assertEqual(screen.result, 'victory')
        self.assertFalse(screen.active)
        self.assertEqual(tracks, [])

    def test_air_down_arrow_and_jump_button_are_distinct_from_slide(self):
        air, _ = self.vision.detect(self.frame('overview', 20), 20)
        self.assertEqual(air.left_control, 'down')
        self.assertEqual(air.right_control, 'up')
        self.assertTrue(air.airborne)
        self.assertFalse(air.grounded)
        landed, _ = self.vision.detect(self.frame('overview', 30), 30)
        self.assertEqual(landed.right_control, 'jump')
        self.assertFalse(landed.airborne)
        self.assertTrue(landed.grounded)


class HayabusaAttackRegressionTest(unittest.TestCase):
    def setUp(self):
        self.vision = HayabusaVision(use_gpu=False, recognize_boards=False, detector=False)

    def tracks(self, seconds):
        raw = cv2.imread(str(FIXTURES/f'frame-{seconds:08.3f}.jpg'))
        self.assertIsNotNone(raw)
        state, tracks = self.vision.detect(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), seconds)
        self.assertTrue(state.active)
        return tracks

    def test_yellow_chapter_gold_damage_and_player_purple_aura_are_not_attacks(self):
        # 19s has yellow chapter letters, gold damage numerals, and purple hands;
        # no hostile fireball or black-fire projectile has been emitted yet.
        self.assertEqual(self.tracks(19), [])

    def test_white_core_validation_retains_the_six_actual_fireballs(self):
        tracks = self.tracks(24)
        fire = [track for track in tracks if track.kind == 'fire']
        self.assertEqual(len(fire), 6)
        self.assertTrue(any(abs(track.x-206) < 20 and abs(track.y-248) < 20 for track in fire))
        self.assertFalse(any(track.kind == 'blackfire' for track in tracks))

    def test_tracking_black_fire_is_retained_but_small_purple_pickup_is_rejected(self):
        self.assertEqual(len([track for track in self.tracks(255) if track.kind == 'blackfire']), 2)
        tracks = self.tracks(260)
        self.assertEqual(len([track for track in tracks if track.kind == 'blackfire']), 3)
        self.assertFalse(any(abs(track.x-551) < 20 and abs(track.y-214) < 20 for track in tracks))
        self.assertFalse(any(track.kind == 'blackfire' for track in self.tracks(291)))

    def test_depth_chapter_letters_and_lit_background_windows_are_rejected(self):
        tracks = self.tracks(302)
        self.assertFalse(any(track.kind == 'fire' for track in tracks))
        self.assertFalse(any(570 <= track.x <= 660 and 290 <= track.y <= 425 for track in tracks))


class HayabusaGpuTest(unittest.TestCase):
    def test_directml_and_cpu_color_rules_agree(self):
        try:
            segmenter = HayabusaSegmenter()
        except RuntimeError as error:
            self.skipTest(str(error))
        image = np.random.default_rng(42).integers(0, 256, (250, 640, 3), dtype=np.uint8)
        actual = segmenter.masks(image)
        for name, expected in color_masks(image).items():
            np.testing.assert_array_equal(actual[name], expected)


if __name__ == '__main__':
    unittest.main()
