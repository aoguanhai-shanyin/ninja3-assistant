from ninja3_assistant.paths import project_root
import os
import unittest
from unittest.mock import patch
from pathlib import Path
import cv2
import numpy as np
from ninja3_assistant.activities.bounty.flow import Flow
from ninja3_assistant.vision.vision import Vision

EVIDENCE = project_root().parent / 'android-icon-clicker/build/evidence'

@unittest.skipUnless((EVIDENCE / 'bounty/19-app-main.png').is_file(), 'local captured evidence unavailable')
class VisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = Vision(use_gpu=False)

    def detect(self, name):
        image = cv2.imread(str(EVIDENCE / name))
        return self.vision.detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def test_login_home_and_account_dialog_are_distinguished(self):
        login = self.detect('app/33-login-back.png')
        self.assertIsNotNone(login.login)
        self.assertIsNone(self.detect('app/31-start-from-home.png').login)
        self.assertIsNotNone(self.detect('game/08-back-home.png').home)
        self.assertIsNotNone(self.detect('game/03-before-battlepass.png').home)
        self.assertIsNone(self.detect('bounty/17-menu.png').home)

    def test_captured_hall_and_network_prompt(self):
        temp = Path(os.environ.get('TEMP', ''))
        hall, network = temp / 'ninja3-hall-reference.png', temp / 'ninja3-network-reference.png'
        if not hall.is_file() or not network.is_file():
            self.skipTest('local startup evidence unavailable')
        def recognize(path):
            return self.vision.detect(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
        self.assertIsNotNone(recognize(hall).hall)
        prompt = recognize(network)
        self.assertTrue(prompt.network_error)
        self.assertIsNotNone(prompt.network_retry)

    def test_replay_startup_to_reverse_bounty(self):
        hall = Path(os.environ.get('TEMP', '')) / 'ninja3-hall-reference.png'
        if not hall.is_file():
            self.skipTest('local hall evidence unavailable')
        flow = Flow()
        flow.entering = True
        frames = [self.detect('app/33-login-back.png'), self.detect('game/03-before-battlepass.png'),
                  self.vision.detect(cv2.cvtColor(cv2.imread(str(hall)), cv2.COLOR_BGR2RGB)),
                  self.detect('bounty/17-menu.png'), self.detect('bounty/01-current.png')]
        for now, (screen, expected) in enumerate(zip(frames, ('login', 'home', 'hall', 'menu', 'ss_off'))):
            decision = flow.step(screen, now)
            self.assertEqual(decision.action, expected)
            flow.completed(decision, now)

    def test_s_team_not_ss_plus(self):
        self.assertIsNone(self.detect('bounty/02-refresh.png').join)
        result = self.detect('bounty/19-app-main.png')
        self.assertTrue(result.board)
        self.assertAlmostEqual(result.join[0], 1151, delta=4)
        self.assertEqual((result.a, result.s, result.ss), ('off', 'on', 'on'))

    def test_failed_join_in_actual_video_and_disbanded_toast(self):
        self.assertTrue(self.detect('video/frame-1s-crop.png').failed)
        self.assertTrue(self.detect('bounty/03-join.png').failed)
        self.assertTrue(self.detect('bounty/37-rapid-join.png').failed)
        self.assertFalse(self.detect('bounty/19-app-main.png').failed)

    def test_s_badge_survives_toast_covering_its_lower_edge(self):
        screenshot = cv2.imread(str(EVIDENCE / 'video/frame-1s-crop.png'), cv2.IMREAD_GRAYSCALE)
        self.assertTrue(self.vision.rank_s(screenshot, 830, 185))
        self.assertFalse(self.vision.rank_s(screenshot, 90, 185))

    def test_team_full_toast_does_not_hide_retry_target(self):
        image = EVIDENCE / 'bounty/attempt-1-1142.png'
        if not image.is_file():
            self.skipTest('local retry evidence unavailable')
        result = self.detect('bounty/attempt-1-1142.png')
        self.assertTrue(result.board)
        self.assertFalse(result.failed)
        self.assertIsNotNone(result.join)
        self.assertAlmostEqual(result.join[0], 400, delta=5)
        self.assertAlmostEqual(result.join[1], 306, delta=5)

    def test_live_card_animation_keeps_same_target(self):
        if not (EVIDENCE / 'bounty/watch-000.png').is_file():
            self.skipTest('local card animation evidence unavailable')
        first = self.detect('bounty/watch-000.png')
        shifted = self.detect('bounty/watch-004.png')
        self.assertGreater(shifted.join[0] - first.join[0], 8)
        flow = Flow()
        action = flow.step(first, 0); self.assertEqual(action.action, 'join')
        flow.completed(action, .1)
        retry = flow.step(shifted, .5)
        self.assertEqual((retry.action, retry.point), ('join', shifted.join))
        flow.completed(retry, .6)
        for index, timestamp in [(7, .9), (12, 1.3), (16, 1.7), (20, 2.1)]:
            current = self.detect(f'bounty/watch-{index:03}.png')
            retry = flow.step(current, timestamp)
            self.assertEqual(retry.action, 'join')
            flow.completed(retry, timestamp)

        image = cv2.imread(str(EVIDENCE / 'bounty/watch-004.png'))
        image[268:343, 360:440] = 0  # Hide the plus while leaving the same card visible.
        covered = self.vision.detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.assertIsNone(covered.join)
        retry = flow.step(covered, 2.2)
        self.assertEqual((retry.action, retry.point), ('join', shifted.join))

    def test_room_and_two_invite_layouts(self):
        self.assertIsNotNone(self.detect('bounty/11-user-state.png').ready)
        self.assertTrue(self.detect('bounty/12-after-ready.png').prepared)
        self.assertIsNotNone(self.detect('bounty/05-current-after-retry.png').invite)
        self.assertIsNotNone(self.detect('bounty/24-perf-screen.png').invite)

    def test_battle_loading_is_distinct_from_board_and_room(self):
        self.assertTrue(self.detect('bounty/30-after-s-join.png').battle_loading)
        self.assertFalse(self.detect('bounty/11-user-state.png').battle_loading)
        self.assertFalse(self.detect('bounty/19-app-main.png').battle_loading)

    def test_battle_active_and_reverse_result(self):
        self.assertTrue(self.detect('bounty/31-battle.png').battle_active)
        result = self.detect('bounty/33-before-result-confirm.png')
        self.assertTrue(result.reverse_result)
        self.assertIsNotNone(result.result)
        self.assertAlmostEqual(result.result[0], 650, delta=5)
        self.assertIsNone(self.detect('bounty/19-app-main.png').result)

    def test_boss_phase_with_down_button_is_still_battle(self):
        video = EVIDENCE / 'battle-inspect/boss-fight.mp4'
        if not video.is_file():
            self.skipTest('local boss recording unavailable')
        capture = cv2.VideoCapture(str(video))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 12 * 30)
            ok, frame = capture.read()
        finally:
            capture.release()
        self.assertTrue(ok, 'could not decode boss frame')
        self.assertTrue(self.vision.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).battle_active)

    def test_replay_s_retry_battle_and_next_refresh(self):
        if not (EVIDENCE / 'bounty/attempt-1-1118.png').is_file():
            self.skipTest('local live-attempt evidence unavailable')
        flow = Flow()
        first = flow.step(self.detect('bounty/35-after-result-wait.png'), 0)
        self.assertEqual(first.action, 'refresh'); flow.completed(first, .1)
        join = flow.step(self.detect('bounty/attempt-1-1118.png'), .3)
        self.assertEqual(join.action, 'join'); flow.completed(join, .4)
        retry = flow.step(self.detect('bounty/attempt-1-1142.png'), .6)
        self.assertEqual(retry.action, 'join'); flow.completed(retry, .7)
        ready = flow.step(self.detect('bounty/11-user-state.png'), 1)
        self.assertEqual(ready.action, 'ready'); flow.completed(ready, 1.1)
        self.assertEqual(flow.step(self.detect('bounty/30-after-s-join.png'), 2).action, 'wait')
        self.assertEqual(flow.step(self.detect('bounty/31-battle.png'), 3).action, 'wait')
        confirm = flow.step(self.detect('bounty/33-before-result-confirm.png'), 4)
        self.assertEqual(confirm.action, 'confirm'); flow.completed(confirm, 4.1)
        self.assertEqual(flow.step(self.detect('bounty/35-after-result-wait.png'), 5).action, 'refresh')

class VisionDarkFrameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = Vision(use_gpu=False)

    @staticmethod
    def put_gray(frame, template, x, y):
        h, w = template.shape
        frame[y:y+h, x:x+w] = template[:, :, None]

    def test_loading_screen_mean_threshold(self):
        logo = self.vision.images['training-loading']
        limit = 5 * 720 * 1280
        for offset in (-1, 0, 1):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            self.put_gray(frame, logo, 1120, 650)
            remaining = limit + offset - int(frame[:, :, 0].sum())
            self.assertGreaterEqual(remaining, 0)
            bright, extra = divmod(remaining, 255)
            top = frame[:300].reshape(-1, 3)
            top[:bright] = 255
            top[bright] = extra
            self.assertEqual(self.vision.detect(frame).game_loading, offset < 0)

    def test_invite_overlay_precedes_battle_buttons(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.put_gray(frame, self.vision.images['battle-slide'], 120, 580)
        self.put_gray(frame, self.vision.images['battle-jump'], 1030, 560)
        self.assertTrue(self.vision.detect(frame).battle_active)
        self.put_gray(frame, self.vision.images['invite-label'], 70, 200)
        self.put_gray(frame, self.vision.images['invite-collapse'], 1130, 140)
        self.put_gray(frame, self.vision.invitations.decline, 368, 492)
        self.put_gray(frame, self.vision.invitations.accept, 708, 493)
        screen = self.vision.detect(frame)
        self.assertIsNotNone(screen.invite)
        self.assertFalse(screen.battle_active)

    def test_float32_frames_keep_black_screen_decision(self):
        def match(gray, name, *args, **kwargs):
            return (1120, 660) if name == 'training-loading' else None
        with patch.object(self.vision, 'match', side_effect=match):
            dark = np.full((720, 1280, 3), 4, dtype=np.float32)
            light = np.full((720, 1280, 3), 6, dtype=np.float32)
            self.assertTrue(self.vision.detect(dark).game_loading)
            self.assertFalse(self.vision.detect(light).game_loading)


if __name__ == '__main__':
    unittest.main()
