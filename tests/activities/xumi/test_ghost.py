"""Replay the opening mirrored-blade hit before attempting a live dodge."""

import unittest

import cv2

from ninja3_assistant.activities.xumi.ghost import GhostAvoidance
from ninja3_assistant.paths import project_root, resource_root


RUN = project_root() / 'data/runtime/xumi/runs/20261119-172722-780500'
UPPER_RUN = project_root() / 'data/runtime/xumi/runs/20261119-175126-264800'
PREVIOUS_UPPER_RUN = project_root() / 'data/runtime/xumi/runs/20261119-173128-841800'
DAQIAN_RUN = project_root() / 'data/runtime/xumi/runs/20261119-175412-705700'
DAQIAN_UPPER_RUN = project_root() / 'data/runtime/xumi/runs/20261119-181037-971800'
TUCKED_RUN = project_root() / 'data/runtime/xumi/runs/20261119-180655-104900'
LATEST_FANGCUN = project_root() / 'data/runtime/xumi/runs/20261119-182024-476800'
LATEST_DAQIAN = project_root() / 'data/runtime/xumi/runs/20261119-182619-858400'
DOWN = resource_root() / 'assets/battle-down.png'
JUMP = resource_root() / 'assets/battle-jump.png'


@unittest.skipUnless((RUN / 'battle-0091.png').is_file() and DOWN.is_file() and JUMP.is_file(),
                     'local Fangcun Ghost replay unavailable')
class GhostAvoidanceTest(unittest.TestCase):
    def setUp(self):
        down = cv2.inRange(cv2.imread(str(DOWN), cv2.IMREAD_GRAYSCALE), 200, 255)
        jump = cv2.inRange(cv2.imread(str(JUMP), cv2.IMREAD_GRAYSCALE), 200, 255)
        self.ghost = GhostAvoidance(down, jump)

    @staticmethod
    def frame(number, run=RUN):
        return cv2.cvtColor(cv2.imread(str(run / f'battle-{number:04d}.png')), cv2.COLOR_BGR2RGB)

    def test_opening_proposes_once_before_player_health_appears_on_0091(self):
        proposals = []
        for number in range(77, 100):
            point = self.ghost.update(self.frame(number), number * .25)
            if point is not None:
                proposals.append((number, point))
                self.ghost.accepted()
        self.assertEqual(proposals, [(89, (1088, 630))])

    def test_missing_boss_upper_control_or_mirrored_blade_clears_the_approach(self):
        for name, (x1, y1, x2, y2) in (
            ('ghost portrait', (384, 32, 436, 78)),
            ('right arrow', (1020, 570, 1155, 690)),
            ('upper blade', (785, 0, 1010, 225)),
        ):
            with self.subTest(missing=name):
                first, second = self.frame(88), self.frame(89)
                second[y1:y2, x1:x2] = (12, 22, 65)
                self.ghost.reset()
                self.assertIsNone(self.ghost.update(first, 1.0))
                self.assertIsNone(self.ghost.update(second, 1.25))
                self.assertIsNone(self.ghost.update(self.frame(89), 1.3))

    def test_visible_hazard_in_upper_landing_corridor_blocks_the_flip(self):
        first, second = self.frame(88), self.frame(89)
        cv2.circle(second, (523, 309), 42, (255, 20, 5), -1)
        self.assertIsNone(self.ghost.update(first, 1.0))
        self.assertIsNone(self.ghost.update(second, 1.25))

    def test_duplicate_and_expired_samples_cannot_propose(self):
        first, second = self.frame(88), self.frame(89)
        self.assertIsNone(self.ghost.update(first, 1.0))
        self.assertIsNone(self.ghost.update(second, 1.0))
        self.assertIsNone(self.ghost.update(second, 1.6))

    def test_acknowledgment_bounds_retry_to_a_fresh_approach(self):
        first, second = self.frame(88), self.frame(89)
        self.ghost.update(first, 1.0)
        self.assertEqual(self.ghost.update(second, 1.25), (1088, 630))
        self.ghost.accepted()
        self.assertIsNone(self.ghost.update(first, 1.5))
        self.assertIsNone(self.ghost.update(second, 1.75))
        self.assertIsNone(self.ghost.update(second, 2.10))
        self.assertIsNone(self.ghost.update(first, 2.25))
        self.assertEqual(self.ghost.update(second, 2.5), (1088, 630))

    def test_other_boss_rope_attack_never_proposes(self):
        for number in range(45, 58):
            with self.subTest(frame=number):
                self.assertIsNone(self.ghost.update(self.frame(number), number * .25))

    @unittest.skipUnless((UPPER_RUN / 'battle-0107.png').is_file(), 'local above-rope replay unavailable')
    def test_above_rope_proposes_down_after_lower_corridor_clears(self):
        proposals = []
        for number in range(96, 111):
            point = self.ghost.update(self.frame(number, UPPER_RUN), number * .25)
            if point is not None:
                proposals.append((number, point))
                self.ghost.accepted()
        self.assertEqual(proposals, [(105, (190, 630))])

    @unittest.skipUnless((PREVIOUS_UPPER_RUN / 'battle-0096.png').is_file(), 'local prior above-rope replay unavailable')
    def test_recent_visible_player_can_be_followed_during_head_occlusion(self):
        for number in (92, 93, 94):
            self.assertIsNone(self.ghost.update(self.frame(number, PREVIOUS_UPPER_RUN), number * .25))
        self.assertEqual(self.ghost.update(self.frame(95, PREVIOUS_UPPER_RUN), 95 * .25), (190, 630))

    @unittest.skipUnless((UPPER_RUN / 'battle-0107.png').is_file(), 'local above-rope replay unavailable')
    def test_lower_hazard_or_missing_jump_glyph_prevents_down(self):
        for missing in ('clear landing', 'jump glyph'):
            with self.subTest(missing=missing):
                self.ghost.reset()
                self.assertIsNone(self.ghost.update(self.frame(104, UPPER_RUN), 1.0))
                target = self.frame(105, UPPER_RUN)
                if missing == 'clear landing':
                    cv2.circle(target, (546, 505), 40, (255, 20, 5), -1)
                else:
                    target[570:690, 1020:1155] = (12, 22, 65)
                self.assertIsNone(self.ghost.update(target, 1.25))

    @unittest.skipUnless((DAQIAN_RUN / 'battle-0062.png').is_file(), 'local Daqian replay unavailable')
    def test_daqian_blue_lead_proposes_up_before_health_bar_appears(self):
        proposals = []
        for number in range(56, 69):
            point = self.ghost.update(self.frame(number, DAQIAN_RUN), number * .25)
            if point is not None:
                proposals.append((number, point))
                self.ghost.accepted()
        self.assertEqual(proposals, [(60, (1088, 630))])

    @unittest.skipUnless((DAQIAN_RUN / 'battle-0062.png').is_file(), 'local Daqian replay unavailable')
    def test_blue_lead_requires_rope_shape_and_avatar_below_it(self):
        for name, (x1, y1, x2, y2) in (
            ('cyan rope shape', (510, 330, 605, 394)),
            ('silver avatar', (466, 399, 574, 479)),
        ):
            with self.subTest(missing=name):
                self.ghost.reset()
                self.assertIsNone(self.ghost.update(self.frame(59, DAQIAN_RUN), 1.0))
                target = self.frame(60, DAQIAN_RUN)
                target[y1:y2, x1:x2] = (12, 22, 65)
                self.assertIsNone(self.ghost.update(target, 1.25))

    @unittest.skipUnless((DAQIAN_UPPER_RUN / 'battle-0071.png').is_file(), 'local above-rope Daqian replay unavailable')
    def test_daqian_blue_lead_proposes_down_after_lower_corridor_clears(self):
        proposals = []
        for number in range(64, 75):
            point = self.ghost.update(self.frame(number, DAQIAN_UPPER_RUN), number * .25)
            if point is not None:
                proposals.append((number, point))
                self.ghost.accepted()
        self.assertEqual(proposals, [(70, (190, 630))])

    @unittest.skipUnless((DAQIAN_UPPER_RUN / 'battle-0071.png').is_file(), 'local above-rope Daqian replay unavailable')
    def test_blue_upper_lead_requires_head_armor_and_clear_landing(self):
        for name, box in (
            ('head and companion', (526, 318, 600, 381)),
            ('lower armor', (507, 387, 538, 418)),
            ('jump glyph', (1020, 570, 1155, 690)),
            ('clear landing', None),
        ):
            with self.subTest(missing=name):
                self.ghost.reset()
                self.assertIsNone(self.ghost.update(self.frame(69, DAQIAN_UPPER_RUN), 1.0))
                target = self.frame(70, DAQIAN_UPPER_RUN)
                if box is None:
                    cv2.circle(target, (530, 505), 40, (255, 20, 5), -1)
                else:
                    x1, y1, x2, y2 = box
                    target[y1:y2, x1:x2] = (12, 22, 65)
                self.assertIsNone(self.ghost.update(target, 1.25))

    @unittest.skipUnless((TUCKED_RUN / 'battle-0102.png').is_file(), 'local tucked-pose replay unavailable')
    def test_fangcun_tucked_pose_proposes_up_before_health_bar_appears(self):
        proposals = []
        for number in range(97, 104):
            point = self.ghost.update(self.frame(number, TUCKED_RUN), number * .25)
            if point is not None:
                proposals.append((number, point))
                self.ghost.accepted()
        self.assertEqual(proposals, [(101, (1088, 630))])

    @unittest.skipUnless((TUCKED_RUN / 'battle-0102.png').is_file(), 'local tucked-pose replay unavailable')
    def test_tucked_pose_requires_boot_clothing_and_face(self):
        for name, (x1, y1, x2, y2) in (
            ('white boot', (510, 398, 548, 423)),
            ('offset clothing', (515, 437, 531, 449)),
            ('warm face', (535, 445, 556, 465)),
        ):
            with self.subTest(missing=name):
                self.ghost.reset()
                self.assertIsNone(self.ghost.update(self.frame(100, TUCKED_RUN), 1.0))
                target = self.frame(101, TUCKED_RUN)
                target[y1:y2, x1:x2] = (12, 22, 65)
                self.assertIsNone(self.ghost.update(target, 1.25))

    @unittest.skipUnless((LATEST_FANGCUN / 'battle-0095.png').is_file(), 'local Fangcun swing replay unavailable')
    def test_fangcun_swing_proposes_up_while_boot_points_left_of_clothing(self):
        proposals = []
        for number in range(90, 96):
            point = self.ghost.update(self.frame(number, LATEST_FANGCUN), number * .25)
            if point is not None:
                proposals.append((number, point))
                self.ghost.accepted()
        self.assertEqual(proposals, [(93, (1088, 630))])

    @unittest.skipUnless((LATEST_DAQIAN / 'battle-0068.png').is_file(), 'local Daqian cloud replay unavailable')
    def test_daqian_cloud_tracks_approach_but_blocks_occupied_lower_landing(self):
        first = self.frame(67, LATEST_DAQIAN)
        next_frame = self.frame(68, LATEST_DAQIAN)
        self.assertIsNone(self.ghost.update(first, 1.0))
        self.assertIsNone(self.ghost.update(next_frame, 1.25))

        self.ghost.reset()
        self.assertIsNone(self.ghost.update(first, 1.0))
        # Isolate the landing guard while retaining the partial rope and the
        # observed merged avatar on the same frame.
        next_frame[442:566, 441:621] = (12, 22, 65)
        self.assertEqual(self.ghost.update(next_frame, 1.25), (190, 630))
