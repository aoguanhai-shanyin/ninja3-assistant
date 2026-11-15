import unittest
from pathlib import Path
from unittest.mock import patch

import cv2

from boss_trial import challenge_ready, wait_for_console


EVIDENCE = Path(__file__).resolve().parents[2] / 'android-icon-clicker/build/evidence/battle-inspect'


@unittest.skipUnless((EVIDENCE / 'clone-current.png').is_file(), 'local challenge screenshot unavailable')
class BossTrialTest(unittest.TestCase):
    def image(self, name):
        frame = cv2.imread(str(EVIDENCE / name))
        self.assertIsNotNone(frame)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def test_starts_only_on_selected_solo_boss_screen(self):
        self.assertTrue(challenge_ready(self.image('clone-current.png')))
        self.assertFalse(challenge_ready(self.image('attack-12.jpg')))
        self.assertFalse(challenge_ready(self.image('full-energy-paused.png')))

    def test_console_can_close_when_input_is_unavailable(self):
        with patch('boss_trial.sys.stdin') as stream, patch('builtins.input', side_effect=EOFError):
            stream.isatty.return_value = True
            wait_for_console()


if __name__ == '__main__':
    unittest.main()
