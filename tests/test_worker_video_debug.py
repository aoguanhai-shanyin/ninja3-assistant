import io
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import worker


class WorkerVideoDebugTest(unittest.TestCase):
    def test_video_dispatch_does_not_require_or_prepare_an_emulator(self):
        commands = io.StringIO('{"command":"frame","frame":42,"request_id":1}\n')
        output = io.StringIO()
        replay = Mock(side_effect=lambda path, emit, stream: emit('video_info', {'path':path}))
        with patch.object(sys, 'argv', ['worker.py', '--mode', 'hayabusa-video', '--video', '测试.mp4']), \
                patch.object(sys, 'stdin', commands), patch.object(sys, 'stdout', output), \
                patch.dict(sys.modules, {'hayabusa_video_debug':SimpleNamespace(run_video_debug=replay)}), \
                patch('worker.RunControl') as control, patch('worker.run_flow') as flow, \
                patch('worker.probe') as probe, patch('worker.threading.Thread') as thread:
            worker.main()
        self.assertEqual(replay.call_args.args[0], '测试.mp4')
        self.assertIs(replay.call_args.args[2], commands)
        self.assertEqual(json.loads(output.getvalue()),
                         {'event':'video_info','value':{'path':'测试.mp4'}})
        for unused in (control, flow, probe, thread):
            unused.assert_not_called()

    def test_video_path_is_required(self):
        with patch.object(sys, 'argv', ['worker.py', '--mode', 'hayabusa-video']), \
                patch.object(sys, 'stderr', io.StringIO()), self.assertRaises(SystemExit):
            worker.main()

    def test_emulator_modes_still_require_device_settings(self):
        for mode in ('run', 'probe', 'hayabusa-debug', 'hayabusa'):
            with self.subTest(mode=mode), \
                    patch.object(sys, 'argv', ['worker.py', '--mode', mode]), \
                    patch.object(sys, 'stderr', io.StringIO()), self.assertRaises(SystemExit):
                worker.main()

    def test_video_cannot_silently_change_an_emulator_mode(self):
        with patch.object(sys, 'argv', ['worker.py', '--mode', 'run', '--folder', 'MuMu',
                                       '--index', '0', '--video', 'test.mp4']), \
                patch.object(sys, 'stderr', io.StringIO()), self.assertRaises(SystemExit):
            worker.main()


if __name__ == '__main__':
    unittest.main()
