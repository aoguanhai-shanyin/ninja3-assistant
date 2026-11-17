import json
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from tools.diagnostics.assistant_control import AssistantController, ControlServer, main
from ninja3_assistant.activities.daily import CHORES


class AssistantControlTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.worker = self.root/'fake_worker.py'
        self.worker.write_text('''import json, sys
print(json.dumps({'event':'state','value':'running'}),flush=True)
for line in sys.stdin:
    command=json.loads(line)['command']
    if command=='stop': break
    state={'pause':'paused','resume':'running'}[command]
    print(json.dumps({'event':'state','value':state}),flush=True)
print(json.dumps({'event':'finished','value':''}),flush=True)
''', encoding='utf-8')
        self.events = []
        self.controller = AssistantController(self.root, 1, python=sys.executable,
                                              worker=self.worker, emit=self.events.append)
        self.addCleanup(self.controller.stop)

    def wait_state(self, expected):
        with self.controller.lock:
            self.assertTrue(self.controller.lock.wait_for(lambda: self.controller.state == expected, timeout=5),
                            self.controller.status())

    def test_worker_lifecycle_reuses_commands_and_rejects_duplicate_start(self):
        with patch.object(self.controller, '_prepare'):
            self.controller.start('observe')
            self.wait_state('running')
            pid = self.controller.status()['worker_pid']
            with self.assertRaisesRegex(RuntimeError, 'already active'):
                self.controller.start()
            self.assertEqual(self.controller.pause()['state'], 'paused')
            with self.assertRaisesRegex(RuntimeError, 'pause requires'):
                self.controller.pause()
            self.assertEqual(self.controller.resume()['state'], 'running')
            self.assertEqual(self.controller.status()['worker_pid'], pid)
            self.assertEqual(self.controller.stop()['state'], 'stopped')
            self.assertIsNone(self.controller.status()['worker_pid'])
            self.assertIsNotNone(self.controller.process.poll())

    def test_packaged_worker_executes_directly_with_the_existing_mode_arguments(self):
        self.controller.worker = self.root/'Ninja3Worker.EXE'
        process = SimpleNamespace(pid=123, poll=lambda: 0, wait=lambda **kwargs: 0,
                                  stdin=StringIO(), stdout=StringIO(), stderr=StringIO())
        with patch.object(self.controller, '_prepare'), \
                patch('tools.diagnostics.assistant_control.subprocess.Popen', return_value=process) as launch:
            self.controller.start('observe')
            self.wait_state('stopped')
        command = launch.call_args.args[0]
        self.assertEqual(command, [str(self.controller.worker), '--mode', 'hayabusa',
                                   '--folder', str(self.root.resolve()), '--index', '1', '--observe-only'])
        self.assertNotIn(self.controller.python, command)

    def test_pause_resume_and_invalid_mode_cannot_start_an_idle_worker(self):
        with self.assertRaises(RuntimeError):
            self.controller.pause()
        with self.assertRaises(RuntimeError):
            self.controller.resume()
        with self.assertRaises(ValueError):
            self.controller.start('unsupported')
        with self.assertRaises(ValueError):
            self.controller.command({'command': 'tap'})
        self.assertEqual(self.controller.stop()['state'], 'idle')
        self.assertIsNone(self.controller.process)

    def test_activity_options_reach_worker_and_status_and_default_off_on_next_start(self):
        self.controller.worker = self.root/'Ninja3Worker.EXE'
        for enabled, mode in ((True, 'daily'), (True, 'continuous'), (False, 'daily')):
            process = SimpleNamespace(pid=123, poll=lambda: 0, wait=lambda **kwargs: 0,
                                      stdin=StringIO(), stdout=StringIO(), stderr=StringIO())
            with self.subTest(enabled=enabled, mode=mode), \
                    patch.object(self.controller, '_prepare'), \
                    patch('tools.diagnostics.assistant_control.subprocess.Popen', return_value=process) as launch:
                self.controller.command({'command': 'start', 'mode': 'run',
                                         'activity_enabled': enabled, 'activity_mode': mode})
                self.wait_state('stopped')
                self.controller.thread.join(timeout=3)
            command = launch.call_args.args[0]
            expected = [str(self.controller.worker), '--mode', 'run',
                        '--folder', str(self.root.resolve()), '--index', '1']
            if enabled:
                expected += ['--activity-enabled', '--activity-mode', mode]
            self.assertEqual(command, expected)
            self.assertEqual(self.controller.status()['activity_enabled'], enabled)
            self.assertEqual(self.controller.status()['activity_mode'], mode)

    def test_activity_validation_cannot_prepare_or_launch_an_invalid_request(self):
        invalid = [dict(mode='observe', activity_enabled=True),
                   dict(mode='hayabusa', activity_enabled=True),
                   dict(mode='run', activity_enabled='false'),
                   dict(mode='run', activity_enabled=True, activity_mode='unknown')]
        with patch.object(self.controller, '_prepare') as prepare:
            for options in invalid:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    self.controller.start(**options)
            prepare.assert_not_called()
        self.assertEqual(self.controller.status()['state'], 'idle')

    def test_daily_only_run_forwards_all_chores_and_exact_bounty_grade(self):
        self.controller.worker = self.root/'Ninja3Worker.EXE'
        process = SimpleNamespace(pid=123,poll=lambda:0,wait=lambda **kwargs:0,
                                  stdin=StringIO(),stdout=StringIO(),stderr=StringIO())
        with patch.object(self.controller,'_prepare'), \
                patch('tools.diagnostics.assistant_control.subprocess.Popen',return_value=process) as launch:
            self.controller.command({'command':'start','mode':'run',
                                     'daily_tasks':list(CHORES),'bounty_grade':'A'})
            self.wait_state('stopped')
            self.controller.thread.join(timeout=3)
        command=launch.call_args.args[0]
        self.assertEqual(command,[str(self.controller.worker),'--mode','run',
            '--folder',str(self.root.resolve()),'--index','1','--daily-tasks',*CHORES,
            '--bounty-grade','A'])
        state=self.controller.status()
        self.assertEqual(state['daily_tasks'],list(CHORES))
        self.assertEqual(state['bounty_grade'],'A')
        self.assertFalse(state['activity_enabled'])
        self.assertFalse(state['weekend_enabled'])
        state['daily_tasks'].clear()
        self.assertEqual(self.controller.status()['daily_tasks'],list(CHORES))

    def test_mixed_task_options_are_cleared_on_the_next_start(self):
        self.controller.worker = self.root/'Ninja3Worker.EXE'
        for enabled in (True,False):
            process = SimpleNamespace(pid=123,poll=lambda:0,wait=lambda **kwargs:0,
                                      stdin=StringIO(),stdout=StringIO(),stderr=StringIO())
            options=(dict(activity_enabled=True,activity_mode='continuous',weekend_enabled=True,
                          weekend_mode='25',daily_tasks=['rank','bounty','rank']) if enabled else {})
            with self.subTest(enabled=enabled),patch.object(self.controller,'_prepare'), \
                    patch('tools.diagnostics.assistant_control.subprocess.Popen',return_value=process) as launch:
                self.controller.start('run',**options)
                self.wait_state('stopped')
                self.controller.thread.join(timeout=3)
            state=self.controller.status()
            command=launch.call_args.args[0]
            self.assertEqual(state['weekend_enabled'],enabled)
            self.assertEqual(state['daily_tasks'],['rank','bounty'] if enabled else [])
            self.assertEqual(state['bounty_grade'],'D' if enabled else None)
            if enabled:
                self.assertEqual(command[-8:],['--weekend-enabled','--weekend-mode','25',
                                             '--daily-tasks','rank','bounty','--bounty-grade','D'])
            else:
                self.assertNotIn('--daily-tasks',command)
                self.assertNotIn('--weekend-enabled',command)
                self.assertNotIn('--bounty-grade',command)

    def test_invalid_daily_or_weekend_requests_never_prepare_or_start(self):
        invalid=[dict(mode='run',daily_tasks='monthly'),dict(mode='run',daily_tasks=['unknown']),
                 dict(mode='run',daily_tasks=[[]]),dict(mode='run',daily_tasks=None),
                 dict(mode='observe',daily_tasks=['rank']),dict(mode='hayabusa',weekend_enabled=True),
                 dict(mode='run',weekend_enabled='false'),dict(mode='run',weekend_mode='continuous'),
                 dict(mode='run',bounty_grade='A'),dict(mode='run',daily_tasks=['rank'],bounty_grade='A'),
                 dict(mode='run',daily_tasks=['bounty'],bounty_grade='S')]
        with patch.object(self.controller,'_prepare') as prepare, \
                patch('tools.diagnostics.assistant_control.subprocess.Popen') as launch:
            for options in invalid:
                with self.subTest(options=options),self.assertRaises(ValueError):
                    self.controller.start(**options)
            prepare.assert_not_called()
            launch.assert_not_called()
        self.assertIsNone(self.controller.thread)
        self.assertEqual(self.controller.status()['state'],'idle')

    def test_exit_stops_only_the_owned_worker_and_can_restart_after_stop(self):
        with patch.object(self.controller, '_prepare'):
            self.controller.start()
            self.wait_state('running')
            first = self.controller.process
            self.controller.command({'command': 'exit'})
            self.assertIsNotNone(first.poll())
            self.controller.start('run')
            self.wait_state('running')
            self.assertNotEqual(self.controller.process.pid, first.pid)
            self.controller.stop()

    def test_screenshot_uses_sdk_rgb_and_closes_its_handle_without_clicks(self):
        path = self.root/'capture.png'
        rgb = np.zeros((720, 1280, 3), np.uint8)
        rgb[:, :] = (255, 128, 0)
        with patch('ninja3_assistant.device.mumu.MuMu') as sdk:
            sdk.return_value.capture.return_value = rgb
            result = self.controller.screenshot(path)
            sdk.assert_called_once_with(str(self.root.resolve()), 1)
            sdk.return_value.close.assert_called_once()
            sdk.return_value.tap.assert_not_called()
        self.assertEqual((result['width'], result['height']), (1280, 720))
        np.testing.assert_array_equal(cv2.imread(str(path))[0, 0], (0, 128, 255))
        with self.assertRaises(FileExistsError):
            self.controller.screenshot(path)

    def test_hide_and_launch_target_only_the_selected_instance(self):
        with patch.object(self.controller, '_manager', return_value={}) as manager:
            self.controller.hide()
            manager.assert_called_once_with('control', '--vmindex', '1', 'hide_window')
        cold = {'is_process_started': False, 'is_android_started': False}
        booting = {'is_process_started': True, 'is_android_started': False}
        ready = {'is_process_started': True, 'is_android_started': True}
        with patch.object(self.controller, '_manager', side_effect=[cold, {}, RuntimeError('window pending'),
                                                                   booting, {}, ready, {}]) as manager, \
                patch('tools.diagnostics.assistant_control.time.sleep'):
            self.assertTrue(self.controller.launch(hide_game_window=True)['is_android_started'])
            self.assertEqual(manager.call_args_list[1].args, ('control', '--vmindex', '1', 'launch'))
            self.assertEqual(manager.call_args_list[2].args, ('control', '--vmindex', '1', 'hide_window'))
            self.assertEqual(manager.call_args_list[4].args, ('control', '--vmindex', '1', 'hide_window'))
            self.assertEqual(manager.call_args_list[-1].args, ('control', '--vmindex', '1', 'hide_window'))

    def test_launch_keeps_window_visible_by_default(self):
        ready = {'is_process_started':True,'is_android_started':True}
        with patch.object(self.controller,'_manager',side_effect=[ready,{}]) as manager:
            self.controller.launch()
        self.assertEqual(manager.call_args_list[-1].args,('control','--vmindex','1','show_window'))

    def test_window_preference_reaches_worker_and_is_reported(self):
        self.controller.worker = self.root/'Ninja3Worker.EXE'
        process = SimpleNamespace(pid=123,poll=lambda:0,wait=lambda **kw:0,
                                  stdin=StringIO(),stdout=StringIO(),stderr=StringIO())
        with patch.object(self.controller,'_prepare'),patch('tools.diagnostics.assistant_control.subprocess.Popen',return_value=process) as launch:
            self.controller.start('run',hide_game_window=True)
            self.wait_state('stopped')
        self.assertIn('--hide-game-window',launch.call_args.args[0])
        self.assertTrue(self.controller.status()['hide_game_window'])

    def test_cold_launch_timeout_is_bounded_and_reports_the_last_startup_state(self):
        clock = SimpleNamespace(value=0.)
        def manager(*arguments, **kwargs):
            return {'is_process_started': True, 'is_android_started': False} if arguments[0] == 'info' else {}
        def sleep(seconds):
            clock.value += seconds
        with patch.object(self.controller, '_manager', side_effect=manager), \
                patch('tools.diagnostics.assistant_control.time.monotonic', side_effect=lambda: clock.value), \
                patch('tools.diagnostics.assistant_control.time.sleep', side_effect=sleep):
            with self.assertRaisesRegex(RuntimeError, '60 seconds.*"is_android_started": false'):
                self.controller.launch()
        self.assertEqual(clock.value, 60.)

    def test_launch_client_timeout_covers_startup_wait_without_changing_other_commands(self):
        for command, timeout in (('launch', 75), ('status', 30)):
            with self.subTest(command=command), \
                    patch.object(sys, 'argv', ['assistant_control.py', '--command', command]), \
                    patch('tools.diagnostics.assistant_control.urlopen', return_value=BytesIO(b'{"ok":true}')) as request, \
                    redirect_stdout(StringIO()):
                self.assertEqual(main(), 0)
            self.assertEqual(request.call_args.kwargs['timeout'], timeout)

    def test_start_cli_serializes_activity_options_without_connecting_to_emulator(self):
        with patch.object(sys, 'argv', ['assistant_control.py', '--command', 'start', '--mode', 'run',
                                       '--activity-enabled', '--activity-mode', 'continuous']), \
                patch('tools.diagnostics.assistant_control.urlopen', return_value=BytesIO(b'{"ok":true}')) as request, \
                redirect_stdout(StringIO()):
            self.assertEqual(main(), 0)
        message = json.loads(request.call_args.args[0].data)
        self.assertEqual(message, {'command': 'start', 'mode': 'run',
                                  'activity_enabled': True, 'activity_mode': 'continuous'})

    def test_start_cli_serializes_daily_and_weekend_selection(self):
        with patch.object(sys,'argv',['assistant_control.py','--command','start','--mode','run',
                     '--daily-tasks','monthly','bounty','rank','--bounty-grade','B',
                     '--weekend-enabled','--weekend-mode','25']), \
                patch('tools.diagnostics.assistant_control.urlopen',return_value=BytesIO(b'{"ok":true}')) as request, \
                redirect_stdout(StringIO()):
            self.assertEqual(main(),0)
        self.assertEqual(json.loads(request.call_args.args[0].data),{
            'command':'start','mode':'run','activity_enabled':False,'activity_mode':'daily',
            'daily_tasks':['monthly','bounty','rank'],'bounty_grade':'B',
            'weekend_enabled':True,'weekend_mode':'25'})

    def test_daily_cli_validation_does_not_issue_a_request(self):
        for arguments in (['--command','start','--mode','observe','--daily-tasks','rank'],
                          ['--command','status','--daily-tasks','rank'],
                          ['--command','start','--mode','run','--bounty-grade','A'],
                          ['--command','start','--mode','run','--daily-tasks','unknown']):
            with self.subTest(arguments=arguments),patch.object(sys,'argv',['assistant_control.py',*arguments]), \
                    patch('tools.diagnostics.assistant_control.urlopen') as request, \
                    patch('sys.stderr',StringIO()),self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code,2)
            request.assert_not_called()

    def test_preparation_failure_is_visible_and_never_spawns_a_worker(self):
        with patch.object(self.controller, '_prepare', side_effect=RuntimeError('network unavailable')):
            self.controller.start()
            self.wait_state('error')
        self.assertEqual(self.controller.status()['error'], 'network unavailable')
        self.assertIsNone(self.controller.process)

    def test_local_http_commands_and_browser_origin_rejection(self):
        server = ControlServer(self.controller, 0)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        thread.start()
        try:
            def request(command, origin=None):
                headers = {'Content-Type': 'application/json'}
                if origin:
                    headers['Origin'] = origin
                return Request(f'http://127.0.0.1:{server.server_port}/command',
                               data=json.dumps({'id': 4, 'command': command}).encode(), headers=headers)
            with urlopen(request('status'), timeout=3) as response:
                payload = json.load(response)
            self.assertEqual(payload['id'], 4)
            self.assertEqual(payload['result']['state'], 'idle')
            with self.assertRaises(HTTPError) as error:
                urlopen(request('start', 'https://example.com'), timeout=3)
            self.assertEqual(error.exception.code, 400)
            self.assertIsNone(self.controller.process)
        finally:
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()


if __name__ == '__main__':
    unittest.main()
