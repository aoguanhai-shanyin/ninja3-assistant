import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from ninja3_assistant.device.mumu import MuMu, instance_serial, prepare_game
from ninja3_assistant.runtime.runner import RunControl

class MuMuInfoTest(unittest.TestCase):
    @staticmethod
    def response(payload, returncode=0):
        return SimpleNamespace(stdout=json.dumps(payload).encode(), returncode=returncode)

    def test_cold_start_waits_for_selected_instance_then_opens_game(self):
        responses = [self.response({'is_process_started': False, 'is_android_started': False}),
                     self.response({'errcode': 0}),
                     self.response({'is_process_started': True, 'is_android_started': True}),
                     self.response({'state': 'stopped'}), self.response({'errcode': 0})]
        control = RunControl()
        with patch.object(control.stop, 'wait', return_value=False), patch('ninja3_assistant.device.mumu.subprocess.run', side_effect=responses) as run:
            self.assertTrue(prepare_game('C:/MuMu', 1, control, lambda _: None))
        commands = [call.args[0][1:] for call in run.call_args_list]
        self.assertEqual(commands[1], ['control', '--vmindex', '1', 'launch'])
        self.assertEqual(commands[-1][:5], ['control', '--vmindex', '1', 'app', 'launch'])
        self.assertTrue(all('--vmindex' in command and command[command.index('--vmindex') + 1] == '1'
                            for command in commands))

    def test_running_game_is_not_restarted(self):
        responses = [self.response({'is_process_started': True, 'is_android_started': True}),
                     self.response({'state': 'running'})]
        with patch('ninja3_assistant.device.mumu.subprocess.run', side_effect=responses) as run:
            self.assertFalse(prepare_game('C:/MuMu', 1, RunControl(), lambda _: None))
        self.assertEqual(run.call_count, 2)

    def test_window_preference_targets_selected_instance_after_game_ready(self):
        for hidden,command in ((False,'show_window'),(True,'hide_window')):
            responses = [self.response({'is_process_started':True,'is_android_started':True}),
                         self.response({'state':'running'}),self.response({'errcode':0})]
            with self.subTest(hidden=hidden),patch('ninja3_assistant.device.mumu.subprocess.run',side_effect=responses) as run:
                prepare_game('C:/MuMu',3,RunControl(hide_game_window=hidden),lambda _:None)
                self.assertEqual(run.call_args.args[0][1:],['control','--vmindex','3',command])

    def test_cold_launch_applies_hidden_preference_only_after_launching_game(self):
        responses = [self.response({'is_process_started':False,'is_android_started':False}),
                     self.response({'errcode':0}),
                     self.response({'is_process_started':True,'is_android_started':True}),
                     self.response({'state':'stopped'}),self.response({'errcode':0}),self.response({'errcode':0})]
        control = RunControl(hide_game_window=True)
        with patch.object(control.stop,'wait',return_value=False),patch('ninja3_assistant.device.mumu.subprocess.run',side_effect=responses) as run:
            prepare_game('C:/MuMu',1,control,lambda _:None)
        self.assertIn('app',run.call_args_list[-2].args[0])
        self.assertEqual(run.call_args.args[0][-1],'hide_window')

    def test_missing_game_and_stopped_task_do_not_launch_game(self):
        running = self.response({'is_process_started': True, 'is_android_started': True})
        with patch('ninja3_assistant.device.mumu.subprocess.run', side_effect=[running, self.response({'state': 'not_installed'})]) as run:
            with self.assertRaisesRegex(RuntimeError, '未安装'):
                prepare_game('C:/MuMu', 1, RunControl(), lambda _: None)
            self.assertEqual(run.call_count, 2)
        control = RunControl()
        control.stop.set()
        with patch('ninja3_assistant.device.mumu.subprocess.run') as run:
            self.assertIsNone(prepare_game('C:/MuMu', 1, control, lambda _: None))
            run.assert_not_called()

    def test_single_instance_and_all_instance_info(self):
        info = {'adb_host_ip': '127.0.0.1', 'adb_port': 16416}
        self.assertEqual(instance_serial(info, 1), '127.0.0.1:16416')
        self.assertEqual(instance_serial({'1': info}, 1), '127.0.0.1:16416')

    def test_missing_instance_does_not_launch(self):
        with self.assertRaises(RuntimeError):
            instance_serial({}, 1)

    def test_reopen_uses_single_instance_adb_address_and_honors_stop(self):
        device = object.__new__(MuMu)
        device.root, device.index, device.handle = Path('C:/MuMu'), 1, 7
        device.dll = SimpleNamespace(nemu_get_display_id=lambda *args: 0)
        stopped = SimpleNamespace(is_set=lambda: False, wait=lambda delay: False)
        output = SimpleNamespace(stdout=b'{"adb_host_ip":"127.0.0.1","adb_port":16416}')
        with patch('ninja3_assistant.device.mumu.subprocess.run', side_effect=[output, SimpleNamespace(stdout=b'connected'), SimpleNamespace(stdout=b'')]) as run:
            device.reopen_game(stopped)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(run.call_args_list[1].args[0][1:], ['connect', '127.0.0.1:16416'])
            self.assertEqual(run.call_args_list[2].args[0][1:3], ['-s', '127.0.0.1:16416'])
            self.assertIn('android.intent.category.LAUNCHER', run.call_args_list[2].args[0])
        stopped = SimpleNamespace(is_set=lambda: True)
        with patch('ninja3_assistant.device.mumu.subprocess.run', return_value=output) as run:
            device.reopen_game(stopped)
            self.assertEqual(run.call_count, 1)

    def test_reopen_does_not_launch_if_adb_connection_fails(self):
        device = object.__new__(MuMu)
        device.root, device.index, device.handle = Path('C:/MuMu'), 1, 7
        stopped = SimpleNamespace(is_set=lambda: False)
        output = SimpleNamespace(stdout=b'{"adb_host_ip":"127.0.0.1","adb_port":16416}')
        with patch('ninja3_assistant.device.mumu.subprocess.run', side_effect=[output, SimpleNamespace(stdout=b'failed to connect')]) as run:
            with self.assertRaisesRegex(RuntimeError, '无法连接'):
                device.reopen_game(stopped)
            self.assertEqual(run.call_count, 2)

if __name__ == '__main__':
    unittest.main()
