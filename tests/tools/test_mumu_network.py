import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from tools.diagnostics.mumu_network import Device, MAPPINGS, PROXY_KEYS, TARGET, apply, digest, hosts_mounts, overlay_hosts, proxy_off, rollback


class MuMuNetworkTest(unittest.TestCase):
    def test_proxy_off_restores_all_original_guest_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'guest-proxy.json'
            original = {key: 'null' for key in PROXY_KEYS}
            path.write_text(json.dumps({'kind': 'guest_proxy', 'serial': 'test',
                'proxy': '10.0.2.2:7890', 'original': original, 'status': 'applied'}), encoding='utf-8')
            current = {**original, 'http_proxy': '10.0.2.2:7890'}

            def setting(device, operation, scope, key, value=None):
                if operation == 'get':
                    return current[key]
                current[key] = 'null' if operation == 'delete' else value
                return ''

            from unittest.mock import patch
            with patch('tools.diagnostics.mumu_network.guest_setting', side_effect=setting):
                self.assertEqual(proxy_off(Mock(serial='test'), path)['status'], 'restored')
            self.assertEqual(current, original)

    def test_proxy_off_preserves_a_later_user_proxy_change(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'guest-proxy.json'
            path.write_text(json.dumps({'kind': 'guest_proxy', 'serial': 'test',
                'proxy': '10.0.2.2:7890', 'original': {key:'null' for key in PROXY_KEYS}}), encoding='utf-8')
            from unittest.mock import patch
            with patch('tools.diagnostics.mumu_network.guest_setting', return_value='user-value') as setting:
                with self.assertRaisesRegex(ValueError, 'outside'):
                    proxy_off(Mock(serial='test'), path)
                self.assertTrue(all(call.args[1]=='get' for call in setting.call_args_list))

    def test_backup_uses_raw_adb_transport_without_windows_newline_translation(self):
        device = Device('adb', 'test')
        device.call = Mock(return_value=b'127.0.0.1 localhost\n')
        self.assertEqual(device.current(), b'127.0.0.1 localhost\n')
        device.call.assert_called_once_with('exec-out', 'su', '0', 'cat', TARGET)

    def test_override_preserves_original_and_only_adds_two_verified_cdns(self):
        original = b'127.0.0.1 localhost\n::1 ip6-localhost\n'
        content = overlay_hosts(original)
        self.assertTrue(content.startswith(original))
        for host, ip in MAPPINGS.items():
            self.assertIn(f'{ip} {host}\n'.encode(), content)
        self.assertNotIn(b'qaq.pandadagames.net', content)

    def test_existing_target_domain_override_is_preserved_by_refusal(self):
        with self.assertRaisesRegex(ValueError, 'already'):
            overlay_hosts(b'127.0.0.1 patchresource.pandadastudio.com\n')

    def test_mount_parser_does_not_confuse_parent_system_mount(self):
        parent = '18 16 8:6 / /system ro shared:2 - ext4 /dev/block/sda6 ro'
        bound = '123 18 8:7 /tmp/hosts /system/etc/hosts rw shared:3 - ext4 /dev/block/sda7 rw'
        self.assertEqual(hosts_mounts(parent+'\n'+bound), [bound])

    def test_rollback_refuses_changed_hosts_before_unmounting(self):
        with tempfile.TemporaryDirectory() as folder:
            state_file = Path(folder)/'state.json'
            state_file.write_text(json.dumps({'serial': 'test', 'target': TARGET, 'status': 'applied',
                'original_mounts': [], 'override_sha256': digest(b'ours')}), encoding='utf-8')
            device = Mock(serial='test')
            device.current.return_value = b'another override'
            with self.assertRaisesRegex(ValueError, 'changed'):
                rollback(device, state_file)
            device.root.assert_not_called()

    def test_existing_hosts_mount_is_never_stacked(self):
        with tempfile.TemporaryDirectory() as folder:
            device = Mock(serial='test')
            device.current.return_value = b'127.0.0.1 localhost\n'
            device.mountinfo.return_value = '123 18 8:7 /other/hosts /system/etc/hosts rw - ext4 /dev/block/sda7 rw'
            with self.assertRaisesRegex(ValueError, 'existing hosts mount'):
                apply(device, folder)
            device.call.assert_not_called()
            device.root.assert_not_called()

    def test_rollback_verifies_original_bytes_and_records_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            state_file = Path(folder)/'state.json'
            mount = '123 18 8:7 /our/hosts /system/etc/hosts rw - ext4 /dev/block/sda7 rw'
            state_file.write_text(json.dumps({'serial': 'test', 'target': TARGET, 'status': 'applied',
                'original_mounts': [], 'override_sha256': digest(b'ours'), 'original_sha256': digest(b'original'),
                'active_mounts': [mount]}), encoding='utf-8')
            device = Mock(serial='test')
            device.current.side_effect = [b'ours', b'original']
            device.mountinfo.return_value = mount
            self.assertEqual(rollback(device, state_file)['status'], 'rolled_back')
            device.root.assert_called_once_with('umount', TARGET)
            self.assertEqual(json.loads(state_file.read_text(encoding='utf-8'))['status'], 'rolled_back')


if __name__ == '__main__':
    unittest.main()
