"""Temporary, reversible MuMu-only hosts binding for the two verified game CDNs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


TARGET = '/system/etc/hosts'
MAPPINGS = {'resourcedistributionms.pandadastudio.com': '27.221.111.88',
            'patchresource.pandadastudio.com': '27.221.111.87'}
PROXY_KEYS = ('http_proxy', 'global_http_proxy_host', 'global_http_proxy_port',
              'global_http_proxy_exclusion_list', 'global_proxy_pac_url')


def digest(content):
    return hashlib.sha256(content).hexdigest()


def overlay_hosts(original):
    text = original.decode('utf-8')
    for line in text.splitlines():
        names = line.split('#', 1)[0].split()[1:]
        if any(name in MAPPINGS for name in names):
            raise ValueError('A game CDN already has a hosts override; refusing to replace it')
    additions = '\n# Temporary Ninja3 CDN connectivity test; rollback with mumu_network.py\n'
    additions += ''.join(f'{ip} {host}\n' for host, ip in MAPPINGS.items())
    return (text.rstrip('\r\n')+'\n'+additions).encode('utf-8')


def hosts_mounts(mountinfo):
    return [line for line in mountinfo.splitlines() if len(line.split()) > 5 and line.split()[4] == TARGET]


class Device:
    def __init__(self, adb, serial):
        self.adb, self.serial = str(adb), serial

    def call(self, *args):
        result = subprocess.run([self.adb, '-s', self.serial, *args], capture_output=True,
            timeout=15, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            raise RuntimeError(result.stderr.decode('utf-8', errors='replace').strip() or
                               result.stdout.decode('utf-8', errors='replace').strip())
        return result.stdout

    def root(self, *args):
        # adb shell reconstructs a command string; quote every remote argument.
        return self.call('shell', shlex.join(['su', '0', *args]))

    def mountinfo(self):
        return self.root('cat', '/proc/1/mountinfo').decode('utf-8')

    def current(self):
        # Windows adb shell adds CRLF; exec-out preserves the actual file bytes.
        return self.call('exec-out', 'su', '0', 'cat', TARGET)


def apply(device, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    state_file = directory/'state.json'
    if state_file.exists():
        raise ValueError('This record directory already has a network state; use a new directory')
    original = device.current()
    mounted = hosts_mounts(device.mountinfo())
    if mounted:
        raise ValueError('An existing hosts mount is present; refusing to stack another binding')
    replacement = overlay_hosts(original)
    remote = '/data/local/tmp/ninja3-network-'+str(time.time_ns())
    (directory/'hosts.original').write_bytes(original)
    (directory/'hosts.override').write_bytes(replacement)
    (directory/'mount-before.txt').write_text(device.mountinfo(), encoding='utf-8')
    state = {'serial': device.serial, 'target': TARGET, 'remote_directory': remote,
             'original_sha256': digest(original), 'override_sha256': digest(replacement),
             'original_mounts': mounted, 'mappings': MAPPINGS, 'status': 'prepared',
             'scope': 'Only an Android bind mount; no system partition, DNS, proxy, route or CA changes.'}
    state_file.write_text(json.dumps(state, indent=2), encoding='utf-8')
    device.call('shell', shlex.join(['mkdir', remote]))
    for name in ('hosts.original', 'hosts.override'):
        device.call('push', str(directory/name), remote+'/'+name)
        device.root('chmod', '644', remote+'/'+name)
    device.root('mount', '--bind', remote+'/hosts.override', TARGET)
    if digest(device.current()) != state['override_sha256']:
        device.root('umount', TARGET)
        state['status'] = 'verification_failed_binding_removed'
        state_file.write_text(json.dumps(state, indent=2), encoding='utf-8')
        raise RuntimeError('Bound hosts content did not match the staged override; binding removed')
    state['status'] = 'applied'
    state['active_mounts'] = hosts_mounts(device.mountinfo())
    state_file.write_text(json.dumps(state, indent=2), encoding='utf-8')
    return state


def rollback(device, state_file):
    state_file = Path(state_file)
    state = json.loads(state_file.read_text(encoding='utf-8'))
    if state.get('serial') != device.serial or state.get('target') != TARGET:
        raise ValueError('State belongs to a different device or target')
    if state.get('original_mounts') or state.get('status') != 'applied':
        raise ValueError('State is not an applied, originally unmounted hosts binding')
    if digest(device.current()) != state['override_sha256']:
        raise ValueError('Current hosts changed since apply; refusing to unmount another override')
    mounted = hosts_mounts(device.mountinfo())
    if mounted != state.get('active_mounts'):
        raise ValueError('Hosts mount changed since apply; refusing to remove it')
    device.root('umount', TARGET)
    if digest(device.current()) != state['original_sha256']:
        raise RuntimeError('Binding removed, but original hosts hash needs manual inspection')
    state['status'] = 'rolled_back'
    state_file.write_text(json.dumps(state, indent=2), encoding='utf-8')
    return state


def guest_setting(device, *args):
    return device.call('shell', shlex.join(['settings', *args])).decode('utf-8').strip()


def proxy_on(device, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    state_file = directory/'guest-proxy.json'
    if state_file.exists():
        raise ValueError('Proxy backup already exists; preserve it and use a new record directory')
    original = {key: guest_setting(device, 'get', 'global', key) for key in PROXY_KEYS}
    state = {'kind': 'guest_proxy', 'serial': device.serial, 'proxy': '10.0.2.2:7890',
             'original': original, 'status': 'prepared'}
    state_file.write_text(json.dumps(state, indent=2), encoding='utf-8')
    guest_setting(device, 'put', 'global', 'http_proxy', state['proxy'])
    if guest_setting(device, 'get', 'global', 'http_proxy') != state['proxy']:
        raise RuntimeError('Guest proxy was not applied')
    state['status'] = 'applied'
    state_file.write_text(json.dumps(state, indent=2), encoding='utf-8')
    return state


def proxy_off(device, state_file):
    state_file = Path(state_file)
    state = json.loads(state_file.read_text(encoding='utf-8'))
    if state.get('kind') != 'guest_proxy' or state.get('serial') != device.serial:
        raise ValueError('Proxy backup belongs to a different device or task')
    current = {key: guest_setting(device, 'get', 'global', key) for key in PROXY_KEYS}
    if current != state['original'] and current['http_proxy'] != state['proxy']:
        raise ValueError('Guest proxy changed outside this task; refusing to overwrite it')
    for key, value in state['original'].items():
        if value == 'null':
            guest_setting(device, 'delete', 'global', key)
        else:
            guest_setting(device, 'put', 'global', key, value)
    restored = {key: guest_setting(device, 'get', 'global', key) for key in PROXY_KEYS}
    if restored != state['original']:
        raise RuntimeError('Guest proxy restoration failed')
    state['status'] = 'restored'
    state_file.write_text(json.dumps(state, indent=2), encoding='utf-8')
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adb', required=True, type=Path)
    parser.add_argument('--serial', required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    apply_parser = commands.add_parser('apply')
    apply_parser.add_argument('--record-dir', required=True, type=Path)
    rollback_parser = commands.add_parser('rollback')
    rollback_parser.add_argument('--state', required=True, type=Path)
    proxy_on_parser = commands.add_parser('proxy-on')
    proxy_on_parser.add_argument('--record-dir', required=True, type=Path)
    proxy_off_parser = commands.add_parser('proxy-off')
    proxy_off_parser.add_argument('--state', required=True, type=Path)
    commands.add_parser('status')
    args = parser.parse_args()
    device = Device(args.adb, args.serial)
    if args.command == 'apply':
        state = apply(device, args.record_dir)
    elif args.command == 'rollback':
        state = rollback(device, args.state)
    elif args.command == 'proxy-on':
        state = proxy_on(device, args.record_dir)
    elif args.command == 'proxy-off':
        state = proxy_off(device, args.state)
    else:
        state = {'serial': args.serial, 'hosts_sha256': digest(device.current()),
                 'hosts_mounts': hosts_mounts(device.mountinfo())}
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
