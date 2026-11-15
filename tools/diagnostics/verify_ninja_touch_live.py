"""Record MuMu touch events during normal stop and an owner-process crash.

Run only while an isolated, controllable game scene is already visible. This
tool does not navigate the emulator or start the assistant. It records its own
subprocess output and Android events in a new directory. Host perf_counter and
Android uptime remain separate clock domains in the resulting report.
"""
from ninja3_assistant.paths import project_root
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
PROJECT = project_root()
DEFAULT_ADB = Path('D:/Program Files/Netease/MuMu/nx_main/adb.exe')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def wait_until(predicate, timeout, message):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.02)
    raise TimeoutError(message)


def read_text(path):
    return path.read_text(encoding='utf-8', errors='replace') if path.exists() else ''


def start_native(executable, folder, index, output):
    stream = output.open('wb')
    proc = subprocess.Popen([str(executable), '--folder', str(folder), '--index', str(index)],
                            stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.DEVNULL,
                            creationflags=FLAGS)
    try:
        wait_until(lambda: 'READY' in read_text(output).splitlines() or proc.poll() is not None,
                   8, 'native process did not become ready')
        if proc.poll() is not None:
            raise RuntimeError('native process exited before READY: ' + read_text(output)[-500:])
    except BaseException:
        proc.stdin.close()
        try:
            proc.wait(timeout=3)
        finally:
            stream.close()
        raise
    return proc, stream


def send(proc, command):
    proc.stdin.write((command + '\n').encode('ascii'))
    proc.stdin.flush()


def owner(args):
    """A disposable process; the verifier terminates it without cleanup."""
    proc, stream = start_native(args.exe, args.folder, args.index, args.output / 'native.log')
    send(proc, 'START 250')
    wait_until(lambda: 'STARTED' in read_text(args.output / 'native.log').splitlines()
               or proc.poll() is not None, 4, 'native process did not start')
    if proc.poll() is not None:
        raise RuntimeError('native process exited before STARTED')
    write_json(args.output / 'owner-ready.json', {'owner_pid': os.getpid(), 'native_pid': proc.pid})
    try:
        while proc.poll() is None:
            send(proc, 'RENEW 250')
            time.sleep(.05)
    finally:
        # Normally unreachable in the crash test; still safe for manual stops.
        proc.stdin.close()
        try:
            proc.wait(timeout=3)
        finally:
            stream.close()


def device_uptime(adb, serial):
    before = time.perf_counter()
    result = subprocess.run([str(adb), '-s', serial, 'shell', 'cat', '/proc/uptime'],
                            capture_output=True, check=True, timeout=5, creationflags=FLAGS)
    after = time.perf_counter()
    return {'android_uptime_seconds': float(result.stdout.split()[0]),
            'host_perf_before': before, 'host_perf_after': after}


def initial_slot(adb, serial, event_device):
    result = subprocess.run([str(adb), '-s', serial, 'shell', 'getevent', '-lp', event_device],
                            capture_output=True, check=True, timeout=5, creationflags=FLAGS)
    text = result.stdout.decode('utf-8', errors='replace')
    if 'Touchscreen' not in text or 'ABS_MT_SLOT' not in text:
        raise RuntimeError('event device is not a multitouch touchscreen')
    match = re.search(r'ABS_MT_SLOT\s*:\s*value\s+(\d+)', text)
    if not match:
        raise RuntimeError('cannot determine the initial Android multitouch slot')
    return int(match.group(1))


def parse_touch_events(path, first_slot, late_cutoff=None):
    """Count starts/releases on Android uptime; do not align them to host time."""
    slot = first_slot
    active = {}
    starts = releases = duplicate_starts = unmatched_releases = late_starts = 0
    last_time = None
    pattern = re.compile(r'\[\s*([0-9.]+)\]\s+\S+\s+(\S+)\s+(\S+)')
    for line in read_text(path).splitlines():
        match = pattern.search(line)
        if not match:
            continue
        stamp, code, value = match.groups()
        stamp = float(stamp)
        last_time = stamp
        if code == 'ABS_MT_SLOT':
            slot = int(value, 16)
        elif code == 'ABS_MT_TRACKING_ID':
            if value.lower() == 'ffffffff':
                releases += 1
                if slot not in active:
                    unmatched_releases += 1
                active.pop(slot, None)
            else:
                starts += 1
                if slot in active:
                    duplicate_starts += 1
                active[slot] = int(value, 16)
                if late_cutoff is not None and stamp > late_cutoff:
                    late_starts += 1
    return {'starts': starts, 'releases': releases, 'active_slots_at_end': active,
            'duplicate_starts': duplicate_starts, 'unmatched_releases': unmatched_releases,
            'late_starts_after_cutoff': late_starts, 'last_event_android_uptime': last_time}


def pid_has_exited(pid):
    """Check the orphaned native process without terminating it."""
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.WaitForSingleObject.restype = ctypes.c_uint
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: no such process.
            return True
        raise OSError(error, 'cannot inspect the orphaned native process')
    try:
        return kernel.WaitForSingleObject(handle, 0) == 0
    finally:
        kernel.CloseHandle(handle)


def record_case(args, case):
    output = args.output / case
    output.mkdir()
    first_slot = initial_slot(args.adb, args.serial, args.event_device)
    event_file = (output / 'getevent.txt').open('wb')
    collector = subprocess.Popen([str(args.adb), '-s', args.serial, 'shell', '-t',
                                  'getevent', '-lt', args.event_device],
                                 stdout=event_file, stderr=subprocess.DEVNULL, creationflags=FLAGS)
    proc = native = stream = None
    result = {'case': case, 'initial_slot': first_slot, 'lease_ms': 250,
              'clocks': 'host_perf is not directly comparable to android_uptime'}
    try:
        time.sleep(.5)
        if collector.poll() is not None:
            raise RuntimeError('Android event collector exited early')
        if case == 'normal-stop':
            native, stream = start_native(args.exe, args.folder, args.index, output / 'native.log')
            send(native, 'START 250')
            until = time.perf_counter() + args.hold
            while time.perf_counter() < until:
                send(native, 'RENEW 250')
                time.sleep(.05)
            result['stop_requested_host_perf'] = time.perf_counter()
            send(native, 'STOP')
            wait_until(lambda: 'STOPPED' in read_text(output / 'native.log').splitlines()
                       or native.poll() is not None, 3, 'STOP was not acknowledged')
            result['stopped_ack_host_perf'] = time.perf_counter()
            result['stop_acknowledged'] = 'STOPPED' in read_text(output / 'native.log').splitlines()
            result['android_uptime_after_ack'] = device_uptime(args.adb, args.serial)
            time.sleep(args.settle)
            send(native, 'QUIT')
            native.stdin.close()
            result['native_exit_code'] = native.wait(timeout=4)
        else:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                     '--internal-owner', '--exe', str(args.exe), '--folder', str(args.folder),
                                     '--index', str(args.index), '--output', str(output)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    creationflags=FLAGS)
            ready = output / 'owner-ready.json'
            wait_until(lambda: ready.exists() or proc.poll() is not None, 12, 'owner did not start')
            if not ready.exists():
                raise RuntimeError('test owner exited before native was ready')
            info = json.loads(ready.read_text(encoding='utf-8'))
            result.update(info)
            time.sleep(args.hold)
            result['owner_terminated_host_perf'] = time.perf_counter()
            proc.terminate()  # Windows TerminateProcess: no Python finally runs.
            result['owner_exit_code'] = proc.wait(timeout=4)
            result['android_uptime_after_owner_exit'] = device_uptime(args.adb, args.serial)
            wait_until(lambda: pid_has_exited(info['native_pid']), 4,
                       'orphaned native process did not exit; do not assume contacts were released')
            result['native_exited_host_perf'] = time.perf_counter()
            time.sleep(args.settle)
    finally:
        if native is not None and native.poll() is None:
            # EOF asks it to release. Do not force-kill a possibly held contact.
            native.stdin.close()
            try:
                native.wait(timeout=4)
            except subprocess.TimeoutExpired:
                result['cleanup_error'] = 'native still running after stdin close'
        if stream is not None:
            stream.close()
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=4)
        if collector.poll() is None:
            collector.terminate()
        collector.wait(timeout=4)
        event_file.close()
        sample = result.get('android_uptime_after_ack') or result.get('android_uptime_after_owner_exit')
        # 300ms is a tolerance after the device-uptime sample, not an E2E latency.
        cutoff = sample['android_uptime_seconds'] + .3 if sample else None
        result['touch'] = parse_touch_events(output / 'getevent.txt', first_slot, cutoff)
        result['late_cutoff_android_uptime'] = cutoff
        result['native_log_tail'] = read_text(output / 'native.log').splitlines()[-12:]
        write_json(output / 'report.json', result)
    touch = result['touch']
    if (not touch['starts'] or touch['active_slots_at_end'] or touch['duplicate_starts'] or
            touch['unmatched_releases'] or touch['late_starts_after_cutoff']):
        raise RuntimeError(f'{case}: Android event evidence is incomplete or contacts were not released')
    if case == 'normal-stop' and (not result['stop_acknowledged'] or result.get('native_exit_code') != 0):
        raise RuntimeError('normal stop was not acknowledged or native exited with an error')
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, default=PROJECT / 'build/ninja-touch/NinjaTouch.exe')
    parser.add_argument('--folder', type=Path, default=Path('D:/Program Files/Netease/MuMu'))
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--serial', default='127.0.0.1:16416')
    parser.add_argument('--adb', type=Path, default=DEFAULT_ADB)
    parser.add_argument('--event-device', default='/dev/input/event4')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hold', type=float, default=1.0)
    parser.add_argument('--settle', type=float, default=.8)
    parser.add_argument('--internal-owner', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal_owner:
        owner(args)
        return
    if os.name != 'nt' or not args.exe.is_file() or not args.adb.is_file():
        parser.error('Windows, an existing NinjaTouch.exe and MuMu adb.exe are required')
    if args.hold < .3 or args.settle < .6:
        parser.error('hold must be >= .3s and settle >= .6s')
    args.output.mkdir(parents=True, exist_ok=False)
    for case in ('normal-stop', 'owner-death'):
        record_case(args, case)


if __name__ == '__main__':
    main()
