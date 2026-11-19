"""Headless JSON-lines control of the existing assistant worker and MuMu SDK.

Run with --folder and --index, then send {"id":1,"command":"status"} on stdin.
Use --serve for a 127.0.0.1 endpoint and --command for separate CLI requests.
start modes are hayabusa (inputs), observe (no inputs), and run (normal assistant).
"""
from ninja3_assistant.paths import project_root
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen

from ninja3_assistant.activities.daily import CHORES, DEFAULT_PET_FEED_REWARD, PET_FEED_REWARDS


ROOT = project_root()


class AssistantController:
    def __init__(self, folder, index, *, python=None, worker=None, emit=None):
        if index < 0:
            raise ValueError('instance index must not be negative')
        self.folder, self.index = Path(folder).resolve(), index
        self.python = python or sys.executable
        self.worker = Path(worker) if worker is not None else ROOT/'src/ninja3_assistant/entrypoints/worker.py'
        self.emit = emit or (lambda value: None)
        self.lock = threading.Condition(threading.RLock())
        self.process = self.thread = self.preparing = None
        self.stop_requested = threading.Event()
        self.paused = False
        self.state, self.mode = 'idle', None
        self.activity_enabled, self.activity_mode = False, 'daily'
        self.weekend_enabled, self.weekend_mode = False, 'daily'
        self.xumi_enabled = False
        self.daily_tasks, self.bounty_grade = [], None
        self.bounty_active_invite = self.bounty_passive_invite = True
        self.pet_feed_reward = None
        self.hide_game_window = False
        self.last_status = self.backend = self.error = None
        self.metrics, self.events = {}, deque(maxlen=30)

    def status(self):
        with self.lock:
            return {'state': self.state, 'mode': self.mode, 'folder': str(self.folder), 'index': self.index,
                    'activity_enabled': self.activity_enabled, 'activity_mode': self.activity_mode,
                    'weekend_enabled': self.weekend_enabled, 'weekend_mode': self.weekend_mode,
                    'xumi_enabled': self.xumi_enabled,
                    'daily_tasks':list(self.daily_tasks), 'bounty_grade':self.bounty_grade,
                    'bounty_active_invite':self.bounty_active_invite,
                    'bounty_passive_invite':self.bounty_passive_invite,
                    'pet_feed_reward':self.pet_feed_reward,
                    'hide_game_window':self.hide_game_window,
                    'worker_pid': self.process.pid if self.process is not None and self.process.poll() is None else None,
                    'status': self.last_status, 'backend': self.backend, 'metrics': dict(self.metrics),
                    'error': self.error, 'recent_events': list(self.events)}

    def _event(self, event):
        with self.lock:
            kind, value = event.get('event'), event.get('value')
            self.events.append(event)
            if kind == 'state' and self.state != 'stopping':
                self.state = value
            elif kind == 'status':
                self.last_status = value
            elif kind == 'backend':
                self.backend = value
            elif kind == 'metrics':
                self.metrics = value
            elif kind == 'error':
                self.error = str(value)
            self.lock.notify_all()
        self.emit({'event': 'worker', 'value': event})

    def _manager(self, *arguments, timeout=10):
        result = subprocess.run([str(self.folder/'nx_main/MuMuManager.exe'), *arguments],
            capture_output=True, encoding='utf-8-sig', errors='replace', timeout=timeout,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        payload = json.loads(result.stdout)
        if result.returncode or payload.get('errcode', 0) or payload.get('error_code', 0):
            raise RuntimeError('MuMu command failed: '+str(payload))
        return payload

    def hide(self):
        return self._manager('control', '--vmindex', str(self.index), 'hide_window')

    def launch(self,hide_game_window=False):
        deadline = time.monotonic()+60
        info = {}

        def remaining():
            seconds = deadline-time.monotonic()
            if seconds <= 0:
                raise RuntimeError('MuMu launch timed out after 60 seconds: '+json.dumps({
                    'is_process_started': info.get('is_process_started', False),
                    'is_android_started': info.get('is_android_started', False)}))
            return seconds

        def call(*arguments):
            return self._manager(*arguments, timeout=min(10, remaining()))

        info = call('info', '--vmindex', str(self.index))
        if not info.get('is_process_started'):
            call('control', '--vmindex', str(self.index), 'launch')
        while not (info.get('is_process_started') and info.get('is_android_started')):
            if hide_game_window:
                try:
                    call('control', '--vmindex', str(self.index), 'hide_window')
                except RuntimeError:
                    pass
            time.sleep(min(.5, remaining()))
            info = call('info', '--vmindex', str(self.index))
        call('control', '--vmindex', str(self.index), 'hide_window' if hide_game_window else 'show_window')
        return info

    def start(self, mode='hayabusa', *, activity_enabled=False, activity_mode='daily',hide_game_window=False,
              weekend_enabled=False, weekend_mode='daily', xumi_enabled=False, daily_tasks=(),
              bounty_grade=None, pet_feed_reward=None,
              bounty_active_invite=True, bounty_passive_invite=True):
        if mode not in ('hayabusa', 'observe', 'run'):
            raise ValueError('mode must be hayabusa, observe, or run')
        if not isinstance(activity_enabled, bool):
            raise ValueError('activity_enabled must be a boolean')
        if not isinstance(hide_game_window,bool):
            raise ValueError('hide_game_window must be a boolean')
        if activity_mode not in ('daily', 'continuous'):
            raise ValueError('activity_mode must be daily or continuous')
        if activity_enabled and mode != 'run':
            raise ValueError('activity is available only in run mode')
        if not isinstance(weekend_enabled,bool):
            raise ValueError('weekend_enabled must be a boolean')
        if weekend_mode not in ('daily','25'):
            raise ValueError('weekend_mode must be daily or 25')
        if weekend_enabled and mode != 'run':
            raise ValueError('weekend is available only in run mode')
        if not isinstance(xumi_enabled, bool):
            raise ValueError('xumi_enabled must be a boolean')
        if xumi_enabled and mode != 'run':
            raise ValueError('xumi is available only in run mode')
        if not isinstance(daily_tasks,(list,tuple)) or any(
                not isinstance(kind,str) or kind not in CHORES for kind in daily_tasks):
            raise ValueError('daily_tasks must be a list of known daily chores')
        if daily_tasks and mode != 'run':
            raise ValueError('daily chores are available only in run mode')
        if bounty_grade is not None and (bounty_grade not in ('D','C','B','A') or
                                        mode != 'run' or 'bounty' not in daily_tasks):
            raise ValueError('bounty_grade must be D, C, B, or A and requires the bounty daily task')
        if not isinstance(bounty_active_invite,bool) or not isinstance(bounty_passive_invite,bool):
            raise ValueError('bounty invitation options must be booleans')
        if not (bounty_active_invite and bounty_passive_invite) and not (
                mode == 'run' and {'bounty','bounty_s'}.intersection(daily_tasks)):
            raise ValueError('bounty invitation options require a bounty daily task in run mode')
        if pet_feed_reward is not None and (not isinstance(pet_feed_reward,str) or
                                           pet_feed_reward not in PET_FEED_REWARDS or
                                           mode != 'run' or 'pet_feed' not in daily_tasks):
            raise ValueError('pet_feed_reward must be a known reward and requires the pet_feed daily task')
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError('an assistant worker is already active')
            self.stop_requested = threading.Event()
            self.process = self.preparing = None
            self.paused = False
            self.state, self.mode = 'starting', mode
            self.activity_enabled, self.activity_mode = activity_enabled, activity_mode
            self.weekend_enabled, self.weekend_mode = weekend_enabled, weekend_mode
            self.xumi_enabled = xumi_enabled
            self.daily_tasks = list(dict.fromkeys(daily_tasks))
            self.bounty_grade = bounty_grade or ('D' if 'bounty' in self.daily_tasks else None)
            self.bounty_active_invite = bounty_active_invite
            self.bounty_passive_invite = bounty_passive_invite
            self.pet_feed_reward = ((pet_feed_reward or DEFAULT_PET_FEED_REWARD)
                                    if 'pet_feed' in self.daily_tasks else None)
            self.hide_game_window = hide_game_window
            self.last_status = self.backend = self.error = None
            self.metrics, self.events = {}, deque(maxlen=30)
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self.status()

    def _prepare(self):
        from ninja3_assistant.device.mumu import prepare_game
        from ninja3_assistant.runtime.runner import RunControl
        control = RunControl(stop=self.stop_requested,hide_game_window=self.hide_game_window)
        with self.lock:
            self.preparing = control
            if self.paused:
                control.pause()

        def status(message):
            self._event({'event': 'status', 'value': message})
        prepare_game(str(self.folder), self.index, control, status)

    def _run(self):
        process = None
        try:
            self._prepare()
            with self.lock:
                if self.stop_requested.is_set():
                    return
                command = ([str(self.worker)] if self.worker.suffix.lower() == '.exe'
                           else [self.python, '-u', str(self.worker)])
                command += ['--mode',
                           'run' if self.mode == 'run' else 'hayabusa',
                           '--folder', str(self.folder), '--index', str(self.index)]
                if self.mode == 'observe':
                    command.append('--observe-only')
                if self.activity_enabled:
                    command += ['--activity-enabled', '--activity-mode', self.activity_mode]
                if self.weekend_enabled:
                    command += ['--weekend-enabled', '--weekend-mode', self.weekend_mode]
                if self.xumi_enabled:
                    command.append('--xumi-enabled')
                if self.daily_tasks:
                    command += ['--daily-tasks',*self.daily_tasks]
                if self.bounty_grade is not None:
                    command += ['--bounty-grade',self.bounty_grade]
                if not self.bounty_active_invite:
                    command.append('--no-bounty-active-invite')
                if not self.bounty_passive_invite:
                    command.append('--no-bounty-passive-invite')
                if self.pet_feed_reward is not None:
                    command += ['--pet-feed-reward',self.pet_feed_reward]
                if self.hide_game_window:
                    command.append('--hide-game-window')
                process = self.process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
                    errors='replace', bufsize=1, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if self.paused:
                    self._send('pause')
            stderr = threading.Thread(target=self._read_errors, args=(process,), daemon=True)
            stderr.start()
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    self._event({'event': 'output', 'value': line.rstrip()})
                    continue
                if not isinstance(event, dict):
                    self._event({'event': 'output', 'value': line.rstrip()})
                    continue
                self._event(event)
            code = process.wait()
            stderr.join(timeout=1)
            if code:
                raise RuntimeError(f'assistant worker exited with code {code}')
        except Exception as error:
            with self.lock:
                self.error = str(error)
            self.emit({'event': 'error', 'value': str(error)})
        finally:
            if process is not None:
                self._finish_process(process)
                for stream in (process.stdin, process.stdout, process.stderr):
                    stream.close()
            with self.lock:
                self.preparing = None
                self.state = 'error' if self.error else 'stopped'
                self.lock.notify_all()
            self.emit({'event': 'controller_state', 'value': self.state})

    def _read_errors(self, process):
        for line in process.stderr:
            self._event({'event': 'stderr', 'value': line.rstrip()})

    def _send(self, command):
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({'command': command})+'\n')
                self.process.stdin.flush()
            except (BrokenPipeError, ValueError):
                if self.process.poll() is None:
                    raise

    def pause(self):
        with self.lock:
            if self.state not in ('starting', 'running'):
                raise RuntimeError('pause requires a starting or running worker')
            self.paused = True
            if self.preparing is not None:
                self.preparing.pause()
            if self.process is None:
                self.state = 'paused'
            else:
                self._send('pause')
                if not self.lock.wait_for(lambda: self.state in ('paused', 'stopped', 'stopping', 'error'), timeout=3):
                    raise RuntimeError('worker has not acknowledged pause')
        return self.status()

    def resume(self):
        with self.lock:
            if self.state != 'paused':
                raise RuntimeError('resume requires a paused worker')
            self.paused = False
            if self.preparing is not None:
                self.preparing.resume()
            if self.process is None:
                self.state = 'starting'
            else:
                self._send('resume')
                if not self.lock.wait_for(lambda: self.state != 'paused', timeout=3):
                    raise RuntimeError('worker has not acknowledged resume')
        return self.status()

    @staticmethod
    def _finish_process(process):
        if process.poll() is not None:
            return
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def stop(self):
        with self.lock:
            thread, process = self.thread, self.process
            if thread is None or not thread.is_alive():
                return self.status()
            self.state = 'stopping'
            self.stop_requested.set()
            self._send('stop')
        if process is not None:
            self._finish_process(process)
        thread.join(timeout=12)
        if thread.is_alive():
            raise RuntimeError('assistant worker has not stopped yet')
        return self.status()

    def screenshot(self, output=None):
        import cv2
        from ninja3_assistant.device.mumu import MuMu
        path = Path(output).resolve() if output else ROOT/'data/assistant-control'/f'frame-{time.time_ns()}.png'
        if path.exists():
            raise FileExistsError('screenshot output already exists')
        device = MuMu(str(self.folder), self.index)
        try:
            frame = device.capture()
        finally:
            device.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            raise RuntimeError('could not save screenshot')
        return {'path': str(path), 'width': frame.shape[1], 'height': frame.shape[0]}

    def command(self, message):
        command = message.get('command')
        if command == 'start':
            return self.start(message.get('mode', 'hayabusa'),
                              activity_enabled=message.get('activity_enabled', False),
                              activity_mode=message.get('activity_mode', 'daily'),
                              hide_game_window=message.get('hide_game_window',False),
                              weekend_enabled=message.get('weekend_enabled',False),
                              weekend_mode=message.get('weekend_mode','daily'),
                              xumi_enabled=message.get('xumi_enabled', False),
                              daily_tasks=message.get('daily_tasks',()),
                              bounty_grade=message.get('bounty_grade'),
                              bounty_active_invite=message.get('bounty_active_invite',True),
                              bounty_passive_invite=message.get('bounty_passive_invite',True),
                              pet_feed_reward=message.get('pet_feed_reward'))
        if command == 'launch':
            return self.launch(hide_game_window=message.get('hide_game_window',False))
        if command == 'screenshot':
            return self.screenshot(message.get('output'))
        if command in ('status', 'pause', 'resume', 'stop', 'hide', 'launch'):
            return getattr(self, command)()
        if command == 'exit':
            return self.stop()
        raise ValueError('unknown command')


class ControlServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, controller, port):
        self.controller = controller
        super().__init__(('127.0.0.1', port), ControlHandler)


class ControlHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        message = {}
        try:
            if (self.path != '/command' or self.headers.get('Origin') is not None or
                    self.headers.get('Host') != f'127.0.0.1:{self.server.server_port}' or
                    self.headers.get('Content-Type', '').split(';')[0] != 'application/json'):
                raise ValueError('only local JSON control requests are accepted')
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 65536:
                raise ValueError('invalid request length')
            message = json.loads(self.rfile.read(length))
            if not isinstance(message, dict):
                raise ValueError('command must be a JSON object')
            result = self.server.controller.command(message)
            response = {'id': message.get('id'), 'ok': True, 'result': result}
            code = 200
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            response = {'id': message.get('id') if isinstance(message, dict) else None,
                        'ok': False, 'error': str(error)}
            code = 400
        payload = json.dumps(response, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        if response['ok'] and message.get('command') == 'exit':
            threading.Thread(target=self.server.shutdown, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder')
    parser.add_argument('--index', type=int)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--worker', help='Python worker script or packaged Ninja3Worker.exe; defaults to the package entrypoint')
    parser.add_argument('--serve', action='store_true', help='serve control requests on 127.0.0.1')
    parser.add_argument('--port', type=int, default=17863)
    parser.add_argument('--command', choices=('start', 'pause', 'resume', 'stop', 'status',
                                             'screenshot', 'hide', 'launch', 'exit'))
    parser.add_argument('--mode', choices=('hayabusa', 'observe', 'run'), default='hayabusa')
    parser.add_argument('--activity-enabled', action='store_true')
    parser.add_argument('--activity-mode', choices=('daily', 'continuous'), default='daily')
    parser.add_argument('--weekend-enabled',action='store_true')
    parser.add_argument('--weekend-mode',choices=('daily','25'),default='daily')
    parser.add_argument('--xumi-enabled', action='store_true')
    parser.add_argument('--daily-tasks',nargs='+',choices=tuple(CHORES),default=[])
    parser.add_argument('--bounty-grade',choices=('D','C','B','A'))
    parser.add_argument('--no-bounty-active-invite',dest='bounty_active_invite',action='store_false')
    parser.add_argument('--no-bounty-passive-invite',dest='bounty_passive_invite',action='store_false')
    parser.add_argument('--pet-feed-reward',choices=tuple(PET_FEED_REWARDS))
    parser.add_argument('--hide-game-window',action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.index is not None and args.index < 0:
        parser.error('index must not be negative')
    if args.command and args.serve:
        parser.error('--command and --serve cannot be combined')
    if not args.command and (args.folder is None or args.index is None):
        parser.error('--folder and --index are required for the controller')
    if args.activity_enabled and (args.command != 'start' or args.mode != 'run'):
        parser.error('activity requires --command start --mode run')
    if args.weekend_enabled and (args.command != 'start' or args.mode != 'run'):
        parser.error('weekend requires --command start --mode run')
    if args.xumi_enabled and (args.command != 'start' or args.mode != 'run'):
        parser.error('xumi requires --command start --mode run')
    if args.daily_tasks and (args.command != 'start' or args.mode != 'run'):
        parser.error('daily chores require --command start --mode run')
    if args.bounty_grade is not None and (args.command != 'start' or args.mode != 'run' or
                                         'bounty' not in args.daily_tasks):
        parser.error('bounty-grade requires the bounty daily task in run mode')
    if not (args.bounty_active_invite and args.bounty_passive_invite) and not (
            args.command == 'start' and args.mode == 'run' and
            {'bounty','bounty_s'}.intersection(args.daily_tasks)):
        parser.error('bounty invitation options require a bounty daily task in run mode')
    if args.pet_feed_reward is not None and (args.command != 'start' or args.mode != 'run' or
                                             'pet_feed' not in args.daily_tasks):
        parser.error('pet-feed-reward requires the pet_feed daily task in run mode')
    if args.hide_game_window and args.command not in ('start','launch'):
        parser.error('hide-game-window requires --command start or launch')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
    output_lock = threading.Lock()

    def emit(value):
        with output_lock:
            print(json.dumps(value, ensure_ascii=False), flush=True)
    if args.command:
        message = {'command': args.command, 'mode': args.mode}
        if args.command == 'start':
            message.update(activity_enabled=args.activity_enabled, activity_mode=args.activity_mode)
            if args.weekend_enabled:
                message.update(weekend_enabled=True,weekend_mode=args.weekend_mode)
            if args.xumi_enabled:
                message['xumi_enabled'] = True
            if args.daily_tasks:
                message['daily_tasks'] = args.daily_tasks
            if args.bounty_grade is not None:
                message['bounty_grade'] = args.bounty_grade
            if not args.bounty_active_invite:
                message['bounty_active_invite'] = False
            if not args.bounty_passive_invite:
                message['bounty_passive_invite'] = False
            if args.pet_feed_reward is not None:
                message['pet_feed_reward'] = args.pet_feed_reward
        if args.command in ('start','launch') and args.hide_game_window:
            message['hide_game_window'] = True
        if args.output:
            message['output'] = args.output
        request = Request(f'http://127.0.0.1:{args.port}/command',
                          data=json.dumps(message).encode('utf-8'), headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=75 if args.command == 'launch' else 30) as response:
                emit(json.load(response))
        except OSError as error:
            if hasattr(error, 'read'):
                emit(json.loads(error.read()))
            else:
                emit({'ok': False, 'error': str(error)})
            return 1
        return 0
    controller = AssistantController(args.folder, args.index, python=args.python, worker=args.worker, emit=emit)
    emit({'event': 'ready', 'value': controller.status()})
    if args.serve:
        server = ControlServer(controller, args.port)
        emit({'event': 'listening', 'value': f'http://127.0.0.1:{server.server_port}'})
        try:
            server.serve_forever(poll_interval=.1)
        finally:
            server.server_close()
            controller.stop()
        return 0
    try:
        for line in sys.stdin:
            message = {}
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError('command must be a JSON object')
                result = controller.command(message)
                emit({'id': message.get('id'), 'ok': True, 'result': result})
                if message.get('command') == 'exit':
                    break
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                emit({'id': message.get('id') if isinstance(message, dict) else None,
                      'ok': False, 'error': str(error)})
    finally:
        controller.stop()


if __name__ == '__main__':
    raise SystemExit(main())
