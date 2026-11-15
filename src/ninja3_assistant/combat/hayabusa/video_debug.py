"""Interactive, read-only video inspection for the WinUI assistant."""
import base64
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path
import tempfile
import threading
import time

import cv2
import numpy as np

from ninja3_assistant.combat.hayabusa.scene import HayabusaScene
from tools.replay.replay_hayabusa import readable_video


class _Cancelled(InterruptedError):
    pass


class _Requests:
    def __init__(self, commands, emit):
        self.condition = threading.Condition()
        self.pending = None
        self.version = 0
        self.stopped = False
        self.commands, self.emit = commands, emit

    def read(self):
        try:
            for line in self.commands:
                try:
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        raise ValueError('无效的视频调试命令')
                    if request.get('command') == 'stop':
                        break
                    if request.get('command') != 'frame':
                        raise ValueError('不支持的视频调试命令')
                    if (type(request.get('frame')) is not int or type(request.get('request_id')) is not int
                            or request['frame'] < 0 or request['request_id'] < 0):
                        raise ValueError('帧号和请求编号必须是非负整数')
                    with self.condition:
                        self.version += 1
                        self.pending = (self.version, request)
                        self.condition.notify_all()
                except (ValueError, TypeError) as error:
                    self.emit('video_error', {'message': str(error)})
        finally:
            with self.condition:
                self.stopped = True
                self.condition.notify_all()

    def take(self):
        with self.condition:
            self.condition.wait_for(lambda: self.stopped or self.pending is not None)
            if self.stopped:
                return None
            result, self.pending = self.pending, None
            return result

    def cancelled(self, version):
        with self.condition:
            return self.stopped or self.version != version


class VideoDebugSession:
    """Keeps causal analysis state only for successive source frames."""
    def __init__(self, video, cancelled=lambda:False):
        self.resources = ExitStack()
        self.scene = self.capture = None
        self.current = None
        self.now = 0.
        self.trails = {}
        source = Path(video).resolve()
        try:
            if cancelled():
                raise _Cancelled()
            if not source.is_file():
                raise ValueError('视频文件不存在')
            normalized = self.resources.enter_context(readable_video(source,cancelled=cancelled))
            if cancelled():
                raise _Cancelled()
            self.capture = cv2.VideoCapture(str(normalized))
            self.resources.callback(self.capture.release)
            if cancelled():
                raise _Cancelled()
            if not self.capture.isOpened():
                raise ValueError('无法打开视频，请检查文件格式或缓存是否完整')
            self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
            count = float(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not isfinite(self.fps) or not 0 < self.fps <= 240 or not isfinite(count) or count < 1:
                raise ValueError('视频没有可靠的帧率或帧数')
            self.total_frames = int(count)
            temporary = self.resources.enter_context(tempfile.TemporaryDirectory(prefix='hayabusa-video-'))
            if cancelled():
                raise _Cancelled()
            self.scene = HayabusaScene(debug=True, clock=lambda: self.now, record_root=temporary)
            self.resources.callback(self.scene.close)
            if cancelled():
                raise _Cancelled()
            self.info = {'path': str(source), 'fps': self.fps, 'total_frames': self.total_frames,
                'duration': self.total_frames/self.fps, 'backend': self.scene.backend}
        except Exception:
            self.resources.close()
            raise

    def close(self):
        self.resources.close()

    def _reset(self):
        self.scene.release()
        if self.scene.log is not None:
            Path(self.scene.log).unlink(missing_ok=True)
        self.trails.clear()
        self.current = None

    def frame(self, target, request_id, cancelled=lambda: False):
        if type(target) is not int or not 0 <= target < self.total_frames:
            raise ValueError('帧号超出视频范围')
        resynced = self.current is None or target != self.current+1
        if resynced:
            self._reset()
            start = max(0, target-int(self.fps*2))
            if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, start):
                raise RuntimeError('视频解码器拒绝跳转')
            self.history_start = start/self.fps
        else:
            start = target
        stride = max(1, round(self.fps/15)) if resynced else 1
        previous_time = None
        before = time.perf_counter()
        try:
            for index in range(start, target+1):
                if cancelled():
                    raise _Cancelled()
                if not self.capture.grab():
                    raise RuntimeError('视频在请求帧之前结束，缓存可能不完整')
                actual_frame = self.capture.get(cv2.CAP_PROP_POS_FRAMES)-1
                stamp = self.capture.get(cv2.CAP_PROP_POS_MSEC)/1000
                if (not isfinite(actual_frame) or abs(actual_frame-index) > .1 or not isfinite(stamp)
                        or abs(stamp-index/self.fps) > max(.1, 2/self.fps)
                        or (previous_time is not None and stamp <= previous_time)):
                    raise RuntimeError('视频跳转或时间戳不可靠，未使用错误的缓存帧')
                previous_time = stamp
                if index != target and (index-start) % stride:
                    continue
                ok, decoded = self.capture.retrieve()
                if not ok:
                    raise RuntimeError('无法解码视频帧')
                frame = cv2.cvtColor(cv2.resize(decoded, (1280, 720)), cv2.COLOR_BGR2RGB)
                self.now = stamp
                decision = self.scene.step(frame, stamp)
                if decision.action is not None:
                    self.scene.observe(decision.action, stamp)
                tracks = self.scene.pending_record['tracks']
                self._track_history(tracks, stamp)
            if cancelled():
                raise _Cancelled()
            self.current = target
            image = self._annotate(frame, decision.state, tracks)
            ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError('无法生成视频预览')
            recipe = self.scene.policy.recipe
            return {'request_id': request_id, 'frame': target, 'seconds': stamp,
                'jpeg_base64': base64.b64encode(jpeg).decode('ascii'), 'width':1280, 'height':720,
                'screen':asdict(decision.state),
                'recipe':{'id':recipe.recipe_id, 'stage':recipe.stage, 'wave':recipe.wave,
                    'mode_active':recipe.mode_active, 'mode_state':recipe.mode_state,
                    'late_entry':recipe.late_entry, 'reason':recipe.last_reason},
                'action':asdict(decision.action) if decision.action is not None else None,
                'analysis_ms':(time.perf_counter()-before)*1000, 'resynced':resynced,
                'history_start_seconds':self.history_start, 'observation_only':True,
                'executed_actions':0, 'status':decision.status}
        except Exception:
            self.current = None
            raise

    def _track_history(self, tracks, stamp):
        for key in list(self.trails):
            trail = self.trails[key]
            while trail and stamp-trail[0][0] > .7:
                trail.popleft()
            if not trail:
                del self.trails[key]
        for track in tracks:
            key = (track['id'], track['kind'])
            if stamp-track['seen'] <= .12:
                self.trails.setdefault(key, deque(maxlen=30)).append((stamp, track['x'], track['y']))

    def _annotate(self, frame, screen, tracks):
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if screen.rope_y is not None:
            cv2.line(image, (0, int(screen.rope_y)), (1279, int(screen.rope_y)), (80, 210, 80), 1)
        if screen.player_x is not None and screen.player_y is not None:
            cv2.circle(image, (int(screen.player_x), int(screen.player_y)), 18, (0, 230, 230), 2)
        colors = {'fire':(0,160,255), 'blackfire':(230,50,230), 'shuriken':(255,180,30),
                  'slash':(255,230,80), 'flash':(255,255,255)}
        for track in tracks:
            color = colors.get(track['kind'], (150,150,150))
            x, y = track['x'], track['y']
            left, top = max(0,int(x-track['width']/2)), max(0,int(y-track['height']/2))
            right, bottom = min(1279,int(x+track['width']/2)), min(719,int(y+track['height']/2))
            cv2.rectangle(image, (left,top), (right,bottom), color, 2)
            cv2.putText(image, f"{track['kind']}#{track['id']}", (left,max(15,top-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            history = self.trails.get((track['id'],track['kind']), ())
            if len(history) > 1:
                points = np.array([(int(px),int(py)) for _,px,py in history],np.int32)
                cv2.polylines(image, [points], False, color, 2)
            if track['hits'] >= 2:
                future = (int(np.clip(x+track['vx']*.3,0,1279)),int(np.clip(y+track['vy']*.3,0,719)))
                cv2.arrowedLine(image, (int(x),int(y)), future, color, 1, tipLength=.15)
        return image


def run_video_debug(video, emit, commands):
    requests = _Requests(commands, emit)
    threading.Thread(target=requests.read,daemon=True).start()
    session = None
    try:
        session = VideoDebugSession(video,cancelled=lambda:requests.stopped)
        if requests.stopped:
            return
        emit('video_info',session.info)
        while (item := requests.take()) is not None:
            version, request = item
            try:
                result = session.frame(request['frame'],request['request_id'],
                    cancelled=lambda:requests.cancelled(version))
                if not requests.cancelled(version):
                    emit('video_frame',result)
            except _Cancelled:
                continue
            except (OSError,RuntimeError,ValueError,cv2.error) as error:
                if not requests.cancelled(version):
                    emit('video_error',{'request_id':request['request_id'],'message':str(error)})
    except InterruptedError:
        pass
    except Exception as error:
        if not requests.stopped:
            emit('video_error',{'message':str(error)})
    finally:
        if session is not None:
            session.close()
        emit('finished','')
