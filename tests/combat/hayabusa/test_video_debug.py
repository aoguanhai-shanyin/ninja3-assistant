import base64
import hashlib
import io
import json
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ninja3_assistant.combat.boss.policy import BossAction
from ninja3_assistant.combat.hayabusa.scene import SceneDecision
from ninja3_assistant.combat.hayabusa.video_debug import VideoDebugSession, _Cancelled, _Requests, run_video_debug
from ninja3_assistant.combat.hayabusa.vision import HayabusaScreen


class Commands:
    def __init__(self):
        self.lines = queue.Queue()
    def __iter__(self):
        while (line := self.lines.get()) is not None:
            yield line
    def send(self, frame, request_id):
        self.lines.put(json.dumps({'command':'frame','frame':frame,'request_id':request_id})+'\n')
    def close(self):
        self.lines.put(None)


class FakeScene:
    backend = 'test CPU'
    def __init__(self, **kwargs):
        self.closed = False
        self.log = None
        self.releases = 0
        self.observations = []
        self.policy = SimpleNamespace(recipe=SimpleNamespace(recipe_id='p1_fire_1',stage=0,
            wave=1,mode_active=True,mode_state='running',late_entry=False,last_reason='test step'))
    def release(self):
        self.releases += 1
        self.policy.recipe.stage = 0
    def step(self, frame, now):
        self.observations.append(now)
        self.pending_record = {'tracks':[{'id':1,'kind':'fire','x':700-now*20,'y':350,
            'width':30,'height':30,'vx':-20,'vy':0,'seen':now,'hits':2}]}
        screen = HayabusaScreen(active=True,rope_y=400,player_x=300,player_y=360,side='above',
            board_id='p1_fire_1',board_name='鬼火一',phase_hint='P1',board_locked=True)
        return SceneDecision(state=screen,active=True,action=BossAction('jump',(1090,630)))
    def observe(self, action, now):
        self.policy.recipe.stage += 1
    def close(self):
        self.closed = True


class VideoDebugTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.video = self.root/'中文录像.mp4'
        writer = cv2.VideoWriter(str(self.video),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,36))
        self.assertTrue(writer.isOpened())
        for value in range(40):
            writer.write(np.full((36,64,3),value*5,np.uint8))
        writer.release()
        self.before = hashlib.sha256(self.video.read_bytes()).digest()
    def tearDown(self):
        self.assertEqual(hashlib.sha256(self.video.read_bytes()).digest(),self.before)
        self.temporary.cleanup()

    def session(self):
        with patch('ninja3_assistant.combat.hayabusa.video_debug.HayabusaScene',FakeScene):
            return VideoDebugSession(self.video)

    def test_native_next_frame_uses_causal_state_and_jpeg_contains_overlay(self):
        session = self.session()
        try:
            self.assertEqual(session.info['fps'],10)
            self.assertEqual(session.info['total_frames'],40)
            first = session.frame(0,10)
            second = session.frame(1,11)
            self.assertEqual((first['frame'],second['frame']),(0,1))
            self.assertAlmostEqual(second['seconds'],.1)
            self.assertTrue(first['resynced'])
            self.assertFalse(second['resynced'])
            self.assertEqual(session.scene.releases,1)
            self.assertEqual(second['screen']['board_name'],'鬼火一')
            self.assertEqual(second['recipe']['stage'],2)
            self.assertEqual(second['action']['kind'],'jump')
            self.assertEqual(second['executed_actions'],0)
            image = cv2.imdecode(np.frombuffer(base64.b64decode(second['jpeg_base64']),np.uint8),1)
            self.assertEqual(image.shape,(720,1280,3))
            self.assertGreater(np.abs(image[335:365,680:720].astype(float)-5).mean(),10)
            self.assertEqual(len(session.trails[(1,'fire')]),2)
            json.dumps(second,ensure_ascii=False)
        finally:
            session.close()
        self.assertTrue(session.scene.closed)

    def test_seek_and_backward_clear_future_state_and_limit_warmup_to_two_seconds(self):
        session = self.session()
        try:
            late = session.frame(30,1)
            self.assertTrue(late['resynced'])
            self.assertAlmostEqual(late['history_start_seconds'],1)
            self.assertGreaterEqual(min(session.scene.observations),1)
            backward = session.frame(3,2)
            self.assertTrue(backward['resynced'])
            self.assertAlmostEqual(backward['seconds'],.3)
            self.assertEqual(backward['recipe']['stage'],4)
            self.assertLessEqual(max(t for t,_,_ in session.trails[(1,'fire')]),.3+1e-9)
            self.assertEqual(session.scene.releases,2)
        finally:
            session.close()

    def test_cancelled_preroll_cannot_be_continued_as_sequential_state(self):
        session = self.session()
        calls = [0]
        def cancelled():
            calls[0] += 1
            return calls[0] > 3
        try:
            with self.assertRaises(_Cancelled):
                session.frame(25,1,cancelled)
            self.assertIsNone(session.current)
            result = session.frame(4,2)
            self.assertTrue(result['resynced'])
            self.assertEqual(result['recipe']['stage'],5)
        finally:
            session.close()

    def test_out_of_range_and_stale_decoder_frame_are_rejected(self):
        session = self.session()
        try:
            for frame in (-1,40,True,.5):
                with self.assertRaises(ValueError):
                    session.frame(frame,1)
            capture = Mock()
            capture.set.return_value = capture.grab.return_value = True
            capture.get.side_effect = lambda prop: {cv2.CAP_PROP_POS_FRAMES:11,
                cv2.CAP_PROP_POS_MSEC:2000}[prop]
            session.capture = capture
            with self.assertRaisesRegex(RuntimeError,'时间戳'):
                session.frame(30,1)
            capture.retrieve.assert_not_called()
        finally:
            session.close()

    def test_only_latest_request_survives_queue_and_eof_stops(self):
        source = Commands()
        requests = _Requests(source,lambda *args:None)
        reader = threading.Thread(target=requests.read,daemon=True)
        reader.start()
        source.send(1,1)
        source.send(2,2)
        with requests.condition:
            self.assertTrue(requests.condition.wait_for(lambda:requests.version==2,timeout=2))
        version, item = requests.take()
        self.assertEqual(item['frame'],2)
        self.assertTrue(requests.cancelled(version-1))
        self.assertFalse(requests.cancelled(version))
        source.close()
        reader.join(timeout=2)
        self.assertTrue(requests.cancelled(version))
        self.assertIsNone(requests.take())

    def test_invalid_commands_are_reported_without_frames(self):
        events = []
        requests = _Requests(io.StringIO('[]\n{"command":"frame","frame":true,"request_id":1}\n'),
            lambda *args:events.append(args))
        requests.read()
        self.assertEqual(len(events),2)
        self.assertTrue(all(kind=='video_error' for kind,_ in events))
        self.assertTrue(requests.stopped)

    def test_worker_opens_without_device_and_eof_releases_video_and_scene(self):
        source = Commands()
        events, done, scenes = [],threading.Event(),[]
        def scene_factory(**kwargs):
            scene = FakeScene(**kwargs)
            scenes.append(scene)
            return scene
        def emit(kind,value):
            events.append((kind,value))
            if kind=='video_info':
                source.send(2,10)
            elif kind=='video_frame':
                source.close()
            elif kind=='finished':
                done.set()
        with patch('ninja3_assistant.combat.hayabusa.video_debug.HayabusaScene',side_effect=scene_factory), \
                patch('ninja3_assistant.device.mumu.MuMu',side_effect=AssertionError('offline must not connect')):
            thread = threading.Thread(target=run_video_debug,args=(self.video,emit,source),daemon=True)
            thread.start()
            self.assertTrue(done.wait(5))
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(scenes[0].closed)
        self.assertEqual([kind for kind,_ in events],['video_info','video_frame','finished'])
        self.assertEqual(events[1][1]['request_id'],10)

    def test_open_error_still_emits_finished(self):
        events = []
        source = Commands()
        run_video_debug(self.root/'missing.mp4',lambda *args:events.append(args),source)
        source.close()
        self.assertEqual([kind for kind,_ in events],['video_error','finished'])

    def test_startup_cancellation_before_open_does_not_initialize_gpu(self):
        with patch('ninja3_assistant.combat.hayabusa.video_debug.HayabusaScene') as scene, \
                patch('ninja3_assistant.combat.hayabusa.video_debug.cv2.VideoCapture') as capture:
            with self.assertRaises(_Cancelled):
                VideoDebugSession(self.video,cancelled=lambda:True)
        scene.assert_not_called()
        capture.assert_not_called()

    def test_cancellation_while_gpu_initializes_releases_every_resource(self):
        stopped, scenes = threading.Event(),[]
        def initialize(**kwargs):
            scene = FakeScene(**kwargs)
            scenes.append(scene)
            stopped.set()
            return scene
        with patch('ninja3_assistant.combat.hayabusa.video_debug.HayabusaScene',side_effect=initialize), \
                self.assertRaises(_Cancelled):
            VideoDebugSession(self.video,cancelled=stopped.is_set)
        self.assertTrue(scenes[0].closed)

    def test_eof_during_initialization_is_normal_cancellation(self):
        events = []
        run_video_debug(self.video,lambda *args:events.append(args),io.StringIO(''))
        self.assertEqual(events,[('finished','')])


if __name__=='__main__':
    unittest.main()
