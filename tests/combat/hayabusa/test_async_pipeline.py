import threading
import time
import unittest
import gc
import weakref
import numpy as np

from ninja3_assistant.combat.hayabusa.async_pipeline import LatestVision


class LatestVisionTest(unittest.TestCase):
    def test_drops_waiting_frame_and_deduplicates(self):
        entered, release, last = threading.Event(), threading.Event(), threading.Event()

        def analyze(frame, captured_at):
            if frame == 1:
                entered.set()
                release.wait(2)
            if frame == 3:
                last.set()
            return frame

        pipeline = LatestVision(analyze)
        try:
            pipeline.submit(1, pipeline.timestamp())
            self.assertTrue(entered.wait(1))
            pipeline.submit(2, pipeline.timestamp())
            final_id = pipeline.submit(3, pipeline.timestamp())
            release.set()
            self.assertTrue(last.wait(1))
            deadline = time.monotonic()+1
            result = None
            while time.monotonic()<deadline:
                result = pipeline.poll(final_id-1)
                if result is not None:
                    break
            self.assertEqual(result.value,3)
            self.assertEqual(pipeline.dropped,1)
            self.assertIsNone(pipeline.poll(final_id))
        finally:
            release.set()
            pipeline.close()

    def test_inflight_result_is_discarded_after_reset_and_stale_is_rejected(self):
        entered, release, processed = threading.Event(), threading.Event(), threading.Event()
        clock_time = [100.0]

        def analyze(frame, captured_at):
            if frame == 'old':
                entered.set()
                release.wait(2)
            processed.set()
            return frame

        resets=[]
        pipeline = LatestVision(analyze,reset_analyzer=lambda:resets.append('reset'),clock=lambda:clock_time[0])
        try:
            pipeline.submit('old',100.)
            self.assertTrue(entered.wait(1))
            pipeline.reset()
            release.set()
            self.assertTrue(processed.wait(1))
            processed.clear()
            new_id = pipeline.submit('new',100.)
            self.assertTrue(processed.wait(1))
            deadline=time.monotonic()+1
            result=None
            while time.monotonic()<deadline:
                result=pipeline.poll(new_id-1)
                if result is not None:
                    break
            self.assertEqual(result.value,'new')
            self.assertEqual(resets,['reset'])
            clock_time[0]=100.201
            self.assertIsNone(pipeline.poll(new_id-1))
        finally:
            release.set()
            pipeline.close()

    def test_waiting_memory_bounded_and_stuck_worker_close_is_bounded(self):
        entered, release = threading.Event(), threading.Event()
        alive = weakref.WeakSet()

        class Frame:
            pass

        def analyze(frame, captured_at):
            entered.set()
            release.wait(2)
            return 'small-result'

        pipeline = LatestVision(analyze)
        try:
            first=Frame()
            alive.add(first)
            pipeline.submit(first,pipeline.timestamp())
            del first
            self.assertTrue(entered.wait(1))
            for _ in range(10000):
                frame=Frame()
                alive.add(frame)
                pipeline.submit(frame,pipeline.timestamp())
            del frame
            gc.collect()
            self.assertLessEqual(len(alive),2)  # in-flight + latest waiting frame
            self.assertEqual(pipeline.dropped,9999)
            started=time.perf_counter()
            self.assertFalse(pipeline.close(timeout=.03))
            self.assertLess(time.perf_counter()-started,.5)
            with self.assertRaises(RuntimeError):
                pipeline.submit(Frame(),pipeline.timestamp())
        finally:
            release.set()
            self.assertTrue(pipeline.close(timeout=1))

    def test_analyzer_exception_is_visible_and_no_output_can_be_used(self):
        raised=threading.Event()

        def analyze(frame, captured_at):
            raised.set()
            raise ValueError('test detection failure')

        pipeline=LatestVision(analyze)
        try:
            pipeline.submit('frame',pipeline.timestamp())
            self.assertTrue(raised.wait(1))
            deadline=time.monotonic()+1
            while pipeline.error is None and time.monotonic()<deadline:
                time.sleep(.001)
            with self.assertRaisesRegex(RuntimeError,'vision worker failed'):
                pipeline.poll(0)
            with self.assertRaises(RuntimeError):
                pipeline.submit('another',pipeline.timestamp())
        finally:
            self.assertTrue(pipeline.close())

    def test_full_size_frames_do_not_accumulate_while_worker_blocks(self):
        entered, release = threading.Event(), threading.Event()
        references=[]

        def analyze(frame, captured_at):
            entered.set()
            release.wait(2)
            return None

        pipeline=LatestVision(analyze)
        try:
            first=np.empty((720,1280,3),dtype=np.uint8)
            references.append(weakref.ref(first))
            pipeline.submit(first,pipeline.timestamp())
            del first
            self.assertTrue(entered.wait(1))
            for _ in range(300):
                frame=np.empty((720,1280,3),dtype=np.uint8)
                references.append(weakref.ref(frame))
                pipeline.submit(frame,pipeline.timestamp())
            del frame
            gc.collect()
            self.assertLessEqual(sum(ref() is not None for ref in references),2)
            self.assertEqual(pipeline.dropped,299)
        finally:
            release.set()
            self.assertTrue(pipeline.close())

    def test_future_and_old_capture_times_fail_closed(self):
        ready=threading.Event()
        now=[100.0]

        def analyze(frame, captured_at):
            ready.set()
            return frame

        pipeline=LatestVision(analyze,clock=lambda:now[0])
        try:
            self.assertEqual(pipeline.timestamp(),100.0)
            frame_id=pipeline.submit('observation',100.01)
            self.assertTrue(ready.wait(1))
            deadline=time.monotonic()+1
            while pipeline.processed<1 and time.monotonic()<deadline:
                time.sleep(.001)
            self.assertIsNone(pipeline.poll(frame_id-1))  # future in this clock
            now[0]=100.209
            self.assertEqual(pipeline.poll(frame_id-1).value,'observation')
            now[0]=100.211
            self.assertIsNone(pipeline.poll(frame_id-1))  # over 200 ms old
        finally:
            self.assertTrue(pipeline.close())


if __name__ == '__main__':
    unittest.main()
