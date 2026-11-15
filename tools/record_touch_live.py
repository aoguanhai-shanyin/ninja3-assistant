"""Record Android display and raw touch events for a single live trial."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)
adb = 'D:/Program Files/Netease/MuMu/nx_main/adb.exe'
base = [adb, '-s', '127.0.0.1:16416']
flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
remote = '/sdcard/ninja3-cpp-' + str(time.time_ns()) + '.mp4'
meta = {'remote_video': remote, 'record_start_perf': time.perf_counter(), 'audio': False}
events = (root/'getevent.txt').open('wb')
capture = recorder = None
try:
    capture = subprocess.Popen([*base, 'shell', '-t', 'getevent', '-lt', '/dev/input/event4'],
                               stdout=events, stderr=subprocess.DEVNULL, creationflags=flags)
    recorder = subprocess.Popen([*base, 'shell', 'screenrecord', '--verbose', '--size', '1280x720',
                                 '--bit-rate', '6000000', '--time-limit', '90', remote],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    time.sleep(2)
    if recorder.poll() is not None or capture.poll() is not None:
        raise RuntimeError('video or Android input recorder exited early')
    (root/'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps({'recording': True, **meta}), flush=True)
    out, err = recorder.communicate(timeout=110)
    meta.update(exit_code=recorder.returncode, stdout=out.decode(errors='replace'),
                stderr=err.decode(errors='replace'), finished_perf=time.perf_counter())
    if recorder.returncode:
        raise RuntimeError('screenrecord failed')
    video = root/'actual-screen.mp4'
    result = subprocess.run([*base, 'pull', remote, str(video)], capture_output=True,
                            timeout=60, creationflags=flags)
    meta.update(pull_exit=result.returncode, video_bytes=video.stat().st_size if video.exists() else 0)
    if result.returncode or not meta['video_bytes']:
        raise RuntimeError('could not retrieve recorded video')
    print(json.dumps({'done':True,'bytes':meta['video_bytes']}), flush=True)
finally:
    if recorder is not None and recorder.poll() is None:
        recorder.terminate()
    if capture is not None and capture.poll() is None:
        capture.terminate()
        capture.wait(timeout=5)
    events.close()
    (root/'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
