"""Offline perception/action replay. Suggested inputs never reach an emulator."""
import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import json
from math import isfinite
from pathlib import Path
import sys
import tempfile
import time

import cv2
import numpy as np


@contextmanager
def readable_video(source, cancelled=None):
    """Remove only the observed Bilibili cache wrapper, in a temporary copy."""
    source = Path(source).resolve()
    normalized = None
    try:
        if cancelled is not None and cancelled():
            raise InterruptedError('Video loading cancelled')
        with source.open('rb') as stream:
            header = stream.read(17)
            if header[:9] == b'000000000':
                if header[13:17] != b'ftyp':
                    raise ValueError('Unrecognized Bilibili cache header')
                with tempfile.NamedTemporaryFile(prefix='hayabusa-replay-', suffix='.mp4', delete=False) as copy:
                    normalized = Path(copy.name)
                    stream.seek(9)
                    while True:
                        if cancelled is not None and cancelled():
                            raise InterruptedError('Video loading cancelled')
                        chunk = stream.read(2*1024*1024)
                        if not chunk:
                            break
                        copy.write(chunk)
        if cancelled is not None and cancelled():
            raise InterruptedError('Video loading cancelled')
        yield normalized or source
    finally:
        if normalized is not None:
            normalized.unlink(missing_ok=True)


def action_record(action):
    if action is None:
        return None
    point = getattr(action, 'point', None)
    return {'kind': action.kind, 'point': [int(value) for value in point] if point is not None else None,
            'threat_id': getattr(action, 'threat_id', None), 'reason': getattr(action, 'reason', '')}


def annotate(frame, screen, tracks, action, timestamp):
    image = frame.copy()
    if screen.rope_y is not None:
        y = int(screen.rope_y)
        cv2.line(image, (0, y), (1279, y), (80, 210, 80), 1)
    if screen.player_x is not None and screen.player_y is not None:
        cv2.circle(image, (int(screen.player_x), int(screen.player_y)), 18, (0, 230, 230), 2)
    colors = {'fire': (0, 160, 255), 'blackfire': (230, 50, 230), 'shuriken': (255, 180, 30),
              'slash': (255, 230, 80), 'flash': (255, 255, 255)}
    for track in tracks:
        left, top = max(0, int(track.x-track.width/2)), max(0, int(track.y-track.height/2))
        right, bottom = min(1279, int(track.x+track.width/2)), min(719, int(track.y+track.height/2))
        color = colors.get(track.kind, (150, 150, 150))
        cv2.rectangle(image, (left, top), (right, bottom), color, 1)
        cv2.putText(image, f'{track.kind}#{track.id}', (left, max(55, top-3)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
    cv2.rectangle(image, (0, 0), (1279, 48), (16, 16, 16), -1)
    board = getattr(screen, 'board_id', None) or screen.board_hint
    phase = getattr(screen, 'phase_hint', None) or '?'
    status = f'{timestamp:08.3f}s active={screen.active} side={screen.side} phase={phase} board={board}'
    cv2.putText(image, status.encode('ascii', 'replace').decode(), (10, 19), cv2.FONT_HERSHEY_SIMPLEX, .5, (230, 230, 230), 1)
    label = 'NONE' if action is None else action.kind.upper()
    cv2.putText(image, f'SUGGESTED: {label}   OFFLINE / NO INPUTS', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (80, 230, 230), 1)
    return image


def analyze_video(video, output, *, start=0, end=None, fps=15, cpu=False):
    if not isfinite(start) or start < 0 or not isfinite(fps) or not 0 < fps <= 120:
        raise ValueError('start must be nonnegative and fps must be in (0, 120]')
    if end is not None and (not isfinite(end) or end <= start):
        raise ValueError('end must be greater than start')
    source, output = Path(video).resolve(), Path(output).resolve()
    if not source.is_file():
        raise ValueError('Video file does not exist')
    output.mkdir(parents=True, exist_ok=True)
    rendered, report_file = output/'annotated.mp4', output/'report.json'
    if source in (rendered, report_file):
        raise ValueError('Output would overwrite the source video')
    # Imports are deliberately deferred: importing this tool does not initialize a GPU.
    from ninja3_assistant.combat.hayabusa.vision import HayabusaVision
    from ninja3_assistant.combat.hayabusa.policy import HayabusaPolicy

    report = {'source': str(source), 'status': 'initializing', 'requested': {'start': start, 'end': end, 'fps': fps, 'cpu': cpu},
              'executed_actions': 0, 'frames': [], 'suggested_actions': [], 'timeline_resets': [],
              'known_limits': [
                  'Suggested actions are hypothetical and do not affect the recorded character.',
                  'Edited cuts and slow-motion replays prevent measurement of clear rate or precise live input timings.',
                  'Resizing the supplied video to 1280x720 does not calibrate another character or emulator layout.',
                  'A retained track or a suggested dodge is not verified attack recognition or collision avoidance.',
                  'The annotated file has a constant sample frame rate; report video_seconds is authoritative across timestamp gaps.',
                  'analysis_ms includes resize, perception and policy; excludes decoding, encoding and emulator input/response.',
              ]}
    cap, writer = None, None
    costs, kinds, suggested = [], Counter(), Counter()
    boards, phases = Counter(), Counter()
    active, located = 0, 0
    try:
        vision, policy = HayabusaVision(use_gpu=not cpu), HayabusaPolicy()
        report['backend'] = vision.backend
        with ExitStack() as resources:
            normalized = resources.enter_context(readable_video(source))
            cap = cv2.VideoCapture(str(normalized))
            resources.callback(cap.release)
            if not cap.isOpened():
                raise RuntimeError('Cannot open video')
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if not isfinite(native_fps) or native_fps <= 0:
                raise RuntimeError('Decoder does not expose a reliable frame rate')
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/native_fps
            report['source_fps'] = native_fps
            report['source_duration_seconds'] = duration
            tolerance = max(.1, 2/native_fps)
            limit = min(end, duration) if end is not None and duration > 0 else end or duration
            if duration > 0 and start >= duration:
                raise ValueError('start is beyond the video duration')
            if not cap.set(cv2.CAP_PROP_POS_MSEC, start*1000):
                raise RuntimeError('Decoder rejected the requested seek')
            last_decoded, last_analyzed, next_sample = None, None, start
            cv2.setNumThreads(1)
            report['status'] = 'running'
            while cap.grab():
                actual = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if not isfinite(actual) or (last_decoded is None and abs(actual-start) > tolerance):
                    raise RuntimeError(f'Seek timestamp mismatch: requested {start:.3f}s, decoded {actual:.3f}s')
                if last_decoded is not None and actual <= last_decoded:
                    vision.reset()
                    policy.reset()
                    raise RuntimeError(f'Non-increasing decoder timestamp: {last_decoded:.3f}s -> {actual:.3f}s')
                last_decoded = actual
                if limit and actual > limit+1e-6:
                    break
                if actual+1e-6 < next_sample:
                    continue
                ok, decoded = cap.retrieve()
                if not ok:
                    raise RuntimeError(f'Cannot retrieve frame at {actual:.3f}s')
                if last_analyzed is not None and actual-last_analyzed > .5:
                    vision.reset()
                    policy.reset()
                    report['timeline_resets'].append({'from_seconds': last_analyzed, 'to_seconds': actual})
                before = time.perf_counter()
                frame = cv2.cvtColor(cv2.resize(decoded, (1280, 720)), cv2.COLOR_BGR2RGB)
                screen, tracks = vision.detect(frame, actual)
                action = policy.decide(screen, tracks, actual)
                if action is not None:
                    policy.completed(action, actual)
                elapsed = (time.perf_counter()-before)*1000
                costs.append(elapsed)
                active += int(screen.active)
                is_located = screen.player_x is not None and screen.player_y is not None
                located += int(is_located)
                kinds.update(track.kind for track in tracks)
                board_id = getattr(screen, 'board_id', None)
                phase = getattr(screen, 'phase_hint', None)
                if board_id:
                    boards.update((board_id,))
                if phase:
                    phases.update((phase,))
                record = action_record(action)
                if record is not None:
                    record['reason'] = getattr(policy, 'last_reason', record['reason'])
                    suggested[action.kind] += 1
                    report['suggested_actions'].append({'video_seconds': actual, **record})
                report['frames'].append({'video_seconds': actual, 'active': screen.active, 'located': is_located,
                    'side': screen.side, 'board_hint': screen.board_hint, 'result': screen.result,
                    'board_id': board_id, 'phase': phase,
                    'board_confidence': getattr(screen, 'board_confidence', 0),
                    'board_margin': getattr(screen, 'board_margin', 0),
                    'board_evidence': getattr(screen, 'board_evidence', ''),
                    'board_locked': getattr(screen, 'board_locked', False),
                    'board_onset': getattr(screen, 'board_onset', False),
                    'board_ended': getattr(screen, 'board_ended', False),
                    'board_epoch': getattr(screen, 'board_epoch', 0),
                    'left_control': getattr(screen, 'left_control', 'unknown'),
                    'right_control': getattr(screen, 'right_control', 'unknown'),
                    'airborne': screen.airborne, 'grounded': getattr(screen, 'grounded', None),
                    'track_count': len(tracks), 'suggested_action': record, 'analysis_ms': elapsed,
                    'decision_reason': getattr(policy, 'last_reason', '')})
                if writer is None:
                    writer = cv2.VideoWriter(str(rendered), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))
                    if not writer.isOpened():
                        raise RuntimeError('Cannot create annotated.mp4')
                image = annotate(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), screen, tracks, action, actual)
                writer.write(image)
                last_analyzed = actual
                next_sample += 1/fps
                if next_sample < actual:
                    next_sample = actual+1/fps
            if not report['frames']:
                raise RuntimeError('No timestamp-verified frames were decoded')
            if limit and last_decoded+tolerance < limit:
                raise RuntimeError(f'Video ended early at {last_decoded:.3f}s; expected {limit:.3f}s (cache may be incomplete)')
            report['status'] = 'completed'
    except Exception as error:
        report['status'] = 'error'
        report['error'] = str(error)
        raise
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        report['summary'] = {'analyzed_frames': len(costs), 'active_frames': active, 'located_frames': located,
            'track_detections_by_kind': dict(kinds), 'suggested_actions_by_kind': dict(suggested),
            'board_observations': dict(boards), 'phase_observations': dict(phases),
            'analysis_ms': ({'mean': float(np.mean(costs)), **{name: float(np.percentile(costs, value))
                            for name, value in [('p50', 50), ('p95', 95), ('max', 100)]}} if costs else {}),
            'annotated_video': str(rendered) if writer is not None and costs else None}
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float)
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        report = analyze_video(args.video, args.output, start=args.start, end=args.end, fps=args.fps, cpu=args.cpu)
    except (OSError, RuntimeError, ValueError) as error:
        print(f'Replay failed: {error}', file=sys.stderr)
        return 1
    print(json.dumps({'status': report['status'], 'backend': report['backend'], 'summary': report['summary']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
