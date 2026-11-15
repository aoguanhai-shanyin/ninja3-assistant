"""Build geometry-only board references and report contiguous held-out coverage."""
from ninja3_assistant.paths import project_root
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

ROOT = project_root()
from ninja3_assistant.combat.hayabusa.recognition import FEATURE_SIZE, FEATURE_VERSION, HayabusaBoardRecognizer, formation_features


def is_between(timestamp, intervals):
    return any(start <= timestamp <= end for start, end in intervals)


def extract(video, catalog, cache, only_titles=False):
    from ninja3_assistant.combat.hayabusa.vision import HayabusaVision
    vision = HayabusaVision(use_gpu=False, recognize_boards=False)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError('Cannot read the normalized analysis proxy')
    cv2.setNumThreads(1)
    samples, features = [], []
    try:
        for label, board in enumerate(catalog['boards']):
            before = len(samples)
            for clip_index, clip in enumerate(board['clips']):
                if clip.get('use_for_training') is False:
                    continue
                start, end = clip['start']+.25, clip['end']-.25
                titles = clip.get('title_intervals', [])
                replays = clip.get('replay_intervals', [])
                split_at = start+(end-start)*.72
                vision.reset()
                capture.set(cv2.CAP_PROP_POS_MSEC, start*1000)
                next_sample = start
                while capture.grab():
                    actual = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
                    if actual > end:
                        break
                    if actual+1e-5 < next_sample:
                        continue
                    ok, bgr = capture.retrieve()
                    if not ok or abs(actual-next_sample) > .21:
                        raise RuntimeError(f'Proxy decoding timestamp mismatch near {next_sample}')
                    next_sample += .2
                    if only_titles and not is_between(actual,titles):
                        continue
                    rgb = cv2.cvtColor(cv2.resize(bgr, (1280, 720)), cv2.COLOR_BGR2RGB)
                    screen, tracks = vision.detect(rgb, actual)
                    feature = formation_features(screen, tracks) if screen.active else np.zeros(FEATURE_SIZE, np.float32)
                    role = ('test' if actual >= split_at+.25 else 'train' if actual <= split_at-.25 else 'gap')
                    if is_between(actual, replays):
                        role = 'replay'
                    samples.append({'label': label, 'board_id': board['id'], 'clip': clip_index,
                        'video_seconds': round(actual, 4), 'progress': (actual-clip['start'])/(clip['end']-clip['start']),
                        'split': role, 'active': bool(screen.active), 'tracks': len(tracks),
                        'title_visible':is_between(actual,titles)})
                    features.append(feature)
            print(json.dumps({'board':board['id'],'frames':len(samples)-before},ensure_ascii=False),flush=True)
    finally:
        capture.release()
    array = np.asarray(features, np.float32)
    np.savez_compressed(cache, features=array)
    cache.with_suffix('.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2),encoding='utf-8')
    return samples, array


def make_bank(samples, features, folder, metadata, training_only=False):
    rows, labels, progress, sequences = [], [], [], []
    groups={}
    for item, feature in zip(samples, features):
        group = (item['label'], item['clip'])
        if training_only and item['split'] != 'train':
            continue
        rows.append(feature)
        labels.append(item['label'])
        progress.append(item['progress'])
        sequences.append(groups.setdefault(group,len(groups)))
    if not rows:
        raise RuntimeError('No attack formation references were extracted')
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'hayabusa-boards.npz',features=np.asarray(rows,np.float32),
        sequences=np.asarray(sequences,np.int32),labels=np.asarray(labels,np.int32),progress=np.asarray(progress,np.float32))
    (folder/'hayabusa-boards.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return Counter(labels)


def evaluate(samples, features, folder, catalog, mode='cold'):
    recognizer = HayabusaBoardRecognizer(folder)
    report = {board['id']:{'phase':board['phase'],'test_frames':0,'nonempty_frames':0,
        'correct':0,'wrong':0,'rejected':0,'phase_correct':0,'confusions':{}} for board in catalog['boards']}
    previous = None
    phase_confusions = {phase:{predicted:0 for predicted in ('P1','P2','P3','P4','unknown')}
                        for phase in ('P1','P2','P3','P4')}
    for item, feature in zip(samples,features):
        if item['split']=='replay' and mode!='full_reference':
            continue
        group = (item['label'],item['clip'])
        if mode=='cold' and item['split']!='test':
            continue
        if group != previous:
            recognizer.reset()
        previous = group
        matched = recognizer._update_features(feature,item['video_seconds'])
        if mode!='full_reference' and item['split']!='test':
            continue
        row = report[item['board_id']]
        row['test_frames'] += 1
        row['nonempty_frames'] += int(np.linalg.norm(feature)>.1)
        if matched.board_id == item['board_id']:
            row['correct'] += 1
            if 'first_confirmation' not in row:
                row['first_confirmation']={'video_seconds':item['video_seconds'],
                    'observed_clip_progress':item['progress'],'matched_reference_progress':matched.progress}
        elif matched.board_id is not None:
            row['wrong'] += 1
            row['confusions'][matched.board_id] = row['confusions'].get(matched.board_id,0)+1
        else:
            row['rejected'] += 1
        row['phase_correct'] += int(matched.phase == row['phase'])
        if row['phase'] in phase_confusions:
            phase_confusions[row['phase']][matched.phase or 'unknown'] += 1
    totals = {key:sum(row[key] for row in report.values()) for key in
              ('test_frames','nonempty_frames','correct','wrong','rejected','phase_correct')}
    return {'mode':mode,'protocol':'Per primary clip, first 72% training, final 28% evaluation, 0.5s gap; labeled replays excluded from holdout. No random adjacent-frame split. Clip starts reset state.',
            'single_video_limit':'Temporal holdouts share one edited recording; this is not an independent run or live control evaluation.',
            'per_board':report,'totals':totals,'phase_confusions':phase_confusions,
            'boards_with_correct_holdout_match':[key for key,row in report.items() if row['correct']],
            'boards_without_correct_holdout_match':[key for key,row in report.items() if not row['correct']]}


def build(video, catalog_file, output, cache, report_file, refresh=False):
    catalog = json.loads(catalog_file.read_text(encoding='utf-8'))
    digest = hashlib.sha256(catalog_file.read_bytes()).hexdigest()
    cache_rows = cache.with_suffix('.json')
    cache_meta=cache.with_suffix('.meta.json')
    extraction_spec=hashlib.sha256(json.dumps([{'id':board['id'],'clips':board['clips']}
        for board in catalog['boards']],sort_keys=True).encode()).hexdigest()
    if cache.is_file() and cache_rows.is_file() and not refresh:
        samples = json.loads(cache_rows.read_text(encoding='utf-8'))
        with np.load(cache,allow_pickle=False) as data:
            features = data['features']
        labels = {row['board_id'] for row in samples}
        if labels != {board['id'] for board in catalog['boards']}:
            raise RuntimeError('Cached feature labels do not match catalog; rebuild with --refresh')
        if cache_meta.is_file():
            spec=json.loads(cache_meta.read_text(encoding='utf-8'))
            if spec.get('extraction_spec')!=extraction_spec or spec.get('geometry_version')!=1:
                raise RuntimeError('Cached clip annotations differ from catalog; rebuild with --refresh')
        else:
            # Upgrade the earlier cache, which omitted the valuable opening/title intervals.
            with tempfile.TemporaryDirectory(prefix='hayabusa-onset-features-') as temporary:
                additions,added_features=extract(video,catalog,Path(temporary)/'onsets.npz',only_titles=True)
            merged={(item['label'],item['clip'],item['video_seconds']):(item,feature)
                    for item,feature in zip(samples,features)}
            merged.update({(item['label'],item['clip'],item['video_seconds']):(item,feature)
                           for item,feature in zip(additions,added_features)})
            ordered=[merged[key] for key in sorted(merged)]
            samples=[item[0] for item in ordered]
            features=np.asarray([item[1] for item in ordered],np.float32)
            np.savez_compressed(cache,features=features)
            cache_rows.write_text(json.dumps(samples,ensure_ascii=False,indent=2),encoding='utf-8')
    else:
        samples,features = extract(video,catalog,cache)
    cache_meta.write_text(json.dumps({'geometry_version':1,'extraction_spec':extraction_spec,
        'retains_title_intervals':True,'source':'attack geometry; raw text pixels are not features'},indent=2),encoding='utf-8')
    metadata = {'feature_version':FEATURE_VERSION,'catalog_sha256':digest,
        'source':'One edited SS+ Hayabusa teaching video; source time labels are build-time only.',
        'min_confidence':.78,'min_margin':.055,
        'sequence_matching':'8-observation subsequence DTW, stay/advance1/advance2; normal and mirror paths independent',
        'boards':[{key:board[key] for key in ('id','name','phase','phase_evidence','families')} for board in catalog['boards']],
        'switch_hysteresis':'An active board requires stronger competing similarity/margin/relative gain and 4-5 consecutive observations before switching; weak same-family variants retain current mode.',
        'runtime_features':'Verified attack kind, rope-relative position, count, width and height; no source video clock, RGB text, HUD, background or chapter labels.',
        'phase_limit':'Phase follows a confirmed matching board; chapter-associated phases are inferred, not independent HP or timer thresholds.',
        'progress_limit':'Approximate matched reference position; never a command schedule or calibrated live game clock.',
        'deployment_fit':'After prefix-only held-out evaluation, deploy all assigned formations including visible gaps for sequence alignment. Reported holdout metrics belong to the prefix-only model.',
        'limitations':['One edited video cannot establish all live variants or dodge success.','Unknown formations and ambiguous classes remain rejected.','Caption filtering depends on upstream attack detector rejecting text-shaped false positives.']}
    with tempfile.TemporaryDirectory(prefix='hayabusa-prefix-bank-') as temporary:
        counts = make_bank(samples,features,Path(temporary),metadata,training_only=True)
        evaluation = evaluate(samples,features,Path(temporary),catalog)
        continuation = evaluate(samples,features,Path(temporary),catalog,mode='continuation')
    deployed = make_bank(samples,features,output,metadata)
    nonempty=Counter(item['label'] for item,feature in zip(samples,features) if np.linalg.norm(feature)>.1)
    full_reference=evaluate(samples,features,output,catalog,mode='full_reference')
    evaluation.update({'catalog_sha256':digest,'feature_version':FEATURE_VERSION,
        'deployed_reference_count':sum(deployed.values()),
        'deployed_per_board':{board['id']:deployed.get(i,0) for i,board in enumerate(catalog['boards'])},
        'nonempty_deployed_per_board':{board['id']:nonempty.get(i,0) for i,board in enumerate(catalog['boards'])},
        'missing_deployed_references':[board['id'] for i,board in enumerate(catalog['boards']) if not deployed.get(i)],
        'missing_nonempty_references':[board['id'] for i,board in enumerate(catalog['boards']) if not nonempty.get(i)],
        'thresholds':{'minimum_similarity':metadata['min_confidence'],'minimum_class_margin':metadata['min_margin']}})
    evaluation['continuation_prefix_observed']=continuation
    evaluation['full_deployed_reference_replay']=full_reference
    evaluation['title_interval_policy']='Retain verified attack geometries during chapter titles; image text is never a feature. Upstream attack detector has real yellow-title negative regressions.'
    report_file.write_text(json.dumps(evaluation,ensure_ascii=False,indent=2),encoding='utf-8')
    return evaluation


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,default=ROOT/'data/hayabusa-video-study/recognition-proxy.mp4')
    parser.add_argument('--catalog',type=Path,default=ROOT/'data/hayabusa-video-study/board-catalog.json')
    parser.add_argument('--output',type=Path,default=ROOT/'assets')
    parser.add_argument('--cache',type=Path,default=ROOT/'data/hayabusa-video-study/board-features.npz')
    parser.add_argument('--report',type=Path,default=ROOT/'data/hayabusa-video-study/board-recognition-evaluation.json')
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    result=build(args.video,args.catalog,args.output,args.cache,args.report,args.refresh)
    print(json.dumps({'totals':result['totals'],'matching_boards':len(result['boards_with_correct_holdout_match']),
        'missing_references':result['missing_deployed_references']},ensure_ascii=False))
