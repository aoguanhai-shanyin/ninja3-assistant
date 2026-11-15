"""Build a grouped, explicitly weak-labelled Hayabusa detector experiment.

No inputs reach an emulator. Known replay/title intervals are excluded; the
source is an edited single video, so held-out clips are not independent runs.
"""
from ninja3_assistant.paths import project_root
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from math import isfinite
from pathlib import Path
import shutil

import cv2
import numpy as np


ROOT = project_root()
CLASSES = ('fire', 'blackfire', 'shuriken', 'slash', 'flash')
CROP = (0, 90, 1280, 500)
VALIDATION = {'p1_fire_2', 'p1_sword_2', 'p1_shuriken_2',
              'p2_blackfire_2', 'p3_ground_slam_1', 'p4_clone_fire'}
TEST = {'p1_fire_4', 'p1_sword_3', 'p1_shuriken_34',
        'p2_blackfire_tracking_short', 'p3_ground_slam_2', 'p4_clone_flash'}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def group_id(board):
    if board in ('p1_shuriken_3', 'p1_shuriken_4', 'p1_shuriken_34_variant'):
        return 'p1_shuriken_34'
    return board


def sample_plan(catalog, interval=.75):
    if not isfinite(interval) or interval <= 0:
        raise ValueError('interval must be positive and finite')
    rows = []
    for board in catalog['boards']:
        group = group_id(board['id'])
        split = 'val' if group in VALIDATION else 'test' if group in TEST else 'train'
        for clip_index, clip in enumerate(board['clips']):
            # Boundaries are approximate. Do not sample the first/last second.
            start, end = float(clip['start']) + 1, float(clip['end']) - 1
            excluded = clip.get('title_intervals', []) + clip.get('replay_intervals', [])
            index = 0
            while start + index * interval < end:
                timestamp = round(start + index * interval, 6)
                index += 1
                if any(float(a) - .5 <= timestamp <= float(b) + .5 for a, b in excluded):
                    continue
                rows.append({'id': f"{board['id']}-{round(timestamp * 1000):09d}",
                    'split': split, 'clip_id': board['id'], 'group_id': group,
                    'clip_index': clip_index, 'video_seconds': timestamp,
                    'expected_families': board['families']})
    rows.sort(key=lambda row: row['video_seconds'])
    if len({row['video_seconds'] for row in rows}) != len(rows):
        raise ValueError('Catalogue clips overlap: assign overlapping source clips to one group first')
    return rows


def clipped_box(kind, full_xyxy):
    if kind not in CLASSES or len(full_xyxy) != 4 or not all(isfinite(v) for v in full_xyxy):
        raise ValueError('Invalid class or box coordinates')
    left, top, right, bottom = CROP
    x1, y1, x2, y2 = full_xyxy
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Box must have positive area')
    x1, y1, x2, y2 = max(left, x1), max(top, y1), min(right, x2), min(bottom, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return {'class': kind, 'xyxy': [round(x1-left, 2), round(y1-top, 2),
                                    round(x2-left, 2), round(y2-top, 2)],
            'full_xyxy': [round(v, 2) for v in (x1, y1, x2, y2)]}


def validate_boxes(boxes):
    width, height = CROP[2]-CROP[0], CROP[3]-CROP[1]
    checked = []
    for box in boxes:
        if box['class'] not in CLASSES:
            raise ValueError('Unknown reviewed class')
        x1, y1, x2, y2 = box['xyxy']
        if not all(isfinite(value) for value in (x1, y1, x2, y2)) or not (
                0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Reviewed coordinates must be within the cropped image')
        checked.append({'class': box['class'], 'xyxy': [x1, y1, x2, y2],
                        'full_xyxy': [x1+CROP[0], y1+CROP[1], x2+CROP[0], y2+CROP[1]]})
    return checked


def write_labels(path, boxes):
    width, height = CROP[2]-CROP[0], CROP[3]-CROP[1]
    lines = []
    for box in validate_boxes(boxes):
        x1, y1, x2, y2 = box['xyxy']
        coordinates = ((x1+x2)/(2*width), (y1+y2)/(2*height),
                       (x2-x1)/width, (y2-y1)/height)
        lines.append(str(CLASSES.index(box['class'])) + ' ' +
                     ' '.join(f'{value:.7f}' for value in coordinates))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_yaml(output, train=None):
    # JSON-quoted paths are valid YAML and preserve Windows spaces safely.
    lines = ['path: ' + json.dumps(output.as_posix()),
             'train: ' + json.dumps(train or 'images/train'),
             'val: images/val', 'test: images/test', 'names:']
    lines += [f'  {i}: {name}' for i, name in enumerate(CLASSES)]
    (output/'dataset.yaml').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def select_reviews(frames, split, count=24):
    groups = defaultdict(list)
    for row in frames:
        if row['split'] == split:
            groups[row['group_id']].append(row)
    candidates = []
    # Spread samples through each held-out group, including empty candidates.
    for key in sorted(groups):
        rows = groups[key]
        landmarks = [round(fraction*(len(rows)-1)) for fraction in (.15,.4,.65,.85)]
        order = list(dict.fromkeys(landmarks+list(range(len(rows)))))
        candidates.append([rows[i] for i in order])
    chosen = []
    while len(chosen) < count and any(candidates):
        for rows in candidates:
            if rows and len(chosen) < count:
                chosen.append(rows.pop(0))
    return chosen


def review_sheets(output, rows, split):
    folder = output/'review'
    folder.mkdir(exist_ok=True)
    for page in range(0, len(rows), 6):
        sheet = np.full((3*245, 2*640, 3), 25, np.uint8)
        for i, row in enumerate(rows[page:page+6]):
            frame = cv2.imread(str(output/row['file']))
            for number, box in enumerate(row['boxes']):
                x1, y1, x2, y2 = map(round, box['xyxy'])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,220,255), 1)
                cv2.putText(frame, f"{number}:{box['class']}", (x1,max(12,y1-3)),
                            cv2.FONT_HERSHEY_SIMPLEX,.4,(0,220,255),1)
            x, y = (i % 2)*640, (i // 2)*245
            sheet[y:y+205,x:x+640] = cv2.resize(frame,(640,205))
            cv2.putText(sheet,row['id'],(x+4,y+221),cv2.FONT_HERSHEY_SIMPLEX,.42,(245,245,245),1)
            cv2.putText(sheet,'UNREVIEWED weak proposals',(x+4,y+237),
                        cv2.FONT_HERSHEY_SIMPLEX,.4,(0,220,255),1)
        cv2.imwrite(str(folder/f'{split}-{page//6+1:02d}.jpg'),sheet)


def prepare(video, catalog_path, output, interval=.75, cpu=False, proxy=None):
    if output.exists():
        raise FileExistsError(f'Preserve existing dataset: {output}')
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    plan = sample_plan(catalog, interval)
    from ninja3_assistant.combat.hayabusa.vision import HayabusaVision
    from tools.replay.replay_hayabusa import readable_video
    cv2.setNumThreads(1)
    vision = HayabusaVision(use_gpu=not cpu, recognize_boards=False, detector=False)
    source_hash = digest(video)
    output.mkdir(parents=True)
    manifest = {'schema_version':1, 'classes':list(CLASSES), 'crop_xyxy':list(CROP),
        'source':{'path':str(video.resolve()),'sha256':source_hash,'catalog_sha256':digest(catalog_path)},
        'label_source':'unreviewed_color_rule_proposals', 'interval_seconds':interval,
        'scope':'Only y=90..500 is visible to this detector; boxes are visual extents, not damage hitboxes.',
        'split_policy':'Board groups held out; related shuriken 3/4/variant grouped; known replay/title ranges excluded with margins.',
        'limits':['One edited teaching video, not independent game runs.',
                  'Unknown replays or near duplicates may remain; exact cropped pixel duplicates are excluded globally.',
                  'All boxes and empty candidates are unreviewed until individually checked.',
                  'The fixed crop removes HUD/buttons and most lower captions; some in-field overlays may remain.'],
        'proposal_backend':vision.backend, 'frames':[], 'excluded_duplicates':[]}
    if proxy is not None:
        manifest['source']['proxy'] = {'path':str(proxy.resolve()),'sha256':digest(proxy),
            'scope':'Existing 1280x720, 10fps analysis proxy of the original source; decoded time is quantized to proxy frames.'}
    seen = {}
    try:
        with readable_video(proxy or video) as normalized:
            capture = cv2.VideoCapture(str(normalized))
            try:
                if not capture.isOpened():
                    raise RuntimeError('Cannot open video')
                for number,row in enumerate(plan):
                    requested = row['video_seconds']
                    if not capture.set(cv2.CAP_PROP_POS_MSEC,requested*1000):
                        raise RuntimeError('Video decoder rejected seek')
                    ok,frame = capture.read()
                    actual = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
                    tolerance = max(.1, 1/capture.get(cv2.CAP_PROP_FPS)+.01)
                    if not ok or not isfinite(actual) or abs(actual-requested) > tolerance:
                        raise RuntimeError(f'Cannot reliably decode requested timestamp {requested}: {actual}')
                    frame = cv2.cvtColor(cv2.resize(frame,(1280,720)),cv2.COLOR_BGR2RGB)
                    cropped = frame[CROP[1]:CROP[3],CROP[0]:CROP[2]]
                    fingerprint = hashlib.sha256(cropped.tobytes()).hexdigest()
                    if fingerprint in seen:
                        manifest['excluded_duplicates'].append({'id':row['id'],'duplicate_of':seen[fingerprint]})
                        continue
                    seen[fingerprint] = row['id']
                    vision.reset()
                    screen,tracks = vision.detect(frame,actual)
                    boxes = []
                    for track in tracks:
                        box = clipped_box(track.kind,(track.x-track.width/2,track.y-track.height/2,
                                                      track.x+track.width/2,track.y+track.height/2))
                        if box is not None:
                            boxes.append(box)
                    relative = Path('images')/row['split']/(row['id']+'.jpg')
                    destination = output/relative
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    if not cv2.imwrite(str(destination),cv2.cvtColor(cropped,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95]):
                        raise RuntimeError('Cannot write extracted frame')
                    label = Path('labels')/row['split']/(row['id']+'.txt')
                    write_labels(output/label,boxes)
                    manifest['frames'].append({**row,'file':relative.as_posix(),'label_file':label.as_posix(),
                        'decoded_seconds':actual,'image_sha256':digest(destination),
                        'reviewed':False,'review_status':'unreviewed','label_source':'weak_color_rules',
                        'active_by_rules':bool(screen.active),'boxes':boxes,'weak_boxes':boxes})
                    if (number+1)%50 == 0:
                        print(json.dumps({'decoded':number+1,'planned':len(plan),'saved':len(manifest['frames'])}),flush=True)
            finally:
                capture.release()
        manifest['status'] = 'completed'
    except BaseException:
        manifest['status'] = 'incomplete'
        save_json(output/'manifest.json',manifest)
        raise
    manifest['counts'] = dict(Counter(row['split'] for row in manifest['frames']))
    manifest['boxes_by_split'] = {split:dict(Counter(box['class'] for row in manifest['frames']
        if row['split']==split for box in row['boxes'])) for split in ('train','val','test')}
    save_json(output/'manifest.json',manifest)
    write_yaml(output)
    selected = select_reviews(manifest['frames'],'val')+select_reviews(manifest['frames'],'test')
    save_json(output/'review.json',{'classes':list(CLASSES),'crop_xyxy':list(CROP),
        'instructions':'Inspect the full cropped image, correct/add/remove boxes including false negatives, then set reviewed=true. Empty boxes require explicit review too.',
        'frames':selected})
    for split in ('val','test'):
        review_sheets(output,[row for row in selected if row['split']==split],split)
    print(json.dumps({'output':str(output),'counts':manifest['counts'],
                      'boxes_by_split':manifest['boxes_by_split'],'review_frames':len(selected)}),flush=True)
    return manifest


def apply_review(output, review_file, *, curated=None, reviewed_train_only=False):
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'completed':
        raise ValueError('Cannot curate incomplete extraction')
    review = json.loads(review_file.read_text(encoding='utf-8'))
    if review['classes'] != list(CLASSES) or review['crop_xyxy'] != list(CROP):
        raise ValueError('Review class/crop contract differs')
    originals = {row['id']:row for row in manifest['frames']}
    checked = {}
    excluded = set()
    used = set()
    for row in review['frames']:
        original = originals[row['id']]
        if row.get('review_status') == 'excluded' and original['split'] == 'train':
            excluded.add(row['id'])
            continue
        if row.get('reviewed') is not True:
            continue
        if row['id'] in used or row['file'] != original['file'] or row['split'] != original['split']:
            raise ValueError('Duplicate review or source/split changed')
        used.add(row['id'])
        if digest(output/original['file']) != original['image_sha256']:
            raise ValueError('Review source image changed')
        boxes = validate_boxes(row['boxes'])
        checked[row['id']] = {**original,'boxes':boxes,'reviewed':True,'review_status':'reviewed',
                             'label_source':'assistant_visual_review','review_notes':row.get('review_notes','')}
    if not all(any(row['split']==split for row in checked.values()) for split in ('val','test')):
        raise ValueError('Both val and test need explicitly reviewed images')
    curated = Path(curated) if curated is not None else output/'reviewed'
    if curated.exists():
        raise FileExistsError(f'Preserve existing reviewed dataset: {curated}')
    curated.mkdir()
    frames, selected = [], []
    for row in manifest['frames']:
        if row['id'] in excluded:
            continue
        if row['id'] in checked:
            selected.append(checked[row['id']])
        elif row['split'] == 'train' and not reviewed_train_only:
            # Chapter context improves weak labels, but does not make them truth.
            allowed = set(row.get('expected_families', CLASSES))
            blue = allowed.intersection(('shuriken', 'slash'))
            boxes = []
            for box in row['boxes']:
                kind = box['class']
                if kind in ('shuriken', 'slash') and len(blue) == 1:
                    kind = next(iter(blue))
                if kind in allowed:
                    boxes.append({**box, 'class': kind})
            selected.append({**row, 'boxes': boxes,
                             'label_source': 'weak_color_rules_with_chapter_context'})
    for row in selected:
        destination = curated/row['file']
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(output/row['file'],destination)
        write_labels(curated/row['label_file'],row['boxes'])
        frames.append(row)
    result = {**manifest,'frames':frames,'review_source':str(review_file.resolve()),
              'review_sha256':digest(review_file),'counts':dict(Counter(row['split'] for row in frames)),
              'boxes_by_split':{split:dict(Counter(box['class'] for row in frames
                  if row['split']==split for box in row['boxes'])) for split in ('train','val','test')},
              'label_source':('assistant-reviewed train/validation/test' if reviewed_train_only else
                              'chapter-filtered weak training with visual overrides; reviewed validation/test'),
              'reviewed_train_only':reviewed_train_only,
              'excluded_training_ids':sorted(excluded),
              'review_limits':'Assistant visual review, approximate visual boxes, no independent adjudication.'}
    save_json(curated/'manifest.json',result)
    write_yaml(curated)
    print(json.dumps({'reviewed_dataset':str(curated),'counts':result['counts']}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--catalog',type=Path,default=ROOT/'data/hayabusa-video-study/board-catalog.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data/hayabusa-detector-v1')
    parser.add_argument('--interval',type=float,default=.75)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--proxy',type=Path,help='Existing analysis proxy; original video is still hashed in provenance')
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--apply-review',type=Path)
    parser.add_argument('--curated-output',type=Path)
    parser.add_argument('--reviewed-train-only',action='store_true')
    args = parser.parse_args()
    if args.apply_review is not None:
        apply_review(args.output,args.apply_review,curated=args.curated_output,
                     reviewed_train_only=args.reviewed_train_only)
    elif args.plan:
        rows = sample_plan(json.loads(args.catalog.read_text(encoding='utf-8')),args.interval)
        print(json.dumps({'counts':dict(Counter(row['split'] for row in rows)),
            'groups':{split:sorted({row['group_id'] for row in rows if row['split']==split})
                      for split in ('train','val','test')}},indent=2))
    else:
        prepare(args.video,args.catalog,args.output,args.interval,args.cpu,args.proxy)


if __name__ == '__main__':
    main()
