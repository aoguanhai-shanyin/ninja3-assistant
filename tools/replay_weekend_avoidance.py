"""Offline race perception benchmark; never connects to the game or sends input."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from weekend_race import RaceAvoider, RaceVision
from race_rope import RaceRopeTraversal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    video = cv2.VideoCapture(args.video)
    if not video.isOpened() or video.get(cv2.CAP_PROP_FPS) <= 0:
        raise RuntimeError('无法读取回放视频')
    cv2.setNumThreads(1)
    fps = video.get(cv2.CAP_PROP_FPS)
    avoider,vision,rope = RaceAvoider(),RaceVision(),RaceRopeTraversal()
    costs,rope_costs,rows = [],[],[]
    index = 0
    try:
        while True:
            okay,bgr = video.read()
            if not okay:
                break
            frame = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            kind = vision.detect(frame)
            now = index/fps
            point = None
            if kind == 'race' and not vision.on_rope and not vision.obscured:
                started = time.perf_counter()
                point = avoider.decide(frame,now)
                costs.append((time.perf_counter()-started)*1000)
                rope = RaceRopeTraversal()
            elif kind == 'race' and vision.on_rope and not vision.obscured:
                started = time.perf_counter()
                avoider.track_player(frame,now)
                actions = rope.decide(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY),now,
                                      avoider.player_x,avoider.player_y)
                if actions:
                    point = actions[0].point
                    rope.scheduled(now)
                rope_costs.append((time.perf_counter()-started)*1000)
            else:
                avoider = RaceAvoider()
                rope = RaceRopeTraversal()
            rows.append({'seconds':now,'scene':kind,'action':point,'hazard':avoider.hazard,
                         'on_rope':vision.on_rope,'upper_platform':rope.platform,
                         'player':[avoider.player_x,avoider.player_y],'ground':avoider.ground,
                         'objects':[{'kind':item.kind,'box':item.box} for item in avoider.detected]})
            if index % round(fps*2) == 0 and kind == 'race':
                for item in avoider.detected:
                    x,y,w,h = item.box
                    cv2.rectangle(bgr,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(bgr,f'{now:.1f}s {avoider.hazard} {point}',(200,100),0,.8,(0,0,255),2)
                cv2.imwrite(str(output/f'{index:05d}.jpg'),bgr)
            index += 1
    finally:
        video.release()
    summary = {'video':str(Path(args.video).resolve()),'video_fps':fps,'frames':index,
               'perception_samples':len(costs),'avoidance_ms':dict(zip(('p50','p95','max'),
                   map(float,np.percentile(costs,(50,95,100))))) if costs else None,
               'rope_samples':len(rope_costs),'rope_ms':dict(zip(('p50','p95','max'),
                   map(float,np.percentile(rope_costs,(50,95,100))))) if rope_costs else None,
               'live_gameplay':False,'accuracy_measured':False}
    (output/'replay.json').write_text(json.dumps({'summary':summary,'frames':rows},
                                               ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
