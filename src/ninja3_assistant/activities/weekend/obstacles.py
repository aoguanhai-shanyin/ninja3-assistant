"""Positive obstacle recognition using observed foreground sprite appearances."""
from ninja3_assistant.paths import resource_root
from dataclasses import dataclass

import cv2


@dataclass(frozen=True, slots=True)
class RaceObstacle:
    kind: str
    box: tuple
    score: float


class RaceObstacleVision:
    def __init__(self):
        root = resource_root()
        self.templates = []
        for kind, foot in (('gate',52),('plant',57),('spike',0),('bomb',0)):
            original = cv2.imread(str(root/'assets/weekend-obstacles'/f'{kind}.png'),0)
            if original is None:
                raise RuntimeError('缺少周末活动障碍模板：'+kind)
            for scale in (.85,1.,1.15):
                template = cv2.resize(original,None,fx=scale/2,fy=scale/2,
                                      interpolation=cv2.INTER_AREA)
                self.templates.append((kind,template,round(foot*scale)))

    def detect(self, frame, player_x, ground):
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        left,right = max(100,player_x-40),min(1160,player_x+460)
        top,bottom = 180,min(580,(ground or 540)+35)
        region = cv2.resize(gray[top:bottom,left:right],None,fx=.5,fy=.5,
                            interpolation=cv2.INTER_AREA)
        found = []
        for kind,template,foot in self.templates:
            h,w = template.shape
            if h >= region.shape[0] or w >= region.shape[1]:
                continue
            scores = cv2.matchTemplate(region,template,cv2.TM_CCOEFF_NORMED)
            for _ in range(3):
                _,score,_,(x,y) = cv2.minMaxLoc(scores)
                if score < .82:
                    break
                box = (left+x*2,top+y*2,w*2,h*2+foot)
                floor = box[1]+box[3]
                if kind == 'bomb' or ground is None or abs(floor-ground) < 105:
                    if not any(old.kind == kind and abs(old.box[0]-box[0]) < 35 and
                               abs(old.box[1]-box[1]) < 35 for old in found):
                        found.append(RaceObstacle(kind,box,score))
                scores[max(0,y-h//2):y+h//2+1,max(0,x-w//2):x+w//2+1] = -1
        return tuple(sorted(found,key=lambda obstacle:obstacle.box[0]))
