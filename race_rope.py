"""Rope traversal decisions; timing and cancellation use the shared kernel."""
from pathlib import Path
import sys

import cv2

from native_touch import TimedTap


class RaceRopeTraversal:
    _templates = None

    def __init__(self):
        self.active = False
        self.last_jump = float('-inf')
        self.platform = None

    @classmethod
    def upper_platform(cls, gray, player_x, player_y):
        if cls._templates is None:
            root = Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent))
            cls._templates = []
            for name in ('platform-grass','platform-snow'):
                original = cv2.imread(str(root/'assets/weekend-rope'/f'{name}.png'),0)
                if original is None:
                    raise RuntimeError('缺少绳索上方地板模板：'+name)
                for scale in (.85,1.,1.15):
                    cls._templates.append(cv2.resize(original,None,fx=scale/2,fy=scale/2,
                                                      interpolation=cv2.INTER_AREA))
        foot = min(580,max(260,player_y+90))
        left,right = max(100,player_x-60),min(1100,player_x+280)
        top,bottom = max(160,foot-280),foot-30
        area = cv2.resize(gray[top:bottom,left:right],None,fx=.5,fy=.5,
                          interpolation=cv2.INTER_AREA)
        candidates = []
        for template in cls._templates:
            h,w = template.shape
            if h >= area.shape[0] or w >= area.shape[1]:
                continue
            _,score,_,(x,y) = cv2.minMaxLoc(cv2.matchTemplate(area,template,cv2.TM_CCOEFF_NORMED))
            if score >= .82 and left+(x+w)*2 >= player_x and left+x*2 <= player_x+180:
                candidates.append((left+x*2,top+y*2,w*2,h*2))
        return max(candidates,key=lambda box:box[1]) if candidates else None

    def decide(self, gray, now, player_x, player_y):
        self.platform = self.upper_platform(gray,player_x,player_y)
        if now-self.last_jump < .48:
            return ()
        jumps = (TimedTap(15,(1090,630),35),)
        if self.platform is not None:
            jumps += (TimedTap(155,(1090,630),35),)
        return jumps

    def scheduled(self, now):
        self.last_jump = now
