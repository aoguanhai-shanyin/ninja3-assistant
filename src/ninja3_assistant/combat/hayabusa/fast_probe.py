"""Offline fast-path experiment. No game input; unknown is not battle evidence."""
from ninja3_assistant.paths import resource_root
from dataclasses import dataclass

import cv2
import numpy as np

from ninja3_assistant.combat.hayabusa.windup import WindupCue


ASSETS = resource_root()/'assets'


@dataclass(frozen=True)
class FastObservation:
    active: bool
    left_control: str
    right_control: str
    cue: WindupCue | None


class FastProbe:
    """Validate fixed UI regions on every frame; only cue-search with fresh player x."""
    def __init__(self, *, max_player_age):
        if max_player_age <= 0:
            raise ValueError('positive player-coordinate freshness required')
        self.max_player_age = max_player_age
        specs = {
            'name':((496,40),(.8,1,1.2)),
            'portrait':((414,43),(.8,1,1.2)),
            'shadow-name':((497,53),(.8,1,1.2)),
            'shadow-portrait':((413,60),(.8,1,1.2)),
            'down':((190,629),(.8,1,1.2)),
            'slide':((190,629),(.65,.8,1,1.2)),
            'right-up':((1095,627),(.8,1,1.2)),
            'right-jump':((1095,627),(.8,1,1.2)),
            'right-jump-live':((1095,627),(.9,1,1.1)),
        }
        self.templates = {}
        for name, (center, scales) in specs.items():
            filename = 'battle-slide.png' if name == 'slide' else f'hayabusa-{name}.png'
            source = cv2.imread(str(ASSETS/filename),cv2.IMREAD_GRAYSCALE)
            if source is None:
                raise FileNotFoundError(filename)
            self.templates[name] = (center,[cv2.resize(source,None,fx=s,fy=s) for s in scales])
        self.reset()

    def reset(self):
        self.previous = None
        self.previous_at = None
        self.fired = False

    def _match(self, gray, name, y_offset):
        (cx,cy), targets = self.templates[name]
        best = (0.,None,None)
        for target in targets:
            h,w = target.shape
            x0 = max(0, int(cx-w/2)-25)
            y0 = max(y_offset, int(cy-h/2)-25)
            x1 = min(1280, int(cx+w/2)+26)
            y1 = min(y_offset+gray.shape[0], int(cy+h/2)+26)
            roi = gray[y0-y_offset:y1-y_offset,x0:x1]
            if roi.shape[0]<h or roi.shape[1]<w:
                continue
            _,score,_,at = cv2.minMaxLoc(cv2.matchTemplate(roi,target,cv2.TM_CCOEFF_NORMED))
            if score > best[0]:
                patch = roi[at[1]:at[1]+h,at[0]:at[0]+w]
                error = float(np.abs(patch.astype(np.float32)-target).mean())
                best = (score,(x0+at[0]+w/2,y0+at[1]+h/2),error)
        return best

    def update(self, rgb, now, *, player_x=None, player_captured_at=None):
        if not isinstance(rgb,np.ndarray) or rgb.shape != (720,1280,3) or rgb.dtype != np.uint8:
            self.reset()
            return FastObservation(False,'unknown','unknown',None)
        top = cv2.cvtColor(rgb[:115],cv2.COLOR_RGB2GRAY)
        active = False
        for a,b,scores,error in (('name','portrait',(.83,.8),32),
                                  ('shadow-name','shadow-portrait',(.9,.9),25)):
            first,second = self._match(top,a,0),self._match(top,b,0)
            if first[0]>=scores[0] and second[0]>=scores[1] and first[2]<error and second[2]<error:
                active = True
                break
        if not active:
            self.reset()
            return FastObservation(False,'unknown','unknown',None)
        bottom = cv2.cvtColor(rgb[550:],cv2.COLOR_RGB2GRAY)
        down,slide = self._match(bottom,'down',550),self._match(bottom,'slide',550)
        left = down if down[0]>=slide[0] else slide
        if left[0]<.82 or left[2]>=35:
            self.reset()
            return FastObservation(False,'unknown','unknown',None)
        left_name = 'down' if left is down else 'slide'
        up = self._match(bottom,'right-up',550)
        jump = max(self._match(bottom,'right-jump',550),self._match(bottom,'right-jump-live',550),key=lambda v:v[0])
        right = up if up[0]>=jump[0] else jump
        right_name = ('up' if right is up else 'jump') if right[0]>=.84 and right[2]<35 else 'unknown'
        fresh = (player_x is not None and player_captured_at is not None and
                 0<=now-player_captured_at<=self.max_player_age and 0<=player_x<1280)
        if not fresh:
            self.previous = self.previous_at = None
            return FastObservation(True,left_name,right_name,None)
        return FastObservation(True,left_name,right_name,self._fire(rgb,now,player_x))

    def _fire(self, rgb, now, player_x):
        if self.fired:
            return None
        hsv = cv2.cvtColor(rgb[230:490],cv2.COLOR_RGB2HSV)
        red1 = cv2.inRange(hsv,(0,161,101),(6,255,255))
        red2 = cv2.inRange(hsv,(173,161,101),(179,255,255))
        if cv2.countNonZero(cv2.bitwise_or(red1,red2)) > .12*260*1280:
            self.previous = self.previous_at = None
            return None
        x0,x1 = ((max(170,int(player_x+350)),1250) if player_x<640
                 else (170,min(1250,int(player_x-350))))
        if x1<=x0:
            self.previous = self.previous_at = None
            return None
        mask = cv2.inRange(hsv[:,x0:x1],(2,65,170),(28,255,255))
        padded = cv2.copyMakeBorder(mask,70,70,90,90,cv2.BORDER_CONSTANT,value=0)
        density = cv2.boxFilter(padded,cv2.CV_32F,(180,140),normalize=False)
        _,value,_,loc = cv2.minMaxLoc(density)
        value /= 255.
        center = (loc[0]+x0-90,loc[1]+160)
        if value<60:
            self.previous = self.previous_at = None
            return None
        if (self.previous is None or self.previous_at is None or not 0<now-self.previous_at<=.45
                or abs(center[0]-self.previous[0])>100 or abs(center[1]-self.previous[1])>90):
            self.previous,self.previous_at = center,now
            return None
        first = self.previous_at
        self.fired = True
        return WindupCue('fire',None,now,first,center,int(round(value)))
