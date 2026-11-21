"""Locate the close control on recognized promotional overlays."""

from functools import lru_cache

import cv2
import numpy as np

from ninja3_assistant.paths import resource_root
from ninja3_assistant.vision.control_match import match_control, text_foreground


@lru_cache(maxsize=2)
def _close_template(scale=1):
    path = resource_root()/'assets/xumi/tasks-close.png'
    template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
    if template is not None and scale != 1:
        template = cv2.resize(template, None, fx=scale, fy=scale)
    return template, text_foreground(template)


def _gift_close(frame):
    # Offer art varies. The dimmed side margins and light gold offer
    # panel are stable, including while the countdown ticks.
    for left, right in ((10, 110), (1200, 1270)):
        if (frame[225:585:4, left:right:4].max(axis=2) < 135).mean() < .88:
            return None
    panel = frame[170:285:3, 185:760:3].astype(np.int16)
    red, green, blue = panel.transpose(2, 0, 1)
    if ((red > 215) & (green > 145) & (red-blue > 20)).mean() < .55:
        return None

    template, mask = _close_template()
    point = match_control(frame, template, (990, 65, 1220, 223),
                          mask=mask, threshold=.95, max_error=40)
    if point is None:
        return None
    hsv = cv2.cvtColor(frame[255:605, 480:880], cv2.COLOR_RGB2HSV)
    fill = cv2.inRange(hsv, (19, 148, 180), (34, 255, 255))
    for contour in cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        _, _, width, height = cv2.boundingRect(contour)
        if 95 <= width <= 180 and 30 <= height <= 54 and cv2.contourArea(contour) > .8*width*height:
            return point
    return None


def promotion_popup_close(frame):
    if frame is None or frame.shape != (720, 1280, 3):
        return None
    # The Abyss season announcement is framed by two cyan rules across
    # the entire screen, with the upper rule partly covered by the title.
    rules = frame[[227, 491], 35:1170:2]
    cyan = ((rules[:, :, 0] < 105) & (rules[:, :, 1] > 165) & (rules[:, :, 2] > 235)).mean(axis=1)
    if cyan[0] > .14 and cyan[1] > .9:
        template, mask = _close_template(1.25)
        point = match_control(frame, template, (1177, 170, 1279, 278),
                              mask=mask, threshold=.95, max_error=45)
        if point is not None:
            return point
    return _gift_close(frame)
