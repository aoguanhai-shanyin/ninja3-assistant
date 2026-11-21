"""Match a control inside its region using only the pixels that identify it."""

import cv2
import numpy as np


def text_foreground(template):
    """Keep light glyphs and a narrow dark outline for shape contrast."""
    if template is None:
        return None
    low, high, _, _ = cv2.minMaxLoc(template)
    if high == low:
        return np.zeros_like(template)
    glyphs = cv2.inRange(template, low + (high - low) * .55, 255)
    return cv2.dilate(glyphs, np.ones((3, 3), dtype=np.uint8))


def match_control(frame, template, box, *, mask=None, threshold=.92, max_error=None):
    if template is None or (mask is not None and not cv2.countNonZero(mask)):
        return None
    x1, y1, x2, y2 = box
    region = frame[y1:y2, x1:x2]
    height, width = template.shape
    if region.shape[0] < height or region.shape[1] < width:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    if mask is not None:
        np.nan_to_num(scores, copy=False, nan=-1, posinf=-1, neginf=-1)
    _, score, _, (x, y) = cv2.minMaxLoc(scores)
    if score < threshold:
        return None
    if max_error is not None:
        error = cv2.absdiff(gray[y:y+height, x:x+width], template)
        if cv2.mean(error, mask=mask)[0] > max_error:
            return None
    return round(x1+x+width/2), round(y1+y+height/2)


def same_text(frame, fresh, box, *, max_error=8):
    """Recheck the glyphs of an OCR hit at its recorded pixel location."""
    if fresh.shape != frame.shape:
        return False
    x, y, width, height = box
    left, top, right, bottom = int(x), int(y), round(x+width), round(y+height)
    if not (0 <= left < right <= frame.shape[1] and 0 <= top < bottom <= frame.shape[0]):
        return False
    before = cv2.cvtColor(frame[top:bottom,left:right],cv2.COLOR_RGB2GRAY)
    after = cv2.cvtColor(fresh[top:bottom,left:right],cv2.COLOR_RGB2GRAY)
    mask = text_foreground(before)
    return bool(cv2.countNonZero(mask) and cv2.mean(cv2.absdiff(before,after),mask=mask)[0] <= max_error)
