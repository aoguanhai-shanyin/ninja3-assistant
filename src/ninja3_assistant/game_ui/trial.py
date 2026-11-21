"""Read the trial page's rice-ball counter from the visible digits."""

import re

import cv2


def trial_counter_glyphs(frame):
    # The icon and add button are outside this box. Colored effects and the
    # dark backing of the counter are removed before scaling the glyphs.
    patch = frame[36:61,1174:1221]
    low, high = patch.min(axis=2), patch.max(axis=2)
    return ((low >= 180) & (high - low <= 35)).astype('uint8') * 255


def parse_repeated_counter(text):
    if text.strip() == '0':
        return 0
    count = re.fullmatch(r'\s*(\d+)\s+\1\s+\1\s*',text.replace('》',' '))
    return int(count[1]) if count else None


def read_trial_counter(frame, ocr, stop):
    foreground = trial_counter_glyphs(frame)
    x,y,width,height = cv2.boundingRect(foreground)
    if not width:
        return {'text':''}
    digits = foreground[y:y+height,x:x+width]

    # Windows OCR can omit narrow digits such as 11 at one scale and polarity.
    # Each rendering must still produce three identical whole numbers.
    for scale,background,padding in ((3,0,32),(5,255,24),(5,0,32)):
        image = 255 - digits if background else digits
        image = cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
        tile = cv2.copyMakeBorder(image,padding,padding,padding,padding,
                                  cv2.BORDER_CONSTANT,value=background)
        row = cv2.cvtColor(cv2.hconcat([tile]*3),cv2.COLOR_GRAY2RGB)
        result = ocr.read(row,stop=stop)
        if result is None or parse_repeated_counter(result.get('text','')) is not None:
            return result
    return result
